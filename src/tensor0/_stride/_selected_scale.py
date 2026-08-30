"""Dynamic scaling over one statically certified affine selection."""

from __future__ import annotations

from functools import lru_cache, partial
from math import prod
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from ._compiler import compile_plan
from ._errors import raise_no_eligible_route
from ._ffi import (
    _batch_only_named_sharding,
    _require_jax_array,
    _selected_scale_ffi_call_v2,
    native_available,
    strided_copy,
)
from ._native_lowering import lower_plan
from ._plan import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    _contiguous_strides,
    build_strided_copy_plan,
)
from ._stablehlo import (
    AffineIntervalProof,
    CompactAffineInterval,
    StableHloProofError,
    _extract_affine_interval,
    _write_affine_interval,
    compile_affine_interval_proof,
    compile_compact_affine_interval,
)
from ._view import StridedView


@lru_cache(maxsize=1_024)
def _selected_scale_native_descriptor(
    plan: StridedCopyPlan,
) -> bytes | None:
    try:
        return lower_plan(plan).descriptor
    except ValueError:
        return None


def _asarray_factor(value: object, dtype: Any) -> Array:
    if isinstance(value, Array) and value.dtype == dtype:
        return value
    return jnp.asarray(value, dtype=dtype)


@lru_cache(maxsize=1_024)
def compile_selected_scale_plan(
    bound: StridedCopyPlan,
) -> StridedCopyPlan:
    """Validate that an affine plan represents identity-address selection."""

    if bound.source_size != bound.output_size:
        raise ValueError("selected scale requires equal source/result storage sizes")
    if bound.source_dtype != bound.result_dtype:
        raise ValueError("selected scale requires one base dtype")
    if bound.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST:
        raise ValueError("selected scale input requires static scale semantics")
    if bound.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("selected scale does not admit reduction records")
    dtype = jnp.dtype(bound.result_dtype)
    for index, record in enumerate(bound.records):
        if (
            record.source_strides != record.destination_strides
            or record.source_offset != record.destination_offset
        ):
            raise ValueError(
                f"record[{index}] selected scale addresses must be identical"
            )
        scale = np.asarray(record.scale, dtype=dtype)
        if scale.shape != () or not bool(scale == np.asarray(1, dtype=dtype)):
            raise ValueError(
                f"record[{index}] selected scale requires static unit scale"
            )
    return build_strided_copy_plan(
        records=bound.records,
        output_size=bound.output_size,
        coverage=bound.coverage,
        source_size=bound.source_size,
        source_dtype=bound.source_dtype,
        result_dtype=bound.result_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
        scalar_kind=StridedScalarKind.DYNAMIC_SCALE,
    )


def _validate_selected_scale_operands(
    base: Array,
    factor: Array,
    plan: StridedCopyPlan,
) -> None:
    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ASSIGN
        or plan.scalar_kind is not StridedScalarKind.DYNAMIC_SCALE
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("selected_scale requires a dynamic-scale update plan")
    if not base.shape or base.shape[-1] != plan.output_size:
        raise ValueError(
            f"expected base shape (*batch, {plan.output_size}), got {base.shape}"
        )
    if base.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError(
            f"expected base dtype {plan.result_dtype}, got {base.dtype.name}"
        )
    if factor.dtype != base.dtype:
        raise TypeError("selected scale factor must use the base dtype")
    if factor.shape not in ((), base.shape[:-1]):
        raise ValueError(
            "selected scale factor must be scalar or match the base batch shape"
        )


@lru_cache(maxsize=1_024)
def _compact_selected_scale_interval(
    plan: StridedCopyPlan,
) -> CompactAffineInterval | None:
    try:
        return compile_compact_affine_interval(compile_plan(plan))
    except StableHloProofError:
        return None


@lru_cache(maxsize=1_024)
def _affine_selected_scale_proof(
    plan: StridedCopyPlan,
) -> AffineIntervalProof | None:
    compiled = compile_plan(plan)
    if len(compiled.records) != 1 or compiled.records[0].logical_elements == 0:
        return None
    record = compiled.records[0]
    try:
        return compile_affine_interval_proof(
            record.logical_shape,
            record.destination_strides,
            offset=record.destination_offset,
            storage_size=plan.output_size,
            field_name="selection",
        )
    except StableHloProofError:
        return None


def _has_selected_scale_structured_hlo(plan: StridedCopyPlan) -> bool:
    return (
        _compact_selected_scale_interval(plan) is not None
        or _affine_selected_scale_proof(plan) is not None
    )


def _execute_selected_scale_structured_hlo(
    base: Array,
    factor: Array,
    plan: StridedCopyPlan,
) -> Array:
    interval = _compact_selected_scale_interval(plan)
    if interval is not None:
        selected = jax.lax.slice_in_dim(
            base,
            interval.destination_offset,
            interval.destination_offset + interval.element_count,
            axis=base.ndim - 1,
        )
        expanded = factor if factor.ndim == 0 else factor[..., None]
        return jax.lax.dynamic_update_slice_in_dim(
            base,
            expanded * selected,
            interval.destination_offset,
            axis=base.ndim - 1,
        )
    proof = _affine_selected_scale_proof(plan)
    if proof is None:
        raise RuntimeError("selected scale has no address-free HLO proof")
    offset = plan.records[0].destination_offset
    selected = _extract_affine_interval(base, offset=offset, proof=proof)
    expanded = factor if factor.ndim == 0 else factor[..., None]
    if proof.logical_shape:
        expanded = jnp.reshape(
            expanded,
            (*factor.shape, *((1,) * len(proof.logical_shape))),
        )
    return _write_affine_interval(
        base,
        expanded * selected,
        offset=offset,
        proof=proof,
    )


@jax.jit
def _selected_scale_jvp_values(
    selected_base: Array,
    selected_base_tangent: Array,
    factor: Array,
    factor_tangent: Array,
) -> Array:
    expanded_factor = factor if factor.ndim == 0 else factor[..., None]
    expanded_tangent = (
        factor_tangent
        if factor_tangent.ndim == 0
        else factor_tangent[..., None]
    )
    return (
        expanded_tangent * selected_base
        + expanded_factor * selected_base_tangent
    )


def _selected_scale_factor_cotangent(
    selected_cotangent: Array,
    selected_base: Array,
) -> Array:
    return jnp.sum(selected_cotangent * selected_base, axis=-1)


def _selected_scale_abstract_eval(
    base_aval: Any,
    factor_aval: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    if not base_aval.shape or base_aval.shape[-1] != plan.output_size:
        raise ValueError("selected scale base size mismatch")
    if base_aval.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError("selected scale base dtype mismatch")
    if factor_aval.dtype != base_aval.dtype:
        raise TypeError("selected scale factor dtype mismatch")
    if factor_aval.shape not in ((), base_aval.shape[:-1]):
        raise ValueError("selected scale factor shape mismatch")
    return base_aval


def _factor_named_sharding(shape: Any) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec

    sharding = getattr(shape, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError(
            "Tensor0 selected scale requires NamedSharding for multi-device input"
        )
    rank = len(shape.shape)
    specification = tuple(sharding.spec)
    if len(specification) > rank:
        raise ValueError("Tensor0 selected scale received invalid factor sharding")
    normalized = specification + (None,) * (rank - len(specification))
    return NamedSharding(sharding.mesh, PartitionSpec(*normalized))


def _partition_selected_scale(
    plan: StridedCopyPlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    base_shape, factor_shape = argument_shapes
    base_sharding = _batch_only_named_sharding(base_shape)
    factor_sharding = _factor_named_sharding(factor_shape)
    result_sharding = _batch_only_named_sharding(result_shape)
    return (
        mesh,
        lambda base, factor: _execute_selected_scale_native(base, factor, plan),
        result_sharding,
        (base_sharding, factor_sharding),
    )


def _infer_selected_scale_sharding(
    plan: StridedCopyPlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> Any:
    del plan, mesh, result_shape
    base_shape, _ = argument_shapes
    return _batch_only_named_sharding(base_shape)


def _propagate_selected_scale_sharding(
    plan: StridedCopyPlan,
    mesh: Any,
    user_shape: Any,
) -> Any:
    del plan, mesh
    return _batch_only_named_sharding(user_shape)


def _create_partitioned_selected_scale() -> Any | None:
    try:
        from jax.experimental.custom_partitioning import custom_partitioning
    except (AttributeError, ImportError):
        return None

    @partial(custom_partitioning, static_argnums=(2,))
    def partitioned(
        base: Array,
        factor: Array,
        plan: StridedCopyPlan,
    ) -> Array:
        del factor, plan
        return jnp.zeros_like(base)

    partitioned.def_partition(
        partition=_partition_selected_scale,
        propagate_user_sharding=_propagate_selected_scale_sharding,
        infer_sharding_from_operands=_infer_selected_scale_sharding,
        decode_shardings=True,
        sharding_rule="... storage, ... -> ... storage",
        need_replication_factors=("storage",),
    )
    return partitioned


_PARTITIONED_SELECTED_SCALE = _create_partitioned_selected_scale()


def _execute_selected_scale_native(
    base: Array,
    factor: Array,
    plan: StridedCopyPlan,
) -> Array:
    descriptor = _selected_scale_native_descriptor(plan)
    if not native_available():
        raise_no_eligible_route(("native_selected_scale_executor_unavailable",))
    if descriptor is None:
        raise_no_eligible_route(("native_selected_scale_projection_unavailable",))
    return _selected_scale_ffi_call_v2(
        base,
        factor,
        descriptor=descriptor,
    )


def _selected_scale_lowering(
    context: Any,
    base: Any,
    factor: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count in (None, 1):
        function = lambda old, value: _execute_selected_scale_native(
            old,
            value,
            plan,
        )
    else:
        partitioned = _PARTITIONED_SELECTED_SCALE
        if partitioned is None:
            raise RuntimeError(
                "multi-device SelectedScale requires custom partitioning support"
            )
        function = lambda old, value: partitioned(old, value, plan)
    return mlir.lower_fun(function, multiple_results=False)(context, base, factor)


def _selected_scale_non_cpu_lowering(
    context: Any,
    base: Any,
    factor: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    del context, base, factor, plan
    raise_no_eligible_route(("native_selected_scale_executor_non_cpu",))


def _selected_scale_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Any, Any],
    *,
    plan: StridedCopyPlan,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    base, factor = primals
    base_tangent, factor_tangent = tangents
    primal = _SELECTED_SCALE_PRIMITIVE.bind(base, factor, plan=plan)
    if not jnp.issubdtype(base.dtype, jnp.inexact):
        return primal, jnp.zeros(primal.shape, dtype=jax.dtypes.float0)
    base_zero = isinstance(base_tangent, ad.Zero)
    factor_zero = isinstance(factor_tangent, ad.Zero)
    if base_zero and factor_zero:
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    if _has_selected_scale_structured_hlo(plan):
        tangent = jax.jvp(
            lambda old, value: _execute_selected_scale_structured_hlo(
                old,
                value,
                plan,
            ),
            (base, factor),
            (
                jnp.zeros_like(base) if base_zero else base_tangent,
                jnp.zeros_like(factor) if factor_zero else factor_tangent,
            ),
        )[1]
        return primal, tangent
    if factor_zero:
        assert not base_zero
        tangent = _SELECTED_SCALE_PRIMITIVE.bind(
            base_tangent,
            factor,
            plan=plan,
        )
    else:
        from ._update import base_assign

        compact = _compact_selected_values_plan(plan)
        selected_base = strided_copy(base, plan=compact)
        if base_zero:
            destination_base = jnp.zeros_like(base)
            expanded_tangent = (
                factor_tangent
                if factor_tangent.ndim == 0
                else factor_tangent[..., None]
            )
            selected_tangent = expanded_tangent * selected_base
        else:
            selected_base_tangent = strided_copy(base_tangent, plan=compact)
            selected_tangent = _selected_scale_jvp_values(
                selected_base,
                selected_base_tangent,
                factor,
                factor_tangent,
            )
            destination_base = base_tangent
        tangent = base_assign(
            destination_base,
            selected_tangent,
            plan=_compact_selected_assign_plan(plan),
        )
    return primal, tangent


def _selected_scale_transpose(
    cotangent: Any,
    base: Any,
    factor: Any,
    *,
    plan: StridedCopyPlan,
) -> list[Any]:
    from jax.interpreters import ad

    base_undefined = ad.is_undefined_primal(base)
    factor_undefined = ad.is_undefined_primal(factor)
    if base_undefined and factor_undefined:
        raise ValueError("SelectedScale is not jointly linear in base and factor")
    if isinstance(cotangent, ad.Zero):
        return [
            ad.Zero(base.aval.to_ct_aval()) if base_undefined else None,
            ad.Zero(factor.aval.to_ct_aval()) if factor_undefined else None,
        ]
    if _has_selected_scale_structured_hlo(plan):
        if base_undefined:
            base_primal = jnp.zeros(base.aval.shape, dtype=base.aval.dtype)
            base_cotangent = jax.linear_transpose(
                lambda old: _execute_selected_scale_structured_hlo(
                    old,
                    factor,
                    plan,
                ),
                base_primal,
            )(cotangent)[0]
            return [base_cotangent, None]
        if factor_undefined:
            factor_primal = jnp.zeros(
                factor.aval.shape,
                dtype=factor.aval.dtype,
            )
            factor_cotangent = jax.linear_transpose(
                lambda value: _execute_selected_scale_structured_hlo(
                    base,
                    value,
                    plan,
                ),
                factor_primal,
            )(cotangent)[0]
            return [None, factor_cotangent]
    if base_undefined:
        base_cotangent = _SELECTED_SCALE_PRIMITIVE.bind(
            cotangent,
            factor,
            plan=plan,
        )
        return [base_cotangent, None]
    if factor_undefined:
        compact = _compact_selected_values_plan(plan)
        selected_base = strided_copy(base, plan=compact)
        selected_cotangent = strided_copy(cotangent, plan=compact)
        factor_cotangent = _selected_scale_factor_cotangent(
            selected_cotangent,
            selected_base,
        )
        if factor.aval.shape == ():
            factor_cotangent = jnp.sum(factor_cotangent)
        return [None, factor_cotangent]
    return [None, None]


@lru_cache(maxsize=1_024)
def _compact_selected_values_plan(
    plan: StridedCopyPlan,
) -> StridedCopyPlan:
    destination_offset = 0
    records: list[StridedCopyRecord] = []
    for record in plan.records:
        records.append(
            StridedCopyRecord(
                logical_shape=record.logical_shape,
                source_strides=record.source_strides,
                source_offset=record.source_offset,
                destination_strides=_contiguous_strides(record.logical_shape),
                destination_offset=destination_offset,
            )
        )
        destination_offset += prod(record.logical_shape)
    return build_strided_copy_plan(
        records=tuple(records),
        output_size=destination_offset,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )


@lru_cache(maxsize=1_024)
def _compact_selected_assign_plan(plan: StridedCopyPlan) -> Any:
    from ._update import compile_base_assign_plan

    source_offset = 0
    records: list[StridedCopyRecord] = []
    for record in plan.records:
        records.append(
            StridedCopyRecord(
                logical_shape=record.logical_shape,
                source_strides=_contiguous_strides(record.logical_shape),
                source_offset=source_offset,
                destination_strides=record.destination_strides,
                destination_offset=record.destination_offset,
            )
        )
        source_offset += prod(record.logical_shape)
    affine = build_strided_copy_plan(
        records=tuple(records),
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=source_offset,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )
    return compile_base_assign_plan(affine)


def _selected_scale_batch(
    arguments: tuple[Array, Array],
    dimensions: tuple[int | None, int | None],
    *,
    plan: StridedCopyPlan,
) -> tuple[Array, int]:
    from jax.interpreters import batching

    base, factor = arguments
    base_dimension, factor_dimension = dimensions
    batch_size = next(
        argument.shape[dimension]
        for argument, dimension in zip(arguments, dimensions, strict=True)
        if dimension is not None
    )
    base = batching.bdim_at_front(base, base_dimension, batch_size)
    factor = batching.bdim_at_front(factor, factor_dimension, batch_size)
    return _SELECTED_SCALE_PRIMITIVE.bind(base, factor, plan=plan), 0


def _create_selected_scale_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_selected_scale")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_selected_scale_abstract_eval)
    ad.primitive_jvps[primitive] = _selected_scale_jvp
    ad.primitive_transposes[primitive] = _selected_scale_transpose
    batching.primitive_batchers[primitive] = _selected_scale_batch
    mlir.register_lowering(primitive, _selected_scale_non_cpu_lowering)
    mlir.register_lowering(primitive, _selected_scale_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_SELECTED_SCALE_PRIMITIVE = _create_selected_scale_primitive()


def selected_scale(
    base: object,
    factor: object,
    *,
    plan: StridedCopyPlan,
) -> Array:
    """Functionally scale certified base addresses by a dynamic operand."""

    base_data = _require_jax_array(base, "selected_scale base")
    factor_data = _asarray_factor(factor, base_data.dtype)
    _validate_selected_scale_operands(base_data, factor_data, plan)
    return _SELECTED_SCALE_PRIMITIVE.bind(base_data, factor_data, plan=plan)


@lru_cache(maxsize=1_024)
def _build_strided_scale_plan(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    source_size: int,
    dtype_name: str,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        logical_shape=sizes,
        source_strides=strides,
        source_offset=offset,
        destination_strides=strides,
        destination_offset=offset,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=source_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source_size,
        source_dtype=dtype_name,
        result_dtype=dtype_name,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
        scalar_kind=StridedScalarKind.DYNAMIC_SCALE,
    )


def strided_scale(
    view: StridedView,
    factor: object,
) -> Array:
    """Scale one static affine subblock by a dynamic operand."""

    if not isinstance(view, StridedView):
        raise TypeError("strided_scale requires a StridedView")
    return selected_scale(
        view.data,
        factor,
        plan=_build_strided_scale_plan(
            view.sizes,
            view.strides,
            view.offset,
            view.storage_size,
            view.data.dtype.name,
        ),
    )


__all__ = [
    "compile_selected_scale_plan",
    "selected_scale",
    "strided_scale",
]
