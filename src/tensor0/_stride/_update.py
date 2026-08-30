"""Functional affine updates with an explicit preserved base operand."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from ._ffi import (
    _base_update_ffi_call_v2,
    _batch_only_named_sharding,
    _require_jax_array,
    native_available,
    strided_copy,
)
from ._errors import raise_no_eligible_route
from ._native_lowering import lower_plan
from ._plan import (
    CompleteMode,
    StridedOutputInit,
    StridedCopyPlan,
    StridedCopyRecord,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_strided_copy_plan,
)
from ._view import StridedView


@lru_cache(maxsize=1_024)
def _base_update_native_descriptor(plan: StridedCopyPlan) -> bytes | None:
    if (
        plan.write_kind is StridedWriteKind.ACCUMULATE
        and plan.source_dtype == "bool"
        and plan.result_dtype == "bool"
    ):
        return None
    try:
        lowered = lower_plan(plan)
    except ValueError:
        return None
    return lowered.descriptor


@lru_cache(maxsize=1_024)
def _build_strided_base_update_plan(
    *,
    sizes: tuple[int, ...],
    source_strides: tuple[int, ...],
    source_offset: int,
    destination_strides: tuple[int, ...],
    destination_offset: int,
    source_size: int,
    source_dtype: DTypeLike,
    base_size: int,
    base_dtype: DTypeLike,
    write_kind: StridedWriteKind,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        logical_shape=sizes,
        source_strides=source_strides,
        source_offset=source_offset,
        destination_strides=destination_strides,
        destination_offset=destination_offset,
        source_broadcast_axes=tuple(
            axis
            for axis, (size, stride) in enumerate(
                zip(sizes, source_strides, strict=True)
            )
            if size > 1 and stride == 0
        ),
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=base_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source_size,
        source_dtype=source_dtype,
        result_dtype=base_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=write_kind,
    )


def _prepare_strided_update(
    destination: StridedView,
    source: StridedView,
    write_kind: StridedWriteKind,
) -> tuple[Array, Array, StridedCopyPlan]:
    if not isinstance(destination, StridedView):
        raise TypeError("strided update destination must be a StridedView")
    if not isinstance(source, StridedView):
        raise TypeError("strided update source must be a StridedView")
    if source.sizes != destination.sizes:
        raise ValueError("strided update source and destination sizes must match")
    if source.batch_shape != destination.batch_shape:
        raise ValueError("strided update source and destination batch shapes must match")
    base_data = destination.data
    source_data = source.data
    bound = _build_strided_base_update_plan(
        sizes=destination.sizes,
        source_strides=source.strides,
        source_offset=source.offset,
        destination_strides=destination.strides,
        destination_offset=destination.offset,
        source_size=source.storage_size,
        source_dtype=source_data.dtype,
        base_size=destination.storage_size,
        base_dtype=base_data.dtype,
        write_kind=write_kind,
    )
    return base_data, source_data, bound


def _compile_base_update_plan(
    bound: StridedCopyPlan,
    write_kind: StridedWriteKind,
) -> StridedCopyPlan:
    if bound.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST:
        raise ValueError("base update requires static scale/cast semantics")
    if bound.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("base update does not admit reduction records")
    return build_strided_copy_plan(
        records=bound.records,
        output_size=bound.output_size,
        coverage=bound.coverage,
        source_size=bound.source_size,
        source_dtype=bound.source_dtype,
        result_dtype=bound.result_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=write_kind,
    )


@lru_cache(maxsize=1_024)
def compile_base_assign_plan(bound: StridedCopyPlan) -> StridedCopyPlan:
    """Freeze a validated unique affine plan for functional assignment."""

    return _compile_base_update_plan(bound, StridedWriteKind.ASSIGN)


@lru_cache(maxsize=1_024)
def compile_base_accumulate_plan(
    bound: StridedCopyPlan,
) -> StridedCopyPlan:
    """Freeze a validated unique affine plan for functional accumulation."""

    return _compile_base_update_plan(bound, StridedWriteKind.ACCUMULATE)


def _validate_operands(
    base: Array,
    source: Array,
    plan: StridedCopyPlan,
) -> None:
    if not base.shape or base.shape[-1] != plan.output_size:
        raise ValueError(
            f"expected base shape (*batch, {plan.output_size}), got {base.shape}"
        )
    if not source.shape or source.shape[-1] != plan.source_size:
        raise ValueError(
            f"expected source shape (*batch, {plan.source_size}), got {source.shape}"
        )
    if base.shape[:-1] != source.shape[:-1]:
        raise ValueError("base and source batch shapes must match")
    if base.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError(
            f"expected base dtype {plan.result_dtype}, got {base.dtype.name}"
        )
    if source.dtype != jnp.dtype(plan.source_dtype):
        raise TypeError(
            f"expected source dtype {plan.source_dtype}, got {source.dtype.name}"
        )


def _base_assign_abstract_eval(
    base_aval: Any,
    source_aval: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    if not base_aval.shape or base_aval.shape[-1] != plan.output_size:
        raise ValueError("base affine assignment size mismatch")
    if not source_aval.shape or source_aval.shape[-1] != plan.source_size:
        raise ValueError("source affine assignment size mismatch")
    if base_aval.shape[:-1] != source_aval.shape[:-1]:
        raise ValueError("base and source batch shapes must match")
    if base_aval.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError("base affine assignment dtype mismatch")
    if source_aval.dtype != jnp.dtype(plan.source_dtype):
        raise TypeError("source affine assignment dtype mismatch")
    return base_aval


def _partition_base_assign(
    plan: StridedCopyPlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    base_shape, source_shape = argument_shapes
    base_sharding = _batch_only_named_sharding(base_shape)
    source_sharding = _batch_only_named_sharding(source_shape)
    result_sharding = _batch_only_named_sharding(result_shape)
    return (
        mesh,
        lambda base, source: _execute_base_update_native(base, source, plan),
        result_sharding,
        (base_sharding, source_sharding),
    )


def _infer_base_assign_sharding(
    plan: StridedCopyPlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> Any:
    del plan, mesh, result_shape
    base_shape, _ = argument_shapes
    return _batch_only_named_sharding(base_shape)


def _propagate_base_assign_sharding(
    plan: StridedCopyPlan,
    mesh: Any,
    user_shape: Any,
) -> Any:
    del plan, mesh
    return _batch_only_named_sharding(user_shape)


def _create_partitioned_base_assign() -> Any | None:
    try:
        from jax.experimental.custom_partitioning import custom_partitioning
    except (AttributeError, ImportError):
        return None

    @partial(custom_partitioning, static_argnums=(2,))
    def partitioned(
        base: Array,
        source: Array,
        plan: StridedCopyPlan,
    ) -> Array:
        del source, plan
        return jnp.zeros_like(base)

    partitioned.def_partition(
        partition=_partition_base_assign,
        propagate_user_sharding=_propagate_base_assign_sharding,
        infer_sharding_from_operands=_infer_base_assign_sharding,
        decode_shardings=True,
        sharding_rule="... base, ... source -> ... result",
    )
    return partitioned


_PARTITIONED_BASE_ASSIGN = _create_partitioned_base_assign()


def _execute_base_update_native(
    base: Array,
    source: Array,
    plan: StridedCopyPlan,
) -> Array:
    descriptor = _base_update_native_descriptor(plan)
    if not native_available():
        raise_no_eligible_route(("native_base_update_executor_unavailable",))
    if descriptor is None:
        raise_no_eligible_route(("native_base_update_projection_unavailable",))
    return _base_update_ffi_call_v2(
        base,
        source,
        descriptor=descriptor,
        operation=plan.write_kind.value,
    )


def _base_assign_lowering(
    context: Any,
    base: Any,
    source: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count in (None, 1):
        function = lambda old, value: _execute_base_update_native(old, value, plan)
    else:
        partitioned = _PARTITIONED_BASE_ASSIGN
        if partitioned is None:
            raise RuntimeError(
                "multi-device BaseAssign requires JAX custom partitioning support"
            )
        function = lambda old, value: partitioned(old, value, plan)
    return mlir.lower_fun(function, multiple_results=False)(context, base, source)


def _base_assign_non_cpu_lowering(
    context: Any,
    base: Any,
    source: Any,
    *,
    plan: StridedCopyPlan,
) -> Any:
    del context, base, source, plan
    raise_no_eligible_route(("native_base_update_executor_non_cpu",))


def _base_assign_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Any, Any],
    *,
    plan: StridedCopyPlan,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    base, source = primals
    base_tangent, source_tangent = tangents
    primal = _BASE_ASSIGN_PRIMITIVE.bind(base, source, plan=plan)
    if not jnp.issubdtype(base.dtype, jnp.inexact):
        return primal, jnp.zeros(primal.shape, dtype=jax.dtypes.float0)
    if isinstance(base_tangent, ad.Zero):
        base_tangent = jnp.zeros_like(base)
    if isinstance(source_tangent, ad.Zero) or (
        getattr(source_tangent, "dtype", None) == jax.dtypes.float0
    ):
        source_tangent = jnp.zeros_like(source)
    return (
        primal,
        _BASE_ASSIGN_PRIMITIVE.bind(
            base_tangent,
            source_tangent,
            plan=plan,
        ),
    )


def _base_assign_transpose(
    cotangent: Any,
    base: Any,
    source: Any,
    *,
    plan: StridedCopyPlan,
) -> list[Any]:
    from jax.interpreters import ad

    base_undefined = ad.is_undefined_primal(base)
    source_undefined = ad.is_undefined_primal(source)
    if isinstance(cotangent, ad.Zero):
        return [
            ad.Zero(base.aval.to_ct_aval()) if base_undefined else None,
            ad.Zero(source.aval.to_ct_aval()) if source_undefined else None,
        ]

    base_cotangent = None
    if base_undefined:
        if plan.write_kind is StridedWriteKind.ASSIGN:
            zero_plan = _zero_destination_base_assign_plan(plan)
            source_shape = (
                source.aval.shape if source_undefined else source.shape
            )
            zero_source = jnp.ones(
                source_shape,
                dtype=jnp.dtype(plan.source_dtype),
            )
            base_cotangent = _BASE_ASSIGN_PRIMITIVE.bind(
                cotangent,
                zero_source,
                plan=zero_plan,
            )
        else:
            base_cotangent = cotangent
    source_cotangent = None
    if source_undefined:
        source_dtype = jnp.dtype(plan.source_dtype)
        if jnp.issubdtype(source_dtype, jnp.inexact):
            source_primal = jnp.zeros(source.aval.shape, dtype=source_dtype)
            source_cotangent = jax.linear_transpose(
                lambda value: strided_copy(
                    value,
                    plan=_fresh_map_projection(plan),
                ),
                source_primal,
            )(cotangent)[0]
        else:
            source_cotangent = ad.Zero(source.aval.to_ct_aval())
    return [base_cotangent, source_cotangent]


@lru_cache(maxsize=1_024)
def _zero_destination_base_assign_plan(
    plan: StridedCopyPlan,
) -> StridedCopyPlan:
    affine = build_strided_copy_plan(
        records=tuple(replace(record, scale=0) for record in plan.records),
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )
    return compile_base_assign_plan(affine)


@lru_cache(maxsize=1_024)
def _fresh_map_projection(plan: StridedCopyPlan) -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=plan.records,
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )


def _base_assign_batch(
    arguments: tuple[Array, Array],
    dimensions: tuple[int | None, int | None],
    *,
    plan: StridedCopyPlan,
) -> tuple[Array, int]:
    from jax.interpreters import batching

    base, source = arguments
    base_dimension, source_dimension = dimensions
    batch_size = next(
        argument.shape[dimension]
        for argument, dimension in zip(arguments, dimensions, strict=True)
        if dimension is not None
    )
    base = batching.bdim_at_front(base, base_dimension, batch_size)
    source = batching.bdim_at_front(source, source_dimension, batch_size)
    return _BASE_ASSIGN_PRIMITIVE.bind(base, source, plan=plan), 0


def _create_base_assign_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad
    from jax.interpreters import batching
    from jax.interpreters import mlir
    from jax.interpreters import xla

    primitive = core.Primitive("tensor0_stride_base_update")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_base_assign_abstract_eval)
    ad.primitive_jvps[primitive] = _base_assign_jvp
    ad.primitive_transposes[primitive] = _base_assign_transpose
    batching.primitive_batchers[primitive] = _base_assign_batch
    mlir.register_lowering(primitive, _base_assign_non_cpu_lowering)
    mlir.register_lowering(primitive, _base_assign_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_BASE_ASSIGN_PRIMITIVE = _create_base_assign_primitive()


def base_assign(
    base: object,
    source: object,
    *,
    plan: StridedCopyPlan,
) -> Array:
    """Functionally assign one certified affine map over a preserved base."""

    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ASSIGN
        or plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("base_assign requires an assign update plan")
    base_data = _require_jax_array(base, "base_assign base")
    source_data = _require_jax_array(source, "base_assign source")
    _validate_operands(base_data, source_data, plan)
    return _BASE_ASSIGN_PRIMITIVE.bind(base_data, source_data, plan=plan)


def base_accumulate(
    base: object,
    source: object,
    *,
    plan: StridedCopyPlan,
) -> Array:
    """Functionally add one certified affine map over a preserved base."""

    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ACCUMULATE
        or plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("base_accumulate requires an accumulate update plan")
    base_data = _require_jax_array(base, "base_accumulate base")
    source_data = _require_jax_array(source, "base_accumulate source")
    _validate_operands(base_data, source_data, plan)
    return _BASE_ASSIGN_PRIMITIVE.bind(base_data, source_data, plan=plan)


def strided_assign(
    destination: StridedView,
    source: StridedView,
) -> Array:
    """Assign one static affine subblock over a preserved base."""

    base_data, source_data, bound = _prepare_strided_update(
        destination,
        source,
        StridedWriteKind.ASSIGN,
    )
    return base_assign(
        base_data,
        source_data,
        plan=bound,
    )


def strided_accumulate(
    destination: StridedView,
    source: StridedView,
) -> Array:
    """Accumulate one static affine subblock over a preserved base."""

    base_data, source_data, bound = _prepare_strided_update(
        destination,
        source,
        StridedWriteKind.ACCUMULATE,
    )
    return base_accumulate(
        base_data,
        source_data,
        plan=bound,
    )


__all__ = [
    "base_accumulate",
    "base_assign",
    "compile_base_accumulate_plan",
    "compile_base_assign_plan",
    "strided_accumulate",
    "strided_assign",
]
