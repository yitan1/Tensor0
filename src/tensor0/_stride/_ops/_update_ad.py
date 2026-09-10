"""Differential expressions and fused linearized update execution."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from math import prod
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from ._dot import native_projection_dotu
from .._errors import raise_no_eligible_route
from .._jax import batch_only_named_sharding, batch_update_operands, require_jax_array
from .._native import _update_tangent_ffi_call, native_available
from .._native_descriptor import lower_plan
from .._plan import (
    AffinePlan,
    AffineRecord,
    CompleteMode,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
    contiguous_strides,
)
from .._map import _execute_map
from .._scalar import product_dtype, update_dtypes
from ._update_support import _scatter_mapped, build_base_assign_plan


_AXPBY_DTYPES = frozenset(
    ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")
)


def _validate_axpby_plan(plan: AffinePlan) -> None:
    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ASSIGN
        or plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("AXPBY requires a static affine assignment plan")
    if any(record.scale is not None for record in plan.records):
        raise ValueError("fused differential updates require a unit mapping scale")
    if plan.source_dtype != plan.result_dtype:
        raise TypeError("AXPBY requires one same-dtype element domain")
    if plan.result_dtype not in _AXPBY_DTYPES:
        raise TypeError("AXPBY requires a supported inexact dtype")


def _validate_axpby_operands(
    base: Any,
    source: Any,
    alpha: Any,
    beta: Any,
    plan: AffinePlan,
) -> None:
    _validate_axpby_plan(plan)
    if not base.shape or base.shape[-1] != plan.output_size:
        raise ValueError("AXPBY base size mismatch")
    if not source.shape or source.shape[-1] != plan.source_size:
        raise ValueError("AXPBY source size mismatch")
    if base.shape[:-1] != source.shape[:-1]:
        raise ValueError("AXPBY base and source batch shapes must match")
    dtype = jnp.dtype(plan.result_dtype)
    if base.dtype != dtype or source.dtype != dtype:
        raise TypeError("AXPBY array dtype mismatch")
    for name, coefficient in (("alpha", alpha), ("beta", beta)):
        if coefficient.shape not in ((), base.shape[:-1]):
            raise ValueError(
                f"AXPBY {name} must be scalar or match the batch shape"
            )


@lru_cache(maxsize=1_024)
def _record_differential_plans(plan: AffinePlan):
    result = []
    for record in plan.records:
        selection = build_base_assign_plan(build_affine_plan(
            records=(record,), source_size=plan.source_size, output_size=plan.output_size,
            source_dtype=plan.source_dtype, result_dtype=plan.result_dtype,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        ))
        source = build_affine_plan(
            records=(replace(record, scale=None,
                             destination_strides=contiguous_strides(record.logical_shape),
                             destination_offset=0),),
            source_size=plan.source_size, output_size=prod(record.logical_shape),
            source_dtype=plan.source_dtype, result_dtype=plan.source_dtype,
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        result.append((source, _compact_base_plan(selection), _compact_assign_plan(selection)))
    return tuple(result)


@lru_cache(maxsize=1_024)
def _compact_base_plan(plan: AffinePlan) -> AffinePlan:
    destination_offset = 0
    records: list[AffineRecord] = []
    for record in plan.records:
        records.append(
            AffineRecord(
                logical_shape=record.logical_shape,
                source_strides=record.destination_strides,
                source_offset=record.destination_offset,
                destination_strides=contiguous_strides(record.logical_shape),
                destination_offset=destination_offset,
            )
        )
        destination_offset += prod(record.logical_shape)
    return build_affine_plan(
        records=tuple(records),
        output_size=destination_offset,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=plan.output_size,
        source_dtype=plan.result_dtype,
        result_dtype=plan.result_dtype,
    )


@lru_cache(maxsize=1_024)
def _compact_assign_plan(plan: AffinePlan) -> AffinePlan:
    source_offset = 0
    records: list[AffineRecord] = []
    for record in plan.records:
        records.append(
            AffineRecord(
                logical_shape=record.logical_shape,
                source_strides=contiguous_strides(record.logical_shape),
                source_offset=source_offset,
                destination_strides=record.destination_strides,
                destination_offset=record.destination_offset,
            )
        )
        source_offset += prod(record.logical_shape)
    affine = build_affine_plan(
        records=tuple(records),
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=source_offset,
        source_dtype=plan.result_dtype,
        result_dtype=plan.result_dtype,
    )
    return build_base_assign_plan(affine)


def _reference_axpby(
    base: Array,
    source: Array,
    alpha: Array,
    beta: Array,
    plan: AffinePlan,
) -> Array:
    alpha, beta = jnp.asarray(alpha), jnp.asarray(beta)
    expanded_beta = beta if beta.ndim == 0 else beta[..., None]
    result = base
    for record, (source_plan, base_plan, assign_plan) in zip(
        plan.records, _record_differential_plans(plan), strict=True,
    ):
        selected_source = _execute_map(source, plan=source_plan)
        coefficient = alpha if record.scale is None else jnp.multiply(alpha, record.scale)
        expanded_coefficient = coefficient if coefficient.ndim == 0 else coefficient[..., None]
        selected_base = _execute_map(base, plan=base_plan)
        selected_result = jnp.asarray(
            expanded_coefficient * selected_source + expanded_beta * selected_base,
            dtype=base.dtype,
        )
        result = _scatter_mapped(result, selected_result, plan=assign_plan)
    return result


def _axpby_abstract_eval(
    base_aval: Any,
    source_aval: Any,
    alpha_aval: Any,
    beta_aval: Any,
    *,
    plan: AffinePlan,
) -> Any:
    _validate_axpby_operands(
        base_aval,
        source_aval,
        alpha_aval,
        beta_aval,
        plan,
    )
    return base_aval


def _execute_update_tangent_native(
    base: Array,
    source: Array,
    alpha: Array,
    beta: Array,
    plan: AffinePlan,
) -> Array:
    if not native_available():
        raise_no_eligible_route(("native_axpby_executor_unavailable",))
    stages = update_dtypes(base, source, alpha, beta, plan.records)
    return _update_tangent_ffi_call(
        base,
        source,
        alpha,
        beta,
        descriptor=lower_plan(plan, coefficient_dtypes=tuple(stage[0] for stage in stages)).descriptor,
        stages=stages,
    )


def _coefficient_named_sharding(shape: Any) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec

    sharding = getattr(shape, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError("Tensor0 AXPBY requires NamedSharding for coefficients")
    rank = len(shape.shape)
    specification = tuple(sharding.spec)
    if len(specification) > rank:
        raise ValueError("Tensor0 AXPBY received invalid coefficient sharding")
    normalized = specification + (None,) * (rank - len(specification))
    return NamedSharding(sharding.mesh, PartitionSpec(*normalized))


def _partition_axpby(
    plan: AffinePlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    base_shape, source_shape, alpha_shape, beta_shape = argument_shapes
    return (
        mesh,
        lambda base, source, alpha, beta: _execute_update_tangent_native(
            base,
            source,
            alpha,
            beta,
            plan,
        ),
        batch_only_named_sharding(result_shape),
        (
            batch_only_named_sharding(base_shape),
            batch_only_named_sharding(source_shape),
            _coefficient_named_sharding(alpha_shape),
            _coefficient_named_sharding(beta_shape),
        ),
    )


def _infer_axpby_sharding(
    plan: AffinePlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> Any:
    del plan, mesh, result_shape
    return batch_only_named_sharding(argument_shapes[0])


def _propagate_axpby_sharding(plan: AffinePlan, mesh: Any, user_shape: Any) -> Any:
    del plan, mesh
    return batch_only_named_sharding(user_shape)


def _differential_sharding_rule(plan, mesh, value_types, result_types):
    from ._update import _sharding_rule

    return _sharding_rule(plan, None, None, mesh, value_types, result_types)


def _create_partitioned_axpby() -> Any | None:
    try:
        from jax.experimental.custom_partitioning import custom_partitioning
    except (AttributeError, ImportError):
        return None

    @partial(custom_partitioning, static_argnums=(4,))
    def partitioned(
        base: Array,
        source: Array,
        alpha: Array,
        beta: Array,
        plan: AffinePlan,
    ) -> Array:
        del source, alpha, beta, plan
        return jnp.zeros_like(base)

    partitioned.def_partition(
        partition=_partition_axpby,
        propagate_user_sharding=_propagate_axpby_sharding,
        infer_sharding_from_operands=_infer_axpby_sharding,
        decode_shardings=True,
        sharding_rule=_differential_sharding_rule,
    )
    return partitioned


_PARTITIONED_AXPBY = _create_partitioned_axpby()


def _axpby_lowering(
    context: Any,
    base: Any,
    source: Any,
    alpha: Any,
    beta: Any,
    *,
    plan: AffinePlan,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count in (None, 1):
        function = lambda dst, src, lhs, rhs: _execute_update_tangent_native(
            dst,
            src,
            lhs,
            rhs,
            plan,
        )
    else:
        partitioned = _PARTITIONED_AXPBY
        if partitioned is None:
            raise RuntimeError("multi-device AXPBY requires custom partitioning")
        function = lambda dst, src, lhs, rhs: partitioned(
            dst,
            src,
            lhs,
            rhs,
            plan,
        )
    return mlir.lower_fun(function, multiple_results=False)(
        context,
        base,
        source,
        alpha,
        beta,
    )


def _axpby_non_cpu_lowering(
    context: Any,
    base: Any,
    source: Any,
    alpha: Any,
    beta: Any,
    *,
    plan: AffinePlan,
) -> Any:
    del context, base, source, alpha, beta, plan
    raise_no_eligible_route(("native_axpby_executor_non_cpu",))


def _axpby_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Any, Any, Any, Any],
    *,
    plan: AffinePlan,
) -> tuple[Array, Array]:
    from jax.interpreters import ad

    base, source, alpha, beta = primals
    primal = _AXPBY_PRIMITIVE.bind(base, source, alpha, beta, plan=plan)
    concrete_tangents = tuple(ad.instantiate_zeros(value) for value in tangents)
    _, tangent = jax.jvp(
        lambda dst, src, lhs, rhs: _reference_axpby(
            dst,
            src,
            lhs,
            rhs,
            plan,
        ),
        primals,
        concrete_tangents,
    )
    return primal, tangent


def _axpby_transpose(
    cotangent: Any,
    base: Any,
    source: Any,
    alpha: Any,
    beta: Any,
    *,
    plan: AffinePlan,
) -> list[Any]:
    from jax.interpreters import ad

    arguments = (base, source, alpha, beta)
    undefined = tuple(ad.is_undefined_primal(value) for value in arguments)
    if undefined[1] and undefined[2]:
        raise ValueError("AXPBY is not jointly linear in source and alpha")
    if undefined[0] and undefined[3]:
        raise ValueError("AXPBY is not jointly linear in base and beta")
    if isinstance(cotangent, ad.Zero):
        return [
            ad.Zero(value.aval.to_ct_aval()) if is_undefined else None
            for value, is_undefined in zip(arguments, undefined, strict=True)
        ]

    if (
        undefined == (False, False, True, False)
        and plan.source_dtype == plan.result_dtype
        and plan.result_dtype in ("float32", "complex64")
        and plan.source_size == plan.output_size
        and all(
            record.source_strides == record.destination_strides
            and record.source_offset == record.destination_offset
            for record in plan.records
        )
        and all(record.scale is None for record in plan.records)
        and all(argument.aval.dtype == jnp.dtype(plan.result_dtype) if missing
                else argument.dtype == jnp.dtype(plan.result_dtype)
                for argument, missing in zip(arguments, undefined, strict=True))
        and native_available()
    ):
        alpha_cotangent = native_projection_dotu(cotangent, source, plan)
        if alpha.aval.shape == ():
            alpha_cotangent = jnp.sum(alpha_cotangent)
        return [None, None, alpha_cotangent, None]

    resolved = tuple(
        jnp.zeros(value.aval.shape, dtype=value.aval.dtype)
        if is_undefined
        else value
        for value, is_undefined in zip(arguments, undefined, strict=True)
    )
    cotangents: list[Any] = []
    for index, (value, is_undefined) in enumerate(
        zip(arguments, undefined, strict=True)
    ):
        if not is_undefined:
            cotangents.append(None)
            continue
        zero = resolved[index]

        def linear(argument: Array, *, position: int = index) -> Array:
            operands = list(resolved)
            operands[position] = argument
            return _reference_axpby(*operands, plan=plan)

        cotangents.append(jax.linear_transpose(linear, zero)(cotangent)[0])
    return cotangents


def _axpby_batch(
    arguments: tuple[Array, Array, Array, Array],
    dimensions: tuple[int | None, int | None, int | None, int | None],
    *,
    plan: AffinePlan,
) -> tuple[Array, int]:
    batched = batch_update_operands(arguments, dimensions)
    return _AXPBY_PRIMITIVE.bind(*batched, plan=plan), 0


def _create_axpby_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_update_tangent")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_axpby_abstract_eval)
    ad.primitive_jvps[primitive] = _axpby_jvp
    ad.primitive_transposes[primitive] = _axpby_transpose
    batching.primitive_batchers[primitive] = _axpby_batch
    mlir.register_lowering(primitive, _axpby_non_cpu_lowering)
    mlir.register_lowering(primitive, _axpby_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_AXPBY_PRIMITIVE = _create_axpby_primitive()


def _execute_update_tangent(
    base: object,
    source: object,
    alpha: object,
    beta: object,
    *,
    plan: AffinePlan,
) -> Array:
    """Fuse a differential expression without coefficient-value short circuits."""

    base_data = require_jax_array(base, "AXPBY base")
    source_data = require_jax_array(source, "AXPBY source")
    alpha_data = jnp.asarray(alpha)
    beta_data = jnp.asarray(beta)
    _validate_axpby_operands(base_data, source_data, alpha_data, beta_data, plan)
    return _AXPBY_PRIMITIVE.bind(
        base_data,
        source_data,
        alpha_data,
        beta_data,
        plan=plan,
    )


def _scale_tangent_native(base, factor, *, plan):
    from .._native import _scale_tangent_ffi_call

    return _scale_tangent_ffi_call(
        base, factor, descriptor=lower_plan(plan, dynamic_dtype=product_dtype(base, factor).name).descriptor,
    )


def _scale_tangent_partition(plan, mesh, argument_shapes, result_shape):
    return (
        mesh, partial(_scale_tangent_native, plan=plan),
        batch_only_named_sharding(result_shape),
        (batch_only_named_sharding(argument_shapes[0]),
         _coefficient_named_sharding(argument_shapes[1])),
    )


def _scale_tangent_sharding_rule(plan, mesh, value_types, result_types):
    from jaxlib.mlir import ir

    del plan, mesh, result_types
    ranks = tuple(ir.RankedTensorType(value).rank for value in value_types)
    batch_axes = tuple(f"batch{axis}" for axis in range(ranks[0] - 1))
    base = " ".join((*batch_axes, "storage"))
    factor = " ".join(batch_axes) if ranks[1] else ""
    return (f"{base}, {factor} -> {base}",
            {"need_replication_factors": ("storage",)})


def _create_partitioned_scale_tangent():
    from jax.experimental.custom_partitioning import custom_partitioning

    @partial(custom_partitioning, static_argnums=(2,))
    def partitioned(base, factor, plan):
        return _scale_tangent_native(base, factor, plan=plan)

    partitioned.def_partition(
        partition=_scale_tangent_partition,
        infer_sharding_from_operands=_infer_axpby_sharding,
        decode_shardings=True,
        sharding_rule=_scale_tangent_sharding_rule,
    )
    return partitioned


_PARTITIONED_SCALE_TANGENT = _create_partitioned_scale_tangent()


def _scale_tangent_lowering(context, base, factor, *, plan):
    from jax.interpreters import mlir

    count = getattr(context.module_context.axis_context, "num_devices", None)
    function = (partial(_scale_tangent_native, plan=plan) if count in (None, 1)
                else lambda data, value: _PARTITIONED_SCALE_TANGENT(data, value, plan))
    return mlir.lower_fun(function, multiple_results=False)(context, base, factor)


def _scale_tangent_abstract(base, factor, *, plan):
    from ._selected_scale import _validate_selected_scale_operands

    _validate_selected_scale_operands(base, factor, plan)
    return base


def _scale_tangent_jvp(primals, tangents, *, plan):
    from ._selected_scale import _selected_scale_jvp

    return _selected_scale_jvp(
        primals, tangents, plan=plan, primitive=_SCALE_TANGENT_PRIMITIVE,
    )


def _scale_tangent_transpose(cotangent, base, factor, *, plan):
    from ._selected_scale import _selected_scale_transpose

    return _selected_scale_transpose(
        cotangent, base, factor, plan=plan, primitive=_SCALE_TANGENT_PRIMITIVE,
    )


def _scale_tangent_batch(arguments, dimensions, *, plan):
    from jax.interpreters import batching

    base, factor = arguments
    base_axis, factor_axis = dimensions
    size = next(value.shape[axis] for value, axis in zip(arguments, dimensions)
                if axis is not None)
    base = batching.bdim_at_front(base, base_axis, size)
    if factor_axis is not None or factor.ndim:
        factor = batching.bdim_at_front(factor, factor_axis, size)
        factor = jnp.broadcast_to(
            factor.reshape((size,) + (1,) * (base.ndim - factor.ndim - 1)
                           + factor.shape[1:]), base.shape[:-1],
        )
    return _SCALE_TANGENT_PRIMITIVE.bind(base, factor, plan=plan), 0


def _create_scale_tangent_primitive():
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_scale_tangent")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_scale_tangent_abstract)
    ad.primitive_jvps[primitive] = _scale_tangent_jvp
    ad.primitive_transposes[primitive] = _scale_tangent_transpose
    batching.primitive_batchers[primitive] = _scale_tangent_batch
    mlir.register_lowering(primitive, _scale_tangent_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_SCALE_TANGENT_PRIMITIVE = _create_scale_tangent_primitive()


def _execute_scale_tangent(base, factor, *, plan):
    return _SCALE_TANGENT_PRIMITIVE.bind(base, factor, plan=plan)


__all__ = ["_execute_update_tangent", "_execute_scale_tangent"]
