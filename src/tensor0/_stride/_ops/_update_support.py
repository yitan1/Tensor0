"""Update plan preparation, specialized lowering, and mapped scatters."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from typing import Any

import jax
from jax import Array
from jax.core import ShapedArray
import jax.numpy as jnp
from jax.typing import DTypeLike

from .._jax import require_jax_array
from .._map import _execute_map
from .._errors import raise_no_eligible_route
from .._plan import (
    CompleteMode,
    StridedOutputInit,
    AffinePlan,
    AffineRecord,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
)
from .._view import StridedView


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
) -> AffinePlan:
    record = AffineRecord(
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
    return build_affine_plan(
        records=(record,),
        output_size=base_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source_size,
        source_dtype=source_dtype,
        result_dtype=base_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
    )


def _prepare_strided_update(
    destination: StridedView,
    source: StridedView,
) -> tuple[Array, Array, AffinePlan]:
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
    plan = _build_strided_base_update_plan(
        sizes=destination.sizes,
        source_strides=source.strides,
        source_offset=source.offset,
        destination_strides=destination.strides,
        destination_offset=destination.offset,
        source_size=source.storage_size,
        source_dtype=source_data.dtype,
        base_size=destination.storage_size,
        base_dtype=base_data.dtype,
    )
    return base_data, source_data, plan


def _build_base_update_plan(
    plan: AffinePlan,
    write_kind: StridedWriteKind,
) -> AffinePlan:
    if plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST:
        raise ValueError("base update requires static scale/cast semantics")
    if plan.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("base update does not admit reduction records")
    return build_affine_plan(
        records=plan.records,
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=write_kind,
    )


@lru_cache(maxsize=1_024)
def build_base_assign_plan(plan: AffinePlan) -> AffinePlan:
    """Freeze a validated unique affine plan for functional assignment."""

    return _build_base_update_plan(plan, StridedWriteKind.ASSIGN)


@lru_cache(maxsize=1_024)
def build_base_accumulate_plan(
    plan: AffinePlan,
) -> AffinePlan:
    """Freeze a validated unique affine plan for functional accumulation."""

    return _build_base_update_plan(plan, StridedWriteKind.ACCUMULATE)


def _validate_operands(
    base: Array | ShapedArray,
    source: Array | ShapedArray,
    plan: AffinePlan,
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
    plan: AffinePlan,
) -> Any:
    _validate_operands(base_aval, source_aval, plan)
    return base_aval


def _base_assign_lowering(
    context: Any,
    base: Any,
    source: Any,
    *,
    plan: AffinePlan,
) -> Any:
    from jax.interpreters import mlir
    from ._update import _execute_update

    function = lambda old, value: _execute_update(
        old, value, source_factor=1, base_factor=0, plan=plan,
    )
    return mlir.lower_fun(function, multiple_results=False)(context, base, source)


def _base_assign_non_cpu_lowering(
    context: Any,
    base: Any,
    source: Any,
    *,
    plan: AffinePlan,
) -> Any:
    del context, base, source, plan
    raise_no_eligible_route(("native_base_update_executor_non_cpu",))


def _base_assign_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Any, Any],
    *,
    plan: AffinePlan,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    base, source = primals
    base_tangent, source_tangent = tangents
    primal = _SCATTER_PRIMITIVE.bind(base, source, plan=plan)
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
        _SCATTER_PRIMITIVE.bind(
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
    plan: AffinePlan,
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
            base_cotangent = _SCATTER_PRIMITIVE.bind(
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
                lambda value: _execute_map(
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
    plan: AffinePlan,
) -> AffinePlan:
    affine = build_affine_plan(
        records=tuple(replace(record, scale=0) for record in plan.records),
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )
    return build_base_assign_plan(affine)


@lru_cache(maxsize=1_024)
def _fresh_map_projection(plan: AffinePlan) -> AffinePlan:
    return build_affine_plan(
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
    plan: AffinePlan,
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
    return _SCATTER_PRIMITIVE.bind(base, source, plan=plan), 0


def _create_base_assign_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad
    from jax.interpreters import batching
    from jax.interpreters import mlir
    from jax.interpreters import xla

    primitive = core.Primitive("tensor0_stride_update_scatter")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_base_assign_abstract_eval)
    ad.primitive_jvps[primitive] = _base_assign_jvp
    ad.primitive_transposes[primitive] = _base_assign_transpose
    batching.primitive_batchers[primitive] = _base_assign_batch
    mlir.register_lowering(primitive, _base_assign_non_cpu_lowering)
    mlir.register_lowering(primitive, _base_assign_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_SCATTER_PRIMITIVE = _create_base_assign_primitive()


def _scatter_mapped(
    base: object,
    source: object,
    *,
    plan: AffinePlan,
) -> Array:
    """Functionally assign one certified affine map over a preserved base."""

    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ASSIGN
        or plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("_scatter_mapped requires an assign update plan")
    base_data = require_jax_array(base, "_scatter_mapped base")
    source_data = require_jax_array(source, "_scatter_mapped source")
    _validate_operands(base_data, source_data, plan)
    return _SCATTER_PRIMITIVE.bind(base_data, source_data, plan=plan)


__all__ = [
    "_scatter_mapped",
    "build_base_accumulate_plan",
    "build_base_assign_plan",
]
