"""Selected-scale kernel preparation and the restricted alias mechanism."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from ._update_ad import _execute_scale_tangent, _execute_update_tangent
from ._dot import _compact_projection_plan, native_projection_dotu
from .._errors import raise_no_eligible_route
from .._jax import require_jax_array
from .._map import _execute_map
from .._scalar import product_dtype
from .._native import (
    _selected_scale_alias_ffi_call,
    native_available,
)
from .._native_descriptor import lower_plan
from .._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
)


@lru_cache(maxsize=1_024)
def _selected_scale_native_descriptor(
    plan: AffinePlan,
    dynamic_dtype: str | None = None,
) -> bytes | None:
    try:
        return lower_plan(plan, dynamic_dtype=dynamic_dtype).descriptor
    except ValueError:
        return None


@lru_cache(maxsize=1_024)
def build_selected_scale_plan(
    plan: AffinePlan,
) -> AffinePlan:
    """Validate that an affine plan represents identity-address selection."""

    if plan.source_size != plan.output_size:
        raise ValueError("selected scale requires equal source/result storage sizes")
    if plan.source_dtype != plan.result_dtype:
        raise ValueError("selected scale requires one base dtype")
    if plan.scalar_kind is not StridedScalarKind.STATIC_SCALE_CAST:
        raise ValueError("selected scale input requires static scale semantics")
    if plan.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("selected scale does not admit reduction records")
    for index, record in enumerate(plan.records):
        if (
            record.source_strides != record.destination_strides
            or record.source_offset != record.destination_offset
        ):
            raise ValueError(
                f"record[{index}] selected scale addresses must be identical"
            )
        if record.scale is not None:
            raise ValueError(
                f"record[{index}] selected scale requires an identity mapping"
            )
    return build_affine_plan(
        records=plan.records,
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
        scalar_kind=StridedScalarKind.DYNAMIC_SCALE,
    )


def _validate_selected_scale_operands(
    base: Array,
    factor: Array,
    plan: AffinePlan,
) -> None:
    if (
        plan.output_init is not StridedOutputInit.PRESERVE_BASE
        or plan.write_kind is not StridedWriteKind.ASSIGN
        or plan.scalar_kind is not StridedScalarKind.DYNAMIC_SCALE
        or plan.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("alias scale requires a dynamic-scale update plan")
    if not base.shape or base.shape[-1] != plan.output_size:
        raise ValueError(
            f"expected base shape (*batch, {plan.output_size}), got {base.shape}"
        )
    if base.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError(
            f"expected base dtype {plan.result_dtype}, got {base.dtype.name}"
        )
    if factor.shape not in ((), base.shape[:-1]):
        raise ValueError(
            "selected scale factor must be scalar or match the base batch shape"
        )


@lru_cache(maxsize=1_024)
def _selected_scale_axpby_plan(plan: AffinePlan) -> AffinePlan:
    return build_affine_plan(
        records=plan.records,
        output_size=plan.output_size,
        coverage=plan.coverage,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
    )


def _selected_scale_abstract_eval(
    base_aval: Any,
    factor_aval: Any,
    *,
    plan: AffinePlan,
) -> Any:
    if not base_aval.shape or base_aval.shape[-1] != plan.output_size:
        raise ValueError("selected scale base size mismatch")
    if base_aval.dtype != jnp.dtype(plan.result_dtype):
        raise TypeError("selected scale base dtype mismatch")
    if factor_aval.shape not in ((), base_aval.shape[:-1]):
        raise ValueError("selected scale factor shape mismatch")
    return base_aval


def _execute_selected_scale_alias_native(
    base: Array,
    factor: Array,
    plan: AffinePlan,
) -> Array:
    descriptor = _selected_scale_native_descriptor(plan, product_dtype(base, factor).name)
    if not native_available():
        raise_no_eligible_route(("native_selected_scale_executor_unavailable",))
    if descriptor is None:
        raise_no_eligible_route(("native_selected_scale_projection_unavailable",))
    return _selected_scale_alias_ffi_call(base, factor, descriptor=descriptor)


def _selected_scale_lowering(
    context: Any,
    base: Any,
    factor: Any,
    *,
    plan: AffinePlan,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count not in (None, 1):
        raise RuntimeError("alias SelectedScale is restricted to one CPU device")
    function = lambda old, value: _execute_selected_scale_alias_native(
        old, value, plan,
    )
    return mlir.lower_fun(function, multiple_results=False)(context, base, factor)


def _selected_scale_non_cpu_lowering(
    context: Any,
    base: Any,
    factor: Any,
    *,
    plan: AffinePlan,
) -> Any:
    del context, base, factor, plan
    raise_no_eligible_route(("native_selected_scale_executor_non_cpu",))


def _selected_scale_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Any, Any],
    *,
    plan: AffinePlan,
    primitive: Any,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    base, factor = primals
    base_tangent, factor_tangent = tangents
    primal = primitive.bind(base, factor, plan=plan)
    if not jnp.issubdtype(base.dtype, jnp.inexact):
        return primal, jnp.zeros(primal.shape, dtype=jax.dtypes.float0)
    base_zero = isinstance(base_tangent, ad.Zero)
    factor_zero = isinstance(factor_tangent, ad.Zero)
    if base_zero and factor_zero:
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    if factor_zero:
        assert not base_zero
        tangent = _execute_scale_tangent(
            base_tangent,
            factor,
            plan=plan,
        )
    else:
        destination_base = jnp.zeros_like(base) if base_zero else base_tangent
        tangent = _execute_update_tangent(
            destination_base,
            base,
            factor_tangent,
            factor,
            plan=_selected_scale_axpby_plan(plan),
        )
    return primal, tangent


def _selected_scale_transpose(
    cotangent: Any,
    base: Any,
    factor: Any,
    *,
    plan: AffinePlan,
    primitive: Any,
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
    if base_undefined:
        base_cotangent = _execute_scale_tangent(
            cotangent,
            factor,
            plan=plan,
        )
        return [base_cotangent, None]
    if factor_undefined:
        if (plan.result_dtype in ("float32", "complex64") and native_available()
                and factor.aval.dtype == base.dtype):
            factor_cotangent = native_projection_dotu(cotangent, base, plan)
        else:
            compact = _compact_selected_values_plan(plan)
            selected_base = _execute_map(base, plan=compact)
            selected_cotangent = _execute_map(cotangent, plan=compact)
            factor_shape = factor.aval.shape
            def expression(coefficient):
                expanded = coefficient if not factor_shape else coefficient[..., None]
                return jnp.asarray(expanded * selected_base, dtype=base.dtype)
            factor_cotangent = jax.linear_transpose(
                expression, jnp.zeros(factor_shape, dtype=factor.aval.dtype),
            )(selected_cotangent)[0]
        if factor.aval.shape == ():
            factor_cotangent = jnp.sum(factor_cotangent)
        return [None, factor_cotangent]
    return [None, None]


@lru_cache(maxsize=1_024)
def _compact_selected_values_plan(
    plan: AffinePlan,
) -> AffinePlan:
    return _compact_projection_plan(plan)


def _selected_scale_batch(
    arguments: tuple[Array, Array],
    dimensions: tuple[int | None, int | None],
    *,
    plan: AffinePlan,
    primitive: Any,
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
    return primitive.bind(base, factor, plan=plan), 0


def _create_selected_scale_alias_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_selected_scale_alias")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_selected_scale_abstract_eval)
    ad.primitive_jvps[primitive] = partial(
        _selected_scale_jvp,
        primitive=primitive,
    )
    ad.primitive_transposes[primitive] = partial(
        _selected_scale_transpose,
        primitive=primitive,
    )
    batching.primitive_batchers[primitive] = partial(
        _selected_scale_batch,
        primitive=primitive,
    )
    mlir.register_lowering(primitive, _selected_scale_non_cpu_lowering)
    mlir.register_lowering(
        primitive,
        _selected_scale_lowering,
        platform="cpu",
    )
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_SELECTED_SCALE_ALIAS_PRIMITIVE = _create_selected_scale_alias_primitive()


def _selected_scale_alias(
    base: object,
    factor: object,
    *,
    plan: AffinePlan,
) -> Array:
    """Execute the internal one-device alias mechanism."""

    base_data = require_jax_array(base, "alias scale base")
    factor_data = jnp.asarray(factor)
    _validate_selected_scale_operands(base_data, factor_data, plan)
    return _SELECTED_SCALE_ALIAS_PRIMITIVE.bind(
        base_data,
        factor_data,
        plan=plan,
    )


@lru_cache(maxsize=1_024)
def _build_strided_scale_plan(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    source_size: int,
    dtype_name: str,
) -> AffinePlan:
    record = AffineRecord(
        logical_shape=sizes,
        source_strides=strides,
        source_offset=offset,
        destination_strides=strides,
        destination_offset=offset,
    )
    return build_affine_plan(
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


__all__ = [
    "build_selected_scale_plan",
]
