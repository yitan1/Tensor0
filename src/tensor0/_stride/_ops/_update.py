"""One forward update boundary with an independent differential rule."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from .._jax import batch_only_named_sharding, batch_update_operands, require_jax_array
from .._scalar import forward_update_dtypes
from .._native import _BASE_UPDATE_SUFFIXES, _SAME_DTYPE_SUFFIXES, _update_ffi_call, native_available
from .._errors import raise_no_eligible_route
from .._native_descriptor import lower_plan
from .._plan import AffinePlan, StridedScalarKind
from ._update_ad import (
    _AXPBY_DTYPES,
    _coefficient_named_sharding,
    _execute_update_tangent,
    _reference_axpby,
)
from ._update_support import (
    _validate_operands,
    build_base_assign_plan,
)


def _native_update(base, source, source_factor, base_factor, *, plan,
                   fixed_factors, identity_source):
    if not native_available():
        raise_no_eligible_route(("native_update_executor_unavailable",))
    del fixed_factors, identity_source
    stages = forward_update_dtypes(base, source, source_factor, base_factor, plan.records)
    return _update_ffi_call(
        base, source, source_factor, base_factor,
        descriptor=lower_plan(plan, coefficient_dtypes=tuple(stage[0] for stage in stages)).descriptor,
        stages=stages,
    )

def _abstract(base, source, source_factor, base_factor, *, plan, **metadata):
    del metadata
    _validate_operands(base, source, plan)
    for factor in (source_factor, base_factor):
        if factor.dtype.name not in _SAME_DTYPE_SUFFIXES:
            raise TypeError("unsupported update coefficient dtype")
        if factor.shape not in ((), base.shape[:-1]):
            raise ValueError("update coefficient must be scalar or match the batch shape")
    return base


def _partition(plan, fixed_factors, identity_source, mesh, argument_shapes,
               result_shape):
    return (
        mesh,
        partial(_native_update, plan=plan, fixed_factors=fixed_factors,
                identity_source=identity_source),
        batch_only_named_sharding(result_shape),
        tuple(batch_only_named_sharding(shape) if index < 2
              else _coefficient_named_sharding(shape)
              for index, shape in enumerate(argument_shapes)),
    )


def _infer(plan, fixed_factors, identity_source, mesh, argument_shapes, result_shape):
    del plan, fixed_factors, identity_source, mesh, result_shape
    return batch_only_named_sharding(argument_shapes[0])


def _sharding_rule(plan, fixed_factors, identity_source, mesh, value_types, result_types):
    from jaxlib.mlir import ir

    del plan, fixed_factors, identity_source, mesh, result_types
    ranks = tuple(ir.RankedTensorType(value).rank for value in value_types)
    batch_axes = tuple(f"batch{axis}" for axis in range(ranks[0] - 1))
    base = " ".join((*batch_axes, "base"))
    source = " ".join((*batch_axes, "source"))
    factors = tuple(" ".join(batch_axes) if rank else "" for rank in ranks[2:])
    return (f"{base}, {source}, {factors[0]}, {factors[1]} -> {base}",
            {"need_replication_factors": ("base", "source")})


def _partitioned_update():
    from jax.experimental.custom_partitioning import custom_partitioning

    @partial(custom_partitioning, static_argnums=(4, 5, 6))
    def partitioned(base, source, source_factor, base_factor, plan,
                    fixed_factors, identity_source):
        return _native_update(
            base, source, source_factor, base_factor, plan=plan,
            fixed_factors=fixed_factors, identity_source=identity_source,
        )

    partitioned.def_partition(
        partition=_partition, infer_sharding_from_operands=_infer,
        decode_shardings=True,
        sharding_rule=_sharding_rule,
    )
    return partitioned


_PARTITIONED_UPDATE = _partitioned_update()


def _lowering(context, *arguments, **metadata):
    from jax.interpreters import mlir

    count = getattr(context.module_context.axis_context, "num_devices", None)
    if count in (None, 1):
        function = partial(_native_update, **metadata)
    else:
        function = lambda *values: _PARTITIONED_UPDATE(
            *values, metadata["plan"], metadata["fixed_factors"],
            metadata["identity_source"],
        )
    return mlir.lower_fun(function, multiple_results=False)(context, *arguments)


def _jvp(primals, tangents, **metadata):
    from jax.interpreters import ad

    primal = _UPDATE_PRIMITIVE.bind(*primals, **metadata)
    if not jnp.issubdtype(primal.dtype, jnp.inexact):
        return primal, jnp.zeros(primal.shape, dtype=jax.dtypes.float0)
    base, source, source_factor, base_factor = primals
    plan = metadata["plan"]
    fixed = metadata["fixed_factors"]
    if (metadata["identity_source"] and fixed[1] == 0
            and plan.result_dtype in _AXPBY_DTYPES):
        from ._selected_scale import build_selected_scale_plan

        base_tangent = tangents[0]
        if isinstance(base_tangent, ad.Zero) and isinstance(tangents[2], ad.Zero):
            return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
        from ._update_ad import _execute_scale_tangent

        if isinstance(tangents[2], ad.Zero):
            tangent = _execute_scale_tangent(
                base_tangent, source_factor, plan=build_selected_scale_plan(plan),
            )
        else:
            tangent = _execute_update_tangent(
                ad.instantiate_zeros(base_tangent), source, tangents[2],
                source_factor, plan=plan,
            )
    else:
        active = tuple(index for index, value in enumerate(tangents)
                       if not isinstance(value, ad.Zero)
                       and getattr(value, "dtype", None) != jax.dtypes.float0)
        if not active:
            return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())

        def reference(*values):
            arguments = list(primals)
            for index, value in zip(active, values, strict=True):
                arguments[index] = value
            return _reference_axpby(*arguments, plan=plan)

        _, tangent = jax.jvp(
            reference, tuple(primals[index] for index in active),
            tuple(tangents[index] for index in active),
        )
    return primal, tangent


def _transpose(cotangent, *arguments, **metadata):
    from ._update_ad import _axpby_transpose

    return _axpby_transpose(cotangent, *arguments, plan=metadata["plan"])


def _batch(arguments, dimensions, **metadata):
    values = batch_update_operands(arguments, dimensions)
    return _UPDATE_PRIMITIVE.bind(*values, **metadata), 0


def _create_primitive():
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_update")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_abstract)
    ad.primitive_jvps[primitive] = _jvp
    ad.primitive_transposes[primitive] = _transpose
    batching.primitive_batchers[primitive] = _batch
    mlir.register_lowering(primitive, _non_cpu_lowering)
    mlir.register_lowering(primitive, _lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


def _non_cpu_lowering(*arguments, **metadata):
    del arguments, metadata
    raise_no_eligible_route(("native_update_executor_non_cpu",))


_UPDATE_PRIMITIVE = _create_primitive()


def _execute_update(base: object, source: object, *, source_factor: object,
                    base_factor: object, plan: AffinePlan) -> Array:
    """Update selected storage; zero terms are absent and unit terms copied."""

    base_data = require_jax_array(base, "update base")
    source_data = require_jax_array(source, "update source")
    if plan.scalar_kind is StridedScalarKind.DYNAMIC_SCALE:
        plan = replace(plan, scalar_kind=StridedScalarKind.STATIC_SCALE_CAST)
    plan = build_base_assign_plan(plan)
    if (plan.source_dtype, plan.result_dtype) not in _BASE_UPDATE_SUFFIXES:
        raise TypeError("unsupported native update source/storage dtype pair")
    factors = tuple(jnp.asarray(value)
                    for value in (source_factor, base_factor))
    fixed_values = tuple(np.asarray(value, dtype=factor.dtype).item()
                         if isinstance(value, (int, float, complex, bool, np.generic))
                         else None for value, factor in zip((source_factor, base_factor), factors, strict=True))
    fixed = tuple(0 if value == 0 else 1 if value == 1 else None
                  for value in fixed_values)
    identity = (base is source and plan.source_dtype == plan.result_dtype
                and all(record.source_offset == record.destination_offset
                        and record.source_strides == record.destination_strides
                        and record.scale is None for record in plan.records))
    return _UPDATE_PRIMITIVE.bind(
        base_data, source_data, *factors, plan=plan,
        fixed_factors=fixed, identity_source=identity,
    )


__all__ = ["_execute_update"]
