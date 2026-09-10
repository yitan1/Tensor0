"""JAX primitive and transformations for affine stride execution."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from ._errors import raise_no_eligible_route
from ._jax import batch_only_named_sharding, require_jax_array
from ._native import (
    _MIXED_DTYPE_SUFFIXES,
    _affine_ffi_call,
    _mixed_ffi_call,
    native_available,
)
from ._native_descriptor import (
    NativeCallKind,
    NativeCallSpec,
    lower_plan,
)
from ._plan import (
    AffinePlan,
    AffineRecord,
    StridedOutputInit,
    PlanValidationError,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
    transpose_plan,
)
from ._map_routing import (
    RouteKind,
    RouteSharding,
    build_affine_route_features,
    decide_fresh_map_route,
)


PrimitivePlan = AffinePlan | NativeCallSpec

def _native_branch(source: Array, plan: NativeCallSpec) -> Array:
    if plan.kind is not NativeCallKind.AFFINE_MAP:
        raise RuntimeError("primitive plan is not an affine native execution")
    if plan.semantic.source_dtype != plan.semantic.result_dtype:
        return _mixed_ffi_call(source, plan)
    return _affine_ffi_call(
        source,
        descriptor=plan.descriptor,
        output_size=plan.semantic.output_size,
        result_dtype=jnp.dtype(plan.semantic.result_dtype),
    )


def _native_primitive_branch(source: Array, plan: PrimitivePlan) -> Array:
    if isinstance(plan, NativeCallSpec):
        return _native_branch(source, plan)
    raise RuntimeError("primitive plan has no native execution projection")


def _primitive_semantic(plan: PrimitivePlan) -> AffinePlan:
    return plan.semantic if isinstance(plan, NativeCallSpec) else plan


def _route_sharding(device_count: int | None) -> RouteSharding:
    if device_count in (None, 1):
        return RouteSharding.UNSHARDED
    # The custom partitioner performs the authoritative NamedSharding check.
    # Routing only selects its batch-prefix candidate at this stage.
    return RouteSharding.BATCH_ONLY


def _build_primitive_route_features(
    plan: PrimitivePlan,
    *,
    source_shape: tuple[int, ...],
    platform: str,
    device_count: int | None,
    native_runtime_available: bool | None = None,
) -> Any:
    semantic = _primitive_semantic(plan)
    execution_plan = plan if isinstance(plan, NativeCallSpec) else None
    semantic_source_shape = (*source_shape[:-1], semantic.source_size)
    if native_runtime_available is None:
        runtime_available = platform == "cpu" and native_available()
    else:
        runtime_available = native_runtime_available
    return build_affine_route_features(
        semantic,
        source_shape=semantic_source_shape,
        platform=platform,
        native_runtime_available=runtime_available,
        execution_plan=execution_plan,
        native_projection_available=execution_plan is not None,
        device_count=device_count,
        sharding=_route_sharding(device_count),
    )


def _partitioned_native_partition(
    plan: PrimitivePlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    (source_shape,) = argument_shapes
    source_sharding = batch_only_named_sharding(source_shape)
    result_sharding = batch_only_named_sharding(result_shape)
    if not isinstance(plan, NativeCallSpec):
        raise_no_eligible_route(("native_cpu_projection_unavailable",))
    return (
        mesh,
        lambda value: _native_primitive_branch(value, plan),
        result_sharding,
        (source_sharding,),
    )


def _partitioned_native_infer_sharding(
    plan: PrimitivePlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> Any:
    del plan, mesh, result_shape
    (source_shape,) = argument_shapes
    return batch_only_named_sharding(source_shape)


def _partitioned_native_propagate_sharding(
    plan: PrimitivePlan,
    mesh: Any,
    user_shape: Any,
) -> Any:
    del plan, mesh
    return batch_only_named_sharding(user_shape)


def _create_partitioned_native_branch() -> Any | None:
    try:
        from jax.experimental.custom_partitioning import custom_partitioning
    except (AttributeError, ImportError):
        return None

    @partial(custom_partitioning, static_argnums=(1,))
    def partitioned(source: Array, plan: PrimitivePlan) -> Array:
        semantic = _primitive_semantic(plan)
        return jnp.zeros(
            (*source.shape[:-1], semantic.output_size),
            dtype=jnp.dtype(semantic.result_dtype),
        )

    partitioned.def_partition(
        partition=_partitioned_native_partition,
        propagate_user_sharding=_partitioned_native_propagate_sharding,
        infer_sharding_from_operands=_partitioned_native_infer_sharding,
        decode_shardings=True,
        sharding_rule="... source -> ... result",
    )
    return partitioned


_PARTITIONED_NATIVE_BRANCH = _create_partitioned_native_branch()


def _primitive_abstract_eval(
    source_aval: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> Any:
    del reverse
    semantic = _primitive_semantic(forward)
    input_size = semantic.source_size
    input_dtype = jnp.dtype(semantic.source_dtype)
    output_size = semantic.output_size
    output_dtype = jnp.dtype(semantic.result_dtype)
    if source_aval.dtype != input_dtype:
        raise TypeError("native stride input dtype does not match the plan")
    if not source_aval.shape or source_aval.shape[-1] != input_size:
        raise ValueError("native stride flat source dimension mismatch")
    return source_aval.update(
        shape=(*source_aval.shape[:-1], output_size),
        dtype=output_dtype,
    )


def _primitive_lowering(
    context: Any,
    source: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> Any:
    del reverse
    from jax.interpreters import mlir

    axis_context = context.module_context.axis_context
    device_count = getattr(axis_context, "num_devices", None)
    source_shape = tuple(context.avals_in[0].shape)
    features = _build_primitive_route_features(
        forward,
        source_shape=source_shape,
        platform="cpu",
        device_count=device_count,
    )
    decision = decide_fresh_map_route(features)
    if decision.route is RouteKind.NO_ROUTE:
        raise_no_eligible_route((decision.reason,))

    assert decision.route is RouteKind.NATIVE
    assert isinstance(forward, NativeCallSpec)
    if device_count in (None, 1):
        function = lambda value: _native_primitive_branch(value, forward)
    else:
        partitioned_native = _PARTITIONED_NATIVE_BRANCH
        if partitioned_native is None:
            raise RuntimeError(
                "multi-device Tensor0 stride execution requires JAX custom "
                "partitioning support"
            )
        function = lambda value: partitioned_native(value, forward)
    return mlir.lower_fun(
        function,
        multiple_results=False,
    )(context, source)


def _primitive_non_cpu_lowering(
    context: Any,
    source: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> Any:
    del reverse
    platforms = tuple(context.module_context.platforms)
    platform = next((value for value in platforms if value != "cpu"), "non_cpu")
    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    features = _build_primitive_route_features(
        forward,
        source_shape=tuple(context.avals_in[0].shape),
        platform=platform,
        device_count=device_count,
    )
    decision = decide_fresh_map_route(features)
    if decision.route is RouteKind.NO_ROUTE:
        raise_no_eligible_route((decision.reason,))

    raise RuntimeError("non-CPU Tensor0 stride selected an invalid native route")


def _bind_primitive(
    source: Array,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> Array:
    if _STRIDE_PRIMITIVE is None:
        raise RuntimeError("the JAX stride transformation primitive is unavailable")
    return _STRIDE_PRIMITIVE.bind(
        source,
        forward=forward,
        reverse=reverse,
    )


def _primitive_jvp(
    primals: tuple[Array],
    tangents: tuple[Any],
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    (source,), (tangent,) = primals, tangents
    primal = _bind_primitive(
        source,
        forward=forward,
        reverse=reverse,
    )
    if isinstance(tangent, ad.Zero):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    semantic = _primitive_semantic(forward)
    input_dtype = jnp.dtype(semantic.source_dtype)
    output_dtype = jnp.dtype(semantic.result_dtype)
    if not (
        jnp.issubdtype(input_dtype, jnp.inexact)
        and jnp.issubdtype(output_dtype, jnp.inexact)
    ):
        return primal, jnp.zeros(primal.shape, dtype=jax.dtypes.float0)
    return (
        primal,
        _bind_primitive(
            tangent,
            forward=forward,
            reverse=reverse,
        ),
    )


def _primitive_transpose(
    cotangent: Any,
    source: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> list[Any]:
    from jax.interpreters import ad

    if not ad.is_undefined_primal(source):
        return [None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source.aval.to_ct_aval())]
    if reverse is None:
        if _primitive_semantic(forward).has_source_broadcast:
            from ._ops._reduction import bind_native_broadcast_transpose

            return [
                bind_native_broadcast_transpose(
                    cotangent,
                    _primitive_semantic(forward),
                )
            ]
        semantic = _primitive_semantic(forward)
        if jnp.issubdtype(
            jnp.dtype(semantic.source_dtype), jnp.inexact
        ) and jnp.issubdtype(jnp.dtype(semantic.result_dtype), jnp.inexact):
            raise NotImplementedError(
                "tensor0 stride transpose is unavailable for an unstructured "
                "repeated source mapping or unsupported dtype policy"
            )
        return [None]
    result = _bind_primitive(
        cotangent,
        forward=reverse,
        reverse=forward,
    )
    return [result]


def _lift_trailing_batch_axis(
    plan: PrimitivePlan,
    batch_size: int,
) -> NativeCallSpec:
    semantic = _primitive_semantic(plan)
    records = tuple(
        AffineRecord(
            logical_shape=(*record.logical_shape, batch_size),
            source_strides=(
                *(stride * batch_size for stride in record.source_strides),
                1,
            ),
            source_offset=record.source_offset * batch_size,
            destination_strides=(
                *(
                    stride * batch_size
                    for stride in record.destination_strides
                ),
                1,
            ),
            destination_offset=record.destination_offset * batch_size,
            scale=record.scale,
            source_broadcast_axes=record.source_broadcast_axes,
            reduction_axes=record.reduction_axes,
        )
        for record in semantic.records
    )
    lifted = build_affine_plan(
        records=records,
        output_size=semantic.output_size * batch_size,
        coverage=semantic.coverage,
        source_size=semantic.source_size * batch_size,
        source_dtype=semantic.source_dtype,
        result_dtype=semantic.result_dtype,
        output_init=semantic.output_init,
        write_kind=semantic.write_kind,
        scalar_kind=semantic.scalar_kind,
        reduction_kind=semantic.reduction_kind,
    )
    return lower_plan(lifted)


def _primitive_batch(
    arguments: tuple[Array],
    dimensions: tuple[int | None],
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
) -> tuple[Array, int | None]:
    (source,), (dimension,) = arguments, dimensions
    if dimension is None:
        return (
            _bind_primitive(
                source,
                forward=forward,
                reverse=reverse,
            ),
            None,
        )
    if dimension < source.ndim - 1:
        return (
            _bind_primitive(
                source,
                forward=forward,
                reverse=reverse,
            ),
            dimension,
        )
    if (
        dimension == source.ndim - 1
        and source.ndim >= 2
        and native_available()
        and jax.default_backend() == "cpu"
        and jax.device_count() == 1
    ):
        batch_size = source.shape[-1]
        try:
            lifted_forward = _lift_trailing_batch_axis(forward, batch_size)
            lifted_reverse = (
                None
                if reverse is None
                else _lift_trailing_batch_axis(reverse, batch_size)
            )
        except (PlanValidationError, ValueError):
            pass
        else:
            batch_prefix = source.shape[:-2]
            flat_source = jnp.reshape(
                source,
                (*batch_prefix, lifted_forward.semantic.source_size),
            )
            flat_result = _bind_primitive(
                flat_source,
                forward=lifted_forward,
                reverse=lifted_reverse,
            )
            return (
                jnp.reshape(
                    flat_result,
                    (*batch_prefix, _primitive_semantic(forward).output_size, batch_size),
                ),
                source.ndim - 1,
            )
    source = jnp.moveaxis(source, dimension, 0)
    return (
        _bind_primitive(
            source,
            forward=forward,
            reverse=reverse,
        ),
        0,
    )


def _create_stride_primitive() -> Any | None:
    try:
        from jax._src import dispatch
        from jax.extend import core
        from jax.interpreters import ad
        from jax.interpreters import batching
        from jax.interpreters import mlir
        from jax.interpreters import xla
    except (AttributeError, ImportError):
        return None

    primitive = core.Primitive("tensor0_stride")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_primitive_abstract_eval)
    ad.primitive_jvps[primitive] = _primitive_jvp
    ad.primitive_transposes[primitive] = _primitive_transpose
    batching.primitive_batchers[primitive] = _primitive_batch
    mlir.register_lowering(primitive, _primitive_non_cpu_lowering)
    mlir.register_lowering(primitive, _primitive_lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_STRIDE_PRIMITIVE = _create_stride_primitive()


def _supports_native_affine_dtype_policy(plan: AffinePlan) -> bool:
    if plan.source_dtype == plan.result_dtype:
        return True
    original_pair = (
        (plan.result_dtype, plan.source_dtype)
        if plan.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else (plan.source_dtype, plan.result_dtype)
    )
    return original_pair in _MIXED_DTYPE_SUFFIXES


def _execute_map(
    source: object,
    *,
    plan: AffinePlan | NativeCallSpec,
) -> Array:
    """Execute one complete unique functional affine copy."""

    lowered: NativeCallSpec | None
    data = require_jax_array(source, "_execute_map source")
    if data.ndim < 1:
        raise ValueError("strided copy requires at least one storage axis")
    semantic = plan.semantic if isinstance(plan, NativeCallSpec) else plan
    if (
        semantic.output_init
        not in (StridedOutputInit.UNINITIALIZED, StridedOutputInit.ZERO)
        or semantic.write_kind is not StridedWriteKind.ASSIGN
        or semantic.scalar_kind
        not in (
            StridedScalarKind.STATIC_SCALE_CAST,
            StridedScalarKind.JAX_TRANSPOSE,
        )
        or semantic.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("_execute_map requires a fresh affine map plan")
    if data.shape[-1] != semantic.source_size:
        raise ValueError(
            f"expected source shape (*batch, {semantic.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(semantic.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    if _supports_native_affine_dtype_policy(semantic):
        try:
            lowered = (
                plan
                if isinstance(plan, NativeCallSpec)
                else lower_plan(semantic)
            )
        except ValueError:
            lowered = None
    else:
        lowered = None

    if _STRIDE_PRIMITIVE is None:
        raise_no_eligible_route(("native_primitive_unavailable",))

    reverse: PrimitivePlan | None
    result_dtype = jnp.dtype(semantic.result_dtype)
    if jnp.issubdtype(expected_dtype, jnp.inexact) and jnp.issubdtype(
        result_dtype, jnp.inexact
    ):
        if semantic.has_source_broadcast:
            reverse = None
        else:
            try:
                reverse_semantic = transpose_plan(semantic)
            except PlanValidationError:
                reverse = None
            else:
                if _supports_native_affine_dtype_policy(reverse_semantic):
                    try:
                        reverse = lower_plan(reverse_semantic)
                    except ValueError:
                        reverse = (
                            reverse_semantic
                            if expected_dtype == result_dtype
                            else None
                        )
                else:
                    reverse = (
                        reverse_semantic
                        if expected_dtype == result_dtype
                        else None
                    )
    else:
        reverse = None
    forward_plan: PrimitivePlan = lowered if lowered is not None else semantic
    return _bind_primitive(
        data,
        forward=forward_plan,
        reverse=reverse,
    )
