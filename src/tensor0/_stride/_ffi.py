"""CPU JAX FFI lowering for the private strided-copy operation."""

from __future__ import annotations

from functools import partial
from importlib.metadata import version
from math import prod
from threading import Lock
from typing import Any, cast

import jax
from jax import Array
from jax.core import Tracer
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np

from .. import _native
from ._compiler import (
    COMPILED_DESCRIPTOR_VERSION,
    CompiledStridePlan,
    compile_plan,
)
from ._errors import raise_no_eligible_route
from ._native_lowering import (
    NativeExecutionKind,
    NativeExecutionPlan,
    lower_compiled_plan,
)
from ._plan import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_strided_copy_plan,
    transpose_plan,
)
from ._routing import (
    RouteKind,
    RouteSharding,
    StableHloCapability,
    compile_affine_route_features,
    decide_fresh_map_route,
)
from ._stablehlo import (
    AffineSingleRecordStableHloRecipe,
    AffineTransposeStableHloRecipe,
    CompactStableHloRecipe,
    CompactTransposeStableHloRecipe,
    FreshMapStableHloRecipe,
    PortableAffineStableHloRecipe,
    execute_fresh_map_stablehlo,
    try_compile_fresh_map_stablehlo,
)


PrimitivePlan = CompiledStridePlan | NativeExecutionPlan


def _require_jax_array(value: object, argument_name: str) -> Array:
    """Require one JAX execution operand without silently transferring it."""

    if not isinstance(value, (Array, Tracer)):
        raise TypeError(
            f"{argument_name} must be a JAX Array or Tracer, got {type(value).__name__}"
        )
    return cast(Array, value)


_SAME_DTYPE_SUFFIXES = {
    "bool": "pred",
    "int8": "s8",
    "int16": "s16",
    "int32": "s32",
    "int64": "s64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
    "complex64": "c64",
    "complex128": "c128",
}
_MIXED_DTYPE_SUFFIXES = {
    ("float16", "float32"): "f16_f32",
    ("float32", "complex64"): "f32_c64",
    ("complex64", "float32"): "c64_f32",
    ("float64", "complex128"): "f64_c128",
    ("complex128", "float64"): "c128_f64",
}
_BASE_UPDATE_SUFFIXES = {
    **{(dtype, dtype): suffix for dtype, suffix in _SAME_DTYPE_SUFFIXES.items()},
    **_MIXED_DTYPE_SUFFIXES,
    ("int32", "bool"): "s32_pred",
    ("int32", "int8"): "s32_s8",
    ("int32", "int16"): "s32_s16",
    ("uint32", "uint8"): "u32_u8",
}
_STRUCTURED_REDUCTION_FORWARD_SUFFIXES = {
    **{(dtype, dtype): suffix for dtype, suffix in _SAME_DTYPE_SUFFIXES.items()},
    **_MIXED_DTYPE_SUFFIXES,
}
_STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES = {
    **{
        (dtype, dtype): suffix
        for dtype, suffix in _SAME_DTYPE_SUFFIXES.items()
        if jnp.issubdtype(jnp.dtype(dtype), jnp.inexact)
    },
    **{
        (result_dtype, source_dtype): (
            f"{_SAME_DTYPE_SUFFIXES[result_dtype]}_"
            f"{_SAME_DTYPE_SUFFIXES[source_dtype]}"
        )
        for source_dtype, result_dtype in _MIXED_DTYPE_SUFFIXES
    },
}
_PREPARED_TYPE = "tensor0.stride.prepared_state.v5"
_PREPARED_TYPE_REGISTRATION_KEY = "prepared-state-type-v5"
_V7_REGISTRATION_KEY = "stride-abi-v7"
_MIXED_REGISTRATION_KEY = "stride-affine-mixed-v2"
_BASE_UPDATE_REGISTRATION_KEYS = {
    "assign": "stride-base-assign-v2",
    "accumulate": "stride-base-accumulate-v2",
}
_SELECTED_SCALE_REGISTRATION_KEY = "stride-selected-scale-v2"
_STRUCTURED_REDUCTION_REGISTRATION_KEY = "stride-structured-reduction-v2"
_RUNTIME_JAXLIB_VERSION = version("jaxlib")
_REGISTRATION_LOCK = Lock()
_REGISTERED: set[str] = set()


def _normalize_signed_zero_impl(value: Array) -> Array:
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        real = jnp.real(value)
        imaginary = jnp.imag(value)
        return jax.lax.complex(
            jnp.where(real == 0, jnp.zeros_like(real), real),
            jnp.where(
                imaginary == 0,
                jnp.zeros_like(imaginary),
                imaginary,
            ),
        )
    return jnp.where(value == 0, jnp.zeros_like(value), value)


def _create_signed_zero_primitive() -> Any | None:
    try:
        from jax.extend import core
        from jax.interpreters import ad
        from jax.interpreters import batching
        from jax.interpreters import mlir
        from jax.interpreters import xla
    except (AttributeError, ImportError):
        return None

    primitive = core.Primitive("tensor0_normalize_signed_zero")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(lambda value: value)
    ad.primitive_jvps[primitive] = lambda primals, tangents: (
        primitive.bind(primals[0]),
        tangents[0],
    )
    ad.primitive_transposes[primitive] = lambda cotangent, value: [cotangent]
    batching.primitive_batchers[primitive] = lambda arguments, dimensions: (
        primitive.bind(arguments[0]),
        dimensions[0],
    )
    mlir.register_lowering(
        primitive,
        mlir.lower_fun(_normalize_signed_zero_impl, multiple_results=False),
    )
    return primitive


_SIGNED_ZERO_PRIMITIVE = _create_signed_zero_primitive()


def _normalize_signed_zero(value: Array) -> Array:
    """Canonicalize zero bits while remaining the identity for AD."""

    if _SIGNED_ZERO_PRIMITIVE is None:
        return _normalize_signed_zero_impl(value)
    return _SIGNED_ZERO_PRIMITIVE.bind(value)


def native_available() -> bool:
    """Return whether this extension contains the current CPU FFI handler."""

    available = bool(
        hasattr(jax, "ffi")
        and hasattr(jax.lax, "platform_dependent")
        and hasattr(_native, "_stride_ffi_available")
        and _native._stride_ffi_available()
        and hasattr(_native, "_stride_ffi_abi_version")
    )
    if not available or not hasattr(_native, "_stride_ffi_build_versions"):
        return False
    return (
        _native._stride_ffi_abi_version() == COMPILED_DESCRIPTOR_VERSION
        and _native._stride_ffi_build_versions()
        == (
            jax.__version__,
            _RUNTIME_JAXLIB_VERSION,
        )
    )


def _ensure_prepared_type_registered() -> None:
    if _PREPARED_TYPE_REGISTRATION_KEY in _REGISTERED:
        return
    if not native_available():
        raise RuntimeError("Tensor0 was built without the JAX CPU stride handler")
    with _REGISTRATION_LOCK:
        if _PREPARED_TYPE_REGISTRATION_KEY in _REGISTERED:
            return
        registration = _native._stride_prepared_registration()
        jax.ffi.register_ffi_type(
            _PREPARED_TYPE,
            {
                "type_id": registration["type_id"],
                "type_info": registration["type_info"],
            },
            platform="cpu",
        )
        _REGISTERED.add(_PREPARED_TYPE_REGISTRATION_KEY)


def _ensure_registered_v7(dtype_name: str) -> str:
    suffix = _SAME_DTYPE_SUFFIXES.get(dtype_name)
    if suffix is None:
        raise RuntimeError(f"Tensor0 has no ABI-v7 stride target for {dtype_name}")
    target = f"tensor0_stride_r2_prepared_{suffix}_cpu_v7"
    _ensure_prepared_type_registered()
    if _V7_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _V7_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_suffix in _SAME_DTYPE_SUFFIXES.values():
            jax.ffi.register_ffi_target(
                f"tensor0_stride_r2_prepared_{registered_suffix}_cpu_v7",
                {
                    "instantiate": registration["instantiate"],
                    "execute": registration[f"execute_{registered_suffix}"],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(_V7_REGISTRATION_KEY)
    return target


def _ensure_mixed_registered(
    source_dtype: str,
    result_dtype: str,
    *,
    transpose: bool,
) -> str:
    pair = (source_dtype, result_dtype)
    if pair not in _MIXED_DTYPE_SUFFIXES:
        raise RuntimeError(f"unsupported mixed affine dtype pair {pair!r}")
    _ensure_prepared_type_registered()
    direction = "transpose" if transpose else "forward"
    target = f"tensor0_stride_affine_{source_dtype}_{result_dtype}_{direction}_cpu_v2"
    if _MIXED_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _MIXED_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for (src, dst), forward_suffix in _MIXED_DTYPE_SUFFIXES.items():
            for is_transpose in (False, True):
                registered_direction = "transpose" if is_transpose else "forward"
                execute_suffix = (
                    f"{_SAME_DTYPE_SUFFIXES[dst]}_{_SAME_DTYPE_SUFFIXES[src]}"
                    if is_transpose
                    else forward_suffix
                )
                jax.ffi.register_ffi_target(
                    (
                        f"tensor0_stride_affine_{src}_{dst}_"
                        f"{registered_direction}_cpu_v2"
                    ),
                    {
                        "instantiate": registration["instantiate"],
                        "execute": registration[
                            f"execute_{execute_suffix}_{registered_direction}"
                        ],
                    },
                    platform="cpu",
                    api_version=1,
                )
        _REGISTERED.add(_MIXED_REGISTRATION_KEY)
    return target


def _ensure_base_update_registered(
    operation: str,
    source_dtype: str,
    result_dtype: str,
) -> str:
    registration_key = _BASE_UPDATE_REGISTRATION_KEYS.get(operation)
    if registration_key is None:
        raise RuntimeError(f"unknown base update operation {operation!r}")
    pair = (source_dtype, result_dtype)
    if pair not in _BASE_UPDATE_SUFFIXES:
        raise RuntimeError(f"unsupported base update dtype pair {pair!r}")
    _ensure_prepared_type_registered()
    target = (
        f"tensor0_stride_base_{operation}_{source_dtype}_{result_dtype}_cpu_v2"
    )
    if registration_key in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if registration_key in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for (registered_source, registered_result), suffix in (
            _BASE_UPDATE_SUFFIXES.items()
        ):
            registered_target = (
                f"tensor0_stride_base_{operation}_{registered_source}_"
                f"{registered_result}_cpu_v2"
            )
            jax.ffi.register_ffi_target(
                registered_target,
                {
                    "instantiate": registration[
                        f"instantiate_base_{operation}"
                    ],
                    "execute": registration[
                        f"execute_base_{operation}_{suffix}"
                    ],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(registration_key)
    return target


def _ensure_selected_scale_registered(dtype_name: str) -> str:
    suffix = _SAME_DTYPE_SUFFIXES.get(dtype_name)
    if suffix is None:
        raise RuntimeError(f"unsupported selected scale dtype {dtype_name!r}")
    _ensure_prepared_type_registered()
    target = f"tensor0_stride_selected_scale_{dtype_name}_cpu_v2"
    if _SELECTED_SCALE_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _SELECTED_SCALE_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_dtype, registered_suffix in _SAME_DTYPE_SUFFIXES.items():
            jax.ffi.register_ffi_target(
                f"tensor0_stride_selected_scale_{registered_dtype}_cpu_v2",
                {
                    "instantiate": registration["instantiate_selected_scale"],
                    "execute": registration[
                        f"execute_selected_scale_{registered_suffix}"
                    ],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(_SELECTED_SCALE_REGISTRATION_KEY)
    return target


def _ensure_structured_reduction_registered(
    input_dtype: str,
    output_dtype: str,
    scalar_kind: StridedScalarKind,
) -> str:
    if scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        suffixes = _STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES
        policy_name = "transpose"
    elif scalar_kind is StridedScalarKind.STATIC_SCALE_CAST:
        suffixes = _STRUCTURED_REDUCTION_FORWARD_SUFFIXES
        policy_name = "forward"
    else:
        raise RuntimeError("unsupported structured reduction scalar policy")
    suffix = suffixes.get((input_dtype, output_dtype))
    if suffix is None:
        raise RuntimeError(
            "unsupported structured reduction dtype pair "
            f"{(input_dtype, output_dtype)!r}"
        )
    _ensure_prepared_type_registered()
    target = (
        f"tensor0_stride_structured_reduction_{input_dtype}_{output_dtype}_"
        f"{policy_name}_cpu_v2"
    )
    if _STRUCTURED_REDUCTION_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _STRUCTURED_REDUCTION_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for policy, policy_suffixes in (
            ("transpose", _STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES),
            ("forward", _STRUCTURED_REDUCTION_FORWARD_SUFFIXES),
        ):
            for (registered_input, registered_output), registered_suffix in (
                policy_suffixes.items()
            ):
                execute_key = (
                    f"execute_structured_reduction_{registered_suffix}"
                    if policy == "transpose"
                    else "execute_structured_reduction_forward_"
                    f"{registered_suffix}"
                )
                jax.ffi.register_ffi_target(
                    (
                        "tensor0_stride_structured_reduction_"
                        f"{registered_input}_{registered_output}_{policy}_cpu_v2"
                    ),
                    {
                        "instantiate": registration[
                            "instantiate_structured_reduction"
                        ],
                        "execute": registration[execute_key],
                    },
                    platform="cpu",
                    api_version=1,
                )
        _REGISTERED.add(_STRUCTURED_REDUCTION_REGISTRATION_KEY)
    return target


def _ffi_call_v7(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    result_dtype: jnp.dtype,
    alias_source_result: bool = False,
) -> Array:
    dtype = jnp.dtype(result_dtype)
    target = _ensure_registered_v7(dtype.name)
    batch_shape = source.shape[:-1]
    batch_count = prod(batch_shape)
    flat_source = jnp.reshape(source, (batch_count * source.shape[-1],))
    result = jax.ShapeDtypeStruct((batch_count * output_size,), dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,),),
        output_layouts=(0,),
        input_output_aliases={0: 0} if alias_source_result else None,
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_source, descriptor=attribute)
    return jnp.reshape(flat_result, (*batch_shape, output_size))


def _mixed_ffi_call_v2(source: Array, plan: NativeExecutionPlan) -> Array:
    semantic = plan.bound
    input_dtype = jnp.dtype(semantic.source_dtype)
    output_dtype = jnp.dtype(semantic.result_dtype)
    output_size = semantic.output_size
    if source.dtype != input_dtype:
        raise TypeError("mixed affine input dtype does not match the plan")
    target = (
        _ensure_mixed_registered(
            semantic.result_dtype,
            semantic.source_dtype,
            transpose=True,
        )
        if semantic.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else _ensure_mixed_registered(
            semantic.source_dtype,
            semantic.result_dtype,
            transpose=False,
        )
    )
    batch_shape = source.shape[:-1]
    batch_count = prod(batch_shape)
    flat_source = jnp.reshape(source, (batch_count * source.shape[-1],))
    result = jax.ShapeDtypeStruct(
        (batch_count * output_size,),
        output_dtype,
    )
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,),),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(plan.descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_source, descriptor=attribute)
    return jnp.reshape(flat_result, (*batch_shape, output_size))


def _base_update_ffi_call_v2(
    base: Array,
    source: Array,
    *,
    descriptor: bytes,
    operation: str,
) -> Array:
    target = _ensure_base_update_registered(
        operation,
        source.dtype.name,
        base.dtype.name,
    )
    batch_shape = base.shape[:-1]
    batch_count = prod(batch_shape)
    flat_base = jnp.reshape(base, (batch_count * base.shape[-1],))
    flat_source = jnp.reshape(source, (batch_count * source.shape[-1],))
    result = jax.ShapeDtypeStruct(flat_base.shape, base.dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,), (0,)),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_base, flat_source, descriptor=attribute)
    return jnp.reshape(flat_result, base.shape)


def _selected_scale_ffi_call_v2(
    base: Array,
    factor: Array,
    *,
    descriptor: bytes,
) -> Array:
    target = _ensure_selected_scale_registered(base.dtype.name)
    batch_shape = base.shape[:-1]
    batch_count = prod(batch_shape)
    flat_base = jnp.reshape(base, (batch_count * base.shape[-1],))
    flat_factor = jnp.reshape(factor, (factor.size,))
    result = jax.ShapeDtypeStruct(flat_base.shape, base.dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,), (0,)),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_base, flat_factor, descriptor=attribute)
    return jnp.reshape(flat_result, base.shape)


def _structured_reduction_ffi_call_v2(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    output_dtype: jnp.dtype,
    scalar_kind: StridedScalarKind = StridedScalarKind.JAX_TRANSPOSE,
) -> Array:
    dtype = jnp.dtype(output_dtype)
    target = _ensure_structured_reduction_registered(
        source.dtype.name,
        dtype.name,
        scalar_kind,
    )
    batch_shape = source.shape[:-1]
    batch_count = prod(batch_shape)
    flat_source = jnp.reshape(source, (batch_count * source.shape[-1],))
    result = jax.ShapeDtypeStruct((batch_count * output_size,), dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,),),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_source, descriptor=attribute)
    return jnp.reshape(flat_result, (*batch_shape, output_size))


def _native_branch(source: Array, plan: NativeExecutionPlan) -> Array:
    if plan.kind is not NativeExecutionKind.AFFINE_MAP:
        raise RuntimeError("primitive plan is not an affine native execution")
    if plan.bound.source_dtype != plan.bound.result_dtype:
        return _mixed_ffi_call_v2(source, plan)
    return _ffi_call_v7(
        source,
        descriptor=plan.descriptor,
        output_size=plan.bound.output_size,
        result_dtype=jnp.dtype(plan.bound.result_dtype),
    )


def _native_primitive_branch(source: Array, plan: PrimitivePlan) -> Array:
    if isinstance(plan, NativeExecutionPlan):
        return _native_branch(source, plan)
    raise RuntimeError("primitive plan has no native execution projection")


def _primitive_bound(plan: PrimitivePlan) -> StridedCopyPlan:
    return plan.bound


def _stablehlo_capability(
    recipe: FreshMapStableHloRecipe | None,
) -> StableHloCapability:
    if isinstance(
        recipe,
        (CompactStableHloRecipe, CompactTransposeStableHloRecipe),
    ):
        return StableHloCapability.COMPACT_AUTOMATIC
    if isinstance(
        recipe,
        (AffineSingleRecordStableHloRecipe, AffineTransposeStableHloRecipe),
    ):
        return StableHloCapability.AFFINE_AUTOMATIC
    if isinstance(recipe, PortableAffineStableHloRecipe):
        return StableHloCapability.PORTABLE_AUTOMATIC
    return StableHloCapability.NONE


def _route_sharding(device_count: int | None) -> RouteSharding:
    if device_count in (None, 1):
        return RouteSharding.UNSHARDED
    # The custom partitioner performs the authoritative NamedSharding check.
    # Routing only selects its batch-prefix candidate at this stage.
    return RouteSharding.BATCH_ONLY


def _compile_primitive_route_features(
    plan: PrimitivePlan,
    *,
    source_shape: tuple[int, ...],
    platform: str,
    recipe: FreshMapStableHloRecipe | None,
    device_count: int | None,
    automatic: bool,
    native_runtime_available: bool | None = None,
    stablehlo_capability: StableHloCapability | None = None,
) -> Any:
    bound = _primitive_bound(plan)
    execution_plan: NativeExecutionPlan | None
    if isinstance(plan, NativeExecutionPlan):
        execution_plan = plan
    else:
        execution_plan = None
    semantic_source_shape = (*source_shape[:-1], bound.source_size)
    if native_runtime_available is None:
        runtime_available = platform == "cpu" and native_available()
    else:
        runtime_available = native_runtime_available
    return compile_affine_route_features(
        bound,
        source_shape=semantic_source_shape,
        platform=platform,
        stablehlo_capability=(
            _stablehlo_capability(recipe)
            if stablehlo_capability is None
            else stablehlo_capability
        ),
        native_runtime_available=runtime_available,
        compiled_plan=(plan if isinstance(plan, CompiledStridePlan) else None),
        execution_plan=execution_plan,
        native_projection_available=execution_plan is not None,
        device_count=device_count,
        sharding=_route_sharding(device_count),
        automatic=automatic,
    )


def _batch_only_named_sharding(shape: Any) -> Any:
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec

    sharding = getattr(shape, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError(
            "Tensor0 stride requires NamedSharding for multi-device fallback"
        )
    rank = len(shape.shape)
    specification = tuple(sharding.spec)
    if len(specification) > rank:
        raise ValueError("Tensor0 stride received an invalid sharding rank")
    normalized = specification + (None,) * (rank - len(specification))
    if normalized[-1] is not None:
        raise ValueError(
            "Tensor0 stride cannot shard the packed storage axis without an "
            "explicit collective"
        )
    return NamedSharding(
        sharding.mesh,
        PartitionSpec(*normalized),
    )


def _partitioned_native_partition(
    plan: PrimitivePlan,
    mesh: Any,
    argument_shapes: tuple[Any, ...],
    result_shape: Any,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    (source_shape,) = argument_shapes
    source_sharding = _batch_only_named_sharding(source_shape)
    result_sharding = _batch_only_named_sharding(result_shape)
    if not isinstance(plan, NativeExecutionPlan):
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
    return _batch_only_named_sharding(source_shape)


def _partitioned_native_propagate_sharding(
    plan: PrimitivePlan,
    mesh: Any,
    user_shape: Any,
) -> Any:
    del plan, mesh
    return _batch_only_named_sharding(user_shape)


def _create_partitioned_native_branch() -> Any | None:
    try:
        from jax.experimental.custom_partitioning import custom_partitioning
    except (AttributeError, ImportError):
        return None

    @partial(custom_partitioning, static_argnums=(1,))
    def partitioned(source: Array, plan: PrimitivePlan) -> Array:
        bound = _primitive_bound(plan)
        return jnp.zeros(
            (*source.shape[:-1], bound.output_size),
            dtype=jnp.dtype(bound.result_dtype),
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
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> Any:
    del (
        reverse,
        forward_stablehlo,
        reverse_stablehlo,
        automatic,
    )
    bound = _primitive_bound(forward)
    input_size = bound.source_size
    input_dtype = jnp.dtype(bound.source_dtype)
    output_size = bound.output_size
    output_dtype = jnp.dtype(bound.result_dtype)
    if source_aval.dtype != input_dtype:
        raise TypeError("native stride input dtype does not match the plan")
    if not source_aval.shape or source_aval.shape[-1] != input_size:
        raise ValueError("native stride flat source dimension mismatch")
    return source_aval.update(
        shape=(*source_aval.shape[:-1], output_size),
        dtype=output_dtype,
    )


def _lower_stablehlo_recipe(
    context: Any,
    source: Any,
    forward: PrimitivePlan,
    recipe: FreshMapStableHloRecipe,
) -> Any:
    from jax.interpreters import mlir

    bound = _primitive_bound(forward)
    if recipe.compiled.bound != bound:
        raise RuntimeError("StableHLO recipe does not match the forward plan")
    return mlir.lower_fun(
        lambda value: execute_fresh_map_stablehlo(value, recipe),
        multiple_results=False,
    )(context, source)


def _primitive_lowering(
    context: Any,
    source: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> Any:
    del reverse, reverse_stablehlo
    from jax.interpreters import mlir

    axis_context = context.module_context.axis_context
    device_count = getattr(axis_context, "num_devices", None)
    source_shape = tuple(context.avals_in[0].shape)
    features = _compile_primitive_route_features(
        forward,
        source_shape=source_shape,
        platform="cpu",
        recipe=forward_stablehlo,
        device_count=device_count,
        automatic=automatic,
    )
    decision = decide_fresh_map_route(features)
    if decision.route is RouteKind.STABLEHLO:
        if forward_stablehlo is None:
            raise RuntimeError("central route selected an unavailable StableHLO recipe")
        return _lower_stablehlo_recipe(
            context,
            source,
            forward,
            forward_stablehlo,
        )
    if decision.route is RouteKind.NO_ROUTE:
        if not automatic:
            raise RuntimeError(
                "native_required Tensor0 stride only supports unsharded CPU input"
            )
        raise_no_eligible_route((decision.reason,))

    if device_count in (None, 1):
        assert decision.route is RouteKind.NATIVE
        assert isinstance(forward, NativeExecutionPlan)
        function = lambda value: _native_primitive_branch(value, forward)
    elif not automatic:
        raise RuntimeError(
            "native_required Tensor0 stride only supports unsharded CPU input"
        )
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
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> Any:
    del reverse, reverse_stablehlo
    platforms = tuple(context.module_context.platforms)
    if not automatic and len(platforms) == 1:
        raise RuntimeError(
            "native_required Tensor0 stride only supports the CPU platform"
        )
    from jax.interpreters import mlir

    platform = next((value for value in platforms if value != "cpu"), "non_cpu")
    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    features = _compile_primitive_route_features(
        forward,
        source_shape=tuple(context.avals_in[0].shape),
        platform=platform,
        recipe=forward_stablehlo,
        device_count=device_count,
        automatic=automatic,
    )
    decision = decide_fresh_map_route(features)
    if decision.route is RouteKind.STABLEHLO:
        if forward_stablehlo is None:
            raise RuntimeError("central route selected an unavailable StableHLO recipe")
        return _lower_stablehlo_recipe(
            context,
            source,
            forward,
            forward_stablehlo,
        )
    if decision.route is RouteKind.NO_ROUTE:
        if not automatic:
            raise RuntimeError(
                "native_required Tensor0 stride only supports the CPU platform"
            )
        raise_no_eligible_route((decision.reason,))

    raise RuntimeError("non-CPU Tensor0 stride selected an invalid native route")


def _bind_primitive(
    source: Array,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> Array:
    if _STRIDE_PRIMITIVE is None:
        raise RuntimeError("the JAX stride transformation primitive is unavailable")
    return _STRIDE_PRIMITIVE.bind(
        source,
        forward=forward,
        reverse=reverse,
        forward_stablehlo=forward_stablehlo,
        reverse_stablehlo=reverse_stablehlo,
        automatic=automatic,
    )


def _primitive_jvp(
    primals: tuple[Array],
    tangents: tuple[Any],
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    (source,), (tangent,) = primals, tangents
    primal = _bind_primitive(
        source,
        forward=forward,
        reverse=reverse,
        forward_stablehlo=forward_stablehlo,
        reverse_stablehlo=reverse_stablehlo,
        automatic=automatic,
    )
    if isinstance(tangent, ad.Zero):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    bound = _primitive_bound(forward)
    input_dtype = jnp.dtype(bound.source_dtype)
    output_dtype = jnp.dtype(bound.result_dtype)
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
            forward_stablehlo=forward_stablehlo,
            reverse_stablehlo=reverse_stablehlo,
            automatic=automatic,
        ),
    )


def _primitive_transpose(
    cotangent: Any,
    source: Any,
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> list[Any]:
    from jax.interpreters import ad

    if not ad.is_undefined_primal(source):
        return [None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source.aval.to_ct_aval())]
    if reverse is None:
        if _primitive_bound(forward).has_source_broadcast:
            from ._native_reduction import bind_native_broadcast_transpose

            return [
                bind_native_broadcast_transpose(
                    cotangent,
                    _primitive_bound(forward),
                )
            ]
        return [None]
    result = _bind_primitive(
        cotangent,
        forward=reverse,
        reverse=forward,
        forward_stablehlo=reverse_stablehlo,
        reverse_stablehlo=forward_stablehlo,
        automatic=automatic,
    )
    if isinstance(reverse, NativeExecutionPlan):
        result = _normalize_signed_zero(result)
    return [result]


def _primitive_batch(
    arguments: tuple[Array],
    dimensions: tuple[int | None],
    *,
    forward: PrimitivePlan,
    reverse: PrimitivePlan | None,
    forward_stablehlo: FreshMapStableHloRecipe | None,
    reverse_stablehlo: FreshMapStableHloRecipe | None,
    automatic: bool,
) -> tuple[Array, int | None]:
    (source,), (dimension,) = arguments, dimensions
    if dimension is None:
        return (
            _bind_primitive(
                source,
                forward=forward,
                reverse=reverse,
                forward_stablehlo=forward_stablehlo,
                reverse_stablehlo=reverse_stablehlo,
                automatic=automatic,
            ),
            None,
        )
    source = jnp.moveaxis(source, dimension, 0)
    return (
        _bind_primitive(
            source,
            forward=forward,
            reverse=reverse,
            forward_stablehlo=forward_stablehlo,
            reverse_stablehlo=reverse_stablehlo,
            automatic=automatic,
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


def _supports_native_affine_dtype_policy(bound: StridedCopyPlan) -> bool:
    if bound.source_dtype == bound.result_dtype:
        return True
    original_pair = (
        (bound.result_dtype, bound.source_dtype)
        if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else (bound.source_dtype, bound.result_dtype)
    )
    return original_pair in _MIXED_DTYPE_SUFFIXES


def strided_copy(
    source: object,
    *,
    records: tuple[StridedCopyRecord, ...] | None = None,
    output_size: int | None = None,
    result_dtype: DTypeLike | None = None,
    plan: StridedCopyPlan | NativeExecutionPlan | None = None,
    native_required: bool = False,
) -> Array:
    """Execute one complete unique functional affine copy."""

    lowered: NativeExecutionPlan | None
    data = _require_jax_array(source, "strided_copy source")
    if data.ndim < 1:
        raise ValueError("strided copy requires at least one storage axis")
    direct_arguments = (
        records is not None or output_size is not None or result_dtype is not None
    )
    if plan is not None and direct_arguments:
        raise TypeError("strided_copy accepts either plan or records, not both")
    if plan is None:
        if records is None or output_size is None:
            raise TypeError("strided_copy requires records and output_size")
        plan = build_strided_copy_plan(
            records=records,
            output_size=output_size,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=data.shape[-1],
            source_dtype=data.dtype,
            result_dtype=(data.dtype if result_dtype is None else result_dtype),
        )
    bound = plan.bound if isinstance(plan, NativeExecutionPlan) else plan
    if (
        bound.output_init
        not in (StridedOutputInit.UNINITIALIZED, StridedOutputInit.ZERO)
        or bound.write_kind is not StridedWriteKind.ASSIGN
        or bound.scalar_kind
        not in (
            StridedScalarKind.STATIC_SCALE_CAST,
            StridedScalarKind.JAX_TRANSPOSE,
        )
        or bound.reduction_kind is not StridedReductionKind.NONE
    ):
        raise ValueError("strided_copy requires a fresh affine map plan")
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    compiled = (
        plan.compiled if isinstance(plan, NativeExecutionPlan) else compile_plan(bound)
    )
    automatic_recipe: FreshMapStableHloRecipe | None = None
    if (
        not native_required
        and bound.scalar_kind is StridedScalarKind.STATIC_SCALE_CAST
    ):
        automatic_recipe = try_compile_fresh_map_stablehlo(compiled)

    if _supports_native_affine_dtype_policy(bound):
        try:
            lowered = (
                plan
                if isinstance(plan, NativeExecutionPlan)
                else lower_compiled_plan(compiled)
            )
        except ValueError:
            if native_required:
                raise
            lowered = None
    else:
        lowered = None

    if _STRIDE_PRIMITIVE is None:
        if native_required:
            raise RuntimeError(
                "the requested plan has no transformation-safe native CPU path"
            )
        raise_no_eligible_route(("native_primitive_unavailable",))
    if native_required and (lowered is None or not native_available()):
        raise RuntimeError(
            "the requested plan has no transformation-safe native CPU path"
        )

    reverse: PrimitivePlan | None
    result_dtype = jnp.dtype(bound.result_dtype)
    if jnp.issubdtype(expected_dtype, jnp.inexact) and jnp.issubdtype(
        result_dtype, jnp.inexact
    ):
        if bound.has_source_broadcast:
            reverse = None
        else:
            reverse_bound = transpose_plan(bound)
            reverse_compiled = compile_plan(reverse_bound)
            if _supports_native_affine_dtype_policy(reverse_bound):
                try:
                    reverse = lower_compiled_plan(reverse_compiled)
                except ValueError:
                    reverse = (
                        reverse_compiled
                        if expected_dtype == result_dtype
                        else None
                    )
            else:
                reverse = reverse_compiled if expected_dtype == result_dtype else None
    else:
        reverse = None
    reverse_automatic_recipe: FreshMapStableHloRecipe | None = None
    if not native_required and isinstance(
        reverse,
        (CompiledStridePlan, NativeExecutionPlan),
    ):
        reverse_compiled = (
            reverse.compiled if isinstance(reverse, NativeExecutionPlan) else reverse
        )
        reverse_automatic_recipe = try_compile_fresh_map_stablehlo(reverse_compiled)
    forward_plan: PrimitivePlan = lowered if lowered is not None else compiled
    return _bind_primitive(
        data,
        forward=forward_plan,
        reverse=reverse,
        forward_stablehlo=automatic_recipe,
        reverse_stablehlo=reverse_automatic_recipe,
        automatic=not native_required,
    )


def _native_call_count_for_tests() -> int:
    if not native_available():
        return 0
    return int(_native._stride_native_call_count())


def _reset_native_call_count_for_tests() -> None:
    if native_available():
        _native._reset_stride_native_call_count()


def _native_worker_counts_for_tests() -> tuple[int, int]:
    if not native_available():
        return 0, 0
    return (
        int(_native._stride_last_worker_count()),
        int(_native._stride_last_available_worker_count()),
    )


def _set_native_worker_limit_for_tests(limit: int | None) -> None:
    if native_available():
        _native._set_stride_worker_limit_for_tests(limit)


def _prepared_native_call_for_tests(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    alias_source_result: bool = False,
) -> Array:
    return _ffi_call_v7(
        jnp.asarray(source),
        descriptor=descriptor,
        output_size=output_size,
        result_dtype=jnp.dtype(source.dtype),
        alias_source_result=alias_source_result,
    )
