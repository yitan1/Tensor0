"""JAX FFI registration and native call boundary for stride execution."""

from __future__ import annotations

from importlib.metadata import version
from math import prod
from typing import Any
from threading import Lock

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from .. import _native
from ._native_descriptor import (
    AFFINE_DESCRIPTOR_VERSION,
    NativeCallSpec,
)
from ._plan import StridedScalarKind

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
_PREPARED_TYPE = "tensor0.stride.prepared_state.v6"
_PREPARED_TYPE_REGISTRATION_KEY = "prepared-state-type-v6"
_AFFINE_REGISTRATION_KEY = "stride-affine-v8"
_MIXED_REGISTRATION_KEY = "stride-affine-mixed-v3"
_AXPBY_REGISTRATION_KEY = "stride-axpby-v1"
_DOT_REGISTRATION_KEY = "stride-dot-v1"
_SELECTED_SCALE_ALIAS_REGISTRATION_KEY = "stride-selected-scale-alias-v2"
_STRUCTURED_REDUCTION_REGISTRATION_KEY = "stride-structured-reduction-v3"
_RUNTIME_JAXLIB_VERSION = version("jaxlib")
_REGISTRATION_LOCK = Lock()
_REGISTERED: set[str] = set()
_UNLIMITED_WORKERS = (1 << 64) - 1


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
        _native._stride_ffi_abi_version() == AFFINE_DESCRIPTOR_VERSION
        and _native._stride_ffi_build_versions()
        == (
            jax.__version__,
            _RUNTIME_JAXLIB_VERSION,
        )
    )


def _require_native_thread_control() -> None:
    if not native_available():
        raise RuntimeError("native CPU stride execution is unavailable")


def get_num_threads() -> int | None:
    """Return the process-wide native stride worker upper bound.

    ``None`` means no additional limit beyond the workers supplied by XLA.
    An operation may use fewer workers than the returned bound. The setting is
    read when asynchronous native execution begins, not when JAX work is
    submitted; wait for outstanding work before changing it.
    """

    _require_native_thread_control()
    limit = _native._stride_worker_limit()
    return None if limit is None else int(limit)


def set_num_threads(n: int) -> None:
    """Set the process-wide native stride worker upper bound to ``n``.

    An operation may use fewer workers according to its workload and the XLA
    thread pool. The setting is read when asynchronous native execution begins,
    not when JAX work is submitted; wait for outstanding work before changing
    it.
    """

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n >= _UNLIMITED_WORKERS:
        raise ValueError("n exceeds the configurable native worker-limit range")
    _require_native_thread_control()
    _native._set_stride_worker_limit(n)


def enable_threads() -> None:
    """Remove the process-wide native stride worker upper bound.

    Native operations may then use the workers supplied by XLA, but can still
    choose fewer. The setting is read when asynchronous native execution begins,
    not when JAX work is submitted; wait for outstanding work before changing
    it.
    """

    _require_native_thread_control()
    _native._set_stride_worker_limit(None)


def disable_threads() -> None:
    """Set the process-wide native stride worker upper bound to one.

    The setting is read when asynchronous native execution begins, not when JAX
    work is submitted; wait for outstanding work before changing it.
    """

    set_num_threads(1)


def native_f16_contiguous_simd_available() -> bool:
    """Return whether the current CPU can execute the native f16 SIMD leaf."""

    return bool(
        native_available()
        and hasattr(_native, "_stride_cpu_supports_f16_contiguous_simd")
        and _native._stride_cpu_supports_f16_contiguous_simd()
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


def _ensure_affine_registered(dtype_name: str) -> str:
    suffix = _SAME_DTYPE_SUFFIXES.get(dtype_name)
    if suffix is None:
        raise RuntimeError(f"Tensor0 has no affine-v8 target for {dtype_name}")
    target = f"tensor0_stride_affine_{suffix}_cpu_v8"
    _ensure_prepared_type_registered()
    if _AFFINE_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _AFFINE_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_suffix in _SAME_DTYPE_SUFFIXES.values():
            jax.ffi.register_ffi_target(
                f"tensor0_stride_affine_{registered_suffix}_cpu_v8",
                {
                    "instantiate": registration["instantiate"],
                    "execute": registration[f"execute_{registered_suffix}"],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(_AFFINE_REGISTRATION_KEY)
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
    target = f"tensor0_stride_affine_{source_dtype}_{result_dtype}_{direction}_cpu_v3"
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
                        f"{registered_direction}_cpu_v3"
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


def _ensure_scale_tangent_registered(dtype_name: str) -> str:
    _ensure_prepared_type_registered()
    target = f"tensor0_stride_scale_tangent_{dtype_name}_cpu_v1"
    with _REGISTRATION_LOCK:
        if target not in _REGISTERED:
            registration = _native._stride_prepared_registration()
            jax.ffi.register_ffi_target(
                target,
                {"instantiate": registration["instantiate_selected_scale"],
                 "execute": registration[
                     f"execute_scale_tangent_{_SAME_DTYPE_SUFFIXES[dtype_name]}"
                 ]},
                platform="cpu", api_version=1,
            )
            _REGISTERED.add(target)
    return target


def _ensure_axpby_registered(dtype_name: str) -> str:
    supported = (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    if dtype_name not in supported:
        raise RuntimeError(f"unsupported native AXPBY dtype {dtype_name!r}")
    _ensure_prepared_type_registered()
    target = f"tensor0_stride_axpby_{dtype_name}_cpu_v1"
    if _AXPBY_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _AXPBY_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_dtype in supported:
            suffix = _SAME_DTYPE_SUFFIXES[registered_dtype]
            jax.ffi.register_ffi_target(
                f"tensor0_stride_axpby_{registered_dtype}_cpu_v1",
                {
                    "instantiate": registration["instantiate_update"],
                    "execute": registration[f"execute_axpby_{suffix}"],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(_AXPBY_REGISTRATION_KEY)
    return target


def _ensure_update_registered(source_dtype: str, result_dtype: str) -> str:
    suffix = _BASE_UPDATE_SUFFIXES.get((source_dtype, result_dtype))
    if suffix is None:
        raise RuntimeError("unsupported native update dtype pair")
    _ensure_prepared_type_registered()
    domain = result_dtype if source_dtype == result_dtype else f"{source_dtype}_{result_dtype}"
    target = f"tensor0_stride_update_{domain}_cpu_v1"
    with _REGISTRATION_LOCK:
        if target not in _REGISTERED:
            registration = _native._stride_prepared_registration()
            jax.ffi.register_ffi_target(
                target,
                {"instantiate": registration["instantiate_update"],
                 "execute": registration[f"execute_update_{suffix}"]},
                platform="cpu", api_version=1,
            )
            _REGISTERED.add(target)
    return target


def _ensure_dot_registered(dtype_name: str, *, conjugate_left: bool) -> str:
    if dtype_name not in ("float32", "complex64"):
        raise RuntimeError(f"unsupported native dot dtype {dtype_name!r}")
    _ensure_prepared_type_registered()
    operation = "dotc" if conjugate_left else "dotu"
    target = f"tensor0_stride_{operation}_{dtype_name}_cpu_v1"
    if _DOT_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _DOT_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_dtype, suffix in (
            ("float32", "f32"),
            ("complex64", "c64"),
        ):
            for registered_operation in ("dotu", "dotc"):
                jax.ffi.register_ffi_target(
                    (
                        f"tensor0_stride_{registered_operation}_"
                        f"{registered_dtype}_cpu_v1"
                    ),
                    {
                        "instantiate": registration["instantiate_dot"],
                        "execute": registration[
                            f"execute_{registered_operation}_{suffix}"
                        ],
                    },
                    platform="cpu",
                    api_version=1,
                )
        _REGISTERED.add(_DOT_REGISTRATION_KEY)
    return target


def _ensure_selected_scale_alias_registered(dtype_name: str) -> str:
    if dtype_name not in ("float32", "complex64"):
        raise RuntimeError(
            f"unsupported diagnostic selected-scale alias dtype {dtype_name!r}"
        )
    _ensure_prepared_type_registered()
    target = f"tensor0_stride_selected_scale_alias_{dtype_name}_cpu_v2"
    if _SELECTED_SCALE_ALIAS_REGISTRATION_KEY in _REGISTERED:
        return target
    with _REGISTRATION_LOCK:
        if _SELECTED_SCALE_ALIAS_REGISTRATION_KEY in _REGISTERED:
            return target
        registration = _native._stride_prepared_registration()
        for registered_dtype, suffix in (
            ("float32", "f32"),
            ("complex64", "c64"),
        ):
            jax.ffi.register_ffi_target(
                f"tensor0_stride_selected_scale_alias_{registered_dtype}_cpu_v2",
                {
                    "instantiate": registration["instantiate_selected_scale"],
                    "execute": registration[
                        f"execute_selected_scale_alias_{suffix}"
                    ],
                },
                platform="cpu",
                api_version=1,
            )
        _REGISTERED.add(_SELECTED_SCALE_ALIAS_REGISTRATION_KEY)
    return target


def _ensure_structured_reduction_registered(
    input_dtype: str,
    output_dtype: str,
    scalar_kind: StridedScalarKind,
    *,
    grouped_output_owner: bool = False,
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
        "tensor0_stride_structured_reduction_"
        f"{'grouped_' if grouped_output_owner else ''}"
        f"{input_dtype}_{output_dtype}_{policy_name}_cpu_"
        "v3"
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
                        f"{registered_input}_{registered_output}_{policy}_cpu_v3"
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
                jax.ffi.register_ffi_target(
                    (
                        "tensor0_stride_structured_reduction_grouped_"
                        f"{registered_input}_{registered_output}_{policy}_cpu_v3"
                    ),
                    {
                        "instantiate": registration[
                            "instantiate_grouped_reduction"
                        ],
                        "execute": registration[execute_key],
                    },
                    platform="cpu",
                    api_version=1,
                )
        _REGISTERED.add(_STRUCTURED_REDUCTION_REGISTRATION_KEY)
    return target


def _affine_ffi_call(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    result_dtype: jnp.dtype,
    alias_source_result: bool = False,
) -> Array:
    dtype = jnp.dtype(result_dtype)
    target = _ensure_affine_registered(dtype.name)
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


def _mixed_ffi_call(source: Array, plan: NativeCallSpec) -> Array:
    semantic = plan.semantic
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


def _scale_tangent_ffi_call(base: Array, factor: Array, *, descriptor: bytes) -> Array:
    return _scale_ffi_call(
        base, factor, descriptor=descriptor,
        target=_ensure_scale_tangent_registered(base.dtype.name),
    )


def _scale_ffi_call(base: Array, factor: Array, *, descriptor: bytes, target: str) -> Array:
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


def _linear_update_ffi_call(
    base: Array,
    source: Array,
    alpha: Array,
    beta: Array,
    *,
    descriptor: bytes,
    target: str,
    stages: tuple[tuple[str, ...], ...],
) -> Array:
    from ._native_descriptor import _dtype_code

    batch_shape = base.shape[:-1]
    batch_count = prod(batch_shape)
    flat_base = jnp.reshape(base, (batch_count * base.shape[-1],))
    flat_source = jnp.reshape(source, (batch_count * source.shape[-1],))
    flat_alpha = jnp.reshape(alpha, (alpha.size,))
    flat_beta = jnp.reshape(beta, (beta.size,))
    result = jax.ShapeDtypeStruct(flat_base.shape, base.dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,), (0,), (0,), (0,)),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    stage_codes = np.asarray(
        [_dtype_code(jnp.dtype(dtype)) for stage in stages for dtype in stage],
        dtype=np.uint64,
    )
    flat_result = call(
        flat_base,
        flat_source,
        flat_alpha,
        flat_beta,
        descriptor=attribute,
        stages=stage_codes,
    )
    return jnp.reshape(flat_result, base.shape)


def _update_tangent_ffi_call(
    base: Array,
    source: Array,
    alpha: Array,
    beta: Array,
    *,
    descriptor: bytes,
    stages: tuple[tuple[str, str, str, str], ...],
) -> Array:
    return _linear_update_ffi_call(
        base, source, alpha, beta, descriptor=descriptor, stages=stages,
        target=_ensure_axpby_registered(base.dtype.name),
    )


def _update_ffi_call(
    base: Array,
    source: Array,
    alpha: Array,
    beta: Array,
    *,
    descriptor: bytes,
    stages: tuple[tuple[str, ...], ...],
) -> Array:
    return _linear_update_ffi_call(
        base, source, alpha, beta, descriptor=descriptor, stages=stages,
        target=_ensure_update_registered(source.dtype.name, base.dtype.name),
    )


def _dot_ffi_call(
    left: Array,
    right: Array,
    *,
    descriptor: bytes,
    conjugate_left: bool,
) -> Array:
    target = _ensure_dot_registered(
        left.dtype.name,
        conjugate_left=conjugate_left,
    )
    batch_shape = left.shape[:-1]
    batch_count = prod(batch_shape)
    flat_left = jnp.reshape(left, (batch_count * left.shape[-1],))
    flat_right = jnp.reshape(right, (batch_count * right.shape[-1],))
    result = jax.ShapeDtypeStruct((batch_count,), left.dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=((0,), (0,)),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    flat_result = call(flat_left, flat_right, descriptor=attribute)
    return jnp.reshape(flat_result, batch_shape)


def _selected_scale_alias_ffi_call(
    base: Array,
    factor: Array,
    *,
    descriptor: bytes,
) -> Array:
    target = _ensure_selected_scale_alias_registered(base.dtype.name)
    base_layout = tuple(range(base.ndim))
    flat_factor = jnp.reshape(factor, (factor.size,))
    result = jax.ShapeDtypeStruct(base.shape, base.dtype)
    call = jax.ffi.ffi_call(
        target,
        result,
        input_layouts=(base_layout, (0,)),
        output_layouts=base_layout,
        input_output_aliases={0: 0},
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    return call(base, flat_factor, descriptor=attribute)


def _structured_reduction_ffi_call(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    output_dtype: jnp.dtype,
    scalar_kind: StridedScalarKind = StridedScalarKind.JAX_TRANSPOSE,
    grouped_output_owner: bool = False,
) -> Array:
    dtype = jnp.dtype(output_dtype)
    target = _ensure_structured_reduction_registered(
        source.dtype.name,
        dtype.name,
        scalar_kind,
        grouped_output_owner=grouped_output_owner,
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
