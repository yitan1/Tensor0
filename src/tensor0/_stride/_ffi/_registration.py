"""Independent, lazy registration of native CPU and CUDA handlers."""

from importlib.metadata import version
from threading import Lock

import jax

from ... import _native


_STORAGE_SUFFIXES = {
    "bool": "pred",
    "int8": "s8", "int16": "s16", "int32": "s32", "int64": "s64",
    "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
    "float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64",
    "complex64": "c64", "complex128": "c128",
}
_LOCK = Lock()
_REGISTERED: set[str] = set()


def operation_target(operation: str, dtype) -> str:
    if operation not in ("copy", "update", "reduction", "dot", "accumulation"):
        raise NotImplementedError(f"native operation {operation} is not migrated yet")
    suffix = _STORAGE_SUFFIXES.get(dtype.name)
    if suffix is None:
        raise NotImplementedError(f"native {operation} does not support {dtype}")
    target = f"tensor0_stride_{operation}_{suffix}_cpu_v1"
    with _LOCK:
        if target in _REGISTERED:
            return target
        if not hasattr(_native, "_stride_native_registration") or not _native._stride_ffi_available():
            raise RuntimeError("native CPU execution is unavailable; rebuild the extension")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_native_registration()
        prefix = operation + "_" if operation in ("reduction", "dot", "accumulation") else ""
        if not _REGISTERED:
            jax.devices("cpu")
            jax.ffi.register_ffi_type(
                "tensor0_stride_prepared_v1",
                {name: registration[name] for name in ("type_id", "type_info")},
                platform="cpu",
            )
        jax.ffi.register_ffi_target(
            target,
            {"instantiate": registration[prefix + "instantiate"], "execute": registration[f"{operation}_{suffix}"]},
            platform="cpu", api_version=1,
        )
        _REGISTERED.add(target)
    return target


_CUDA_REGISTERED: set[str] = set()


def cuda_copy_target(source_dtype, result_dtype) -> str:
    if source_dtype != result_dtype or result_dtype.name not in (
            "float32", "float64", "complex64", "complex128"):
        raise NotImplementedError(
            "native CUDA copy supports only same-dtype F32/F64/C64/C128 storage; "
            f"got {source_dtype} -> {result_dtype}")
    suffix = _STORAGE_SUFFIXES[result_dtype.name]
    target = f"tensor0_stride_copy_{suffix}_cuda_v1"
    with _LOCK:
        if target in _CUDA_REGISTERED:
            return target
        if (not hasattr(_native, "_stride_cuda_registration")
                or not hasattr(_native, "_stride_cuda_available")
                or not _native._stride_cuda_available()):
            raise RuntimeError("native CUDA copy is unavailable; rebuild the extension with CUDA support")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_cuda_registration()
        if not any(name.startswith("tensor0_stride_copy_") for name in _CUDA_REGISTERED):
            jax.ffi.register_ffi_type(
                "tensor0_stride_cuda_copy_prepared_v1",
                {name: registration[f"copy_{name}"] for name in ("type_id", "type_info")},
                platform="CUDA",
            )
        jax.ffi.register_ffi_target(
            target, {"instantiate": registration["copy_instantiate"],
                     "execute": registration[f"copy_{suffix}"]},
            platform="CUDA", api_version=1,
        )
        _CUDA_REGISTERED.add(target)
    return target


_CUDA_UPDATE_COEFFICIENTS = {
    "float32": ("int32", "float32"),
    "float64": ("int32", "float64"),
    "complex64": ("int32", "float32", "complex64"),
    "complex128": ("int32", "float64", "complex128"),
}


def cuda_update_target(source_dtype, base_dtype, alpha_dtype, beta_dtype) -> str:
    if source_dtype != base_dtype or base_dtype.name not in _CUDA_UPDATE_COEFFICIENTS:
        raise NotImplementedError(
            "native CUDA update supports only same-dtype F32/F64/C64/C128 storage; "
            f"got {source_dtype} -> {base_dtype}")
    allowed = _CUDA_UPDATE_COEFFICIENTS[base_dtype.name]
    for name, dtype in (("alpha", alpha_dtype), ("beta", beta_dtype)):
        if dtype.name not in allowed:
            raise NotImplementedError(
                f"native CUDA update {name} coefficient dtype {dtype} is not supported "
                f"for {base_dtype}; expected one of {allowed}")
    suffix = _STORAGE_SUFFIXES[base_dtype.name]
    target = f"tensor0_stride_update_{suffix}_cuda_v1"
    with _LOCK:
        if target in _CUDA_REGISTERED:
            return target
        if (not hasattr(_native, "_stride_cuda_registration")
                or not hasattr(_native, "_stride_cuda_available")
                or not _native._stride_cuda_available()):
            raise RuntimeError("native CUDA update is unavailable; rebuild the extension with CUDA support")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_cuda_registration()
        if not any(name.startswith("tensor0_stride_update_") for name in _CUDA_REGISTERED):
            jax.ffi.register_ffi_type(
                "tensor0_stride_cuda_update_prepared_v1",
                {name: registration[f"update_{name}"] for name in ("type_id", "type_info")},
                platform="CUDA",
            )
        jax.ffi.register_ffi_target(
            target, {"instantiate": registration["update_instantiate"],
                     "execute": registration[f"update_{suffix}"]},
            platform="CUDA", api_version=1,
        )
        _CUDA_REGISTERED.add(target)
    return target


def cuda_accumulation_target(source_dtype, result_dtype, *coefficient_dtypes) -> str:
    if source_dtype != result_dtype or result_dtype.name not in _CUDA_UPDATE_COEFFICIENTS:
        raise NotImplementedError(
            "native CUDA accumulation supports only same-dtype F32/F64/C64/C128 storage; "
            f"got {source_dtype} -> {result_dtype}")
    allowed = _CUDA_UPDATE_COEFFICIENTS[result_dtype.name]
    for name, dtype in enumerate(coefficient_dtypes):
        if dtype.name not in allowed:
            raise NotImplementedError(
                f"native CUDA accumulation {name} coefficient dtype {dtype} is not supported "
                f"for {result_dtype}; expected one of {allowed}")
    suffix = _STORAGE_SUFFIXES[result_dtype.name]
    target = f"tensor0_stride_accumulation_{suffix}_cuda_v1"
    with _LOCK:
        if target in _CUDA_REGISTERED:
            return target
        if (not hasattr(_native, "_stride_cuda_registration")
                or not hasattr(_native, "_stride_cuda_available")
                or not _native._stride_cuda_available()):
            raise RuntimeError("native CUDA accumulation is unavailable; rebuild the extension with CUDA support")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_cuda_registration()
        if not any(name.startswith("tensor0_stride_accumulation_") for name in _CUDA_REGISTERED):
            jax.ffi.register_ffi_type(
                "tensor0_stride_cuda_accumulation_prepared_v1",
                {name: registration[f"accumulation_{name}"] for name in ("type_id", "type_info")},
                platform="CUDA",
            )
        jax.ffi.register_ffi_target(
            target, {"instantiate": registration["accumulation_instantiate"],
                     "execute": registration[f"accumulation_{suffix}"]},
            platform="CUDA", api_version=1,
        )
        _CUDA_REGISTERED.add(target)
    return target


def cuda_dot_target(source_dtype, right_dtype, result_dtype) -> str:
    if right_dtype != result_dtype or source_dtype != result_dtype or result_dtype.name not in (
            "float32", "float64", "complex64", "complex128"):
        raise NotImplementedError(
            "native CUDA dot supports only same-dtype F32/F64/C64/C128 inputs and result; "
            f"got {source_dtype}, {right_dtype} -> {result_dtype}")
    suffix = _STORAGE_SUFFIXES[result_dtype.name]
    target = f"tensor0_stride_dot_{suffix}_cuda_v1"
    with _LOCK:
        if target in _CUDA_REGISTERED:
            return target
        if (not hasattr(_native, "_stride_cuda_registration")
                or not hasattr(_native, "_stride_cuda_available")
                or not _native._stride_cuda_available()):
            raise RuntimeError("native CUDA dot is unavailable; rebuild the extension with CUDA support")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_cuda_registration()
        if not any(name.startswith("tensor0_stride_dot_") for name in _CUDA_REGISTERED):
            jax.ffi.register_ffi_type(
                "tensor0_stride_cuda_dot_prepared_v1",
                {name: registration[f"dot_{name}"] for name in ("type_id", "type_info")},
                platform="CUDA",
            )
        jax.ffi.register_ffi_target(
            target, {"instantiate": registration["dot_instantiate"],
                     "execute": registration[f"dot_{suffix}"]},
            platform="CUDA", api_version=1,
        )
        _CUDA_REGISTERED.add(target)
    return target


def cuda_reduction_target(source_dtype, result_dtype, *coefficient_dtypes) -> str:
    if source_dtype != result_dtype or result_dtype.name not in _CUDA_UPDATE_COEFFICIENTS:
        raise NotImplementedError(
            "native CUDA reduction supports only same-dtype F32/F64/C64/C128 storage; "
            f"got {source_dtype} -> {result_dtype}")
    allowed = _CUDA_UPDATE_COEFFICIENTS[result_dtype.name]
    for name, dtype in enumerate(coefficient_dtypes):
        if dtype.name not in allowed:
            raise NotImplementedError(
                f"native CUDA reduction {name} coefficient dtype {dtype} is not supported "
                f"for {result_dtype}; expected one of {allowed}")
    suffix = _STORAGE_SUFFIXES[result_dtype.name]
    target = f"tensor0_stride_reduction_{suffix}_cuda_v1"
    with _LOCK:
        if target in _CUDA_REGISTERED:
            return target
        if (not hasattr(_native, "_stride_cuda_registration")
                or not hasattr(_native, "_stride_cuda_available")
                or not _native._stride_cuda_available()):
            raise RuntimeError("native CUDA reduction is unavailable; rebuild the extension with CUDA support")
        if _native._stride_ffi_build_versions() != (jax.__version__, version("jaxlib")):
            raise RuntimeError("native was built for different JAX versions; rebuild the extension")
        registration = _native._stride_cuda_registration()
        if not any(name.startswith("tensor0_stride_reduction_") for name in _CUDA_REGISTERED):
            jax.ffi.register_ffi_type(
                "tensor0_stride_cuda_reduction_prepared_v1",
                {name: registration[f"reduction_{name}"] for name in ("type_id", "type_info")},
                platform="CUDA",
            )
        jax.ffi.register_ffi_target(
            target, {"instantiate": registration["reduction_instantiate"],
                     "execute": registration[f"reduction_{suffix}"]},
            platform="CUDA", api_version=1,
        )
        _CUDA_REGISTERED.add(target)
    return target
