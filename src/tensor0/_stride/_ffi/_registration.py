"""Independent, lazy registration of native CPU operation handlers."""

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
