"""Load the optional CUDA companion only when native CUDA execution is requested."""

import ctypes
import importlib.metadata
import os
from pathlib import Path
import re
from threading import Lock

_libraries: tuple[ctypes.CDLL, ctypes.CDLL] | None = None
_lock = Lock()
# Keep runtime NOLOAD references and even rejected companions alive through exit.
_retained: list[ctypes.CDLL] = []
_failure: str | None = None
_REQUIRED_SYMBOLS = (
    "cudaGetErrorString", "cudaGetLastError", "cudaMemcpyAsync", "cudaMemsetAsync",
    "cudaLaunchKernel", "__cudaPushCallConfiguration", "__cudaPopCallConfiguration",
    "__cudaRegisterFatBinary", "__cudaRegisterFatBinaryEnd", "__cudaRegisterFunction",
    "__cudaUnregisterFatBinary",
)


def _cuda_companion_present() -> bool:
    """Probe distribution presence without runtime, ELF or device initialization.

    A broken/mismatched installed addon remains visible so registration reports
    the error rather than silently treating the installation as CPU-only.
    """
    try:
        importlib.metadata.distribution("tensor0-cuda12")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _companion_path() -> Path:
    try:
        addon = importlib.metadata.distribution("tensor0-cuda12")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Tensor0 CUDA companion is missing; pip install 'tensor0[cuda12]'."
        ) from exc
    # Use installed distribution metadata, not the Cargo/native version string.
    core_version = importlib.metadata.version("tensor0")
    if addon.version != core_version:
        raise RuntimeError(
            f"Tensor0 CUDA companion version mismatch: tensor0={core_version}, "
            f"tensor0-cuda12={addon.version}; install matching distributions."
        )
    relative = "tensor0_cuda12/_stride_cuda.so"
    if not addon.files or relative not in {str(file) for file in addon.files}:
        raise RuntimeError("Tensor0 CUDA companion library is missing from distribution records.")
    companion = Path(str(addon.locate_file(relative))).resolve()
    if not companion.is_file():
        raise RuntimeError("Tensor0 CUDA companion library is missing from this installation.")
    return companion


def _load_cuda_library() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Preflight failures are retryable; constructor/status failures require restart."""
    global _libraries, _failure
    with _lock:
        if _libraries is not None:
            return _libraries
        if _failure is not None:
            raise RuntimeError(_failure)
        companion = _companion_path()
        try:
            distribution = importlib.metadata.distribution("nvidia-cuda-runtime-cu12")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Tensor0 CUDA runtime is missing; pip install 'tensor0[cuda12]'."
            ) from exc
        version = distribution.version
        if not re.fullmatch(r"\d+(?:\.\d+)*", version):
            raise RuntimeError(f"Unsupported NVIDIA CUDA runtime version: {version}")
        parts = tuple(map(int, version.split(".")))
        parts += (0,) * max(0, 3 - len(parts))
        if not ((12, 9, 79) <= parts < (13,)):
            raise RuntimeError(
                f"Tensor0 requires nvidia-cuda-runtime-cu12 >=12.9.79,<13 "
                f"(found {version}); pip install 'tensor0[cuda12]'."
            )
        runtime = Path(str(distribution.locate_file(
            "nvidia/cuda_runtime/lib/libcudart.so.12"
        ))).resolve()
        if not runtime.is_file():
            raise RuntimeError(
                "Tensor0 CUDA runtime library is missing; "
                "pip install 'tensor0[cuda12]'."
            )
        try:
            runtime_library = ctypes.CDLL(str(runtime), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
            _retained.append(runtime_library)
            # Probe the resident SONAME without searching for or loading another runtime.
            resident_runtime = ctypes.CDLL(
                "libcudart.so.12", mode=os.RTLD_NOW | os.RTLD_NOLOAD
            )
            _retained.append(resident_runtime)
            if resident_runtime._handle != runtime_library._handle:
                raise RuntimeError(
                    "Tensor0 CUDA runtime conflict: another libcudart.so.12 is already "
                    "loaded. Restart in a clean environment using the pip CUDA runtime."
                )
            for symbol in _REQUIRED_SYMBOLS:
                try:
                    getattr(runtime_library, symbol)
                except AttributeError as exc:
                    raise RuntimeError(f"Tensor0 CUDA runtime is missing required symbol {symbol}.") from exc
            companion_library = ctypes.CDLL(str(companion), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        except OSError as exc:
            raise RuntimeError(f"Unable to load Tensor0 CUDA libraries: {exc}") from exc
        _retained.append(companion_library)
        try:
            status = companion_library.Tensor0CudaForwarderStatusV1
            status.argtypes = []
            status.restype = ctypes.c_int
            bound = companion_library.Tensor0CudaForwarderRuntimeV1
            bound.argtypes = []
            bound.restype = ctypes.c_void_p
            error = companion_library.Tensor0CudaForwarderErrorV1
            error.argtypes = []
            error.restype = ctypes.c_char_p
            if status() != 0:
                raise RuntimeError(f"forwarder initialization failed: {error()!r}")
            if bound() != runtime_library._handle:
                raise RuntimeError("forwarder runtime identity mismatch")
        except (AttributeError, RuntimeError) as exc:
            _failure = f"Tensor0 CUDA companion validation failed: {exc}. Restart in a clean environment."
            raise RuntimeError(_failure) from exc
        _libraries = runtime_library, companion_library
        return _libraries
