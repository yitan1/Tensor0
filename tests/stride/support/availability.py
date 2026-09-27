"""Native availability for tests of the CPU execution boundary."""

from importlib.metadata import version

import jax
import pytest

from tensor0 import _native


def cuda_device_or_skip():
    """Skip optional GPU tests when the extension or device is unavailable."""
    if not _native._stride_cuda_available():
        pytest.skip("CUDA extension unavailable")
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    if not devices:
        pytest.skip("CUDA device is unavailable")
    return devices[0]


def native_available() -> bool:
    return (
        hasattr(_native, "_stride_native_registration")
        and _native._stride_ffi_available()
        and _native._stride_ffi_build_versions() == (jax.__version__, version("jaxlib"))
    )
