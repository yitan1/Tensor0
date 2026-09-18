"""Native availability for tests of the CPU execution boundary."""

from importlib.metadata import version

import jax

from tensor0 import _native


def native_available() -> bool:
    return (
        hasattr(_native, "_stride_native_registration")
        and _native._stride_ffi_available()
        and _native._stride_ffi_build_versions() == (jax.__version__, version("jaxlib"))
    )
