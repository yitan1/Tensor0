"""Process-wide native worker limits, read at execution rather than tracing.

Wait for outstanding work before changing the limit. XLA pool capacity and
workload size may reduce the actual number of workers below this upper bound.
"""

from .. import _native


def _require_native_thread_control() -> None:
    if not _native._stride_ffi_available():
        raise RuntimeError("native CPU stride execution is unavailable")


def get_num_threads() -> int | None:
    """Return the worker upper bound, or None for no additional limit."""
    _require_native_thread_control()
    return _native._stride_native_worker_limit()


def set_num_threads(n: int) -> None:
    """Set the worker upper bound without recompiling existing executables."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n >= (1 << 64) - 1:
        raise ValueError("n exceeds the configurable native worker-limit range")
    _require_native_thread_control()
    _native._set_stride_native_worker_limit(n)


def enable_threads() -> None:
    """Remove the additional worker limit; workload and XLA still bound it."""
    _require_native_thread_control()
    _native._set_stride_native_worker_limit(None)


def disable_threads() -> None:
    """Limit native execution to one worker."""
    set_num_threads(1)
