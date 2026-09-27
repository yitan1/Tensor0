"""Isolated CUDA Copy destruction checks, including outstanding launch results."""

import gc

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord


# Keep live executables through interpreter shutdown in the exit case.
_EXIT_KEEPALIVE = None


def _collect_until(expected):
    # Call only after synchronizing every result. No sleep or assumptions about
    # whether CUDA dispatch happened to be synchronous on this machine.
    for _ in range(10):
        gc.collect()
        if _native._stride_cuda_copy_prepared_stats() == expected:
            return
    assert _native._stride_cuda_copy_prepared_stats() == expected


def _check_lifecycle(destruction):
    global _EXIT_KEEPALIVE
    assert destruction in ("local", "global", "exit", "compile_only", "lazy_registration")
    if destruction == "lazy_registration":
        from tensor0._stride._ffi import _registration
        assert not _registration._CUDA_REGISTERED
        _registration.cuda_copy_target(np.dtype("float32"), np.dtype("float32"))
        assert not _registration._REGISTERED
        destruction = "local"
    device = jax.devices("cuda")[0]
    assert _native._stride_cuda_available()
    jax.clear_caches()
    gc.collect()
    baseline = _native._stride_cuda_copy_prepared_stats()
    cpu_baseline = _native._stride_native_prepared_stats()
    size = 257
    host = np.arange(1024 * size, dtype=np.float32).reshape(1024, size)
    source = jax.device_put(host, device)
    source.block_until_ready()
    layout = encode_layout((AffineRecord((size,), (-1,), size - 1, (2,), 1),),
                           source_size=size, output_size=2 * size + 1)
    expected = np.zeros((1024, 2 * size + 1), np.float32)
    expected[:, 1:2 * size:2] = host[:, ::-1]

    def copy(value):
        return execute_copy(value, layout=layout, output_size=2 * size + 1,
                            platform="cuda")

    operation = jax.jit(copy)
    compiled = operation.lower(source).compile()
    assert _native._stride_cuda_copy_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "compile_only":
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 1, baseline[1] + 1))
        assert _native._stride_native_prepared_stats() == cpu_baseline
        return
    np.testing.assert_array_equal(compiled(source), expected)
    results = [compiled(source) for _ in range(8)]
    assert _native._stride_cuda_copy_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "exit":
        for result in results:
            result.block_until_ready()
            np.testing.assert_array_equal(result, expected)
        _EXIT_KEEPALIVE = (operation, compiled, source, results)
        assert _native._stride_cuda_copy_prepared_stats() == (baseline[0] + 1, baseline[1])
    else:
        # Drop the executable and cache BEFORE waiting on any queued output.
        # The test makes no claim that a particular launch is still in flight;
        # correctness must hold regardless of GPU scheduling or allocator mode.
        del compiled
        if destruction == "local":
            operation.clear_cache()
        else:
            jax.clear_caches()
        del operation
        gc.collect()
        for result in results:
            result.block_until_ready()
            assert result.devices() == {device}
            np.testing.assert_array_equal(result, expected)
        _collect_until((baseline[0] + 1, baseline[1] + 1))

        # A fresh compilation owns a fresh state, not the released one.
        operation = jax.jit(copy)
        compiled = operation.lower(source).compile()
        assert _native._stride_cuda_copy_prepared_stats() == (baseline[0] + 2, baseline[1] + 1)
        np.testing.assert_array_equal(compiled(source), expected)
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 2, baseline[1] + 2))
    assert _native._stride_native_prepared_stats() == cpu_baseline
