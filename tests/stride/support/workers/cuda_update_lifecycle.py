"""Isolated CUDA Update destruction checks, including outstanding launch results."""

import gc

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord


_EXIT_KEEPALIVE = None


def _collect_until(expected):
    # Every queued result is synchronized before checking finalizer counters.
    for _ in range(10):
        gc.collect()
        if _native._stride_cuda_update_prepared_stats() == expected:
            return
    assert _native._stride_cuda_update_prepared_stats() == expected


def _check_lifecycle(destruction):
    global _EXIT_KEEPALIVE
    assert destruction in ("local", "global", "exit", "compile_only", "lazy_registration")
    if destruction == "lazy_registration":
        from tensor0._stride._ffi import _registration
        assert not _registration._CUDA_REGISTERED
        dtype = np.dtype("float32")
        _registration.cuda_update_target(dtype, dtype, dtype, dtype)
        assert not _registration._REGISTERED
        destruction = "local"
    device = jax.devices("cuda")[0]
    assert _native._stride_cuda_available()
    jax.clear_caches()
    gc.collect()
    baseline = _native._stride_cuda_update_prepared_stats()
    copy_baseline = _native._stride_cuda_copy_prepared_stats()
    cpu_baseline = _native._stride_native_prepared_stats()
    size = 257
    host_source = np.arange(1024 * size, dtype=np.float32).reshape(1024, size)
    host_base = np.full((1024, 2 * size + 1), 3, np.float32)
    source = jax.device_put(host_source, device)
    base = jax.device_put(host_base, device)
    alpha = jax.device_put(np.array(2, np.float32), device)
    beta = jax.device_put(np.array(3, np.float32), device)
    for value in (source, base, alpha, beta):
        value.block_until_ready()
    layout = encode_layout((AffineRecord((size,), (-1,), size - 1, (2,), 1),),
                           source_size=size, output_size=2 * size + 1)
    expected = host_base.copy()
    expected[:, 1:2 * size:2] = 2 * host_source[:, ::-1] + 3 * host_base[:, 1:2 * size:2]

    def update(s, b, a, c):
        return execute_update(s, b, a, c, layout=layout, platform="cuda")

    operation = jax.jit(update)
    compiled = operation.lower(source, base, alpha, beta).compile()
    assert _native._stride_cuda_update_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "compile_only":
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 1, baseline[1] + 1))
        assert _native._stride_cuda_copy_prepared_stats() == copy_baseline
        assert _native._stride_native_prepared_stats() == cpu_baseline
        return
    np.testing.assert_array_equal(compiled(source, base, alpha, beta), expected)
    results = [compiled(source, base, alpha, beta) for _ in range(8)]
    assert _native._stride_cuda_update_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "exit":
        for result in results:
            result.block_until_ready()
            np.testing.assert_array_equal(result, expected)
        _EXIT_KEEPALIVE = (operation, compiled, source, base, alpha, beta, results)
        assert _native._stride_cuda_update_prepared_stats() == (baseline[0] + 1, baseline[1])
    else:
        # Drop executable and cache before waiting on any queued output.
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

        operation = jax.jit(update)
        compiled = operation.lower(source, base, alpha, beta).compile()
        assert _native._stride_cuda_update_prepared_stats() == (baseline[0] + 2, baseline[1] + 1)
        np.testing.assert_array_equal(compiled(source, base, alpha, beta), expected)
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 2, baseline[1] + 2))
    assert _native._stride_cuda_copy_prepared_stats() == copy_baseline
    assert _native._stride_native_prepared_stats() == cpu_baseline
