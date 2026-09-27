"""Isolated CUDA Accumulation executable destruction with queued results."""

import gc

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_accumulation
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord


_EXIT_KEEPALIVE = None


def _collect_until(expected):
    for _ in range(10):
        gc.collect()
        if _native._stride_cuda_accumulation_prepared_stats() == expected:
            return
    assert _native._stride_cuda_accumulation_prepared_stats() == expected


def _check_lifecycle(destruction):
    global _EXIT_KEEPALIVE
    assert destruction in ("local", "global", "exit", "compile_only", "lazy_registration")
    if destruction == "lazy_registration":
        from tensor0._stride._ffi import _registration
        assert not _registration._CUDA_REGISTERED
        dtype = np.dtype("float32")
        _registration.cuda_accumulation_target(dtype, dtype, dtype)
        assert not _registration._REGISTERED
        destruction = "local"
    device = jax.devices("cuda")[0]
    assert _native._stride_cuda_available()
    jax.clear_caches()
    gc.collect()
    baseline = _native._stride_cuda_accumulation_prepared_stats()
    unrelated = (_native._stride_native_prepared_stats(),
                 _native._stride_cuda_copy_prepared_stats(),
                 _native._stride_cuda_update_prepared_stats())
    size = 257
    host = np.resize(np.array([1, 2, 3], np.float32), (1024, size))
    source = jax.device_put(host, device)
    coefficient = jax.device_put(np.array(2, np.float32), device)
    source.block_until_ready()
    coefficient.block_until_ready()
    layout = encode_layout((AffineRecord((size,), (-1,), size - 1, (0,), 1),),
                           source_size=size, output_size=3)
    expected = np.zeros((1024, 3), np.float32)
    expected[:, 1] = 2 * host.sum(axis=1)

    def accumulate(value, factor):
        return execute_accumulation(value, (factor,), coefficient_records=(0,),
                                    layout=layout, output_size=3, platform="cuda")

    operation = jax.jit(accumulate)
    compiled = operation.lower(source, coefficient).compile()
    assert _native._stride_cuda_accumulation_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "compile_only":
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 1, baseline[1] + 1))
        assert (_native._stride_native_prepared_stats(),
                _native._stride_cuda_copy_prepared_stats(),
                _native._stride_cuda_update_prepared_stats()) == unrelated
        return
    np.testing.assert_array_equal(compiled(source, coefficient), expected)
    results = [compiled(source, coefficient) for _ in range(8)]
    assert _native._stride_cuda_accumulation_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "exit":
        for result in results:
            result.block_until_ready()
            np.testing.assert_array_equal(result, expected)
        _EXIT_KEEPALIVE = (operation, compiled, source, coefficient, results)
        assert _native._stride_cuda_accumulation_prepared_stats() == (baseline[0] + 1, baseline[1])
    else:
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

        operation = jax.jit(accumulate)
        compiled = operation.lower(source, coefficient).compile()
        assert _native._stride_cuda_accumulation_prepared_stats() == (baseline[0] + 2, baseline[1] + 1)
        np.testing.assert_array_equal(compiled(source, coefficient), expected)
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 2, baseline[1] + 2))
    assert (_native._stride_native_prepared_stats(),
            _native._stride_cuda_copy_prepared_stats(),
            _native._stride_cuda_update_prepared_stats()) == unrelated
