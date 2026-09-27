"""Isolated CUDA Dot prepared executable destruction and queued results."""

import gc

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_dot
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord


_EXIT_KEEPALIVE = None


def _collect_until(expected):
    for _ in range(10):
        gc.collect()
        if _native._stride_cuda_dot_prepared_stats() == expected:
            return
    assert _native._stride_cuda_dot_prepared_stats() == expected


def _check_lifecycle(destruction):
    global _EXIT_KEEPALIVE
    assert destruction in ("local", "global", "exit", "compile_only", "lazy_registration")
    if destruction == "lazy_registration":
        from tensor0._stride._ffi import _registration
        assert not _registration._CUDA_REGISTERED
        dtype = np.dtype("complex64")
        _registration.cuda_dot_target(dtype, dtype, dtype)
        assert not _registration._REGISTERED
        destruction = "local"
    device = jax.devices("cuda")[0]
    assert _native._stride_cuda_available()
    jax.clear_caches()
    gc.collect()
    baseline = _native._stride_cuda_dot_prepared_stats()
    unrelated = (_native._stride_native_prepared_stats(),
                 _native._stride_cuda_copy_prepared_stats(),
                 _native._stride_cuda_update_prepared_stats(),
                 _native._stride_cuda_accumulation_prepared_stats())
    size = 1025
    host_left = np.full((256, size), 1 + 2j, np.complex64)
    host_right = np.full((256, size), 2 - 1j, np.complex64)
    left, right = (jax.device_put(v, device) for v in (host_left, host_right))
    left.block_until_ready()
    right.block_until_ready()
    layout = encode_layout((AffineRecord((0,), (1,), 0, (1,), 0),
                            AffineRecord((size,), (-1,), size - 1, (1,), 0)),
                           source_size=size, output_size=size)
    expected = np.full(256, size * (1 - 2j) * (2 - 1j), np.complex64)

    def dot(a, b):
        return execute_dot(a, b, layout=layout, conjugate_left=True, platform="cuda")

    operation = jax.jit(dot)
    compiled = operation.lower(left, right).compile()
    assert _native._stride_cuda_dot_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "compile_only":
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 1, baseline[1] + 1))
        assert (_native._stride_native_prepared_stats(),
                _native._stride_cuda_copy_prepared_stats(),
                _native._stride_cuda_update_prepared_stats(),
                _native._stride_cuda_accumulation_prepared_stats()) == unrelated
        return
    np.testing.assert_array_equal(compiled(left, right), expected)
    results = [compiled(left, right) for _ in range(8)]
    assert _native._stride_cuda_dot_prepared_stats() == (baseline[0] + 1, baseline[1])
    if destruction == "exit":
        for result in results:
            result.block_until_ready()
            np.testing.assert_array_equal(result, expected)
        _EXIT_KEEPALIVE = (operation, compiled, left, right, results)
        assert _native._stride_cuda_dot_prepared_stats() == (baseline[0] + 1, baseline[1])
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

        operation = jax.jit(dot)
        compiled = operation.lower(left, right).compile()
        assert _native._stride_cuda_dot_prepared_stats() == (baseline[0] + 2, baseline[1] + 1)
        np.testing.assert_array_equal(compiled(left, right), expected)
        del compiled
        operation.clear_cache()
        del operation
        _collect_until((baseline[0] + 2, baseline[1] + 2))
    assert (_native._stride_native_prepared_stats(),
            _native._stride_cuda_copy_prepared_stats(),
            _native._stride_cuda_update_prepared_stats(),
            _native._stride_cuda_accumulation_prepared_stats()) == unrelated
