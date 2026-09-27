"""Isolated executable lifecycle for CUDA Reduction."""
import gc
import jax
import numpy as np
from tensor0 import _native
from tensor0._stride._ffi import _calls, _registration
from tensor0._stride._ffi._descriptor import encode_reduction_layout
from tensor0._stride._layout import AffineRecord

_KEEPALIVE = None


def _check_lifecycle(destruction):
    global _KEEPALIVE
    if destruction == "lazy_registration":
        assert not _registration._CUDA_REGISTERED
        _registration.cuda_reduction_target(np.dtype("float32"), np.dtype("float32"))
        assert not _registration._REGISTERED
        destruction = "local"
    device = jax.devices("cuda")[0]
    jax.clear_caches()
    gc.collect()
    before = _native._stride_cuda_reduction_prepared_stats()
    other = (_native._stride_native_prepared_stats(), *(
        getattr(_native, f"_stride_cuda_{name}_prepared_stats")()
        for name in ("copy", "update", "accumulation", "dot")))
    record = AffineRecord((2, 2), (2, 1), 0, (0, 1), 0)
    layout = encode_reduction_layout((record,), output_shapes=((1, 2),),
        reduction_axes=((True, False),), source_size=4, output_size=2)
    source = jax.device_put(np.ones((64, 4), np.float32), device)
    op = jax.jit(lambda x: _calls.execute_reduction(x, layout=layout, output_size=2, platform="cuda"))
    compiled = op.lower(source).compile()
    assert _native._stride_cuda_reduction_prepared_stats() == (before[0] + 1, before[1])
    if destruction == "compile_only":
        del compiled
        op.clear_cache()
        del op
    else:
        outputs = [compiled(source) for _ in range(8)]
        if destruction == "exit":
            for value in outputs: np.testing.assert_array_equal(value, np.full((64, 2), 2, np.float32))
            _KEEPALIVE = (op, compiled, outputs, source)
            return
        del compiled
        if destruction == "global": jax.clear_caches()
        else: op.clear_cache()
        del op
        gc.collect()
        for value in outputs: np.testing.assert_array_equal(value, np.full((64, 2), 2, np.float32))
    for _ in range(10):
        gc.collect()
        if _native._stride_cuda_reduction_prepared_stats() == (before[0] + 1, before[1] + 1): break
    assert _native._stride_cuda_reduction_prepared_stats() == (before[0] + 1, before[1] + 1)
    assert (_native._stride_native_prepared_stats(), *(
        getattr(_native, f"_stride_cuda_{name}_prepared_stats")()
        for name in ("copy", "update", "accumulation", "dot"))) == other
