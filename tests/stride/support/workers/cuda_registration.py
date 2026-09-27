"""Check fresh CUDA FFI registration without mutating the pytest process."""

import gc
from unittest.mock import patch

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride._ffi import _registration


def check_fresh_registration():
    assert not _registration._CUDA_REGISTERED
    gc.collect()
    cuda_before = _native._stride_cuda_copy_prepared_stats()
    cpu_before = _native._stride_native_prepared_stats()
    devices = jax.devices
    register_type = jax.ffi.register_ffi_type
    register_target = jax.ffi.register_ffi_target
    types, targets = [], []

    def checked_devices(backend=None):
        assert backend != "cpu", "CUDA registration must not initialize CPU"
        return devices(backend)

    def record_type(name, handler, **kwargs):
        types.append((name, handler, kwargs))
        return register_type(name, handler, **kwargs)

    def record_target(name, handler, **kwargs):
        targets.append((name, handler, kwargs))
        return register_target(name, handler, **kwargs)

    with (patch.object(jax, "devices", checked_devices),
          patch.object(jax.ffi, "register_ffi_type", record_type),
          patch.object(jax.ffi, "register_ffi_target", record_target)):
        dtype = np.dtype("float32")
        _registration.cuda_copy_target(dtype, dtype)
        _registration.cuda_copy_target(dtype, dtype)
        _registration.cuda_update_target(dtype, dtype, dtype, dtype)
        _registration.cuda_accumulation_target(dtype, dtype)
        _registration.cuda_dot_target(dtype, dtype, dtype)
        _registration.cuda_reduction_target(dtype, dtype)
    assert len(types) == 5
    assert [name for name, _, _ in types] == [
        "tensor0_stride_cuda_copy_prepared_v1", "tensor0_stride_cuda_update_prepared_v1",
        "tensor0_stride_cuda_accumulation_prepared_v1", "tensor0_stride_cuda_dot_prepared_v1",
        "tensor0_stride_cuda_reduction_prepared_v1"]
    assert all(set(handler) == {"type_id", "type_info"} for _, handler, _ in types)
    assert all(options["platform"].lower() == "cuda" for _, _, options in types)
    assert len(targets) == 5
    assert all(set(handler) == {"instantiate", "execute"} for _, handler, _ in targets)
    assert all(options["platform"].lower() == "cuda" for _, _, options in targets)
    gc.collect()
    assert _native._stride_cuda_copy_prepared_stats() == cuda_before
    assert _native._stride_native_prepared_stats() == cpu_before
