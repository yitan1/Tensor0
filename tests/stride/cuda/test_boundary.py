"""CUDA FFI host-boundary rejection before device submission.

Descriptor contents are an internal ABI: the lowering supplies identical words
as an immutable device constant and a host attribute. These tests deliberately
bypass the lowering only to exercise independently checkable boundary metadata.
"""

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi._registration import cuda_copy_target


@pytest.fixture(scope="module")
def cuda_target():
    if not getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("extension was built without CUDA")
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    return cuda_copy_target(np.dtype("float32"), np.dtype("float32")), devices[0]


def invoke(cuda_target, words, *, source_shape=(1, 4), output_shape=(1, 4),
           source_dtype=np.float32, descriptor_size=None, alias_source=False):
    target, device = cuda_target
    words = np.asarray(words, dtype=np.int64)
    table = words if descriptor_size is None else np.zeros(descriptor_size, dtype=np.int64)
    with jax.enable_x64(), jax.default_device(device):
        operation = jax.jit(lambda source: jax.ffi.ffi_call(
            target, jax.ShapeDtypeStruct(output_shape, np.float32),
            input_output_aliases={0: 0} if alias_source else None,
        )(source, table, layout=words))
        return operation(jax.numpy.zeros(source_shape, dtype=source_dtype)).block_until_ready()


@pytest.mark.parametrize("words, message", [
    ([2, 4, 4, 0], "unsupported native layout version"),
    ([1, 4, 4, 1], "record count"),
    ([1, 4, 4, 1, 1, 0, 0, 4, 1, 0], "injective"),
    ([1, 4, 4, 1, 1, 1, 0, 4, 1, 1], "source address"),
    ([1, 4, 4, 1, 1, 0, 1, 4, 1, 1], "destination address"),
    ([1, 4, 4, 1, 1, 0, 0, 4, -1, 1], "source address"),
    ([1, 4, 4, 0, 0], "trailing words"),
    ([1, 4, 4, 1, 2, 0, 0, 1], "rank exceeds descriptor length"),
    ([1, 4, 4, 1, 1, 0, 0, -1, 1, 1], "size or offset is negative"),
    ([1, 4, 4, 1, 1, 0, 0, 3, (1 << 63) - 1, 1], "address arithmetic overflow"),
    ([1, 4, 4, 1, 2, 0, 0, (1 << 63) - 1, 3, 1, 1, 1, 1], "element count overflows"),
])
def test_invalid_host_layout(cuda_target, words, message):
    with pytest.raises(Exception, match=message):
        invoke(cuda_target, words)


@pytest.mark.parametrize("options, message", [
    ({"source_shape": (4,)}, "rank-two"),
    ({"source_shape": (2, 4)}, "dimensions"),
    ({"source_shape": (1, 3)}, "dimensions"),
    ({"output_shape": (1, 3)}, "dimensions"),
    ({"source_dtype": np.float64}, "same-dtype"),
    ({"descriptor_size": 3}, "descriptor operand shape"),
    ({"alias_source": True}, "input and output buffers overlap"),
])
def test_invalid_buffer_metadata(cuda_target, options, message):
    with pytest.raises(Exception, match=message):
        invoke(cuda_target, [1, 4, 4, 0], **options)
