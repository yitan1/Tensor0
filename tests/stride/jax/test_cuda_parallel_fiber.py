"""Long fibers with multiple independent output owners on the shared CUDA strategy."""

import jax
import numpy as np
import pytest

from tensor0 import _native
from tests.stride.support.availability import cuda_device_or_skip
from tensor0._stride import _jax
from tensor0._stride._layout import AffineRecord


@pytest.mark.parametrize("operation", ("reduction", "accumulation"))
@pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
def test_parallel_multiowner_coefficients_records_and_zero(cuda_device, operation, dtype):
    records = (
        AffineRecord((2, 4096), (4096, -1), 4095, (2, 0), 1),
        AffineRecord((2, 1), (4096, 0), 0, (2, 0), 1),
    )
    host = (np.arange(2 * 8192).reshape(2, 8192) % 7 - 3).astype(dtype)
    if dtype.startswith("complex"):
        host += 1j * (np.arange(2 * 8192).reshape(2, 8192) % 5 - 2)
    coefficient = (np.asarray([0, 2 + 1j], dtype=dtype) if dtype.startswith("complex")
                   else np.asarray([0, 2], dtype=np.int32))
    expected = np.zeros((2, 5), dtype=dtype)
    for batch in range(2):
        for owner in range(2):
            start = owner * 4096
            value = np.sum(host[batch, start:start + 4096],
                           dtype=np.complex128 if dtype.startswith("complex") else np.float64)
            expected[batch, 1 + owner * 2] = coefficient[batch] * value + host[batch, start]

    def operation_call(source, factor):
        if operation == "accumulation":
            return _jax.accumulation_p.bind(source, factor, records=records,
                output_size=5, dtype=source.dtype, coefficient_records=(0,))
        return _jax.reduction_p.bind(source, factor, records=records,
            output_shapes=((2, 1), (2, 1)), reduction_axes=((False, True), (False, True)),
            output_size=5, dtype=source.dtype, coefficient_records=(0,))

    with jax.enable_x64():
        source = jax.device_put(host, cuda_device)
        factor = jax.device_put(coefficient, cuda_device)
        result = jax.jit(operation_call)(source, factor)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
        assert result.devices() == {cuda_device}


@pytest.mark.parametrize("operation", ("reduction", "accumulation"))
@pytest.mark.parametrize("fiber", (1023, 1024, 1025), ids=("below", "at", "above"))
def test_sum_fiber_threshold_values(cuda_device, operation, fiber):
    record = AffineRecord((2, fiber), (fiber, 1), 0, (1, 0), 0)
    host = (np.arange(2 * fiber, dtype=np.int32) % 7 - 3).astype(np.float32)
    expected = np.array([np.sum(host[i * fiber:(i + 1) * fiber], dtype=np.float64)
                         for i in range(2)], dtype=np.float32)
    with jax.enable_x64():
        source = jax.device_put(host[None, :], cuda_device)
        if operation == "reduction":
            result = jax.jit(lambda x: _jax.reduction_p.bind(
                x, records=(record,), output_shapes=((2, 1),),
                reduction_axes=((False, True),), output_size=2, dtype=x.dtype,
                coefficient_records=()))(source)
        else:
            result = jax.jit(lambda x: _jax.accumulation_p.bind(
                x, records=(record,), output_size=2, dtype=x.dtype,
                coefficient_records=()))(source)
        np.testing.assert_array_equal(result, expected[None, :])
        assert result.devices() == {cuda_device}


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()
