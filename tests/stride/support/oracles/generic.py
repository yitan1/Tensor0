"""Generic fixtures, reference, and lowering checks."""

from math import prod

import jax.numpy as jnp
import numpy as np

from tensor0._stride._layout import AffineRecord, contiguous_strides

from tests.stride.support.oracles.addresses import record_addresses as addresses


def blocked_record(shape):
    strides = contiguous_strides(shape)
    source_strides = (-strides[0], *strides[1:]) if shape else ()
    offset = (shape[0] - 1) * strides[0] if shape else 0
    return AffineRecord(shape, source_strides, offset, tuple(prod(shape[:axis]) for axis in range(len(shape))), 0)


def dtype_values(dtype, size):
    raw = np.arange(size) % 17
    if dtype == jnp.bool_:
        return jnp.asarray(raw % 3 != 0)
    result = jnp.asarray(raw, dtype=dtype)
    return result + 1j * result[::-1] if jnp.issubdtype(dtype, jnp.complexfloating) else result


def cast(value, dtype):
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
        value = jnp.real(value)
    return value.astype(dtype)


def reference_map(source, record, coefficient, dtype, output_size):
    source_indices, destinations = addresses(record)
    selected = source[..., source_indices] * coefficient
    return jnp.zeros((*source.shape[:-1], output_size), dtype=dtype).at[..., destinations].set(cast(selected, dtype))


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if jnp.issubdtype(actual.dtype, jnp.integer) or actual.dtype == jnp.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 2e-3 if actual.dtype in (jnp.float16, jnp.bfloat16) else 2e-6
        for component in (jnp.real, jnp.imag):
            np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                       np.asarray(component(expected)).astype(np.float64),
                                       rtol=tolerance, atol=1e-7, equal_nan=True)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


HALF_LAYOUTS = [
    (AffineRecord((257,), (1,), 0, (1,), 0), 257),
    (AffineRecord((128, 257), (512, 1), 0, (257, 1), 0), 65536),
    (AffineRecord((65, 129), (1, 65), 0, (129, 1), 0), 8385),
    (AffineRecord((65, 129), (129, 1), 0, (1, 65), 0), 8385),
    (AffineRecord((512, 512), (1, 512), 0, (512, 1), 0), 262144),
    (AffineRecord((512, 512), (512, 1), 0, (1, 512), 0), 262144),
]


def complex_values(size):
    components = [0., -0., 1.25, -1.25, .125, 1e10, -1e10, 1e-10]
    return jnp.asarray(np.resize(np.asarray([complex(real, imaginary) for real in components
                                             for imaginary in components], dtype=np.complex64), size))
