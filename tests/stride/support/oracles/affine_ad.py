"""Shared affine ad fixtures and references."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._layout import AffineRecord


PARTITIONS = (
    AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
    AffineRecord((4, 2), (4, 1), 2, (4, 1), 0),
)


def reference_transform(source, records, factors, output_size, dtype):
    result = jnp.zeros(source.shape[:-1] + (output_size,), dtype=dtype)
    for record, factor in zip(records, factors, strict=True):
        source_indices = np.full(record.logical_shape, record.source_offset, dtype=np.int32)
        destination_indices = np.full(record.logical_shape, record.destination_offset, dtype=np.int32)
        for axis, size in enumerate(record.logical_shape):
            shape = (1,) * axis + (size,) + (1,) * (len(record.logical_shape) - axis - 1)
            indices = np.arange(size, dtype=np.int32).reshape(shape)
            source_indices += indices * record.source_strides[axis]
            destination_indices += indices * record.destination_strides[axis]
        values = source[..., source_indices.ravel()] * factor
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            values = jnp.real(values)
        result = result.at[..., destination_indices.ravel()].set(values.astype(dtype))
    return result


def assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = 2e-3 if actual.dtype == jnp.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def assert_native_calls(lowered, count):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == count
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text
