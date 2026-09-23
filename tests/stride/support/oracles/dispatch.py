"""Shared dispatch fixtures and references."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._jax import accumulation_p, copy_p


def addresses(record):
    source = np.full(record.logical_shape, record.source_offset, dtype=np.int64)
    destination = np.full(record.logical_shape, record.destination_offset, dtype=np.int64)
    for axis, extent in enumerate(record.logical_shape):
        coordinate = np.arange(extent).reshape((1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1))
        source += coordinate * record.source_strides[axis]
        destination += coordinate * record.destination_strides[axis]
    return source.ravel(), destination.ravel()


def fresh(source, record, factor, output_size, dtype):
    if factor is None:
        return copy_p.bind(source, records=(record,), output_size=output_size, dtype=jnp.dtype(dtype))
    return accumulation_p.bind(source, factor, records=(record,), coefficient_records=(0,),
                               output_size=output_size, dtype=jnp.dtype(dtype))


def selected_reference(source, record, factor, dtype):
    source_indices, _ = addresses(record)
    selected = source[source_indices]
    if factor is not None:
        if factor == 0:
            selected = jnp.zeros_like(selected)
        elif factor != 1:
            selected = selected * factor
    return selected.astype(dtype)


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    tolerance = 2e-3 if actual.dtype == jnp.float16 else 2e-6
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(component(actual), component(expected), rtol=tolerance, atol=1e-6, equal_nan=True)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1 and "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text
