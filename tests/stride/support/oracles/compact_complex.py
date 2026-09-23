"""Shared compact complex fixtures and references."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from tensor0._stride._jax import accumulation_p
from tensor0._stride._layout import AffineRecord, contiguous_strides


def compact_record(shape, source_offset=0):
    strides = contiguous_strides(shape)
    return AffineRecord(shape, strides, source_offset, strides, 0)


def complex_values(shape):
    values = jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)
    return (values / 32 + 1j * (values / 64 - 1)).astype(jnp.complex64)


def mapped(source, record, factor):
    return accumulation_p.bind(source, jnp.asarray(factor, dtype=source.dtype), records=(record,),
                               coefficient_records=(0,), output_size=prod(record.logical_shape), dtype=source.dtype)


def reference(source, record, factor):
    elements = prod(record.logical_shape)
    return (source[..., record.source_offset:record.source_offset + elements]
            * jnp.asarray(factor, dtype=source.dtype)).astype(source.dtype)


def assert_native(function, argument):
    text = jax.jit(function).lower(argument).as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text and "iota" not in text
    return text


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 2e-3 if actual.dtype in (jnp.float16, jnp.bfloat16) else 2e-6
        for component in (jnp.real, jnp.imag):
            np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                       np.asarray(component(expected)).astype(np.float64),
                                       rtol=tolerance, atol=1e-6)
