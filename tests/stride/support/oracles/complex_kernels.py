"""Shared complex kernels fixtures and references."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._jax import accumulation_p


def values(size):
    indices = jnp.arange(size, dtype=jnp.float32)
    return ((indices % 4093) / 1024 - 2 + 1j * (.5 - (indices % 4079) / 2048)).astype(jnp.complex64)


def mapped(source, record, factor):
    return accumulation_p.bind(source, jnp.complex64(factor), records=(record,), coefficient_records=(0,),
                               output_size=int(np.prod(record.logical_shape)), dtype=jnp.dtype(jnp.complex64))


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1 and "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text
