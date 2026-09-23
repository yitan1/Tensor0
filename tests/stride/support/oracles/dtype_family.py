"""Shared dtype family fixtures and references."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._ffi._registration import operation_target


def values(dtype, size, shift=0):
    indices = np.arange(size, dtype=np.int64) + shift
    if dtype == jnp.bool_:
        host = indices % 3 != 0
    elif jnp.issubdtype(dtype, jnp.unsignedinteger):
        host = indices * 71 + 200
    elif jnp.issubdtype(dtype, jnp.signedinteger):
        host = indices * 37 - 120
    else:
        host = indices.astype(np.float64) / 3 - 2
        if jnp.issubdtype(dtype, jnp.complexfloating):
            host = host + 1j * host[::-1]
    return jnp.asarray(np.asarray(host).astype(np.dtype(dtype)))


def assert_result(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == jnp.bool_ or jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = (2e-14 if actual.dtype in (jnp.float64, jnp.complex128) else
                     .008 if actual.dtype == jnp.bfloat16 else .002 if actual.dtype == jnp.float16 else 2e-6)
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex128), np.asarray(expected).astype(np.complex128),
                                   rtol=tolerance, atol=tolerance)


def assert_native(lowered, operation, dtype):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert operation_target(operation, np.dtype(dtype)) in text
    assert "gather" not in text and "scatter" not in text
