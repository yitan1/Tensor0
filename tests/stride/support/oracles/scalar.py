"""Shared scalar fixtures and references."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._layout import AffineRecord


PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)


def assert_components(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == jnp.bool_ or jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(np.asarray(actual.real).astype(np.float64), np.asarray(expected.real).astype(np.float64),
                                   rtol=3e-3, atol=0, equal_nan=True)
        np.testing.assert_allclose(np.asarray(actual.imag).astype(np.float64), np.asarray(expected.imag).astype(np.float64),
                                   rtol=3e-3, atol=0, equal_nan=True)
