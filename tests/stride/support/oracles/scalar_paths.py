"""Scalar path fixtures, reference storage, and result checks."""

import jax.numpy as jnp
import numpy as np




_CASES = [
    ("float16", "float16", "float32"),
    ("bfloat16", "bfloat16", "float32"),
    ("float32", "float32", "float64"),
    ("complex64", "complex64", "complex128"),
    ("float16", "float32", "float64"),
    ("float32", "complex64", "complex128"),
    ("complex64", "float32", "complex64"),
    ("complex128", "float64", "complex128"),
]


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                   np.asarray(component(expected)).astype(np.float64), rtol=2e-3, atol=0)


def store(base, selected):
    if not jnp.issubdtype(base.dtype, jnp.complexfloating):
        selected = jnp.real(selected)
    return base.at[1::2].set(selected.astype(base.dtype))
