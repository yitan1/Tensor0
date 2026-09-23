"""Shared coefficient differentiation checks."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(0, 3))
def _coefficient_ad(run, arguments, cotangent, coefficient_count):
    """Keep primals dynamic and compile the three AD checks together."""
    directions = tuple(jnp.ones_like(value) for value in arguments)
    forward = jax.jvp(run, arguments, directions)
    reverse = jax.vjp(run, *arguments)[1](cotangent)
    fixed, coefficients = arguments[:-coefficient_count], arguments[-coefficient_count:]
    linear = lambda *factors: run(*fixed, *factors)
    transpose = jax.linear_transpose(linear, *coefficients)(cotangent)
    return forward, reverse, transpose


def check_coefficient_ad(run, reference, arguments, *, coefficient_count, complex_cotangent):
    """Compare JIT JVP, joint VJP and coefficient transpose to an eager oracle."""
    directions = tuple(jnp.ones_like(value) for value in arguments)
    expected_forward = jax.jvp(reference, arguments, directions)
    output = expected_forward[0]
    cotangent = jnp.full_like(output, complex_cotangent if jnp.iscomplexobj(output) else 2)
    expected_reverse = jax.vjp(reference, *arguments)[1](cotangent)
    forward, reverse, transpose = _coefficient_ad(run, arguments, cotangent, coefficient_count)
    for actual, expected in zip(forward, expected_forward, strict=True):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    for actual, expected, original in zip(reverse, expected_reverse, arguments, strict=True):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    for actual, expected, original in zip(transpose, expected_reverse[-coefficient_count:],
                                          arguments[-coefficient_count:], strict=True):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
