"""Deterministic differentiable sample arrays for stride tests."""

from math import prod

import jax.numpy as jnp


def values(shape, dtype):
    result = (jnp.arange(prod(shape)) % 5 / 4).astype(dtype).reshape(shape)
    return result + 1j * (result - .5) if dtype.startswith("complex") else result
