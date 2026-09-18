"""JAX operand type resolution without layout or coefficient serialization."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(maxsize=1_024)
def _multiplication_dtype(source_dtype, coefficient_dtype, coefficient_weak, x64, promotion):
    # Configuration participates in the cache key; eval_shape reads the active context.
    del x64, promotion
    return jax.eval_shape(
        jnp.multiply,
        jax.ShapeDtypeStruct((), jnp.dtype(source_dtype)),
        jax.ShapeDtypeStruct((), jnp.dtype(coefficient_dtype), weak_type=coefficient_weak),
    ).dtype


def normalize_coefficient(source_dtype, factor):
    """Resolve a weak coefficient against concrete storage before update branches.

    Strong coefficients retain their dtype. Real coefficients remain real when
    the product is complex; normalization never classifies coefficient values.
    """
    coefficient = jnp.asarray(factor)
    result_dtype = _multiplication_dtype(
        jnp.dtype(source_dtype).name, coefficient.dtype.name, coefficient.weak_type,
        jax.config.values["jax_enable_x64"],
        jax.config.values["jax_numpy_dtype_promotion"],
    )
    if not coefficient.weak_type:
        return coefficient
    if (jnp.issubdtype(result_dtype, jnp.complexfloating)
            and not jnp.issubdtype(coefficient.dtype, jnp.complexfloating)):
        result_dtype = result_dtype.type(0).real.dtype
    return coefficient.astype(result_dtype)
