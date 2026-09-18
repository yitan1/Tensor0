"""JAX operand type resolution without layout or coefficient serialization."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(maxsize=1_024)
def _binary_type(operation, left_dtype, left_weak, right_dtype, right_weak, x64, promotion):
    del x64, promotion
    return jax.eval_shape(
        operation,
        jax.ShapeDtypeStruct((), jnp.dtype(left_dtype), weak_type=left_weak),
        jax.ShapeDtypeStruct((), jnp.dtype(right_dtype), weak_type=right_weak),
    )


def _binary_shape(operation, left, right):
    return _binary_type(
        operation,
        left.dtype.name, getattr(left, "weak_type", False),
        right.dtype.name, getattr(right, "weak_type", False),
        jax.config.values["jax_enable_x64"],
        jax.config.values["jax_numpy_dtype_promotion"],
    )


def _product_shape(left, right):
    return _binary_shape(jnp.multiply, left, right)


def mapping_dtype(source_dtype, scale):
    """Resolve a single mapping stage without a destination-storage constraint."""

    source = jax.ShapeDtypeStruct((), jnp.dtype(source_dtype))
    if scale is None:
        return source.dtype
    return _product_shape(source, jnp.asarray(scale)).dtype


def product_dtype(base, factor):
    return _product_shape(base, factor).dtype


def normalize_coefficient(source_dtype, factor):
    """Resolve a weak coefficient against concrete storage before update branches.

    Strong coefficients retain their dtype. Real coefficients remain real when
    the product is complex; normalization never classifies coefficient values.
    """
    coefficient = jnp.asarray(factor)
    result_dtype = mapping_dtype(source_dtype, coefficient)
    if not coefficient.weak_type:
        return coefficient
    if (jnp.issubdtype(result_dtype, jnp.complexfloating)
            and not jnp.issubdtype(coefficient.dtype, jnp.complexfloating)):
        result_dtype = result_dtype.type(0).real.dtype
    return coefficient.astype(result_dtype)
