"""JAX scalar typing shared by affine mapping and its execution descriptors."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


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


def scalar_key(scale):
    """Preserve scalar type, weak typing, signed zero, and nonfinite payloads."""

    if scale is None:
        return None
    value = jnp.asarray(scale)
    if value.shape != ():
        raise ValueError("record scale must be scalar")
    dtype = value.dtype
    payload = np.asarray(scale, dtype=dtype).astype(
        dtype.newbyteorder("<"), copy=False,
    ).tobytes()
    return dtype.name, value.weak_type, payload


def product_dtype(base, factor):
    return _product_shape(base, factor).dtype


def _update_shapes(base, source, source_factor, base_factor, records):
    def scalar_shape(value):
        return jax.ShapeDtypeStruct((), value.dtype,
                                   weak_type=getattr(value, "weak_type", False))

    base_shape = scalar_shape(base)
    source_shape = scalar_shape(source)
    source_factor_shape = scalar_shape(source_factor)
    base_factor_shape = scalar_shape(base_factor)
    base_product = _product_shape(base_factor_shape, base_shape)
    for record in records:
        coefficient = (source_factor_shape if record.scale is None else
                       _product_shape(source_factor_shape, jnp.asarray(record.scale)))
        source_product = _product_shape(coefficient, source_shape)
        yield coefficient, source_shape, base_shape, source_product, base_product


def update_dtypes(base, source, source_factor, base_factor, records):
    """Resolve the ordinary, unshortened differential expression."""

    return tuple(
        (coefficient.dtype.name, source_product.dtype.name, base_product.dtype.name,
         _binary_shape(jnp.add, source_product, base_product).dtype.name)
        for coefficient, _, _, source_product, base_product in
        _update_shapes(base, source, source_factor, base_factor, records)
    )


def forward_update_dtypes(base, source, source_factor, base_factor, records):
    """Resolve products and the four additions of actual forward terms."""

    result = []
    for coefficient, source_shape, base_shape, source_product, base_product in \
            _update_shapes(base, source, source_factor, base_factor, records):
        additions = tuple(
            _binary_shape(jnp.add, first, second).dtype.name
            for first in (source_shape, source_product)
            for second in (base_shape, base_product)
        )
        result.append((coefficient.dtype.name, source_product.dtype.name,
                       base_product.dtype.name, *additions))
    return tuple(result)
