"""Dot layouts, operands and differentiable reference."""

from math import prod

import jax.numpy as jnp
import numpy as np

from tensor0._stride import StridedView, dotc


def operands(left_dtype, right_dtype, batch_shape):
    left = (jnp.arange(prod(batch_shape) * 8) % 7 / 4).astype(left_dtype).reshape((*batch_shape, 8))
    right = (jnp.arange(prod(batch_shape) * 10) % 5 / 4).astype(right_dtype).reshape((*batch_shape, 10))
    if jnp.iscomplexobj(left):
        left = left + 1j * (left - 0.5)
    if jnp.iscomplexobj(right):
        right = right + 1j * (1 - right)
    return (left, right)


LAYOUTS = [
    ((2, 3), (1, 2), 1, (-3, -1), 6),
    ((2, 3), (0, 1), 1, (1, 1), 1),
    ((0,), (1,), 0, (1,), 0),
    ((), (), 2, (), 3),
]


def dot_functions(operation, layout, dtype):
    shape, left_strides, left_offset, right_strides, right_offset = layout
    left_indices = np.asarray([left_offset + sum((index * stride for index, stride in zip(coordinates, left_strides, strict=True))) for coordinates in np.ndindex(shape)], dtype=np.int32)
    right_indices = np.asarray([right_offset + sum((index * stride for index, stride in zip(coordinates, right_strides, strict=True))) for coordinates in np.ndindex(shape)], dtype=np.int32)

    def run(left, right):
        return operation(StridedView(left, shape, left_strides, left_offset), StridedView(right, shape, right_strides, right_offset), dtype=dtype)

    def reference(left, right):
        values = left[..., left_indices]
        if operation is dotc:
            values = jnp.conj(values)
        products = values * right[..., right_indices]
        result_dtype = products.dtype if dtype is None else jnp.dtype(dtype)
        accumulator_dtype = jnp.promote_types(products.dtype, result_dtype)
        result = jnp.sum(products, axis=-1, dtype=accumulator_dtype)
        if jnp.issubdtype(result_dtype, jnp.floating):
            result = jnp.real(result)
        return result.astype(result_dtype)
    return (run, reference)


INEXACT_DTYPES = ('float32', 'float64', 'complex64', 'complex128')
