"""Signed affine layouts and the native operation used to exercise them."""

from math import prod

import jax.numpy as jnp
import numpy as np

from tensor0._stride import enable_threads, set_num_threads
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord, contiguous_strides


def destination_layout(shape, fastest_first, signs):
    absolute = [0] * len(shape)
    stride = 1
    for axis in fastest_first:
        absolute[axis] = stride
        stride *= shape[axis]
    strides = tuple(sign * stride for sign, stride in zip(signs, absolute, strict=True))
    offset = sum((extent - 1) * stride for extent, stride, sign in zip(shape, absolute, signs, strict=True)
                 if sign < 0)
    return strides, offset


def signed_record(physical_shape=(2, 3), logical_to_physical=(0, 1), source_signs=(-1, 1),
                  destination_fastest=(1, 0), destination_signs=(1, 1)):
    physical_strides = contiguous_strides(physical_shape)
    shape = tuple(physical_shape[axis] for axis in logical_to_physical)
    source_strides = tuple(sign * physical_strides[axis]
                           for sign, axis in zip(source_signs, logical_to_physical, strict=True))
    source_offset = sum((physical_shape[axis] - 1) * physical_strides[axis]
                        for sign, axis in zip(source_signs, logical_to_physical, strict=True) if sign < 0)
    strides, offset = destination_layout(shape, destination_fastest, destination_signs)
    return AffineRecord(shape, source_strides, source_offset, strides, offset)


def native(source, record, factor, dtype):
    output = jnp.zeros((*source.shape[:-1], prod(record.logical_shape)), dtype=dtype)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if isinstance(factor, complex) else jnp.float32)
    return update_p.bind(source, output, coefficient, jnp.float32(0), records=(record,))


MIXED = [("float16", "float32", -1.25), ("float32", "complex64", .5 + .75j),
         ("complex64", "float32", -.75)]


def broadcast_record(shape=(2, 3), broadcast_axes=(0,), source_signs=(-1,),
                     destination_fastest=(1, 0), destination_signs=(1, 1)):
    axes = tuple(axis for axis, extent in enumerate(shape) if extent > 1 and axis not in broadcast_axes)
    source_shape = tuple(shape[axis] for axis in axes)
    source_strides = [0] * len(shape)
    source_offset = 0
    for axis, sign, stride in zip(axes, source_signs, contiguous_strides(source_shape), strict=True):
        source_strides[axis] = sign * stride
        if sign < 0:
            source_offset += (shape[axis] - 1) * stride
    strides, offset = destination_layout(shape, destination_fastest, destination_signs)
    return AffineRecord(shape, tuple(source_strides), source_offset, strides, offset), prod(source_shape)


def values(dtype, size):
    real = np.linspace(-2, 3, size, dtype=np.float32)
    return jnp.asarray(real + 1j * real[::-1] if dtype.startswith("complex") else real, dtype=dtype)


def indices(record):
    coordinates = np.indices(record.logical_shape, dtype=np.int64)
    source = np.full(record.logical_shape, record.source_offset, dtype=np.int64)
    destination = np.full(record.logical_shape, record.destination_offset, dtype=np.int64)
    for axis, coordinate in enumerate(coordinates):
        source += coordinate * record.source_strides[axis]
        destination += coordinate * record.destination_strides[axis]
    return source.ravel(), destination.ravel()


def reference(source, record, factor, dtype):
    source_indices, destination_indices = indices(record)
    selected = source[..., source_indices] * jnp.asarray(factor)
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
        selected = jnp.real(selected)
    return jnp.zeros((*source.shape[:-1], prod(record.logical_shape)), dtype=dtype).at[
        ..., destination_indices].set(selected.astype(dtype))


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(component(actual), component(expected), rtol=2e-3, atol=1e-6, equal_nan=True)


def restore_threads(previous):
    if previous is None:
        enable_threads()
    else:
        set_num_threads(previous)
