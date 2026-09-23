"""Record-wise Reduction references and derivative checks."""

from functools import cache
from itertools import product
from math import prod

import jax.numpy as jnp
import numpy as np

from tensor0._stride._jax import accumulation_p, reduction_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.ad import check_coefficient_ad


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if jnp.issubdtype(actual.dtype, jnp.integer) or actual.dtype == jnp.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = .008 if actual.dtype == jnp.bfloat16 else .002 if actual.dtype == jnp.float16 else 2e-5
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex128), np.asarray(expected).astype(np.complex128),
                                   rtol=tolerance, atol=tolerance)


@cache
def functions(operation, dtype):
    records = (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1),
               AffineRecord((2, 2), (0, -1), 3, (-1, 1), 3),
               AffineRecord((0,), (1,), 5, (1,), 5))
    def run(source, first, second):
        if operation == "accumulation":
            return accumulation_p.bind(source, first, second, records=records,
                coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
        return reduction_p.bind(source, first, second, records=records,
            output_shapes=((2, 1), (2, 1), (1,)), reduction_axes=((False, True), (False, True), (True,)),
            coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
    # Gather each record at once. Cast each mapped contribution before the
    # scatter-add, retaining repeated destinations and record ordering.
    indices = []
    for record in records[:2]:
        coordinates = tuple(np.ndindex(record.logical_shape))
        inputs = [record.source_offset + sum(i * stride for i, stride in
                  zip(coordinate, record.source_strides, strict=True)) for coordinate in coordinates]
        outputs = [record.destination_offset + sum(i * stride for axis, (i, stride) in
                   enumerate(zip(coordinate, record.destination_strides, strict=True))
                   if operation == "accumulation" or axis == 0) for coordinate in coordinates]
        indices.append((np.asarray(inputs, dtype=np.int32), np.asarray(outputs, dtype=np.int32)))

    def reference(source, first, second):
        factors = tuple(factor.reshape(()) if factor.shape == (1,) else factor for factor in (first, second))
        result = jnp.zeros((*source.shape[:-1], 6), dtype=dtype)
        for (inputs, outputs), factor in zip(indices, factors, strict=True):
            contribution = source[..., inputs] * factor[..., None]
            if not dtype.startswith("complex"):
                contribution = jnp.real(contribution)
            result = result.at[..., outputs].add(contribution.astype(dtype))
        return result
    return run, reference


def compare_derivatives(run, reference, arguments):
    check_coefficient_ad(run, reference, arguments,
                         coefficient_count=len(arguments) - 1, complex_cotangent=2 - 1j)


RECORDS = (
    AffineRecord((2, 0), (1, 1), 8, (2, 7), 1),
    AffineRecord((2, 2), (1, 1), 1, (2, 1), 1),
    AffineRecord((2, 2), (-1, 1), 3, (100, 1), 2),
    AffineRecord((2, 3), (0, -1), 5, (3, 100), 0),
)


OUTPUT_SHAPES = ((2, 1), (2, 1), (1, 2), (2, 1))


AXES = ((False, True), (False, True), (True, False), (False, True))


COEFFICIENT_RECORDS = (0, 2, 3)


def execute(source, empty, first, second):
    return reduction_p.bind(
        source, empty, first, second, records=RECORDS, output_shapes=OUTPUT_SHAPES,
        reduction_axes=AXES, coefficient_records=COEFFICIENT_RECORDS, output_size=7, dtype=source.dtype,
    )


def reference(source, empty, first, second):
    coefficients = tuple(value.reshape(()) if value.shape == (1,) else value for value in (empty, first, second))
    result = jnp.zeros((*source.shape[:-1], 7), dtype=source.dtype)
    for index, (record, axes) in enumerate(zip(RECORDS, AXES, strict=True)):
        factor = coefficients[COEFFICIENT_RECORDS.index(index)] if index in COEFFICIENT_RECORDS else None
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum(axis * stride for axis, stride in zip(coordinates, record.source_strides))
            output_index = record.destination_offset + sum(axis * stride for axis, stride, reduced in
                zip(coordinates, record.destination_strides, axes, strict=True) if not reduced)
            value = source[..., source_index]
            result = result.at[..., output_index].add(value if factor is None else value * factor)
    return result


def operands(dtype, batch_shape, coefficient_shape):
    source = (jnp.arange(prod(batch_shape) * 8) % 5).astype(dtype).reshape((*batch_shape, 8))
    if dtype in ("complex64", "complex128"):
        source = source + 1j * (source - 2)
    empty = jnp.full(coefficient_shape, 1, dtype=dtype)
    first = jnp.full(coefficient_shape, 2 + 1j if dtype in ("complex64", "complex128") else 2, dtype=dtype)
    second = jnp.full(batch_shape, 3 - 2j if dtype in ("complex64", "complex128") else 3, dtype=dtype)
    return source, empty, first, second


def addresses(shape, strides, offset):
    indices = np.full(shape, offset, dtype=np.int64)
    for axis, (size, stride) in enumerate(zip(shape, strides, strict=True)):
        broadcast_shape = (1,) * axis + (size,) + (1,) * (len(shape) - axis - 1)
        indices += np.arange(size).reshape(broadcast_shape) * stride
    return indices


def reference_sum(source, factors, *, records, output_shapes, reduction_axes, output_size, dtype):
    result = jnp.zeros((*source.shape[:-1], output_size), dtype=dtype)
    for record, shape, axes, factor in zip(records, output_shapes, reduction_axes, factors, strict=True):
        values = source[..., addresses(record.logical_shape, record.source_strides, record.source_offset)]
        if factor is not None:
            values = values * factor
        summed = jnp.sum(values, axis=tuple(source.ndim - 1 + axis for axis, reduced in enumerate(axes) if reduced), keepdims=True)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            summed = jnp.real(summed)
        destinations = addresses(shape, record.destination_strides, record.destination_offset).ravel()
        result = result.at[..., destinations].add(summed.astype(dtype).reshape(*source.shape[:-1], destinations.size))
    return result


CROSS = [types for types in product(("float32", "float64", "complex64", "complex128"), repeat=3)
         if len({dtype.startswith("complex") for dtype in types}) > 1]


CROSS_SHAPE_TYPES = [
    ("complex128", "float32", "float64"),
    ("float32", "complex128", "float64"),
    ("float64", "float32", "complex128"),
    ("complex64", "complex128", "float32"),
    ("complex128", "float64", "complex64"),
    ("float32", "complex64", "complex128"),
]
