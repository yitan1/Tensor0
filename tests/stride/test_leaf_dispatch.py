"""Equivalent affine operations share numerical contracts across entry points."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord, contiguous_strides

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


def layout_fields(name):
    if name == "contiguous":
        return (33,), (1,)
    if name == "transpose":
        return (8, 16), (1, 8)
    if name == "broadcast":
        return (8, 16), (0, 1)
    return (2, 3, 16, 24), (384, 768, 1, 16)


def addresses(record):
    source = np.full(record.logical_shape, record.source_offset, dtype=np.int64)
    destination = np.full(record.logical_shape, record.destination_offset, dtype=np.int64)
    for axis, extent in enumerate(record.logical_shape):
        coordinate = np.arange(extent).reshape((1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1))
        source += coordinate * record.source_strides[axis]
        destination += coordinate * record.destination_strides[axis]
    return source.ravel(), destination.ravel()


def fresh(source, record, factor, output_size, dtype):
    if factor is None:
        return copy_p.bind(source, records=(record,), output_size=output_size, dtype=jnp.dtype(dtype))
    return accumulation_p.bind(source, factor, records=(record,), coefficient_records=(0,),
                               output_size=output_size, dtype=jnp.dtype(dtype))


def selected_reference(source, record, factor, dtype):
    source_indices, _ = addresses(record)
    selected = source[source_indices]
    if factor is not None:
        if factor == 0:
            selected = jnp.zeros_like(selected)
        elif factor != 1:
            selected = selected * factor
    return selected.astype(dtype)


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    tolerance = 2e-3 if actual.dtype == jnp.float16 else 2e-6
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(component(actual), component(expected), rtol=tolerance, atol=1e-6, equal_nan=True)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1 and "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "broadcast", "rank4"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "complex64"])
@pytest.mark.parametrize("factor_value", [None, 1., -.75])
@pytest.mark.parametrize("partial", [False, True])
def test_fresh_and_overwrite_share_selected_formula(layout, dtype, factor_value, partial):
    shape, strides = layout_fields(layout)
    source_size = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True))
    selected_size = int(np.prod(shape))
    offset = 3 if partial else 0
    output_size = selected_size + 2 * offset
    record = AffineRecord(shape, strides, 0, contiguous_strides(shape), offset)
    factor = None if factor_value is None else jnp.asarray(factor_value, dtype=dtype)
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504], dtype=dtype), (source_size,))
    base = jnp.full(output_size, -7., dtype=dtype)
    source_factor = jnp.asarray(1, dtype=dtype) if factor is None else factor
    mapping = jax.jit(lambda values: fresh(values, record, factor, output_size, dtype))
    update = jax.jit(lambda old, values: update_p.bind(values, old, source_factor, jnp.int32(0), records=(record,)))
    selected = selected_reference(source, record, factor, dtype)
    expected_fresh = jnp.zeros(output_size, dtype=dtype).at[offset:offset + selected_size].set(selected)
    expected_update = base.at[offset:offset + selected_size].set(selected)
    assert_close(mapping(source), expected_fresh)
    actual = update(base, source)
    assert_close(actual, expected_update)
    if partial:
        np.testing.assert_array_equal(actual[:offset], base[:offset])
        np.testing.assert_array_equal(actual[-offset:], base[-offset:])
    assert_native(mapping.lower(source))
    assert_native(update.lower(base, source))


@pytest.mark.parametrize("source_dtype,result_dtype", [
    ("float16", "float32"), ("float32", "float32"), ("float32", "complex64"),
])
@pytest.mark.parametrize("factor_value", [None, 1., 1.00000003, 1e-8])
def test_unit_dynamic_factor_preserves_explicit_coefficient_composition(source_dtype, result_dtype, factor_value):
    with jax.enable_x64():
        source = jnp.asarray([-0., 1., 65504, .125, -3.5, 128.], dtype=source_dtype)
        base = jnp.full(source.size + 2, np.nan, dtype=result_dtype)
        record = AffineRecord((source.size,), (1,), 0, (1,), 1)

        def operation(old, values, coefficient):
            effective = coefficient if factor_value is None else coefficient * jnp.float64(factor_value)
            return update_p.bind(values, old, effective, jnp.int32(0), records=(record,))

        actual = jax.jit(operation)(base, source, jnp.float32(1))
        effective = jnp.float32(1) if factor_value is None else jnp.float32(1) * jnp.float64(factor_value)
        assert_close(actual[1:-1], selected_reference(source, record, effective, result_dtype))
        assert bool(jnp.isnan(actual[0]) & jnp.isnan(actual[-1]))


def test_tiny_strong_coefficient_survives_closed_and_dynamic_binding():
    with jax.enable_x64():
        source = jnp.asarray([1e30], dtype=jnp.float32)
        record = AffineRecord((1,), (1,), 0, (1,), 0)
        coefficient = jnp.float64(1e-46)
        operation = lambda values, factor: update_p.bind(values, jnp.zeros_like(values), factor,
                                                          jnp.int32(0), records=(record,))
        expected = (coefficient * source).astype(source.dtype)
        for actual in (jax.jit(lambda values: operation(values, coefficient))(source),
                       jax.jit(operation)(source, coefficient)):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
            assert bool(jnp.all(actual != 0))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_mixed_complex_fresh_and_update_mapping(transpose, partial):
    shape = (9, 17)
    strides = (1, 9) if transpose else (17, 1)
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504], dtype=jnp.float32), (153,))
    offset = 3 if partial else 0
    base = jnp.full(153 + 2 * offset, complex(np.nan, np.nan), dtype=jnp.complex64)
    factor = jnp.complex64(1.25 - .5j)
    record = AffineRecord(shape, strides, 0, (17, 1), offset)
    expected = selected_reference(source, record, factor, base.dtype)
    mapped = jax.jit(lambda values: fresh(values, record, factor, base.size, base.dtype))(source)
    updated = jax.jit(lambda old, values: update_p.bind(values, old, factor, jnp.int32(0), records=(record,)))(base, source)
    for actual in (mapped, updated):
        assert_close(actual[offset:offset + source.size], expected)
    if partial:
        np.testing.assert_array_equal(mapped[:offset], 0)
        np.testing.assert_array_equal(mapped[-offset:], 0)
        assert bool(jnp.isnan(updated[:offset]).all() & jnp.isnan(updated[-offset:]).all())


@pytest.mark.parametrize("explicit", [False, True])
def test_half_storage_derivatives_do_not_narrow_factor(explicit):
    source = jnp.ones(1, dtype=jnp.float16)
    factor = jnp.float32(1e-8)
    direction = jnp.full_like(source, 65504)
    operation = lambda values, coefficient: scale(StridedView(values, (1,), (1,), 0), coefficient).data

    def derivative(values, coefficient):
        if explicit:
            tangent = jax.jvp(operation, (values, coefficient), (direction, jnp.zeros_like(coefficient)))[1]
        else:
            tangent = jax.jvp(lambda data: operation(data, coefficient), (values,), (direction,))[1]
        cotangent = jax.vjp(lambda data: operation(data, coefficient), values)[1](direction)[0]
        return tangent, cotangent

    for actual in jax.jit(derivative)(source, factor):
        assert_close(actual, (factor * direction).astype(source.dtype))
        assert bool(jnp.all(actual != 0))


@pytest.mark.parametrize("dtype,factor_value", [
    ("float16", 0.), ("float16", 1.), ("float16", .75),
    ("complex64", 0.), ("complex64", 1.), ("complex64", .75), ("complex64", 1.25 - .5j),
])
@pytest.mark.parametrize("layout", ["contiguous", "forward", "reverse"])
def test_closed_fresh_and_dynamic_update_coefficients_agree(dtype, layout, factor_value):
    shape = (9, 17)
    source_strides = (1, 9) if layout == "forward" else (17, 1)
    destination_strides = (1, 9) if layout == "reverse" else (17, 1)
    record = AffineRecord(shape, source_strides, 0, destination_strides, 0)
    values = [-0., 1., -2., .125, -3.5, 128., 65504]
    if dtype == "complex64":
        values.extend([1.5 - 2.25j, .125 - 3.5j, complex(-0., -0.)])
    source = jnp.resize(jnp.asarray(values, dtype=dtype), (153,))
    factor = jnp.asarray(factor_value, dtype=dtype)
    mapping = jax.jit(lambda data: fresh(data, record, factor, 153, dtype))
    update = jax.jit(lambda data, coefficient: update_p.bind(data, jnp.zeros_like(data), coefficient,
                                                            jnp.int32(0), records=(record,)))
    _, destinations = addresses(record)
    expected = jnp.zeros_like(source).at[destinations].set(selected_reference(source, record, factor, dtype))
    assert_close(mapping(source), expected)
    assert_close(update(source, factor), expected)
    assert_native(mapping.lower(source))
    assert_native(update.lower(source, factor))


@pytest.mark.parametrize("layout", ["contiguous", "forward", "reverse"])
def test_half_permutation_tails_and_zero_cotangents(layout):
    shape = (11, 19)
    record = AffineRecord(shape, (1, 11) if layout == "forward" else (19, 1), 0,
                          (1, 11) if layout == "reverse" else (19, 1), 0)
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504], dtype=jnp.float16), (209,))
    factor = jnp.float16(.75)
    operation = lambda values: fresh(values, record, factor, 209, values.dtype)
    _, destinations = addresses(record)
    expected = jnp.zeros_like(source).at[destinations].set(selected_reference(source, record, factor, source.dtype))
    assert_close(jax.jit(operation)(source), expected)
    pullback = jax.jit(lambda values, cot: jax.vjp(operation, values)[1](cot)[0])
    np.testing.assert_array_equal(pullback(source, jnp.full_like(source, -0.)), jnp.zeros_like(source))


@pytest.mark.parametrize("dtype", ["float16", "complex64"])
@pytest.mark.parametrize("factor_value", [0., 1.])
def test_data_derivative_zero_one_scaling(dtype, factor_value):
    source = jnp.ones(153, dtype=dtype)
    direction = jnp.full_like(source, 1.25 - .5j if dtype == "complex64" else 1.25)
    coefficient = jnp.asarray(factor_value, dtype=dtype)
    derivative = lambda values, factor, tangent: jax.jvp(
        lambda data: scale(StridedView(data, (153,), (1,), 0), factor).data, (values,), (tangent,))[1]
    lowered = jax.jit(derivative).lower(source, coefficient, direction)
    assert_native(lowered)
    assert_close(lowered.compile()(source, coefficient, direction), coefficient * direction)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("factor", [0, 1])
def test_short_copy_and_zero_preserve_storage_bits(dtype, transpose, factor):
    shape = (11, 19)
    count = int(np.prod(shape))
    component_count = count * (2 if dtype == "complex64" else 1)
    bits = np.resize(np.asarray([0x80000000, 0x7FC12345, 0xFFC23456, 0x7F800000, 0xFF800000, 0x3FA00000],
                               dtype=np.uint32), component_count)
    source = jnp.asarray(bits.view(dtype))
    source_bits = np.asarray(source).copy().view(np.uint32)
    base = jnp.resize(source, (count + 4,))
    base_bits = np.asarray(base).copy().view(np.uint32)
    record = AffineRecord(shape, (1, 11) if transpose else (19, 1), 0, (19, 1), 2)
    operation = jax.jit(lambda old, values, coefficient: update_p.bind(values, old, coefficient,
                                                                      jnp.int32(0), records=(record,)))
    actual = operation(base, source, jnp.asarray(factor, dtype=dtype))
    assert_native(operation.lower(base, source, jnp.asarray(factor, dtype=dtype)))
    selected = np.asarray(actual)[2:-2].copy()
    expected = np.asarray(source).reshape(19, 11).T.reshape(-1) if transpose else np.asarray(source)
    if factor:
        np.testing.assert_array_equal(selected.view(np.uint32), expected.copy().view(np.uint32))
        copied = jax.jit(lambda values: fresh(values, record, None, count + 4, dtype))(source)
        np.testing.assert_array_equal(np.asarray(copied)[2:-2].copy().view(np.uint32), expected.copy().view(np.uint32))
    else:
        np.testing.assert_array_equal(selected.view(np.uint32), np.zeros(selected.view(np.uint32).shape, dtype=np.uint32))
    for region in (slice(None, 2), slice(-2, None)):
        np.testing.assert_array_equal(np.asarray(actual)[region].copy().view(np.uint32),
                                      np.asarray(base)[region].copy().view(np.uint32))
    np.testing.assert_array_equal(np.asarray(source).view(np.uint32), source_bits)
    np.testing.assert_array_equal(np.asarray(base).view(np.uint32), base_bits)
