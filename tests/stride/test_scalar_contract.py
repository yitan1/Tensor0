"""Concrete mixed arithmetic, explicit coefficient composition and final casts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._dtype import normalize_coefficient
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)


def assert_components(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == jnp.bool_ or jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(np.asarray(actual.real).astype(np.float64), np.asarray(expected.real).astype(np.float64),
                                   rtol=3e-3, atol=0, equal_nan=True)
        np.testing.assert_allclose(np.asarray(actual.imag).astype(np.float64), np.asarray(expected.imag).astype(np.float64),
                                   rtol=3e-3, atol=0, equal_nan=True)


def term(values, factor):
    if factor == 0:
        return None
    return values if factor == 1 else values * factor


def reference_update(base, source, alpha, beta):
    source_term, base_term = term(source, alpha), term(base[1::2], beta)
    if source_term is None:
        selected = jnp.zeros_like(base[1::2]) if base_term is None else base_term
    elif base_term is None:
        selected = source_term
    else:
        selected = source_term + base_term
    if not jnp.issubdtype(base.dtype, jnp.complexfloating):
        selected = jnp.real(selected)
    return base.at[1::2].set(selected.astype(base.dtype))


@pytest.mark.parametrize("alpha,beta", [(1., 0.), (0., 1.), (1., 1.), (1., 2.), (2., 1.), (2., 2.), (0., 0.)])
def test_update_short_circuit_branch_preserves_integer_precision(alpha, beta):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 6, 8, 2**24 + 9, 9, 2**24 + 10, 10], dtype=jnp.int32)
    alpha, beta = jnp.float32(alpha), jnp.float32(beta)
    actual = jax.jit(lambda old, new, first, second: update_p.bind(new, old, first, second, records=PARTIAL))(
        base, source, alpha, beta)
    np.testing.assert_array_equal(actual, reference_update(base, source, alpha, beta))


@pytest.mark.parametrize("static_factor", [.5, 1., 2.])
def test_explicit_coefficient_composition_batch_vmap_and_multirecord_layout(static_factor):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 7, 8, 2**24 + 9, 9, 2**24 + 11, 10], dtype=jnp.int32)
    records = tuple(AffineRecord((1,), (1,), index, (1,), 2 * index + 1) for index in range(3))
    pairs = [(first, second) for first in (0., 1., 2.) for second in (0., 1., 2.)]
    alpha = jnp.asarray([first for first, _ in pairs], dtype=jnp.float32)
    beta = jnp.asarray([second for _, second in pairs], dtype=jnp.float32)
    bases, sources = jnp.broadcast_to(base, (9, 7)), jnp.broadcast_to(source, (9, 3))
    operation = lambda old, new, first, second: update_p.bind(
        new, old, first * jnp.float32(static_factor), second, records=records)
    expected = jnp.stack([reference_update(base, source, jnp.float32(first) * jnp.float32(static_factor), jnp.float32(second))
                          for first, second in pairs])
    for function in (operation, jax.vmap(operation)):
        compiled = jax.jit(function)
        np.testing.assert_array_equal(compiled(bases, sources, alpha, beta), expected)
        assert compiled.lower(bases, sources, alpha, beta).as_text().count("custom_call") == 1


@pytest.mark.parametrize("static_factor", [None, .5])
@pytest.mark.parametrize("coefficient", [0., 1., 2.])
def test_coefficient_branches_keep_algebraic_first_and_second_derivatives(static_factor, coefficient):
    with jax.enable_x64():
        source, base = jnp.asarray([1.25, 3.5, -2.75], dtype=jnp.float32), jnp.arange(7, dtype=jnp.float32)
        arguments = (base, source, jnp.float64(coefficient), jnp.float64(1))
        directions = tuple(jnp.ones_like(value) for value in arguments)

        def operation(old, new, alpha, beta):
            effective = alpha if static_factor is None else alpha * jnp.float64(static_factor)
            return update_p.bind(new, old, effective, beta, records=PARTIAL)

        def reference(old, new, alpha, beta):
            effective = alpha if static_factor is None else alpha * jnp.float64(static_factor)
            return old.at[1::2].set((effective * new + beta * old[1::2]).astype(old.dtype))

        def derivatives(function, *values):
            first = lambda *items: jax.jvp(function, items, directions)[1]
            return (first(*values), jax.grad(lambda *items: jnp.sum(function(*items)), argnums=(0, 1, 2, 3))(*values),
                    jax.jvp(first, values, directions)[1])

        actual = jax.jit(lambda *values: derivatives(operation, *values))(*arguments)
        expected = derivatives(reference, *arguments)
        for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(result, wanted, rtol=1e-6, atol=1e-6)


def test_copy_and_unit_scale_preserve_complex_identity_bits():
    source = jnp.asarray([complex(np.inf, 0), complex(0, np.inf), complex(-0., -0.)], dtype=jnp.complex64)
    copied = jax.jit(lambda value: copy_p.bind(value, records=PARTIAL, output_size=7, dtype=value.dtype))(source)
    np.testing.assert_array_equal(np.asarray(copied)[1::2].copy().view(np.uint32), np.asarray(source).view(np.uint32))
    np.testing.assert_array_equal(copied[::2], 0)
    for factor in (1, np.float32(1), np.complex64(1)):
        actual = jax.jit(lambda value, coefficient: scale(StridedView(value, (3,), (1,), 0), coefficient).data)(source, factor)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(source).view(np.uint32))


def test_weak_normalization_preserves_strong_dtype_and_real_coefficient_form():
    with jax.enable_x64():
        assert normalize_coefficient(jnp.float16, 1e-8).dtype == jnp.float16
        assert normalize_coefficient(jnp.complex64, 2.).dtype == jnp.float32
        for dtype in (jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex128):
            coefficient = jnp.asarray(1, dtype=dtype)
            actual = normalize_coefficient(jnp.float16, coefficient)
            assert actual.dtype == coefficient.dtype and not actual.weak_type
        for value in (np.float64(1 + 1e-8), np.float64(1 + 2e-8), np.float64(-0.)):
            actual = normalize_coefficient(jnp.float16, value)
            assert np.asarray(actual).tobytes() == np.asarray(value).tobytes()


@pytest.mark.parametrize("factor", [1e-8, np.float32(1e-8), np.float64(1e-8)])
def test_closed_and_dynamic_scale_do_not_narrow_strong_coefficients(factor):
    with jax.enable_x64():
        source = jnp.asarray([65504, 32752, -65504], dtype=jnp.float16)
        expected = (factor * source).astype(source.dtype)
        closed = jax.jit(lambda value: scale(StridedView(value, (3,), (1,), 0), factor).data)
        dynamic = jax.jit(lambda value, coefficient: scale(StridedView(value, (3,), (1,), 0), coefficient).data)
        assert_components(closed(source), expected)
        assert_components(dynamic(source, factor), expected)


def test_weak_and_strong_coefficients_are_normalized_before_zero_one_binding():
    with jax.enable_x64():
        operation = jax.jit(lambda data, factor: scale(StridedView(data, (3,), (1,), 0), factor).data)
        source = jnp.asarray([1e30, np.inf, np.nan], dtype=jnp.float32)
        np.testing.assert_array_equal(operation(source, 1e-46), jnp.zeros_like(source))
        strong = operation(source, jnp.float64(1e-46))
        expected = (source.astype(jnp.float64) * jnp.float64(1e-46)).astype(source.dtype)
        assert_components(strong, expected)
        finite = jnp.asarray([65504, 32752, -65504], dtype=jnp.float32)
        np.testing.assert_array_equal(operation(finite, 1.00000001), finite)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32), (jnp.float32, jnp.float32),
    (jnp.float64, jnp.float64), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32),
    (jnp.bool_, jnp.bool_), (jnp.int32, jnp.int8), (jnp.uint32, jnp.uint8)])
@pytest.mark.parametrize("static_factor", [None, 1, np.float32(.75), np.complex64(1 + 2j)])
def test_update_types_follow_bound_branches_and_final_conversion(source_dtype, result_dtype, static_factor):
    with jax.enable_x64():
        base, source = jnp.asarray([3, 120, 4, 100, 5, 80, 6], dtype=result_dtype), jnp.asarray([8, 9, 10], dtype=source_dtype)
        operation = jax.jit(lambda old, new, first, second: update_p.bind(new, old, first, second, records=PARTIAL))
        for alpha, beta in ((np.int64(1), np.int64(1)), (np.float64(.125), np.float32(2)),
                            (np.float64(0), np.complex64(1)), (np.float32(1), np.float64(0))):
            effective = jnp.asarray(alpha) if static_factor is None else jnp.asarray(alpha) * jnp.asarray(static_factor)
            actual = operation(base, source, effective, jnp.asarray(beta))
            assert_components(actual, reference_update(base, source, effective, jnp.asarray(beta)))
            np.testing.assert_array_equal(actual[::2], base[::2])


@pytest.mark.parametrize("coefficient_dtype", [jnp.float32, jnp.float64, jnp.complex128])
def test_mixed_scale_first_and_higher_derivatives(coefficient_dtype):
    with jax.enable_x64():
        base, coefficient = jnp.asarray([1.25, 2.5, 3.75, 4.25, 5.5], dtype=jnp.float32), jnp.asarray(1.25, dtype=coefficient_dtype)
        native = lambda data, factor: scale(StridedView(data, (2,), (2,), 1), factor).data
        reference = lambda data, factor: data.at[1::2].set(jnp.real(factor * data[1::2]).astype(data.dtype))
        for result, wanted in zip(jax.vjp(native, base, coefficient)[1](jnp.ones_like(base)),
                                   jax.vjp(reference, base, coefficient)[1](jnp.ones_like(base)), strict=True):
            assert_components(result, wanted)
        for explicit in (False, True):
            directions = (jnp.ones_like(base), jnp.ones_like(coefficient) if explicit else jnp.zeros_like(coefficient))
            derivative = lambda data, factor: jax.jvp(native, (data, factor), directions)[1]
            assert_components(jax.jit(derivative)(base, coefficient), jax.jvp(reference, (base, coefficient), directions)[1])
            second = jax.jvp(lambda factor: derivative(base, factor), (coefficient,), (jnp.ones_like(coefficient),))[1]
            assert_components(second, jnp.zeros_like(base).at[1::2].set(1))


@pytest.mark.parametrize("factor", [np.float64(.125), np.complex128(1 + 2j)])
def test_mapping_final_cast_preserves_derivatives(factor):
    with jax.enable_x64():
        source = jnp.asarray([1.25, 2.5, -3.75], dtype=jnp.float32)
        native = lambda value: accumulation_p.bind(value, jnp.asarray(factor), records=PARTIAL,
                                                   coefficient_records=(0,), output_size=7, dtype=value.dtype)
        reference = lambda value: jnp.zeros(7, dtype=value.dtype).at[1::2].set(jnp.real(factor * value).astype(value.dtype))
        for result, wanted in zip(jax.jvp(native, (source,), (jnp.ones_like(source),)),
                                   jax.jvp(reference, (source,), (jnp.ones_like(source),)), strict=True):
            assert_components(result, wanted)
        cotangent = jnp.ones(7, dtype=source.dtype)
        assert_components(jax.vjp(native, source)[1](cotangent)[0], jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.parametrize("factor", [0, 1])
def test_strong_complex_zero_one_finite_data_derivative(factor):
    with jax.enable_x64():
        source = jnp.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=jnp.complex64)
        direction = jnp.full_like(source, 1.25 - .5j)
        coefficient = jnp.asarray(factor, dtype=jnp.complex128)
        operation = lambda value: scale(StridedView(value, (3,), (1,), 0), coefficient).data
        actual = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,))[1])(source, direction)
        assert_components(actual, (coefficient * direction).astype(source.dtype))


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int64, jnp.uint64, jnp.bool_])
def test_final_integer_cast_of_nonfinite_and_out_of_range_products(dtype):
    with jax.enable_x64():
        source = jnp.asarray([0, 1, 2, 63, 127], dtype=dtype)
        for factor in (np.float64(np.inf), np.float64(np.nan), np.float64(-1e30), np.float64(1e30), np.float64(.75)):
            actual = scale(StridedView(source, (5,), (1,), 0), factor).data
            expected = (factor * source.astype(jnp.result_type(factor, source))).astype(source.dtype)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("explicit", [False, True])
def test_scale_jvp_keeps_coefficient_precision_before_product(explicit):
    with jax.enable_x64():
        base = jnp.asarray([1., 2., 3.], dtype=jnp.float32)
        direction, coefficient = jnp.full_like(base, 1e30), jnp.float64(1e-46)
        operation = lambda data, factor: scale(StridedView(data, (3,), (1,), 0), factor).data
        def derivative(data, factor):
            if explicit:
                return jax.jvp(operation, (data, factor), (direction, jnp.zeros_like(factor)))[1]
            return jax.jvp(lambda value: operation(value, factor), (data,), (direction,))[1]
        actual = jax.jit(derivative)(base, coefficient)
        expected = (coefficient * direction).astype(base.dtype)
        np.testing.assert_array_equal(actual, expected)
        assert bool(jnp.all(actual != 0))


@pytest.mark.parametrize("size", [3, 9, 33])
def test_identity_half_conversion_tails_and_special_value_bits(size):
    source = jnp.resize(jnp.asarray([-0., 1., -2., np.inf, -np.inf, np.nan, 65504], dtype=jnp.float16), (size,))
    actual = jax.jit(lambda value: copy_p.bind(value, records=(AffineRecord((size,), (1,), 0, (1,), 0),),
                                              output_size=size, dtype=np.dtype(jnp.float32)))(source)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(source.astype(jnp.float32)).view(np.uint32))


@pytest.mark.parametrize("dtype,step", [(jnp.float16, 2**-10), (jnp.bfloat16, 2**-7)])
@pytest.mark.parametrize("base_factor", [0., 1., .25])
def test_final_update_write_rounds_low_precision_after_combining_terms(dtype, step, base_factor):
    with jax.enable_x64():
        midpoints = np.asarray([1 + step / 2, 1 + 1.5 * step], dtype=np.float32)
        values = np.concatenate((np.nextafter(midpoints, np.float32(0)), midpoints, np.nextafter(midpoints, np.float32(2))))
        values = np.concatenate((values, -values))
        source, base = jnp.ones((values.size, 3), dtype=dtype), jnp.full((values.size, 7), .5, dtype=dtype)
        coefficients, beta = jnp.asarray(values, dtype=jnp.float64), jnp.float64(base_factor)
        actual = jax.jit(lambda old, new, alpha, beta: update_p.bind(new, old, alpha, beta, records=PARTIAL))(
            base, source, coefficients, beta)
        expected = (coefficients[:, None] * source.astype(jnp.float64) + beta * .5).astype(dtype)
        np.testing.assert_array_equal(actual[:, 1::2], expected)
        np.testing.assert_array_equal(actual[:, ::2], base[:, ::2])


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                                   "float16", "bfloat16", "float32", "float64", "complex64", "complex128"])
def test_batched_coefficients_final_conversion_matches_jax(dtype):
    with jax.enable_x64():
        target = jnp.dtype(dtype)
        values = [0., 1., 2., 3., 7., 15., 31.] if jnp.issubdtype(target, jnp.unsignedinteger) else [-2., -1., 0., 1., 2., 3., 7.]
        if jnp.issubdtype(target, jnp.inexact):
            values += [1.00048828125, 1.00390625, 1.01171875, -.125]
        source = jnp.asarray(values, dtype=target)
        promoted = source.astype(jnp.complex128 if jnp.issubdtype(target, jnp.complexfloating) else jnp.float64)
        if jnp.issubdtype(target, jnp.complexfloating):
            promoted += 1.25j
            source = promoted.astype(target)
        count = source.size
        records = (AffineRecord((count,), (1,), 0, (1,), 0),)
        actual = jax.jit(lambda coefficients: update_p.bind(
            jnp.ones((count, count), dtype=target), jnp.zeros((count, count), dtype=target),
            coefficients, jnp.int32(0), records=records))(promoted)
        np.testing.assert_array_equal(actual, jnp.broadcast_to(source[:, None], actual.shape))


@pytest.mark.parametrize("source_dtype,result_dtype", [("int32", "int8"), ("int32", "int16"), ("uint32", "uint8"),
                                                       ("int64", "int64"), ("uint64", "uint64")])
def test_integer_conversion_preserves_boundary_bits(source_dtype, result_dtype):
    with jax.enable_x64():
        limits = np.iinfo(source_dtype)
        source = jnp.asarray(np.asarray([limits.min, limits.min + 1, 0, 1, limits.max - 1, limits.max], dtype=source_dtype))
        records = (AffineRecord((source.size,), (1,), 0, (1,), 0),)
        actual = jax.jit(lambda value: update_p.bind(value, jnp.zeros(value.shape, dtype=result_dtype),
                                                    jnp.int32(1), jnp.int32(0), records=records))(source)
        np.testing.assert_array_equal(actual, source.astype(result_dtype))
