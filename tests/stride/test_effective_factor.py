"""Explicit coefficient composition precedes native update binding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._dtype import normalize_coefficient
from tensor0._stride._jax import accumulation_p, update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)
SINGLE = (AffineRecord((1,), (1,), 0, (1,), 0),)


@pytest.mark.parametrize("first,second", [(2., .5), (0., 2.), (.75, 2.)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_effective_factor_short_circuits_after_coefficient_product(first, second, dtype):
    source = jnp.asarray([0., 40000., -3.5], dtype=dtype)
    base = jnp.full((7,), -7., dtype=dtype)
    coefficient = jnp.asarray(second, dtype=dtype)

    def operation(old, values, factor):
        return update_p.bind(values, old, jnp.asarray(first, dtype=dtype) * factor,
                             jnp.asarray(0, dtype=dtype), records=PARTIAL)

    effective = first * second
    selected = jnp.zeros_like(source) if effective == 0 else source if effective == 1 else source * effective
    expected = base.at[1::2].set(selected)
    for function in (operation, jax.jit(operation)):
        np.testing.assert_allclose(function(base, source, coefficient), expected, rtol=1e-3, atol=0)
    text = jax.jit(operation).lower(base, source, coefficient).as_text()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text


def test_mapping_and_update_receive_independent_coefficients():
    source = jnp.asarray([30000], dtype=jnp.float16)
    fresh = accumulation_p.bind(source, jnp.float16(2), records=SINGLE,
                                coefficient_records=(0,), output_size=1, dtype=source.dtype)
    updated = update_p.bind(source, jnp.zeros_like(source), jnp.float16(2) * jnp.float16(.5),
                            jnp.float16(0), records=SINGLE)
    np.testing.assert_allclose(fresh, source * jnp.float16(2), rtol=1e-3, atol=0)
    np.testing.assert_array_equal(updated, source)


def test_data_derivative_uses_the_composed_factor_without_source_prescaling():
    source = jnp.asarray([30000], dtype=jnp.float16)
    direction = jnp.asarray([65504], dtype=jnp.float16)
    operation = lambda values: update_p.bind(
        values, jnp.zeros_like(values), jnp.float16(2) * jnp.float16(.5),
        jnp.float16(0), records=SINGLE)
    tangent = jax.jit(lambda values, delta: jax.jvp(operation, (values,), (delta,))[1])(source, direction)
    cotangent = jax.jit(lambda values, delta: jax.vjp(operation, values)[1](delta)[0])(source, direction)
    np.testing.assert_array_equal(tangent, direction)
    np.testing.assert_array_equal(cotangent, direction)


@pytest.mark.parametrize("strong", [False, True])
def test_coefficient_product_precedes_weak_normalization(strong):
    with jax.enable_x64():
        first = np.float32(1.0003) if strong else 1.0003
        second = np.float32(1.0003) if strong else 1.0003
        source = jnp.ones((1,), dtype=jnp.float16)

        def operation(values, factor):
            effective = normalize_coefficient(values.dtype, jnp.asarray(first) * factor)
            return update_p.bind(values, jnp.zeros_like(values), effective, jnp.float16(0), records=SINGLE)

        product = jnp.asarray(first) * jnp.asarray(second)
        coefficient = product if strong else product.astype(source.dtype)
        expected = (source * coefficient).astype(source.dtype)
        for function in (operation, jax.jit(operation)):
            actual = function(source, jnp.asarray(second))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
            assert float(actual[0]) > 1


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bool_])
def test_integer_coefficient_product_precedes_source_promotion(dtype):
    records = (AffineRecord((2,), (1,), 0, (1,), 0),)
    source = jnp.asarray([1, 3], dtype=dtype)
    actual = jax.jit(lambda values, factor: update_p.bind(
        values, jnp.zeros_like(values), jnp.int8(100) * factor, jnp.int8(0), records=records))(
            source, jnp.int8(2))
    expected = jnp.asarray([True, True] if dtype == jnp.bool_ else [-56, -168], dtype=dtype)
    np.testing.assert_array_equal(actual, expected)


def test_boolean_coefficient_product_selects_zero_without_reading_source():
    source = jnp.asarray([jnp.nan], dtype=jnp.float32)
    actual = jax.jit(lambda values, factor: update_p.bind(
        values, jnp.zeros_like(values), jnp.asarray(False) * factor, jnp.float32(0), records=SINGLE))(
            source, jnp.float32(3.5))
    np.testing.assert_array_equal(actual, jnp.zeros_like(actual))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.complex64])
def test_independent_effective_factors_batch_and_differentiate(dtype):
    with jax.enable_x64():
        factors = (np.float64(1.0003), np.float32(-.75))
        records = tuple(AffineRecord((2,), (2,), offset, (2,), offset + 1) for offset in range(2))
        base = jnp.ones((3, 6), dtype=dtype)
        source = jnp.asarray([[1., 2., 3., 4.]] * 3, dtype=dtype)
        alpha, beta = jnp.asarray([0., 1., 1.0003]), jnp.asarray([1., 0., -.25])

        def operation(old, values, first, second):
            result = old
            for record, factor in zip(records, factors, strict=True):
                result = update_p.bind(values, result, first * factor, second, records=(record,))
            return result

        def reference(old, values, first, second):
            result = old
            for offset, factor in enumerate(factors):
                selected = (first[:, None] * factor * values[:, offset::2]
                            + second[:, None] * old[:, offset + 1:offset + 4:2])
                result = result.at[:, offset + 1:offset + 4:2].set(selected.astype(dtype))
            return result

        arguments = (base, source, alpha, beta)
        tangents = tuple(jnp.ones_like(value) for value in arguments)
        expected = reference(*arguments)
        np.testing.assert_allclose(jax.jit(operation)(*arguments), expected, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(jax.jit(jax.vmap(operation))(*arguments), expected, rtol=1e-3, atol=1e-6)
        actual_jvp = jax.jit(lambda *values: jax.jvp(operation, values, tangents)[1])(*arguments)
        expected_jvp = jax.jvp(reference, arguments, tangents)[1]
        np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=1e-3, atol=1e-6)
        cotangent = jnp.ones_like(base)
        actual_vjp = jax.jit(lambda *values: jax.vjp(operation, *values)[1](cotangent))(*arguments)
        expected_vjp = jax.vjp(reference, *arguments)[1](cotangent)
        for actual_part, expected_part in zip(actual_vjp, expected_vjp, strict=True):
            np.testing.assert_allclose(actual_part, expected_part, rtol=1e-3, atol=1e-6)
        loss = lambda coefficient: jnp.real(operation(base, source, coefficient, beta)).sum()
        expected_loss = lambda coefficient: jnp.real(reference(base, source, coefficient, beta)).sum()
        np.testing.assert_allclose(jax.jacfwd(jax.grad(loss))(alpha),
                                   jax.jacfwd(jax.grad(expected_loss))(alpha), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
@pytest.mark.parametrize("transpose", [False, True])
def test_composed_factor_uses_one_update_for_contiguous_and_transposed_layouts(dtype, transpose):
    strides = (1, 9) if transpose else (17, 1)
    records = (AffineRecord((9, 17), strides, 0, (17, 1), 1),)
    source = jnp.arange(153, dtype=jnp.float32).astype(dtype)
    base = jnp.full((155,), -7, dtype=dtype)
    factor = jnp.asarray(.5, dtype=dtype)
    lowered = jax.jit(lambda old, values, coefficient: update_p.bind(
        values, old, coefficient * jnp.asarray(1.5, dtype=dtype), jnp.asarray(0, dtype=dtype),
        records=records)).lower(base, source, factor)
    text = lowered.as_text()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text
    values = source.reshape(17, 9).T.reshape(-1) if transpose else source
    expected = base.at[1:-1].set(values * jnp.asarray(.75, dtype=dtype))
    actual = lowered.compile()(base, source, factor)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
    np.testing.assert_array_equal(actual[jnp.asarray([0, 154])], base[jnp.asarray([0, 154])])
