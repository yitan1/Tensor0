"""Concrete and weak operand type resolution against JAX, without Native execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import _dtype


DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
)


def outcome(function):
    try:
        result = function()
    except (TypeError, jax.dtypes.TypePromotionError) as error:
        return type(error), str(error)
    return result.dtype.name, result.weak_type


def assert_normalized_type(source_dtype, factor):
    coefficient = jnp.asarray(factor)
    source = jax.ShapeDtypeStruct((), jnp.dtype(source_dtype))
    coefficient_shape = jax.ShapeDtypeStruct((), coefficient.dtype, weak_type=coefficient.weak_type)

    def reference():
        result_dtype = jax.eval_shape(jnp.multiply, source, coefficient_shape).dtype
        if not coefficient.weak_type:
            return coefficient
        if (jnp.issubdtype(result_dtype, jnp.complexfloating)
                and not jnp.issubdtype(coefficient.dtype, jnp.complexfloating)):
            result_dtype = jnp.real(jnp.zeros((), dtype=result_dtype)).dtype
        return coefficient.astype(result_dtype)

    assert outcome(lambda: _dtype.normalize_coefficient(source_dtype, factor)) == outcome(reference)


@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("factor_dtype", DTYPES)
def test_concrete_types(source_dtype, factor_dtype, promotion):
    with jax.enable_x64(), jax.numpy_dtype_promotion(promotion):
        assert_normalized_type(source_dtype, jnp.asarray(1, dtype=factor_dtype))


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("source_dtype", ["bool", "int32", "float16", "float32", "complex64"])
@pytest.mark.parametrize("factor", [True, 2, 1.5, 0.5 + 1j, np.float16(0.5), np.float32(0.5)])
def test_weak_and_strong_coefficients(source_dtype, factor, promotion, x64):
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        assert_normalized_type(source_dtype, factor)


def test_cache_respects_configuration():
    _dtype._multiplication_dtype.cache_clear()
    source_dtype = jnp.dtype("float32")
    factor = jnp.asarray(1, dtype=source_dtype)
    for x64 in (False, True):
        for promotion in ("standard", "strict"):
            with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
                assert _dtype.normalize_coefficient(source_dtype, factor).dtype == source_dtype
    assert _dtype._multiplication_dtype.cache_info().misses == 4
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        assert _dtype.normalize_coefficient(source_dtype, factor).dtype == source_dtype
    assert _dtype._multiplication_dtype.cache_info().hits == 1


def test_cache_ignores_values_but_not_weakness():
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        _dtype._multiplication_dtype.cache_clear()
        for factor in (0.0, 1.0, -0.0, float("nan"), float("inf")):
            assert _dtype.normalize_coefficient(jnp.float16, factor).dtype == jnp.dtype("float16")
        assert _dtype._multiplication_dtype.cache_info().misses == 1
        assert _dtype._multiplication_dtype.cache_info().hits == 4
        factor = jnp.asarray(1.0, dtype=jnp.float32)
        assert _dtype.normalize_coefficient(jnp.float16, factor).dtype == factor.dtype
        assert _dtype._multiplication_dtype.cache_info().misses == 2


def test_normalization_accepts_strong_traced_coefficients():
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        coefficient = jnp.asarray([0, 0.5, 1], dtype=jnp.float32)
        result = jax.jit(lambda value: _dtype.normalize_coefficient(jnp.float16, value))(coefficient)
        np.testing.assert_array_equal(result, coefficient)
        assert result.dtype == coefficient.dtype


@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("factor_dtype", DTYPES)
def test_normalization_preserves_strong_dtype(source_dtype, factor_dtype):
    with jax.enable_x64():
        factor = jnp.asarray([0, 1, 2], dtype=factor_dtype)
        normalized = _dtype.normalize_coefficient(source_dtype, factor)
        assert normalized.dtype == factor.dtype
        np.testing.assert_array_equal(normalized, factor)


@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("factor", [2, .5, 2j])
def test_normalization_resolves_weak_dtype(source_dtype, factor):
    with jax.enable_x64():
        original = jnp.asarray(factor)
        source = jax.ShapeDtypeStruct((), jnp.dtype(source_dtype))
        result_dtype = jax.eval_shape(lambda values: values * factor, source).dtype
        expected_dtype = (jnp.real(jnp.zeros((), dtype=result_dtype)).dtype
                          if jnp.issubdtype(result_dtype, jnp.complexfloating) and not isinstance(factor, complex)
                          else result_dtype)
        normalized = _dtype.normalize_coefficient(source_dtype, original)
        assert normalized.dtype == expected_dtype
        assert not normalized.weak_type
        np.testing.assert_array_equal(normalized, original.astype(expected_dtype))


@pytest.mark.parametrize("x64", [False, True])
def test_normalization_tracks_x64_and_tracers(x64):
    with jax.enable_x64(x64):
        execute = jax.jit(lambda value: _dtype.normalize_coefficient(jnp.int8, value))
        assert execute(.5).dtype == jnp.dtype("float64" if x64 else "float32")
        np.testing.assert_array_equal(execute(257), np.int8(1))
        np.testing.assert_array_equal(execute(256), np.int8(0))
