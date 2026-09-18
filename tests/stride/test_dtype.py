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


def assert_product_types(left, right):
    left_shape = jax.ShapeDtypeStruct((), left.dtype, weak_type=left.weak_type)
    right_shape = jax.ShapeDtypeStruct((), right.dtype, weak_type=right.weak_type)
    expected = outcome(lambda: jax.eval_shape(lambda first, second: first * second,
                                            left_shape, right_shape))
    actual = outcome(lambda: _dtype._product_shape(left, right))
    assert actual == expected
    if isinstance(expected[0], str):
        assert _dtype.product_dtype(left, right) == jnp.dtype(expected[0])
    else:
        with pytest.raises(expected[0]):
            _dtype.product_dtype(left, right)


@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("left_dtype", DTYPES)
@pytest.mark.parametrize("right_dtype", DTYPES)
def test_concrete_types(left_dtype, right_dtype, promotion):
    with jax.enable_x64(), jax.numpy_dtype_promotion(promotion):
        left = jax.ShapeDtypeStruct((), jnp.dtype(left_dtype))
        right = jax.ShapeDtypeStruct((), jnp.dtype(right_dtype))
        assert_product_types(left, right)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("source_dtype", ["bool", "int32", "float16", "float32", "complex64"])
@pytest.mark.parametrize("factor", [True, 2, 1.5, 0.5 + 1j, np.float16(0.5), np.float32(0.5)])
def test_weak_and_strong_mapping(source_dtype, factor, promotion, x64):
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        source = jax.ShapeDtypeStruct((), jnp.dtype(source_dtype))
        coefficient = jnp.asarray(factor)
        assert_product_types(source, coefficient)
        expected = outcome(lambda: jax.eval_shape(lambda value: value * coefficient, source))
        if isinstance(expected[0], str):
            assert _dtype.mapping_dtype(source.dtype, factor) == jnp.dtype(expected[0])
        else:
            with pytest.raises(expected[0]) as actual:
                _dtype.mapping_dtype(source.dtype, factor)
            assert str(actual.value) == expected[1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_absent_mapping_preserves_dtype(dtype):
    with jax.enable_x64():
        assert _dtype.mapping_dtype(jnp.dtype(dtype), None) == jnp.dtype(dtype)


@pytest.mark.parametrize("x64", [False, True])
def test_weak_source_and_strong_factor(x64):
    with jax.enable_x64(x64), jax.numpy_dtype_promotion("strict"):
        source = jnp.asarray(1.0003)
        factor = jnp.asarray(1, dtype=jnp.float16)
        assert source.weak_type and not factor.weak_type
        assert_product_types(source, factor)
        assert _dtype.product_dtype(source, factor) == jnp.dtype("float16")


def test_cache_respects_configuration():
    _dtype._binary_type.cache_clear()
    source = jax.ShapeDtypeStruct((), jnp.dtype("float32"))
    for x64 in (False, True):
        for promotion in ("standard", "strict"):
            with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
                assert _dtype.product_dtype(source, source) == jnp.dtype("float32")
    assert _dtype._binary_type.cache_info().misses == 4
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        assert _dtype.product_dtype(source, source) == jnp.dtype("float32")
    assert _dtype._binary_type.cache_info().hits == 1


def test_cache_ignores_values_but_not_weakness():
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        _dtype._binary_type.cache_clear()
        source = jnp.ones(3, dtype=jnp.float16)
        for factor in (0.0, 1.0, -0.0, float("nan"), float("inf")):
            assert _dtype.product_dtype(source, jnp.asarray(factor)) == jnp.dtype("float16")
        assert _dtype._binary_type.cache_info().misses == 1
        assert _dtype._binary_type.cache_info().hits == 4
        assert _dtype.product_dtype(source, jnp.asarray(1.0, dtype=jnp.float32)) == jnp.dtype("float32")
        assert _dtype._binary_type.cache_info().misses == 2


def test_type_resolution_accepts_traced_operands():
    def operation(source, coefficient):
        result_dtype = _dtype.product_dtype(source, coefficient)
        assert result_dtype == _dtype.mapping_dtype(source.dtype, coefficient)
        return jnp.asarray(source * coefficient, dtype=result_dtype)

    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        source = jnp.asarray([1, 2, 3], dtype=jnp.float16)
        coefficient = jnp.asarray(0.5, dtype=jnp.float32)
        result = jax.jit(operation)(source, coefficient)
        np.testing.assert_array_equal(result, source * coefficient)
        assert result.dtype == jnp.dtype("float32")


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
