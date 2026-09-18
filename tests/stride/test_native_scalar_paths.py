"""Native-valued updates preserve arithmetic stages and final conversions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)


_CASES = [
    ("float16", "float16", "float32"),
    ("bfloat16", "bfloat16", "float32"),
    ("float32", "float32", "float64"),
    ("complex64", "complex64", "complex128"),
    ("float16", "float32", "float64"),
    ("float32", "complex64", "complex128"),
    ("complex64", "float32", "complex64"),
    ("complex128", "float64", "complex128"),
]


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                   np.asarray(component(expected)).astype(np.float64), rtol=2e-3, atol=0)


def store(base, selected):
    if not jnp.issubdtype(base.dtype, jnp.complexfloating):
        selected = jnp.real(selected)
    return base.at[1::2].set(selected.astype(base.dtype))


@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
def test_native_computation_batch_branches_and_final_cast(
    source_dtype, result_dtype, coefficient_dtype,
):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=source_dtype)
        if jnp.issubdtype(source.dtype, jnp.complexfloating):
            source = source + jnp.asarray(.25j, dtype=source.dtype)
        base = jnp.arange(7, dtype=jnp.float32).astype(result_dtype)
        first = jnp.asarray([0, 1, .75, .75, .75], dtype=coefficient_dtype)
        second = jnp.asarray([.5, .5, 0, 1, .5], dtype=coefficient_dtype)
        sources = jnp.broadcast_to(source, (5, 3))
        bases = jnp.broadcast_to(base, (5, 7))
        def operation(old, values, alpha, beta):
            return update_p.bind(values, old, alpha, beta, records=PARTIAL)
        expected_rows = []
        for alpha, beta in zip(first, second, strict=True):
            source_term = source if alpha == 1 else alpha * source
            base_term = base[1::2] if beta == 1 else beta * base[1::2]
            selected = base_term if alpha == 0 else source_term if beta == 0 else source_term + base_term
            expected_rows.append(store(base, selected))
        expected = jnp.stack(expected_rows)
        for function in (operation, jax.vmap(operation)):
            lowered = jax.jit(function).lower(bases, sources, first, second)
            assert lowered.as_text().count("custom_call") == 1
            assert "tensor0_stride_update_" in lowered.as_text()
            actual = lowered.compile()(bases, sources, first, second).block_until_ready()
            assert_close(actual, expected)
            np.testing.assert_array_equal(actual[:, ::2], bases[:, ::2])


@pytest.mark.parametrize("dtype,coefficient_dtype,value,increment", [
    ("float16", "float32", 65504., .001),
    ("bfloat16", "float32", 256., 1 / 256),
    ("float32", "float64", 2**24, 2**-25),
    ("complex64", "complex128", 2**24 * (1+1j), 2**-25 * (1+1j)),
])
def test_native_update_does_not_round_source_term_to_storage(
    dtype, coefficient_dtype, value, increment,
):
    with jax.enable_x64():
        source = jnp.full((3,), value, dtype=dtype)
        base = jnp.full((7,), value, dtype=dtype)
        first, second = jnp.asarray(1 + increment, dtype=coefficient_dtype), jnp.asarray(-1, dtype=coefficient_dtype)
        actual = jax.jit(lambda old, values, alpha, beta: update_p.bind(
            values, old, alpha, beta, records=PARTIAL,
        ))(base, source, first, second)
        expected = store(base, first * source + second * base[1::2])
        assert_close(actual, expected)
        assert bool(jnp.all(actual[1::2] != 0))


@pytest.mark.parametrize("coefficient_dtype", ["float16", "bfloat16"])
def test_native_low_precision_coefficients_with_weak_float_storage(coefficient_dtype):
    with jax.enable_x64(False):
        source = jnp.broadcast_to(jnp.asarray(1.0003), (3,))
        base = jnp.broadcast_to(jnp.asarray(.2503), (7,))
        assert source.weak_type and base.weak_type
        first, second = jnp.asarray(.75, dtype=coefficient_dtype), jnp.asarray(.5, dtype=coefficient_dtype)
        actual = jax.jit(lambda old, values, alpha, beta: update_p.bind(
            values, old, alpha, beta, records=PARTIAL,
        ))(base, source, first, second)
        expected = store(base, first.astype(source.dtype) * source.astype(source.dtype)
                         + second.astype(base.dtype) * base[1::2].astype(base.dtype))
        assert not actual.weak_type
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
def test_native_computation_explicit_differentials(source_dtype, result_dtype, coefficient_dtype):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=source_dtype)
        base = jnp.arange(7, dtype=jnp.float32).astype(result_dtype)
        first, second = jnp.asarray(.75, dtype=coefficient_dtype), jnp.asarray(.5, dtype=coefficient_dtype)
        def operation(old, values, alpha, beta):
            return update_p.bind(values, old, alpha, beta, records=PARTIAL)
        def reference(old, values, alpha, beta):
            return store(old, alpha * values + beta * old[1::2])
        arguments = (base, source, first, second)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        def derivative(function, *values):
            return jax.jvp(function, values, directions)[1]
        actual = jax.jit(lambda *values: derivative(operation, *values))(*arguments)
        expected = derivative(reference, *arguments)
        assert_close(actual, expected)
        for actual_value, expected_value in zip(
            jax.vjp(operation, *arguments)[1](jnp.ones_like(base)),
            jax.vjp(reference, *arguments)[1](jnp.ones_like(base)), strict=True,
        ):
            assert_close(actual_value, expected_value)
        actual = jax.jvp(lambda *values: derivative(operation, *values), arguments, directions)[1]
        expected = jax.jvp(lambda *values: derivative(reference, *values), arguments, directions)[1]
        assert_close(actual, expected)
