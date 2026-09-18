"""Update products retain their concrete types before final storage conversion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)


@pytest.mark.parametrize("source_is_narrow", [False, True])
def test_update_preserves_each_product_rounding(source_is_narrow):
    with jax.enable_x64():
        narrow_value = np.float32(1 - 2 ** -23)
        narrow_factor = jnp.asarray(1 + 2 ** -23, dtype=jnp.float32)
        wide_factor = jnp.asarray(2, dtype=jnp.float64)
        source = jnp.full((3,), narrow_value if source_is_narrow else -.5, dtype=jnp.float32)
        base = jnp.full((7,), -.5 if source_is_narrow else narrow_value, dtype=jnp.float32)
        alpha, beta = (narrow_factor, wide_factor) if source_is_narrow else (wide_factor, narrow_factor)
        function = jax.jit(lambda old, values, first, second: update_p.bind(
            values, old, first, second, records=PARTIAL,
        ))
        source_term, base_term = alpha * source, beta * base[1::2]
        assert (source_term.dtype, base_term.dtype) == (
            (jnp.float32, jnp.float64) if source_is_narrow else (jnp.float64, jnp.float32))
        expected = base.at[1::2].set((source_term + base_term).astype(base.dtype))
        before_source, before_base = np.asarray(source).copy(), np.asarray(base).copy()
        lowered = function.lower(base, source, alpha, beta)
        assert lowered.as_text().count("custom_call") == 1
        assert "tensor0_stride_update_f32" in lowered.as_text()
        actual = lowered.compile()(base, source, alpha, beta).block_until_ready()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(actual[::2], before_base[::2])
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(base, before_base)


@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("coefficient_values", [(0, .25), (1, .25), (.75, 1), (.75, .25)])
def test_mixed_product_stage_differentials(source_is_narrow, coefficient_values):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=jnp.float32)
        base = jnp.arange(7, dtype=jnp.float32)
        alpha = jnp.asarray(coefficient_values[0], dtype=jnp.float32 if source_is_narrow else jnp.float64)
        beta = jnp.asarray(coefficient_values[1], dtype=jnp.float64 if source_is_narrow else jnp.float32)
        def operation(old, values, first, second):
            return update_p.bind(values, old, first, second, records=PARTIAL)

        def reference(old, values, first, second):
            return old.at[1::2].set((first * values + second * old[1::2]).astype(old.dtype))

        arguments = (base, source, alpha, beta)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        for function in (operation, jax.jit(operation)):
            actual = jax.jvp(function, arguments, directions)[1]
            expected = jax.jvp(reference, arguments, directions)[1]
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
            actual_vjp = jax.vjp(function, *arguments)[1](jnp.ones_like(base))
            expected_vjp = jax.vjp(reference, *arguments)[1](jnp.ones_like(base))
            for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
            actual = jax.jvp(lambda *values: jax.jvp(function, values, directions)[1], arguments, directions)[1]
            expected = jax.jvp(lambda *values: jax.jvp(reference, values, directions)[1], arguments, directions)[1]
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("promotion", ["standard", "strict"])
def test_weak_storage_uses_concrete_dtype_at_native_boundary(promotion):
    with jax.enable_x64(False), jax.numpy_dtype_promotion(promotion):
        source = jnp.broadcast_to(jnp.asarray(1.0003), (3,))
        base = jnp.broadcast_to(jnp.asarray(.25), (7,))
        assert source.weak_type and base.weak_type
        alpha = jnp.asarray(1, dtype=jnp.float16)
        beta = jnp.asarray(.3, dtype=jnp.float16)
        actual = jax.jit(lambda old, values, first, second: update_p.bind(
            values, old, first, second, records=PARTIAL,
        ))(base, source, alpha, beta)
        expected = np.asarray(base).copy()
        expected[1::2] = np.asarray(source) + np.float32(beta) * np.asarray(base)[1::2]
        assert actual.dtype == base.dtype and not actual.weak_type
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("factor", [None, .5])
def test_mixed_product_stages_batch_zero_one_and_records(source_is_narrow, factor):
    with jax.enable_x64():
        source = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 8
        base = jnp.arange(35, dtype=jnp.float32).reshape(5, 7) / 16
        alpha = jnp.asarray([0, 1, .75, 1, .75], dtype=jnp.float32 if source_is_narrow else jnp.float64)
        beta = jnp.asarray([.25, 1, 0, .25, 1], dtype=jnp.float64 if source_is_narrow else jnp.float32)
        records = (AffineRecord((2,), (1,), 0, (2,), 1),
                   AffineRecord((1,), (1,), 2, (1,), 5))

        def operation(old, values, first, second):
            effective = first if factor is None else first * jnp.asarray(factor, dtype=first.dtype)
            return update_p.bind(values, old, effective, second, records=records)

        effective = alpha if factor is None else alpha * jnp.asarray(factor, dtype=alpha.dtype)
        expected = base.at[:, 1::2].set(
            (effective[:, None] * source + beta[:, None] * base[:, 1::2]).astype(base.dtype))
        direct = jax.jit(operation)(base, source, alpha, beta)
        mapped = jax.jit(jax.vmap(operation))(base, source, alpha, beta)
        np.testing.assert_allclose(direct, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(mapped, expected, rtol=2e-6, atol=1e-7)
