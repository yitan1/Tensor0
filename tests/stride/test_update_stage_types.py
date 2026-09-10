"""Reachable update stages must retain rounding and weak-type promotion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ops._update import _execute_update
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._scalar import forward_update_dtypes
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_force_generic_for_tests,
)

from ._oracle import execute_update_reference


def _plan():
    return build_affine_plan(
        records=(AffineRecord((3,), (1,), 0, (2,), 1),),
        source_size=3, output_size=7, source_dtype="float32",
        result_dtype="float32", coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )


@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("generic", [False, True])
def test_update_preserves_each_product_rounding(source_is_narrow, generic):
    with jax.enable_x64():
        narrow_value = np.float32(1 - 2 ** -23)
        narrow_factor = jnp.asarray(1 + 2 ** -23, dtype=jnp.float32)
        wide_factor = jnp.asarray(2, dtype=jnp.float64)
        source = jnp.full((3,), narrow_value if source_is_narrow else -.5, dtype=jnp.float32)
        base = jnp.full((7,), -.5 if source_is_narrow else narrow_value, dtype=jnp.float32)
        alpha, beta = (narrow_factor, wide_factor) if source_is_narrow else (wide_factor, narrow_factor)
        plan = _plan()
        stages = forward_update_dtypes(base, source, alpha, beta, plan.records)[0]
        assert stages[1:3] == (("float32", "float64") if source_is_narrow else ("float64", "float32"))
        assert stages[-1] == "float64"
        function = jax.jit(lambda old, values, first, second: _execute_update(
            old, values, source_factor=first, base_factor=second, plan=plan,
        ))
        expected = execute_update_reference(base, source, alpha, beta, plan)
        np.testing.assert_array_equal(expected[1::2], 0)
        widened = (alpha.astype(jnp.float64) * source.astype(jnp.float64)
                   + beta.astype(jnp.float64) * base[1::2].astype(jnp.float64)).astype(jnp.float32)
        assert bool(jnp.all(widened != expected[1::2]))
        before_source, before_base = np.asarray(source).copy(), np.asarray(base).copy()
        _set_native_force_generic_for_tests(generic)
        try:
            executable = function.lower(base, source, alpha, beta).compile()
            executable(base, source, alpha, beta).block_until_ready()
            _reset_native_call_count_for_tests()
            actual = executable(base, source, alpha, beta).block_until_ready()
            assert _native_call_count_for_tests() == 1
        finally:
            _set_native_force_generic_for_tests(False)
        np.testing.assert_array_equal(actual, expected)
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
        plan = _plan()

        def operation(old, values, first, second):
            return _execute_update(old, values, source_factor=first, base_factor=second, plan=plan)

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
def test_weak_unit_term_can_narrow_at_actual_addition(promotion):
    with jax.enable_x64(False), jax.numpy_dtype_promotion(promotion):
        source = jnp.broadcast_to(jnp.asarray(1.0003), (3,))
        base = jnp.broadcast_to(jnp.asarray(.25), (7,))
        assert source.weak_type and base.weak_type
        alpha = jnp.asarray(1, dtype=jnp.float16)
        beta = jnp.asarray(.3, dtype=jnp.float16)
        plan = _plan()
        stages = forward_update_dtypes(base, source, alpha, beta, plan.records)[0]
        assert stages[1:3] == ("float16", "float16")
        assert stages[4] == "float16"
        actual = jax.jit(lambda old, values, first, second: _execute_update(
            old, values, source_factor=first, base_factor=second, plan=plan,
        ))(base, source, alpha, beta)
        expected = execute_update_reference(base, source, alpha, beta, plan)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_mixed_product_stages_batch_zero_one_and_records(source_is_narrow, static):
    with jax.enable_x64():
        source = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 8
        base = jnp.arange(35, dtype=jnp.float32).reshape(5, 7) / 16
        alpha = jnp.asarray([0, 1, .75, 1, .75], dtype=jnp.float32 if source_is_narrow else jnp.float64)
        beta = jnp.asarray([.25, 1, 0, .25, 1], dtype=jnp.float64 if source_is_narrow else jnp.float32)
        plan = build_affine_plan(
            records=(AffineRecord((2,), (1,), 0, (2,), 1, scale=np.float32(.5) if static else None),
                     AffineRecord((1,), (1,), 2, (1,), 5, scale=np.float64(2) if static else None)),
            source_size=3, output_size=7, source_dtype="float32", result_dtype="float32",
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        )

        def operation(old, values, first, second):
            return _execute_update(old, values, source_factor=first, base_factor=second, plan=plan)

        expected = execute_update_reference(base, source, alpha, beta, plan)
        direct = jax.jit(operation)(base, source, alpha, beta)
        mapped = jax.jit(jax.vmap(operation))(base, source, alpha, beta)
        np.testing.assert_array_equal(direct, expected)
        np.testing.assert_array_equal(mapped, expected)
