"""Native-valued updates preserve arithmetic stages and final conversions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ops._update import _execute_update
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_force_generic_for_tests,
)

from ._oracle import execute_update_reference


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


def _plan(source_dtype, result_dtype):
    return build_affine_plan(
        records=(AffineRecord((3,), (1,), 0, (2,), 1),),
        source_size=3, output_size=7, source_dtype=source_dtype,
        result_dtype=result_dtype, coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )


@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
@pytest.mark.parametrize("generic", [False, True])
def test_native_computation_batch_branches_and_final_cast(
    source_dtype, result_dtype, coefficient_dtype, generic,
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
        plan = _plan(source.dtype, base.dtype)
        def operation(old, values, alpha, beta):
            return _execute_update(old, values, source_factor=alpha, base_factor=beta, plan=plan)
        expected = execute_update_reference(bases, sources, first, second, plan)
        _set_native_force_generic_for_tests(generic)
        try:
            executable = jax.jit(jax.vmap(operation)).lower(bases, sources, first, second).compile()
            _reset_native_call_count_for_tests()
            actual = executable(bases, sources, first, second).block_until_ready()
            assert _native_call_count_for_tests() == 1
            memory = executable.memory_analysis()
            assert memory is not None
            assert memory.temp_size_in_bytes == 0
            for component in (jnp.real, jnp.imag):
                np.testing.assert_allclose(component(actual), component(expected), rtol=2e-3, atol=0)
            np.testing.assert_array_equal(actual[:, ::2], bases[:, ::2])
        finally:
            _set_native_force_generic_for_tests(False)


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
        plan = _plan(source.dtype, base.dtype)
        actual = jax.jit(lambda old, values, alpha, beta: _execute_update(
            old, values, source_factor=alpha, base_factor=beta, plan=plan,
        ))(base, source, first, second)
        expected = execute_update_reference(base, source, first, second, plan)
        np.testing.assert_array_equal(actual, expected)
        assert bool(jnp.all(actual[1::2] != 0))


@pytest.mark.parametrize("compute_dtype", ["float16", "bfloat16"])
def test_native_low_precision_computation_with_weak_float_storage(compute_dtype):
    with jax.enable_x64(False):
        source = jnp.broadcast_to(jnp.asarray(1.125), (3,))
        base = jnp.broadcast_to(jnp.asarray(.25), (7,))
        assert source.weak_type and base.weak_type
        first, second = jnp.asarray(.75, dtype=compute_dtype), jnp.asarray(.5, dtype=compute_dtype)
        plan = _plan(source.dtype, base.dtype)
        actual = jax.jit(lambda old, values, alpha, beta: _execute_update(
            old, values, source_factor=alpha, base_factor=beta, plan=plan,
        ))(base, source, first, second)
        expected = execute_update_reference(base, source, first, second, plan)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
def test_native_computation_explicit_differentials(source_dtype, result_dtype, coefficient_dtype):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=source_dtype)
        base = jnp.arange(7, dtype=jnp.float32).astype(result_dtype)
        first, second = jnp.asarray(.75, dtype=coefficient_dtype), jnp.asarray(.5, dtype=coefficient_dtype)
        plan = _plan(source.dtype, base.dtype)
        def operation(old, values, alpha, beta):
            return _execute_update(old, values, source_factor=alpha, base_factor=beta, plan=plan)
        def reference(old, values, alpha, beta):
            return old.at[1::2].set((alpha * values + beta * old[1::2]).astype(old.dtype))
        arguments = (base, source, first, second)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        def derivative(function, *values):
            return jax.jvp(function, values, directions)[1]
        actual = jax.jit(lambda *values: derivative(operation, *values))(*arguments)
        expected = derivative(reference, *arguments)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=0)
        for actual_value, expected_value in zip(
            jax.vjp(operation, *arguments)[1](jnp.ones_like(base)),
            jax.vjp(reference, *arguments)[1](jnp.ones_like(base)), strict=True,
        ):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-3, atol=0)
        actual = jax.jvp(lambda *values: derivative(operation, *values), arguments, directions)[1]
        expected = jax.jvp(lambda *values: derivative(reference, *values), arguments, directions)[1]
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=0)
