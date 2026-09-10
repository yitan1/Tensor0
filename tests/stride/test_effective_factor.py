import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._map import _execute_map
from tensor0._stride._ops._update import _execute_update
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._testing import (
    _native_call_count_for_tests, _reset_native_call_count_for_tests,
    _native_leaf_kernel_masks_for_tests, _observe_native_leaf_kernels_for_tests,
)

from ._oracle import execute_update_reference


@pytest.mark.parametrize("static,dynamic", [(2., .5), (0., 2.), (.75, 2.)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_effective_factor_short_circuits_after_coefficient_product(static, dynamic, dtype):
    plan = build_affine_plan(
        records=(AffineRecord((3,), (1,), 0, (2,), 1, scale=np.dtype(dtype).type(static)),),
        source_size=3, output_size=7, source_dtype=dtype, result_dtype=dtype,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    source = jnp.asarray([0., 40000., -3.5], dtype=dtype)
    base = jnp.full((7,), -7., dtype=dtype)
    coefficient = jnp.asarray(dynamic, dtype=dtype)
    execute = jax.jit(lambda old, values, factor: _execute_update(
        old, values, source_factor=factor, base_factor=0, plan=plan,
    ))
    expected = execute_update_reference(base, source, coefficient, 0, plan)
    for actual in (execute(base, source, coefficient), _execute_update(
        base, source, source_factor=np.dtype(dtype).type(dynamic), base_factor=0, plan=plan,
    )):
        for component in (np.real, np.imag):
            np.testing.assert_array_equal(component(actual), component(expected))
    _reset_native_call_count_for_tests()
    execute(base, source, coefficient).block_until_ready()
    assert _native_call_count_for_tests() == 1


def test_fresh_mapping_retains_its_independent_static_multiply():
    plan = build_affine_plan(
        records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=2),),
        source_size=1, output_size=1, source_dtype=jnp.float16, result_dtype=jnp.float16,
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source = jnp.asarray([30000], dtype=jnp.float16)
    fresh = _execute_map(source, plan=plan)
    updated = _execute_update(jnp.zeros_like(source), source,
                              source_factor=jnp.float16(.5), base_factor=0, plan=plan)
    np.testing.assert_array_equal(fresh, source * jnp.float16(2))
    np.testing.assert_array_equal(updated, source)


def test_data_derivative_uses_the_composed_factor_without_source_prescaling():
    plan = build_affine_plan(
        records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=np.float16(2)),),
        source_size=1, output_size=1, source_dtype=jnp.float16, result_dtype=jnp.float16,
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source = jnp.asarray([30000], dtype=jnp.float16)
    direction = jnp.asarray([65504], dtype=jnp.float16)
    operation = lambda values: _execute_update(
        jnp.zeros_like(values), values, source_factor=jnp.float16(.5), base_factor=0, plan=plan,
    )
    tangent = jax.jit(lambda values, delta: jax.jvp(operation, (values,), (delta,))[1])(
        source, direction,
    )
    cotangent = jax.jit(lambda values, delta: jax.vjp(operation, values)[1](delta)[0])(
        source, direction,
    )
    np.testing.assert_array_equal(tangent, direction)
    np.testing.assert_array_equal(cotangent, direction)


@pytest.mark.parametrize("strong", [False, True])
def test_effective_factor_preserves_weak_coefficient_product_stage(strong):
    with jax.enable_x64():
        static = np.float32(1.0003) if strong else 1.0003
        dynamic = np.float32(1.0003) if strong else 1.0003
        plan = build_affine_plan(
            records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=static),),
            source_size=1, output_size=1, source_dtype=jnp.float16,
            result_dtype=jnp.float16, coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        source = jnp.ones((1,), dtype=jnp.float16)
        base = jnp.zeros_like(source)
        expected = execute_update_reference(base, source, dynamic, 0, plan)
        for execute in (lambda: _execute_update(base, source, source_factor=dynamic,
                                                base_factor=0, plan=plan),
                        lambda: jax.jit(lambda values, coefficient: _execute_update(
                            base, values, source_factor=coefficient, base_factor=0, plan=plan,
                        ))(source, jnp.asarray(dynamic))):
            actual = execute()
            np.testing.assert_array_equal(actual, expected)
            assert float(actual[0]) > 1


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bool_])
def test_integer_coefficient_product_precedes_source_promotion(dtype):
    plan = build_affine_plan(
        records=(AffineRecord((2,), (1,), 0, (1,), 0, scale=np.int8(100)),),
        source_size=2, output_size=2, source_dtype=dtype, result_dtype=dtype,
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source = jnp.asarray([1, 3], dtype=dtype)
    base = jnp.zeros_like(source)
    factor = jnp.asarray(2, dtype=jnp.int8)
    actual = jax.jit(lambda values, coefficient: _execute_update(
        base, values, source_factor=coefficient, base_factor=0, plan=plan,
    ))(source, factor)
    expected = execute_update_reference(base, source, factor, 0, plan)
    np.testing.assert_array_equal(actual, expected)
    if dtype != jnp.bool_:
        np.testing.assert_array_equal(actual, jnp.asarray([-56, -168], dtype=dtype))


def test_boolean_static_factor_is_converted_before_ordinary_coefficient_multiply():
    plan = build_affine_plan(
        records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=False),),
        source_size=1, output_size=1, source_dtype=jnp.float32, result_dtype=jnp.float32,
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source, base = jnp.ones((1,)), jnp.zeros((1,))
    factor = jnp.float32(3.5)
    actual = jax.jit(lambda values, coefficient: _execute_update(
        base, values, source_factor=coefficient, base_factor=0, plan=plan,
    ))(source, factor)
    expected = execute_update_reference(base, source, factor, 0, plan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, jnp.zeros_like(actual))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.complex64])
def test_multirecord_effective_factors_batch_and_differentiate(dtype):
    with jax.enable_x64():
        factors = (np.float64(1.0003), np.float32(-.75))
        plan = build_affine_plan(
            records=tuple(AffineRecord((2,), (2,), offset, (2,), offset + 1, scale=factor)
                          for offset, factor in enumerate(factors)),
            source_size=4, output_size=6, source_dtype=dtype, result_dtype=dtype,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        )
        base = jnp.ones((3, 6), dtype=dtype)
        source = jnp.asarray([[1., 2., 3., 4.]] * 3, dtype=dtype)
        alpha, beta = jnp.asarray([0., 1., 1.0003]), jnp.asarray([1., 0., -.25])
        operation = lambda old, values, first, second: _execute_update(
            old, values, source_factor=first, base_factor=second, plan=plan,
        )

        def reference(old, values, first, second):
            result = old
            for offset, factor in enumerate(factors):
                coefficient = first * factor
                selected = (coefficient[:, None] * values[:, offset::2]
                            + second[:, None] * old[:, offset + 1:offset + 4:2])
                result = result.at[:, offset + 1:offset + 4:2].set(selected.astype(dtype))
            return result

        arguments = (base, source, alpha, beta)
        tangents = tuple(jnp.ones_like(value) for value in arguments)
        actual = jax.jit(operation)(*arguments)
        expected = execute_update_reference(*arguments, plan)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(jax.jit(jax.vmap(operation))(*arguments), expected)
        actual_jvp = jax.jit(lambda *values: jax.jvp(operation, values, tangents)[1])(*arguments)
        expected_jvp = jax.jvp(reference, arguments, tangents)[1]
        np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=1e-3)
        cotangent = jnp.ones_like(base)
        actual_vjp = jax.jit(lambda *values: jax.vjp(operation, *values)[1](cotangent))(*arguments)
        expected_vjp = jax.vjp(reference, *arguments)[1](cotangent)
        for actual_part, expected_part in zip(actual_vjp, expected_vjp, strict=True):
            np.testing.assert_allclose(actual_part, expected_part, rtol=1e-3)
        loss = lambda coefficient: jnp.real(operation(base, source, coefficient, beta)).sum()
        expected_loss = lambda coefficient: jnp.real(reference(base, source, coefficient, beta)).sum()
        np.testing.assert_allclose(jax.jacfwd(jax.grad(loss))(alpha),
                                   jax.jacfwd(jax.grad(expected_loss))(alpha))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
@pytest.mark.parametrize("transpose", [False, True])
def test_composed_factor_reuses_single_scale_leaf(dtype, transpose):
    strides = (1, 9) if transpose else (17, 1)
    plan = build_affine_plan(
        records=(AffineRecord((9, 17), strides, 0, (17, 1), 1,
                              scale=np.dtype(dtype).type(1.5)),),
        source_size=153, output_size=155, source_dtype=dtype, result_dtype=dtype,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    source = jnp.arange(153, dtype=jnp.float32).astype(dtype)
    base = jnp.full((155,), -7, dtype=dtype)
    factor = jnp.asarray(.5, dtype=dtype)
    execute = jax.jit(lambda old, values, coefficient: _execute_update(
        old, values, source_factor=coefficient, base_factor=0, plan=plan,
    )).lower(base, source, factor).compile()
    expected = execute_update_reference(base, source, factor, 0, plan)
    try:
        _observe_native_leaf_kernels_for_tests(True)
        _reset_native_call_count_for_tests()
        actual = execute(base, source, factor)
        actual.block_until_ready()
        assert _native_call_count_for_tests() == 1
        mask, supported = _native_leaf_kernel_masks_for_tests()
        expected_bit = (2 if transpose else 1) if dtype == jnp.float16 else (8 if transpose else 4)
        if supported & expected_bit:
            assert mask & expected_bit
        np.testing.assert_array_equal(actual, expected)
    finally:
        _observe_native_leaf_kernels_for_tests(False)
