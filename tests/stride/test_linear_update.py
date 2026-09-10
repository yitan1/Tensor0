from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, add, scale
from tensor0._stride._ops._update import _execute_update
from tensor0._stride._native import _BASE_UPDATE_SUFFIXES
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
)

from ._fixtures import two_record_noncompact_plan
from ._oracle import execute_base_assign_reference


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("source_factor,base_factor",
                         [(source, base) for source in (0, 1, 2)
                          for base in (0, 1, -2)])
def test_zero_one_contract_is_identical_for_static_and_batched_factors(
    dtype, source_factor, base_factor,
):
    with jax.enable_x64():
        previous = jnp.asarray([[jnp.nan, -0.0, jnp.inf, -0.0],
                                [jnp.inf, -0.0, jnp.nan, -0.0]], dtype=dtype)
        source = jnp.asarray([[3, -0.0], [-0.0, 4]], dtype=dtype)
        left = StridedView(previous, (2,), (2,), 0)
        right = StridedView(source, (2,), (1,), 0)
        factors = (jnp.full((2,), base_factor, dtype=dtype),
                   jnp.full((2,), source_factor, dtype=dtype))
        function = jax.jit(lambda old, new, lhs, rhs:
                           add(old, new, alpha=lhs, beta=rhs).data)
        outputs = (add(left, right, alpha=base_factor, beta=source_factor).data,
                   function(left, right, *factors))
        terms = []
        for factor, values in ((source_factor, source),
                               (base_factor, previous[:, ::2])):
            if factor != 0:
                terms.append(values if factor == 1 else factor * values)
        selected = (jnp.zeros_like(source) if not terms else terms[0]
                    if len(terms) == 1 else terms[0] + terms[1])
        expected = previous.at[:, ::2].set(selected)
        for output in outputs:
            np.testing.assert_allclose(output, expected, equal_nan=True)
            np.testing.assert_array_equal(
                np.asarray(output)[:, 1::2].copy().view(np.uint8),
                np.asarray(previous)[:, 1::2].copy().view(np.uint8),
            )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("factor", [0, 1])
def test_scale_short_circuit_preserves_unit_bits_and_zeroes_nonfinite(dtype, factor):
    values = jnp.asarray([complex(jnp.inf, -0.0), complex(jnp.nan, 2),
                          complex(-0.0, -0.0)], dtype=jnp.complex64)
    if dtype == jnp.float32:
        values = values.real
    view = StridedView.from_dense(values, (3,))
    expected = values if factor else jnp.zeros_like(values)
    for output in (scale(view, factor).data,
                   jax.jit(lambda value: scale(view, value).data)(
                       jnp.asarray(factor, dtype=dtype))):
        np.testing.assert_array_equal(np.asarray(output).view(np.uint8),
                                      np.asarray(expected).view(np.uint8))


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float32, jnp.float32),
                                                      (jnp.float16, jnp.float32)])
@pytest.mark.parametrize("source_factor,base_factor", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nonunit_record_scale_keeps_original_coefficient_derivatives(
    source_dtype, result_dtype, source_factor, base_factor,
):
    plan = build_affine_plan(
        records=(AffineRecord((2,), (1,), 0, (2,), 1, scale=2),),
        source_size=2, output_size=5, source_dtype=source_dtype,
        result_dtype=result_dtype, coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    base = jnp.arange(5, dtype=result_dtype)
    source = jnp.asarray([3, 5], dtype=source_dtype)
    function = lambda old, new, lhs, rhs: _execute_update(
        old, new, source_factor=lhs, base_factor=rhs, plan=plan,
    )
    oracle = lambda old, new, lhs, rhs: old.at[1::2].set(
        (lhs * 2) * new + rhs * old[1::2],
    )
    primals = (base, source, jnp.asarray(source_factor, dtype=result_dtype),
               jnp.asarray(base_factor, dtype=result_dtype))
    tangents = tuple(jnp.ones_like(value) for value in primals)
    actual = jax.jit(lambda *values: jax.jvp(function, values, tangents))(*primals)
    expected = jax.jvp(oracle, primals, tangents)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference)
    actual_vjp = jax.jit(jax.grad(lambda *values: function(*values).sum(),
                                  argnums=(0, 1, 2, 3)))(*primals)
    expected_vjp = jax.grad(lambda *values: oracle(*values).sum(),
                            argnums=(0, 1, 2, 3))(*primals)
    for observed, reference in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(observed, reference)
    second = jax.jacfwd(jax.grad(lambda new, factor:
                                function(base, new, factor, primals[3]).sum(), 1), 0)
    np.testing.assert_allclose(second(source, primals[2]), jnp.full((2,), 2))
    _reset_native_call_count_for_tests()
    jax.jit(function)(*primals).block_until_ready()
    assert _native_call_count_for_tests() == 1


def test_effective_factor_is_composed_before_source_multiplication():
    plan = build_affine_plan(
        records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=2),),
        source_size=1, output_size=1, source_dtype=jnp.float16,
        result_dtype=jnp.float16, coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    base = jnp.asarray([jnp.nan], dtype=jnp.float16)
    source = jnp.asarray([40000], dtype=jnp.float16)
    result = jax.jit(lambda factor: _execute_update(
        base, source, source_factor=factor, base_factor=0, plan=plan,
    ))(jnp.asarray(0.5, dtype=jnp.float16))
    np.testing.assert_array_equal(result, source)
    np.testing.assert_array_equal(_execute_update(
        base, source, source_factor=0, base_factor=0, plan=plan,
    ), jnp.zeros_like(base))


def test_multirecord_mapping_and_vmap_share_one_update_primitive():
    plan = two_record_noncompact_plan()
    base = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    source = base + 2
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    function = lambda old, new, factor: _execute_update(
        old, new, source_factor=factor, base_factor=1, plan=plan,
    )
    batched = jax.jit(function)(base, source, factors)
    mapped = jax.jit(jax.vmap(function))(base, source, factors)
    np.testing.assert_array_equal(batched, mapped)
    primitives = jax.make_jaxpr(function)(base, source, factors).jaxpr.eqns
    assert [equation.primitive.name for equation in primitives].count(
        "tensor0_stride_update") == 1


def test_broadcast_source_transpose_sums_repeated_reads():
    plan = build_affine_plan(
        records=(AffineRecord((3,), (0,), 0, (1,), 1, scale=2,
                               source_broadcast_axes=(0,)),),
        source_size=1, output_size=5, source_dtype=jnp.float32,
        result_dtype=jnp.float32, coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    base = jnp.arange(5, dtype=jnp.float32)
    source = jnp.asarray([3.0])
    function = lambda old, new, factor: _execute_update(
        old, new, source_factor=factor, base_factor=0, plan=plan,
    )
    gradients = jax.grad(lambda old, new, factor: function(old, new, factor).sum(),
                        argnums=(0, 1, 2))(base, source, jnp.asarray(1.0))
    np.testing.assert_array_equal(gradients[0], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(gradients[1], [6])
    np.testing.assert_array_equal(gradients[2], 18)


@pytest.mark.parametrize("source_dtype,result_dtype", tuple(_BASE_UPDATE_SUFFIXES))
@pytest.mark.parametrize("factor", [0, 1, 2])
def test_dynamic_update_preserves_native_dtype_matrix(source_dtype, result_dtype, factor):
    with jax.enable_x64():
        plan = build_affine_plan(
            records=(AffineRecord((3,), (1,), 0, (1,), 0, scale=2),),
            source_size=3, output_size=3, source_dtype=source_dtype,
            result_dtype=result_dtype, coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        base = jnp.asarray([1, 2, 3], dtype=result_dtype)
        source = jnp.asarray([4, 5, 6], dtype=source_dtype)
        coefficient = jnp.asarray(factor, dtype=result_dtype)
        mapped = execute_base_assign_reference(base, source, plan)
        expected = base if coefficient == 0 else (
            mapped if coefficient == 1 else coefficient * mapped) + base
        function = jax.jit(lambda value: _execute_update(
            base, source, source_factor=value, base_factor=1, plan=plan,
        ))
        _reset_native_call_count_for_tests()
        output = function(coefficient)
        output.block_until_ready()
        assert _native_call_count_for_tests() == 1
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("factor", [0, 1])
def test_scale_zero_one_coefficient_vjp_across_inexact_dtypes(dtype, factor):
    with jax.enable_x64():
        values = jnp.asarray([1, 2, 3], dtype=dtype)
        coefficient = jnp.asarray(factor, dtype=dtype)
        function = lambda value: scale(StridedView(values, (2,), (1,), 0), value).data
        transpose = jax.jit(lambda value: jax.vjp(function, value)[1](
            jnp.ones_like(values))[0])
        np.testing.assert_array_equal(transpose(coefficient), jnp.asarray(3, dtype=dtype))


def test_mixed_dtype_direct_coefficient_transpose():
    plan = build_affine_plan(
        records=(AffineRecord((2,), (1,), 0, (1,), 0),),
        source_size=2, output_size=2, source_dtype=jnp.float16,
        result_dtype=jnp.float32, coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    base = jnp.ones(2, dtype=jnp.float32)
    source = jnp.asarray([3, 5], dtype=jnp.float16)
    function = lambda factor: _execute_update(
        base, source, source_factor=factor, base_factor=0, plan=plan,
    )
    np.testing.assert_array_equal(jax.linear_transpose(function, jnp.asarray(0.))(
        jnp.ones_like(base))[0], 8)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float32, jnp.float32),
                                                      (jnp.float16, jnp.float32),
                                                      (jnp.complex64, jnp.complex64)])
def test_complete_contiguous_updates_use_each_batch_coefficient(source_dtype, result_dtype):
    plan = build_affine_plan(
        records=(AffineRecord((4,), (1,), 0, (1,), 0),),
        source_size=4, output_size=4, source_dtype=source_dtype,
        result_dtype=result_dtype, coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    base = jnp.asarray([[jnp.nan] * 4, [10] * 4, [7] * 4], dtype=result_dtype)
    source = jnp.asarray([[1] * 4, [jnp.nan] * 4, [2] * 4], dtype=source_dtype)
    source_factor = jnp.asarray([1, 0, 2], dtype=result_dtype)
    base_factor = jnp.asarray([0, 1, 3], dtype=result_dtype)
    function = lambda old, new, lhs, rhs: _execute_update(
        old, new, source_factor=lhs, base_factor=rhs, plan=plan,
    )
    expected = jnp.asarray([[1] * 4, [10] * 4, [25] * 4], dtype=result_dtype)
    for operation in (jax.jit(function), jax.jit(jax.vmap(function))):
        np.testing.assert_array_equal(operation(base, source, source_factor, base_factor),
                                      expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_complete_contiguous_scale_tangents_use_each_batch_coefficient(dtype):
    with jax.enable_x64():
        plan = build_affine_plan(
            records=(AffineRecord((4,), (1,), 0, (1,), 0),),
            source_size=4, output_size=4, source_dtype=dtype,
            result_dtype=dtype, coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        values = jnp.arange(12, dtype=jnp.float32).reshape(3, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        factor_tangents = jnp.asarray([1, 0, -1], dtype=dtype)
        function = lambda data, coefficient: _execute_update(
            data, data, source_factor=coefficient, base_factor=0, plan=plan,
        )
        primal, tangent = jax.jit(lambda data, coefficient: jax.jvp(
            function, (data, coefficient), (jnp.ones_like(data), factor_tangents),
        ))(values, factors)
        np.testing.assert_array_equal(primal, values * factors[:, None])
        np.testing.assert_array_equal(tangent, factors[:, None]
                                      + values * factor_tangents[:, None])
        coefficient_vjp = jax.jit(lambda data, coefficient: jax.vjp(
            function, data, coefficient)[1](jnp.ones_like(data))[1])(values, factors)
        np.testing.assert_array_equal(coefficient_vjp, values.sum(-1))


def test_vmap_preserves_existing_batch_axes_for_scalar_coefficients():
    values = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4)
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    function = lambda data, factor: scale(StridedView(data, (4,), (1,), 0), factor).data
    mapped = jax.jit(jax.vmap(function))
    nested = jax.jit(jax.vmap(jax.vmap(function, in_axes=(0, None))))
    expected = values * factors[:, None, None]
    np.testing.assert_array_equal(mapped(values, factors), expected)
    np.testing.assert_array_equal(nested(values, factors), expected)
    shared = jax.jit(jax.vmap(lambda data: function(data, 2)))
    np.testing.assert_array_equal(shared(values), values * 2)
    for operation in (mapped, nested):
        tangent = jax.jvp(lambda coefficient: operation(values, coefficient),
                          (factors,), (jnp.ones_like(factors),))[1]
        np.testing.assert_array_equal(tangent, values)
        gradient = jax.grad(lambda coefficient: operation(values, coefficient).sum())(factors)
        np.testing.assert_array_equal(gradient, values.sum((1, 2)))


def test_general_python_coefficients_share_update_lowering(monkeypatch):
    from tensor0._stride._ops import _update

    jax.clear_caches()
    lowerings = []
    original = _update._native_update

    def counted(*arguments, **metadata):
        lowerings.append(metadata["fixed_factors"])
        return original(*arguments, **metadata)

    monkeypatch.setattr(_update, "_native_update", counted)
    values = jnp.arange(4, dtype=jnp.float32)
    view = StridedView(values, (4,), (1,), 0)
    for factor in (2, 3, 4, 2):
        np.testing.assert_array_equal(scale(view, factor).data, values * factor)
    assert lowerings == [(None, 0)]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("factor", [0, 1])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_scale_derivative_uses_ordinary_zero_one_multiplication(
    dtype, factor, alias, explicit,
):
    from tensor0._stride._ops._selected_scale import (
        _build_strided_scale_plan, _selected_scale_alias,
    )

    data = jnp.asarray([2, 3, 4, 5], dtype=dtype)
    direction = jnp.asarray([complex(1.25, -.5), complex(-3.5, .25),
                             complex(-0.0, -0.0), complex(2, 0)]
                            if dtype == jnp.complex64 else
                            [1.25, -3.5, -0.0, 2], dtype=dtype)
    coefficient = jnp.asarray(factor, dtype=dtype)
    plan = _build_strided_scale_plan((2,), (2,), 0, 4, jnp.dtype(dtype).name)

    def operation(values, value):
        if alias:
            return _selected_scale_alias(values, value, plan=plan)
        return scale(StridedView(values, (2,), (2,), 0), value).data

    def derivative(values, tangent, value):
        if explicit:
            return jax.jvp(operation, (values, value),
                           (tangent, jnp.zeros_like(value)))[1]
        return jax.jvp(lambda current: operation(current, value),
                       (values,), (tangent,))[1]

    compiled = jax.jit(derivative).lower(data, direction, coefficient).compile()
    _reset_native_call_count_for_tests()
    actual = compiled(data, direction, coefficient).block_until_ready()
    assert _native_call_count_for_tests() == 1
    selected = coefficient * direction[::2]
    if explicit:
        selected = selected + jnp.zeros_like(coefficient) * data[::2]
    expected = direction.at[::2].set(selected)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("factor", [0, 1])
def test_explicit_zero_coefficient_tangent_is_not_structurally_absent(alias, factor):
    from tensor0._stride._ops._selected_scale import (
        _build_strided_scale_plan, _selected_scale_alias,
    )

    data = jnp.asarray([2, 7, -3, 9], dtype=jnp.float32)
    direction = jnp.ones_like(data)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    plan = _build_strided_scale_plan((2,), (2,), 0, 4, "float32")

    def operation(values, value):
        if alias:
            return _selected_scale_alias(values, value, plan=plan)
        return scale(StridedView(values, (2,), (2,), 0), value).data

    single_product = lambda values, value: jax.jvp(
        lambda current: operation(current, value), (values,), (direction,),
    )[1]
    two_products = lambda values, value: jax.jvp(
        operation, (values, value), (direction, jnp.zeros_like(value)),
    )[1]
    single_jaxpr = jax.make_jaxpr(single_product)(data, coefficient).jaxpr
    double_jaxpr = jax.make_jaxpr(two_products)(data, coefficient).jaxpr
    assert sum(equation.primitive.name == "tensor0_stride_scale_tangent"
               for equation in single_jaxpr.eqns) == 1
    assert sum(equation.primitive.name == "tensor0_stride_update_tangent"
               for equation in double_jaxpr.eqns) == 1
    single = jax.jit(single_product)(data, coefficient)
    double = jax.jit(two_products)(data, coefficient)
    np.testing.assert_array_equal(single, direction.at[::2].set(factor))
    np.testing.assert_array_equal(double, single)


@pytest.mark.parametrize("factor,expected", [
    (0, 0), (-0.0, 0), (1, 1), (np.float64(1 + 1e-9), 1),
    (2, None), (np.float32(3), None), (float("nan"), None),
    (float("inf"), None),
])
def test_update_specialization_metadata_only_records_normalized_zero_one(factor, expected):
    values = jnp.ones(4, dtype=jnp.float32)
    traced = jax.make_jaxpr(
        lambda data: scale(StridedView(data, (4,), (1,), 0), factor).data,
    )(values)
    update = next(equation for equation in traced.jaxpr.eqns
                  if equation.primitive.name == "tensor0_stride_update")
    assert update.params["fixed_factors"] == (expected, 0)
    dynamic = jax.make_jaxpr(
        lambda data, coefficient: scale(StridedView(data, (4,), (1,), 0), coefficient).data,
    )(values, jnp.asarray(factor, dtype=values.dtype))
    update = next(equation for equation in dynamic.jaxpr.eqns
                  if equation.primitive.name == "tensor0_stride_update")
    assert update.params["fixed_factors"] == (None, 0)


@pytest.mark.parametrize("factor", [0, 1])
def test_closed_python_scale_factor_keeps_complex_derivative_product(factor):
    data = jnp.ones(4, dtype=jnp.complex64)
    tangent = jnp.full(4, complex(np.inf, 0), dtype=data.dtype)
    operation = lambda values: scale(StridedView(values, (4,), (1,), 0), factor).data
    actual = jax.jit(lambda values, direction:
                     jax.jvp(operation, (values,), (direction,))[1])(data, tangent)
    expected = jnp.asarray(factor, dtype=data.dtype) * tangent
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_single_scale_tangent_batching_and_nested_ad(dtype):
    from tensor0._stride._ops._selected_scale import _build_strided_scale_plan
    from tensor0._stride._ops._update_ad import _execute_scale_tangent

    with jax.enable_x64():
        data = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        plan = _build_strided_scale_plan((2,), (2,), 0, 4, jnp.dtype(dtype).name)
        operation = lambda values, factor: _execute_scale_tangent(values, factor, plan=plan)
        mapped = jax.jit(jax.vmap(operation))
        actual = mapped(data, factors)
        expected = data.at[..., ::2].set(data[..., ::2] * factors[:, None, None])
        np.testing.assert_array_equal(actual, expected)
        direction = jnp.ones_like(data)
        first = lambda values, coefficient: jax.jvp(
            mapped, (values, coefficient), (direction, jnp.ones_like(coefficient)),
        )[1]
        second = jax.jit(lambda values, coefficient: jax.jvp(
            first, (values, coefficient), (direction, jnp.ones_like(coefficient)),
        )[1])(data, factors)
        np.testing.assert_array_equal(second, jnp.zeros_like(data).at[..., ::2].set(2))


@pytest.mark.parametrize("factor", [0, 1])
def test_alias_data_transpose_uses_single_ordinary_product(factor):
    from tensor0._stride._ops._selected_scale import (
        _build_strided_scale_plan, _selected_scale_alias,
    )

    data = jnp.ones(4, dtype=jnp.complex64)
    cotangent = jnp.asarray([complex(np.inf, 0)] * 4, dtype=data.dtype)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    plan = _build_strided_scale_plan((2,), (2,), 0, 4, "complex64")
    transpose = jax.linear_transpose(
        lambda values: _selected_scale_alias(values, coefficient, plan=plan), data,
    )
    actual = jax.jit(transpose)(cotangent)[0]
    expected = cotangent.at[::2].set(coefficient * cotangent[::2])
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)
