from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, add, scale
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord


DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "bfloat16", "float32", "float64", "complex64", "complex128")
UPDATE_DTYPES = tuple((dtype, dtype) for dtype in DTYPES) + (
    ("float16", "float32"), ("float32", "float16"),
    ("float32", "complex64"), ("complex64", "float32"),
    ("float64", "complex128"), ("complex128", "float64"),
    ("int32", "bool"), ("int32", "int8"), ("int32", "int16"), ("uint32", "uint8"),
)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("source_factor,base_factor",
                         [(source, base) for source in (0, 1, 2)
                          for base in (0, 1, -2)])
def test_finite_zero_one_contract_is_identical_for_static_and_batched_factors(
    dtype, source_factor, base_factor,
):
    with jax.enable_x64():
        previous = jnp.asarray([[7, -0.0, -3, -0.0],
                                [-2, -0.0, 5, -0.0]], dtype=dtype)
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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("source_factor,base_factor", [(0, 0), (0, 1), (1, 0)])
def test_zero_terms_skip_nonfinite_input_and_preserve_unselected_bits(dtype, source_factor, base_factor):
    with jax.enable_x64():
        source = jnp.asarray([[1, 2], [3, 4]] if source_factor else
                             [[jnp.nan, jnp.inf], [jnp.inf, jnp.nan]], dtype=dtype)
        selected_base = jnp.asarray([[7, 8], [9, 10]] if base_factor else
                                    [[jnp.inf, jnp.nan], [jnp.nan, jnp.inf]], dtype=dtype)
        base = jnp.asarray([[0, -0.0, 0, jnp.nan], [0, jnp.nan, 0, -0.0]], dtype=dtype)
        base = base.at[:, ::2].set(selected_base)
        left = StridedView(base, (2,), (2,), 0)
        right = StridedView(source, (2,), (1,), 0)
        expected = source if source_factor else selected_base if base_factor else jnp.zeros_like(source)
        dynamic = jax.jit(lambda old, new, alpha, beta: add(old, new, alpha=alpha, beta=beta).data)
        results = (add(left, right, alpha=base_factor, beta=source_factor).data,
                   dynamic(left, right, jnp.full((2,), base_factor, dtype=dtype),
                           jnp.full((2,), source_factor, dtype=dtype)))
        for result in results:
            np.testing.assert_array_equal(result[:, ::2], expected)
            np.testing.assert_array_equal(np.asarray(result)[:, 1::2].copy().view(np.uint8),
                                          np.asarray(base)[:, 1::2].copy().view(np.uint8))


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
def test_explicit_coefficient_composition_keeps_original_derivatives(
    source_dtype, result_dtype, source_factor, base_factor,
):
    records = (AffineRecord((2,), (1,), 0, (2,), 1),)
    base = jnp.arange(5, dtype=result_dtype)
    source = jnp.asarray([3, 5], dtype=source_dtype)
    function = lambda old, new, lhs, rhs: update_p.bind(
        new, old, lhs * 2, rhs, records=records,
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
    jax.jit(function)(*primals).block_until_ready()
    lowered = jax.jit(function).lower(*primals).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_update_f32_cpu_v1" in lowered


def test_effective_factor_is_composed_before_source_multiplication():
    records = (AffineRecord((1,), (1,), 0, (1,), 0),)
    base = jnp.asarray([jnp.nan], dtype=jnp.float16)
    source = jnp.asarray([40000], dtype=jnp.float16)
    result = jax.jit(lambda factor: update_p.bind(
        source, base, factor * 2, jnp.int32(0), records=records,
    ))(jnp.asarray(0.5, dtype=jnp.float16))
    np.testing.assert_array_equal(result, source)
    np.testing.assert_array_equal(update_p.bind(
        source, base, jnp.int32(0), jnp.int32(0), records=records,
    ), jnp.zeros_like(base))


def test_multirecord_mapping_and_vmap_share_one_update_primitive():
    records = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
               AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))
    base = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    source = base + 2
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    function = lambda old, new, factor: update_p.bind(
        new, old, factor, jnp.int32(1), records=records,
    )
    batched = jax.jit(function)(base, source, factors)
    mapped = jax.jit(jax.vmap(function))(base, source, factors)
    np.testing.assert_array_equal(batched, mapped)
    primitives = jax.make_jaxpr(function)(base, source, factors).jaxpr.eqns
    assert [equation.primitive.name for equation in primitives].count(
        "tensor0_stride_update") == 1
    selected = source.reshape(3, 4, 4)[..., [2, 3, 0, 1]].reshape(3, 16)
    np.testing.assert_array_equal(batched, factors[:, None] * selected + base)


def test_broadcast_source_transpose_sums_repeated_reads():
    records = (AffineRecord((3,), (0,), 0, (1,), 1),)
    base = jnp.arange(5, dtype=jnp.float32)
    source = jnp.asarray([3.0])
    function = lambda old, new, factor: update_p.bind(
        new, old, factor * 2, jnp.int32(0), records=records,
    )
    gradients = jax.grad(lambda old, new, factor: function(old, new, factor).sum(),
                        argnums=(0, 1, 2))(base, source, jnp.asarray(1.0))
    np.testing.assert_array_equal(gradients[0], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(gradients[1], [6])
    np.testing.assert_array_equal(gradients[2], 18)


@pytest.mark.parametrize("source_dtype,result_dtype", UPDATE_DTYPES)
@pytest.mark.parametrize("factor", [0, 1, 2])
def test_dynamic_update_preserves_native_dtype_matrix(source_dtype, result_dtype, factor):
    with jax.enable_x64():
        records = (AffineRecord((3,), (1,), 0, (1,), 0),)
        base = jnp.asarray([1, 2, 3], dtype=result_dtype)
        source = jnp.asarray([4, 5, 6], dtype=source_dtype)
        coefficient = jnp.asarray(factor, dtype=result_dtype)
        expected = jnp.real((coefficient * 2) * source + base).astype(result_dtype)
        function = jax.jit(lambda value: update_p.bind(
            source, base, value * 2, jnp.int32(1), records=records,
        ))
        output = function(coefficient)
        output.block_until_ready()
        assert function.lower(coefficient).as_text().count("stablehlo.custom_call") == 1
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
    records = (AffineRecord((2,), (1,), 0, (1,), 0),)
    base = jnp.ones(2, dtype=jnp.float32)
    source = jnp.asarray([3, 5], dtype=jnp.float16)
    function = lambda factor: update_p.bind(
        source, base, factor, jnp.int32(0), records=records,
    )
    np.testing.assert_array_equal(jax.linear_transpose(function, jnp.asarray(0.))(
        jnp.ones_like(base))[0], 8)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float32, jnp.float32),
                                                      (jnp.float16, jnp.float32),
                                                      (jnp.complex64, jnp.complex64)])
def test_complete_contiguous_updates_use_each_batch_coefficient(source_dtype, result_dtype):
    records = (AffineRecord((4,), (1,), 0, (1,), 0),)
    base = jnp.asarray([[jnp.nan] * 4, [10] * 4, [7] * 4], dtype=result_dtype)
    source = jnp.asarray([[1] * 4, [jnp.nan] * 4, [2] * 4], dtype=source_dtype)
    source_factor = jnp.asarray([1, 0, 2], dtype=result_dtype)
    base_factor = jnp.asarray([0, 1, 3], dtype=result_dtype)
    function = lambda old, new, lhs, rhs: update_p.bind(
        new, old, lhs, rhs, records=records,
    )
    expected = jnp.asarray([[1] * 4, [10] * 4, [25] * 4], dtype=result_dtype)
    for operation in (jax.jit(function), jax.jit(jax.vmap(function))):
        np.testing.assert_array_equal(operation(base, source, source_factor, base_factor),
                                      expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_complete_contiguous_scale_tangents_use_each_batch_coefficient(dtype):
    with jax.enable_x64():
        values = jnp.arange(12, dtype=jnp.float32).reshape(3, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        factor_tangents = jnp.asarray([1, 0, -1], dtype=dtype)
        function = lambda data, coefficient: scale(StridedView(data, (4,), (1,), 0), coefficient).data
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


def test_general_coefficients_share_compiled_update():
    values = jnp.arange(4, dtype=jnp.float32)
    traced = []

    @jax.jit
    def operation(data, coefficient):
        traced.append(1)
        return scale(StridedView(data, (4,), (1,), 0), coefficient).data

    for factor in (2, 3, 4, 2):
        np.testing.assert_array_equal(operation(values, jnp.float32(factor)), values * factor)
    assert traced == [1]
    lowered = operation.lower(values, jnp.float32(2)).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_update_f32_cpu_v1" in lowered


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("factor", [0, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_scale_derivative_zero_one_finite_values(
    dtype, factor, explicit,
):
    data = jnp.asarray([2, 3, 4, 5], dtype=dtype)
    direction = jnp.asarray([complex(1.25, -.5), complex(-3.5, .25),
                             complex(-0.0, -0.0), complex(2, 0)]
                            if dtype == jnp.complex64 else
                            [1.25, -3.5, -0.0, 2], dtype=dtype)
    coefficient = jnp.asarray(factor, dtype=dtype)
    def operation(values, value):
        return scale(StridedView(values, (2,), (2,), 0), value).data

    def derivative(values, tangent, value):
        if explicit:
            return jax.jvp(operation, (values, value),
                           (tangent, jnp.zeros_like(value)))[1]
        return jax.jvp(lambda current: operation(current, value),
                       (values,), (tangent,))[1]

    compiled = jax.jit(derivative).lower(data, direction, coefficient).compile()
    actual = compiled(data, direction, coefficient).block_until_ready()
    selected = coefficient * direction[::2]
    if explicit:
        selected = selected + jnp.zeros_like(coefficient) * data[::2]
    expected = direction.at[::2].set(selected)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("factor", [0, 1])
def test_explicit_zero_coefficient_tangent_preserves_finite_derivative(factor):
    data = jnp.asarray([2, 7, -3, 9], dtype=jnp.float32)
    direction = jnp.ones_like(data)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    def operation(values, value):
        return scale(StridedView(values, (2,), (2,), 0), value).data

    single_product = lambda values, value: jax.jvp(
        lambda current: operation(current, value), (values,), (direction,),
    )[1]
    two_products = lambda values, value: jax.jvp(
        operation, (values, value), (direction, jnp.zeros_like(value)),
    )[1]
    single = jax.jit(single_product)(data, coefficient)
    double = jax.jit(two_products)(data, coefficient)
    np.testing.assert_array_equal(single, direction.at[::2].set(factor))
    np.testing.assert_array_equal(double, single)


@pytest.mark.parametrize("factor", [
    0, -0.0, 1, np.float64(1 + 1e-9), 2, np.float32(3), float("nan"), float("inf"),
])
def test_closed_and_dynamic_coefficients_use_update(factor):
    values = jnp.ones(4, dtype=jnp.float32)
    closed = jax.jit(lambda data: scale(StridedView(data, (4,), (1,), 0), factor).data)
    dynamic = jax.jit(lambda data, coefficient: scale(StridedView(data, (4,), (1,), 0), coefficient).data)
    coefficient = jnp.asarray(factor)
    np.testing.assert_allclose(closed(values), dynamic(values, coefficient), rtol=2e-6, atol=1e-6)
    for lowered in (closed.lower(values), dynamic.lower(values, coefficient)):
        assert "tensor0_stride_update_f32_cpu_v1" in lowered.as_text()
        assert lowered.as_text().count("stablehlo.custom_call") == 1


@pytest.mark.parametrize("factor", [0, 1])
def test_closed_python_scale_factor_keeps_finite_complex_derivative(factor):
    data = jnp.ones(4, dtype=jnp.complex64)
    tangent = jnp.full(4, complex(2, -3), dtype=data.dtype)
    operation = lambda values: scale(StridedView(values, (4,), (1,), 0), factor).data
    actual = jax.jit(lambda values, direction:
                     jax.jvp(operation, (values,), (direction,))[1])(data, tangent)
    expected = jnp.asarray(factor, dtype=data.dtype) * tangent
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_single_scale_tangent_batching_and_nested_ad(dtype):
    with jax.enable_x64():
        data = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        operation = lambda values, factor: scale(StridedView(values, (2,), (2,), 0), factor).data
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
def test_selected_scale_data_transpose_preserves_unselected_cotangent(factor):
    data = jnp.ones(4, dtype=jnp.complex64)
    cotangent = jnp.asarray([complex(2, -3)] * 4, dtype=data.dtype)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    transpose = jax.linear_transpose(
        lambda values: scale(StridedView(values, (2,), (2,), 0), coefficient).data, data,
    )
    actual = jax.jit(transpose)(cotangent)[0]
    expected = cotangent.at[::2].set(coefficient * cotangent[::2])
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)
