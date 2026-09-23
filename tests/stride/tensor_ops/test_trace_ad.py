"""Packed trace differentiation and batching contracts."""

from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._tensor_ops import _strided_tensortrace

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.reduction import CROSS, CROSS_SHAPE_TYPES, compare_derivatives
from tests.stride.support.oracles.trace import packed_trace, reference_trace, scalar_trace
from tests.stride.support.samples import values


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype", [("float16", "float32"), ("float32", "float16"),
    ("float64", "float32"), ("complex64", "complex128"), ("complex128", "complex64")])
@pytest.mark.parametrize("batch_shape", [(), (2,), (0,)])
def test_mixed_reduction_multirecord_trace_mixed_source_ad(source_dtype, result_dtype, batch_shape):
    with jax.enable_x64():
        inputs = (SimpleNamespace(sizes=(2, 2, 2), strides=(4, 2, 1), offset=0),
                  SimpleNamespace(sizes=(2, 2, 2), strides=(0, -2, -1), offset=3),
                  SimpleNamespace(sizes=(2, 0, 0), strides=(1, 1, 1), offset=12))
        outputs = (SimpleNamespace(sizes=(2,), strides=(1,), offset=1),
                   SimpleNamespace(sizes=(2,), strides=(-1,), offset=3))
        source = (jnp.arange(prod(batch_shape) * 12, dtype=jnp.float32) % 7 / 4).astype(source_dtype).reshape((*batch_shape, 12))
        if source_dtype.startswith("complex"):
            source = source * (1 + .5j)
        factor = jnp.asarray(1.5 + .5j if source_dtype.startswith("complex") else 1.5, dtype=source_dtype)
        second = jnp.float32(-2)
        run = lambda values: _strided_tensortrace(
            values, destination_size=5, source_subblocks=inputs, destination_subblocks=outputs,
            entries=((2, 0, None), (0, 0, None), (1, 1, factor), (0, 1, second)),
            result_dtype=result_dtype, permutation=(), num_open_out=1, num_open_in=0, trace_count=1)
        def oracle(values):
            first = values[..., jnp.asarray([[0, 3], [4, 7]])]
            repeated = values[..., jnp.asarray([[3, 0], [3, 0]])]
            result = jnp.zeros((*batch_shape, 5), dtype=result_dtype)
            result = result.at[..., 1:3].add(jnp.sum(first, axis=-1, dtype=result_dtype))
            result = result.at[..., jnp.asarray([3, 2])].add(jnp.sum(repeated * factor, axis=-1, dtype=result_dtype))
            return result.at[..., jnp.asarray([3, 2])].add(jnp.sum(first * second, axis=-1, dtype=result_dtype))
        tolerance = 8 * max(float(jnp.finfo(source_dtype).eps), float(jnp.finfo(result_dtype).eps))
        tangent = jnp.ones_like(source)
        for actual, expected in zip(jax.jit(lambda values: jax.jvp(run, (values,), (tangent,)))(source),
                                    jax.jvp(oracle, (source,), (tangent,)), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        cotangent = jnp.full((*batch_shape, 5), 1 + .5j if result_dtype.startswith("complex") else 1, dtype=result_dtype)
        actual = jax.jit(jax.vjp(run, source)[1])(cotangent)[0]
        assert actual.dtype == source.dtype
        np.testing.assert_allclose(actual, jax.vjp(oracle, source)[1](cotangent)[0], rtol=tolerance, atol=tolerance)
        text = jax.jit(jax.linear_transpose(run, source)).lower(cotangent).as_text()
        assert "stride_accumulation_" in text
        assert "stablehlo.gather" not in text and "stablehlo.scatter" not in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_packed_trace_vmap_and_ad_boundary():
    execute = lambda source, factor: scalar_trace(source, (factor,), jnp.float32)
    np.testing.assert_array_equal(jax.jit(jax.vmap(execute))(
        jnp.asarray([[2.], [3.]]), jnp.asarray([3., 4.])), [[6], [12]])
    np.testing.assert_array_equal(
        jax.grad(lambda source: execute(source, jnp.float32(2)).sum())(jnp.ones(1)), [2])
    np.testing.assert_array_equal(jax.grad(lambda factor: execute(jnp.ones(1), factor).sum())(jnp.float32(2)), 1)


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("fixed_dtype,factor,dtype", [
    (fixed, factor, dtype) for fixed in ("int32", "float64")
    for factor in (0, 1, 2) for dtype in ("float32", "float64", "complex64", "complex128")
    if dtype in ("float32", "complex64") or
    (factor == 2 and (dtype, fixed) in (("float64", "int32"), ("complex128", "float64")))
])
def test_trace_higher_derivatives_with_fixed_mixed_coefficient(dtype, factor, fixed_dtype):
    with jax.enable_x64():
        source = (jnp.arange(12) % 5).astype(dtype)
        if dtype in ("complex64", "complex128"):
            source = source + 1j * (source - 2)
        coefficient = jnp.asarray(factor, dtype=dtype)
        fixed = jnp.asarray(2, dtype=fixed_dtype)
        def objective(function, data, value):
            result = function(data, value, fixed)
            return jnp.real(jnp.sum(result * jnp.conj(result)))
        run = lambda data, value: objective(packed_trace, data, value)
        oracle = lambda data, value: objective(reference_trace, data, value)
        gradient, expected_gradient = jax.grad(run, argnums=(0, 1)), jax.grad(oracle, argnums=(0, 1))
        arguments = source, coefficient
        directions = jnp.ones_like(source), jnp.ones_like(coefficient)
        for actual, expected in zip(jax.jit(gradient)(*arguments), expected_gradient(*arguments), strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(jax.jit(lambda *values: jax.jvp(gradient, values, directions)[1])(*arguments),
                                    jax.jvp(expected_gradient, arguments, directions)[1], strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(jax.jit(jax.vjp(gradient, *arguments)[1])(directions),
                                    jax.vjp(expected_gradient, *arguments)[1](directions), strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_trace_shared_coefficient_and_nonzero_at_zero():
    source = jnp.asarray([1, 2, 3], jnp.float32)
    run = lambda value: scalar_trace(source, (value, None, value), source.dtype)
    for factor in (0, 1, 2):
        np.testing.assert_array_equal(jax.grad(lambda value: run(value).sum())(jnp.float32(factor)), 4)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("dtype,coefficient_dtype", [("float16", "float16"), ("bfloat16", "bfloat16")])
def test_low_precision_coefficient_dtypes(dtype, coefficient_dtype):
    with jax.enable_x64():
        source = jnp.ones(1, dtype=dtype)
        factor = jnp.asarray(2, dtype=coefficient_dtype)
        run = lambda value: scalar_trace(source, (value,), source.dtype)
        primal, tangent = jax.jvp(run, (factor,), (jnp.ones_like(factor),))
        np.testing.assert_array_equal(primal, [2])
        np.testing.assert_array_equal(tangent, [1])
        np.testing.assert_array_equal(jax.linear_transpose(run, factor)(source)[0], 1)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype", CROSS)
def test_trace_cross_kind_contract(source_dtype, coefficient_dtype, dtype):
    source = values((2,), source_dtype)
    factor = jnp.asarray(2 + 1j if coefficient_dtype.startswith("complex") else 2, dtype=coefficient_dtype)
    def run(data, alpha):
        return scalar_trace(data, (alpha, None), jnp.dtype(dtype))
    def reference(data, alpha):
        first = data[0] * alpha
        second = data[1]
        if dtype.startswith("float"):
            first, second = jnp.real(first), jnp.real(second)
        return (first.astype(dtype) + second.astype(dtype)).reshape(1)
    if (source_dtype, coefficient_dtype, dtype) in CROSS_SHAPE_TYPES:
        compare_derivatives(run, reference, (source, factor))
    else:
        actual, expected = jax.jit(run)(source, factor), reference(source, factor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


TRACE_COEFFICIENT_DTYPES = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
]


TRACE_DTYPE_PAIRS = [(dtype, dtype) for dtype in ("float16", "bfloat16", "float64", "complex128")]

TRACE_DTYPE_PAIRS += [(dtype, coefficient) for dtype in ("float32", "complex64") for coefficient in TRACE_COEFFICIENT_DTYPES]


# Dtype pairs and batch handling are separate coverage dimensions.
TRACE_BATCH_PAIRS = [(dtype, dtype) for dtype in
                     ("float16", "bfloat16", "float64", "complex128")] + [
    ("float32", "int32"), ("float32", "complex128"),
    ("complex64", "uint64"), ("complex64", "float64"),
]


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,coefficient_dtype,batch_shape", [
    pytest.param(source, coefficient, batch, id=f"{source}-{coefficient}-batch_shape{i}")
    for i, (source, coefficient, batch) in enumerate(
        [(*pair, ()) for pair in TRACE_DTYPE_PAIRS]
        + [(*pair, batch) for pair in TRACE_BATCH_PAIRS for batch in ((2, 3), (0,))])
    if (not batch and (source != "complex64" or coefficient in
        ("bool", "uint64", "float64", "complex64", "complex128")))
    or (source, coefficient, batch) in (
        ("float16", "float16", (2, 3)), ("bfloat16", "bfloat16", (2, 3)),
        ("float32", "int32", (0,)), ("float32", "complex128", (2, 3)),
        ("complex64", "uint64", (2, 3)), ("complex64", "float64", (0,)))
])
def test_packed_trace_typed_coefficients_source_ad(source_dtype, coefficient_dtype, batch_shape):
    with jax.enable_x64():
        source = (jnp.arange(prod(batch_shape) * 12).reshape((*batch_shape, 12)) % 5).astype(source_dtype)
        complex_source = jnp.issubdtype(source.dtype, jnp.complexfloating)
        if complex_source:
            source = source * (1 + 1j)
        factor = jnp.asarray(2 + 1j if coefficient_dtype.startswith("complex") else 2, dtype=coefficient_dtype)
        second = jnp.asarray(-1 + 1j if complex_source else -1, dtype=source_dtype)
        tangent = jnp.ones_like(source)
        cotangent = jnp.broadcast_to(jnp.arange(5), (*batch_shape, 5)).astype(source_dtype)
        if complex_source:
            cotangent = cotangent * (1 - 2j)

        def execute(data):
            return packed_trace(data, factor, second)

        def reference(data):
            return reference_trace(data, factor, second)

        result, derivative = jax.jit(lambda data, scale, delta: jax.jvp(
            lambda value: packed_trace(value, scale, second), (data,), (delta,)))(source, factor, tangent)
        expected, expected_derivative = jax.jvp(reference, (source,), (tangent,))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(derivative, expected_derivative)
        reverse = jax.jit(lambda data, scale, value: jax.vjp(
            lambda inputs: packed_trace(inputs, scale, second), data)[1](value)[0])
        np.testing.assert_array_equal(reverse(source, factor, cotangent), jax.vjp(reference, source)[1](cotangent)[0])

        def transpose(value):
            return jax.linear_transpose(execute, source)(value)[0]

        twice = jax.jit(lambda value: jax.linear_transpose(transpose, cotangent)(value)[0])(tangent)
        np.testing.assert_array_equal(twice, execute(tangent))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("source_axis,coefficient_axis", [(1, None), (None, 0), (1, 0)])
def test_packed_trace_vmap_shared_and_mapped_coefficients(batch_size, source_axis, coefficient_axis):
    source = jnp.arange(12, dtype=jnp.float32) if source_axis is None else jnp.arange(
        12 * batch_size, dtype=jnp.float32).reshape(12, batch_size)
    factor = jnp.float32(2) if coefficient_axis is None else jnp.arange(batch_size, dtype=jnp.float32)
    second = jnp.float32(-1)
    execute = jax.vmap(lambda data, scale: packed_trace(data, scale, second),
                       in_axes=(source_axis, coefficient_axis))
    reference = jax.vmap(lambda data, scale: reference_trace(data, scale, second),
                         in_axes=(source_axis, coefficient_axis))
    np.testing.assert_array_equal(jax.jit(execute)(source, factor), reference(source, factor))
    reverse = jax.jit(jax.grad(lambda data, scale: execute(data, scale).sum()))
    np.testing.assert_array_equal(reverse(source, factor), jax.grad(
        lambda data: reference(data, factor).sum())(source))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_packed_trace_nested_vmap_and_second_source_derivative():
    source = jnp.arange(72, dtype=jnp.float32).reshape(2, 3, 12)
    factors = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    execute = jax.vmap(jax.vmap(lambda data, scale: packed_trace(data, scale, jnp.float32(-1))))
    reference = jax.vmap(jax.vmap(lambda data, scale: reference_trace(data, scale, jnp.float32(-1))))
    np.testing.assert_array_equal(jax.jit(execute)(source, factors), reference(source, factors))
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda data: execute(data, factors).sum()))(source),
                                  jax.grad(lambda data: reference(data, factors).sum())(source))
    row = source[0, 0]
    np.testing.assert_array_equal(jax.jit(jax.hessian(
        lambda data: (packed_trace(data, jnp.float32(2), jnp.float32(-1)) ** 2).sum()))(row),
        jax.hessian(lambda data: (reference_trace(data, jnp.float32(2), jnp.float32(-1)) ** 2).sum())(row))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_packed_trace_scalar_coefficient_derivatives_and_bilinear_transpose_boundary():
    source, factor = jnp.ones(1), jnp.float32(2)
    execute = lambda data, scale: scalar_trace(data, (scale,), data.dtype)
    np.testing.assert_array_equal(jax.jvp(lambda scale: execute(source, scale),
        (factor,), (jnp.float32(1),))[1], [1])
    source_gradient, factor_gradient = jax.vjp(execute, source, factor)[1](jnp.ones(1))
    np.testing.assert_array_equal(source_gradient, [2])
    np.testing.assert_array_equal(factor_gradient, 1)
    np.testing.assert_array_equal(jax.linear_transpose(lambda scale: execute(source, scale), factor)(jnp.ones(1))[0], 1)
    with pytest.raises(NotImplementedError, match="known source or known coefficients"):
        jax.linear_transpose(execute, source, factor)(jnp.ones(1))
    _, tangent = jax.jvp(lambda data: execute(jax.lax.stop_gradient(data), factor), (source,), (source,))
    np.testing.assert_array_equal(tangent, [0])
    np.testing.assert_array_equal(jax.vjp(lambda data: execute(data, factor), source)[1](jnp.zeros(1))[0], [0])


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_packed_trace_mixed_storage_source_ad():
    source = jnp.ones(1, dtype=jnp.float32)
    execute = lambda data: scalar_trace(data, (jnp.float32(2),), jnp.float16)
    assert execute(source).dtype == jnp.float16
    gradient = jax.vjp(execute, source)[1](jnp.ones(1, dtype=jnp.float16))[0]
    assert gradient.dtype == source.dtype
    np.testing.assert_allclose(gradient, [2], rtol=1e-6, atol=1e-6)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_packed_trace_zero_and_real_complex_one_shortcuts():
    source = jnp.ones(1, dtype=jnp.float32)
    zero = lambda data: scalar_trace(data, (jnp.float32(0),), data.dtype)
    infinity = jnp.asarray([np.inf], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jvp(zero, (source,), (infinity,))[1], [0])
    np.testing.assert_array_equal(jax.linear_transpose(zero, source)(infinity)[0], [0])
    complex_source = source.astype(jnp.complex64)
    cotangent = jnp.asarray([complex(np.inf, 3)], dtype=jnp.complex64)
    for factor in (jnp.float32(1), jnp.complex64(1)):
        execute = lambda data: scalar_trace(data, (factor,), data.dtype)
        actual = jax.linear_transpose(execute, complex_source)(cotangent)[0]
        assert np.isinf(actual.real[0])
        assert actual.imag[0] == 3


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_packed_trace_strong_coefficient_precision_and_reverse_record_carry():
    with jax.enable_x64():
        factor = jnp.float64(1 + 2**-24)
        source = jnp.asarray([-1, 1], dtype=jnp.float32)
        execute = lambda data: scalar_trace(data, (None, factor), data.dtype)
        value, tangent = jax.jvp(execute, (source,), (source,))
        np.testing.assert_array_equal(value, [2**-24])
        np.testing.assert_array_equal(tangent, [2**-24])

        def split(data):
            return _strided_tensortrace(
                data, destination_size=2,
                source_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
                destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),
                                      SimpleNamespace(sizes=(), strides=(), offset=1)),
                entries=((0, 0, None), (0, 1, factor)), result_dtype=data.dtype,
                permutation=(), num_open_out=0, num_open_in=0, trace_count=0)

        reverse = jax.jit(lambda value: jax.linear_transpose(split, jnp.ones(1, dtype=jnp.float32))(value)[0])
        np.testing.assert_array_equal(reverse(source), [2**-24])
        text = reverse.lower(source).as_text()
        assert text.count("stablehlo.custom_call @tensor0_stride_accumulation_f32_cpu_v1") == 1
        assert "scatter" not in text
