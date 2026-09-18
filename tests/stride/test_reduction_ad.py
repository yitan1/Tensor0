"""Reduction and packed trace coefficient AD through existing native operations."""

from itertools import product
from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, reduce_sum
from tensor0._stride._jax import accumulation_p, reduction_p
from tensor0._stride._tensor_ops import _strided_tensortrace
from tensor0._stride._layout import AffineRecord
from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


def scalar_trace(source, factors, dtype):
    return _strided_tensortrace(
        source, destination_size=1,
        source_subblocks=tuple(SimpleNamespace(sizes=(), strides=(), offset=index)
                              for index in range(source.shape[-1])),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
        entries=((index, 0, factor) for index, factor in enumerate(factors)),
        result_dtype=dtype, permutation=(), num_open_out=0, num_open_in=0, trace_count=0)


def packed_trace(source, factor, second):
    return _strided_tensortrace(
        source, destination_size=5,
        source_subblocks=(
            SimpleNamespace(sizes=(2, 0, 0), strides=(1, 1, 1), offset=12),
            SimpleNamespace(sizes=(2, 2, 2), strides=(4, 2, 1), offset=0),
            SimpleNamespace(sizes=(2, 2, 2), strides=(0, -2, -1), offset=3),
        ),
        destination_subblocks=(SimpleNamespace(sizes=(2,), strides=(1,), offset=1),
                              SimpleNamespace(sizes=(2,), strides=(-1,), offset=3)),
        entries=((0, 0, None), (1, 0, None), (2, 1, factor), (1, 1, second)),
        result_dtype=source.dtype, permutation=(), num_open_out=1, num_open_in=0, trace_count=1,
    )


def reference_trace(source, factor, second):
    first = source[..., jnp.asarray([[0, 3], [4, 7]])]
    broadcast = source[..., jnp.asarray([[3, 0], [3, 0]])]
    output = jnp.zeros((*source.shape[:-1], 5), dtype=source.dtype)
    output = output.at[..., 1:3].add(first.sum(axis=-1))
    contribution = (broadcast * factor).sum(axis=-1) + (first * second).sum(axis=-1)
    if not jnp.issubdtype(source.dtype, jnp.complexfloating):
        contribution = jnp.real(contribution)
    return output.at[..., jnp.asarray([3, 2])].add(contribution.astype(source.dtype))


def values(shape, dtype):
    result = (jnp.arange(prod(shape)) % 5 / 4).astype(dtype).reshape(shape)
    return result + 1j * (result - .5) if dtype.startswith("complex") else result


def functions(operation, dtype):
    records = (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1),
               AffineRecord((2, 2), (0, -1), 3, (-1, 1), 3),
               AffineRecord((0,), (1,), 5, (1,), 5))
    def run(source, first, second):
        if operation == "accumulation":
            return accumulation_p.bind(source, first, second, records=records,
                coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
        return reduction_p.bind(source, first, second, records=records,
            output_shapes=((2, 1), (2, 1), (1,)), reduction_axes=((False, True), (False, True), (True,)),
            coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
    def reference(source, first, second):
        factors = tuple(factor.reshape(()) if factor.shape == (1,) else factor for factor in (first, second))
        result = jnp.zeros((*source.shape[:-1], 6), dtype=dtype)
        for record, factor in zip(records[:2], factors, strict=True):
            for coordinates in np.ndindex(record.logical_shape):
                source_index = record.source_offset + sum(index * stride for index, stride in
                    zip(coordinates, record.source_strides, strict=True))
                destination_index = record.destination_offset + sum(index * stride for axis, (index, stride) in
                    enumerate(zip(coordinates, record.destination_strides, strict=True))
                    if operation == "accumulation" or axis == 0)
                contribution = source[..., source_index] * factor
                if not dtype.startswith("complex"):
                    contribution = jnp.real(contribution)
                result = result.at[..., destination_index].add(contribution.astype(dtype))
        return result
    return run, reference


def compare_derivatives(run, reference, arguments):
    directions = tuple(jnp.ones_like(value) for value in arguments)
    for value, expected in zip(jax.jit(lambda *inputs: jax.jvp(run, inputs, directions))(*arguments),
                               jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_allclose(value, expected, rtol=3e-6, atol=3e-6)
    output = run(*arguments)
    cotangent = jnp.full_like(output, 2 - 1j if jnp.iscomplexobj(output) else 2)
    for value, expected, original in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent),
                                         jax.vjp(reference, *arguments)[1](cotangent), arguments, strict=True):
        assert value.dtype == original.dtype and value.shape == original.shape
        np.testing.assert_allclose(value, expected, rtol=3e-6, atol=3e-6)
    coefficients = arguments[1:]
    coefficient_run = lambda *factors: run(arguments[0], *factors)
    for value, expected in zip(jax.jit(jax.linear_transpose(coefficient_run, *coefficients))(cotangent),
                               jax.vjp(reference, *arguments)[1](cotangent)[1:], strict=True):
        np.testing.assert_allclose(value, expected, rtol=3e-6, atol=3e-6)


@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield


RECORDS = (
    AffineRecord((2, 0), (1, 1), 8, (2, 7), 1),
    AffineRecord((2, 2), (1, 1), 1, (2, 1), 1),
    AffineRecord((2, 2), (-1, 1), 3, (100, 1), 2),
    AffineRecord((2, 3), (0, -1), 5, (3, 100), 0),
)
OUTPUT_SHAPES = ((2, 1), (2, 1), (1, 2), (2, 1))
AXES = ((False, True), (False, True), (True, False), (False, True))
COEFFICIENT_RECORDS = (0, 2, 3)


def execute(source, empty, first, second):
    return reduction_p.bind(
        source, empty, first, second, records=RECORDS, output_shapes=OUTPUT_SHAPES,
        reduction_axes=AXES, coefficient_records=COEFFICIENT_RECORDS, output_size=7, dtype=source.dtype,
    )


def reference(source, empty, first, second):
    coefficients = tuple(value.reshape(()) if value.shape == (1,) else value for value in (empty, first, second))
    result = jnp.zeros((*source.shape[:-1], 7), dtype=source.dtype)
    for index, (record, axes) in enumerate(zip(RECORDS, AXES, strict=True)):
        factor = coefficients[COEFFICIENT_RECORDS.index(index)] if index in COEFFICIENT_RECORDS else None
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum(axis * stride for axis, stride in zip(coordinates, record.source_strides))
            output_index = record.destination_offset + sum(axis * stride for axis, stride, reduced in
                zip(coordinates, record.destination_strides, axes, strict=True) if not reduced)
            value = source[..., source_index]
            result = result.at[..., output_index].add(value if factor is None else value * factor)
    return result


def operands(dtype, batch_shape, coefficient_shape):
    source = (jnp.arange(prod(batch_shape) * 8) % 5).astype(dtype).reshape((*batch_shape, 8))
    if dtype in ("complex64", "complex128"):
        source = source + 1j * (source - 2)
    empty = jnp.full(coefficient_shape, 1, dtype=dtype)
    first = jnp.full(coefficient_shape, 2 + 1j if dtype in ("complex64", "complex128") else 2, dtype=dtype)
    second = jnp.full(batch_shape, 3 - 2j if dtype in ("complex64", "complex128") else 3, dtype=dtype)
    return source, empty, first, second


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("shape_kind", ["scalar", "one", "batch"])
def test_joint_jvp_and_vjp(dtype, batch_shape, shape_kind):
    shape = {"scalar": (), "one": (1,), "batch": batch_shape}[shape_kind]
    arguments = operands(dtype, batch_shape, shape)
    directions = tuple(jnp.full_like(value, 1 + 2j if dtype in ("complex64", "complex128") else 2) for value in arguments)
    for actual, expected in zip(jax.jit(lambda *values: jax.jvp(execute, values, directions))(*arguments),
                                jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_array_equal(actual, expected)
    cotangent = (jnp.arange(prod(batch_shape) * 7) % 7).astype(dtype).reshape((*batch_shape, 7))
    if dtype in ("complex64", "complex128"):
        cotangent = cotangent * (1 - 1j)
    for actual, expected in zip(jax.jit(jax.vjp(execute, *arguments)[1])(cotangent),
                                jax.vjp(reference, *arguments)[1](cotangent), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("active", [(0,), (1,), (2,), (3,), (1, 2, 3), (1, 3)])
def test_selected_tangents_and_direct_transpose(dtype, active):
    arguments = operands(dtype, (2, 3), ())
    def bind(function, *values):
        operands = list(arguments)
        for index, value in zip(active, values, strict=True):
            operands[index] = value
        return function(*operands)
    run, oracle = lambda *values: bind(execute, *values), lambda *values: bind(reference, *values)
    primals = tuple(arguments[index] for index in active)
    tangents = tuple(jnp.ones_like(value) for value in primals)
    np.testing.assert_array_equal(jax.jvp(run, primals, tangents)[1], jax.jvp(oracle, primals, tangents)[1])
    cotangent = jnp.broadcast_to(jnp.arange(7).astype(dtype), (2, 3, 7))
    expected = jax.vjp(oracle, *primals)[1](cotangent)
    for gradients in (jax.jit(jax.vjp(run, *primals)[1])(cotangent),
                      jax.jit(jax.linear_transpose(run, *primals))(cotangent)):
        for actual, value in zip(gradients, expected, strict=True):
            np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("factor", [0, 1, 2])
@pytest.mark.parametrize("fixed_dtype", ["int32", "float64"])
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


@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("source_axis,coefficient_axis", [(1, None), (None, 0), (1, 0)])
def test_vmap_shared_source_and_coefficients(batch_size, source_axis, coefficient_axis):
    source = jnp.arange(8, dtype=jnp.float32) if source_axis is None else jnp.arange(
        8 * batch_size, dtype=jnp.float32).reshape((8, batch_size))
    first = jnp.float32(2) if coefficient_axis is None else jnp.arange(batch_size, dtype=jnp.float32)
    second = jnp.float32(3)
    arguments = source, first, second
    run, oracle = (jax.vmap(lambda data, alpha, beta: execute(data, jnp.float32(1), alpha, beta),
                           in_axes=(source_axis, coefficient_axis, None)),
                   jax.vmap(lambda data, alpha, beta: reference(data, jnp.float32(1), alpha, beta),
                            in_axes=(source_axis, coefficient_axis, None)))
    cotangent = jnp.ones((batch_size, 7), dtype=jnp.float32)
    for actual, expected in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent),
                                jax.vjp(oracle, *arguments)[1](cotangent), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_trace_shared_coefficient_and_nonzero_at_zero():
    source = jnp.asarray([1, 2, 3], jnp.float32)
    run = lambda value: scalar_trace(source, (value, None, value), source.dtype)
    for factor in (0, 1, 2):
        np.testing.assert_array_equal(jax.grad(lambda value: run(value).sum())(jnp.float32(factor)), 4)


def test_coefficient_only_lowering_and_unselected_nonfinite_values():
    record = AffineRecord((2, 2), (2, 1), 1, (2, 100), 1)
    def run(data, factor):
        return reduction_p.bind(data, factor, records=(record,), output_shapes=((2, 1),),
            reduction_axes=((False, True),), coefficient_records=(0,), output_size=5, dtype=data.dtype)
    source = jnp.asarray([np.nan, 2, 3, 4, 5, np.inf], jnp.float32)
    cotangent = jnp.asarray([np.nan, 2, np.inf, 3, np.nan], jnp.float32)
    reverse = jax.jit(lambda data, value: jax.vjp(lambda factor: run(data, factor), jnp.float32(0))[1](value)[0])
    np.testing.assert_array_equal(reverse(source, cotangent), 37)
    text = reverse.lower(source, cotangent).as_text()
    assert text.count("stablehlo.custom_call @tensor0_stride_dot_f32_cpu_v1") == 1
    for operation in ("tensor0_stride_reduction", "tensor0_stride_accumulation", "stablehlo.gather", "stablehlo.scatter"):
        assert operation not in text


@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,), (2, 0)])
def test_empty_input_and_symbolic_zero(batch_shape):
    source = jnp.zeros((*batch_shape, 0), jnp.float32)
    def run(data, factor):
        return reduction_p.bind(data, factor, records=(AffineRecord((0,), (1,), 0, (1,), 1),),
            output_shapes=((1,),), reduction_axes=((True,),), coefficient_records=(0,), output_size=3, dtype=data.dtype)
    arguments = source, jnp.float32(2)
    output, pullback = jax.vjp(run, *arguments)
    np.testing.assert_array_equal(output, jnp.zeros((*batch_shape, 3)))
    source_gradient, factor_gradient = jax.jit(pullback)(jnp.ones_like(output))
    np.testing.assert_array_equal(source_gradient, source)
    np.testing.assert_array_equal(factor_gradient, 0)
    frozen = lambda *values: run(*(jax.lax.stop_gradient(value) for value in values))
    np.testing.assert_array_equal(jax.jvp(frozen, arguments, tuple(jnp.ones_like(value) for value in arguments))[1], output)


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


def test_trace_vector_coefficients_remain_rejected():
    source = jnp.ones((2, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="scalars shared"):
        scalar_trace(source, (jnp.ones(2),), source.dtype)


def test_joint_bilinear_direct_transpose_remains_rejected():
    source, empty, first, second = operands("float32", (), ())
    with pytest.raises(NotImplementedError, match="known source or known coefficients"):
        jax.linear_transpose(lambda data, value: execute(data, empty, value, second), source, first)(jnp.ones(7, dtype=source.dtype))


CROSS = [types for types in product(("float32", "float64", "complex64", "complex128"), repeat=3)
         if len({dtype.startswith("complex") for dtype in types}) > 1]


@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype", CROSS)
def test_trace_cross_kind_derivatives(source_dtype, coefficient_dtype, dtype):
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
    compare_derivatives(run, reference, (source, factor))


@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype", CROSS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("shape_kind", ["scalar", "one", "batch"])
def test_cross_kind_derivatives(source_dtype, coefficient_dtype, dtype, batch_shape, shape_kind):
    source = values((*batch_shape, 5), source_dtype)
    shape = {"scalar": (), "one": (1,), "batch": batch_shape}[shape_kind]
    first = jnp.full(shape, 2 + 1j if coefficient_dtype.startswith("complex") else 2, dtype=coefficient_dtype)
    second = jnp.full(batch_shape, .5 + .5j if dtype.startswith("complex") else .5, dtype=dtype)
    run, reference = functions("reduction", dtype)
    compare_derivatives(run, reference, (source, first, second))


@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype", [
    ("float32", "complex128", "float64"), ("complex64", "float64", "complex128"),
    ("complex128", "complex64", "float32"), ("float64", "float32", "complex64")])
@pytest.mark.parametrize("factor", [0, 1, 2])
def test_higher_derivatives_and_vmap(source_dtype, coefficient_dtype, dtype, factor):
    run, reference = functions("reduction", dtype)
    source, coefficient = values((5,), source_dtype), jnp.asarray(factor, dtype=coefficient_dtype)
    fixed = jnp.asarray(.5, dtype=dtype)
    def objective(function, data, alpha):
        output = function(data, alpha, fixed)
        return jnp.real(jnp.sum(output * jnp.conj(output)))
    gradient = jax.grad(lambda *inputs: objective(run, *inputs), argnums=(0, 1))
    expected_gradient = jax.grad(lambda *inputs: objective(reference, *inputs), argnums=(0, 1))
    arguments = source, coefficient
    directions = jnp.ones_like(source), jnp.ones_like(coefficient)
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(gradient, inputs, directions)[1])(*arguments),
                                jax.jvp(expected_gradient, arguments, directions)[1], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    for actual, expected in zip(jax.jit(jax.vjp(gradient, *arguments)[1])(directions),
                                jax.vjp(expected_gradient, *arguments)[1](directions), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    compare_derivatives(jax.vmap(run, in_axes=(0, 0, None)), jax.vmap(reference, in_axes=(0, 0, None)),
                        (jnp.stack((source, source * 2)), jnp.stack((coefficient, coefficient)), fixed))


def test_projection_after_multiplication():
    record = AffineRecord((2,), (0,), 0, (1,), 1)
    def run(source, factor):
        return reduction_p.bind(source, factor, records=(record,), output_shapes=((1,),),
            reduction_axes=((True,),), coefficient_records=(0,), output_size=3, dtype=jnp.dtype("complex128"))
    source, factor = jnp.asarray([2], jnp.float32), jnp.complex128(3 + 4j)
    cotangent = jnp.asarray([0, 1 + 2j, 0], jnp.complex128)
    gradients = jax.jit(jax.vjp(run, source, factor)[1])(cotangent)
    np.testing.assert_array_equal(gradients[0], [-10])
    np.testing.assert_array_equal(gradients[1], 4 + 8j)
    text = jax.jit(lambda data, alpha, value: jax.vjp(run, data, alpha)[1](value)).lower(source, factor, cotangent).as_text()
    assert "tensor0_stride_accumulation_" in text and "tensor0_stride_dot_" in text
    for name in ("stablehlo.real", "stablehlo.gather", "stablehlo.scatter", "tensor0_stride_update"):
        assert name not in text


@pytest.mark.parametrize("source_dtype,dtype", [("float32", "complex128"), ("complex64", "float64")])
@pytest.mark.parametrize("shape,strides,offset", [((2, 2), (1, 1), 0), ((2, 2), (0, -1), 2), ((2, 0), (1, 1), 3)])
def test_public_sum_and_empty_reduction(source_dtype, dtype, shape, strides, offset):
    source = values((3,), source_dtype)
    def run(data):
        return reduce_sum(StridedView(data, shape, strides, offset), axes=(1,), dtype=dtype)
    def reference(data):
        logical = jnp.stack([data[offset + outer * strides[0] + inner * strides[1]]
                             for outer in range(shape[0]) for inner in range(shape[1])]) if shape[1] else jnp.zeros(0, source_dtype)
        output = logical.reshape(shape).sum(axis=1)
        return (jnp.real(output) if dtype.startswith("float") else output).astype(dtype)
    for actual, expected in zip(jax.jit(lambda data: jax.jvp(run, (data,), (data,)))(source),
                                jax.jvp(reference, (source,), (source,)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    cotangent = jnp.ones_like(run(source))
    actual = jax.jit(jax.vjp(run, source)[1])(cotangent)[0]
    expected = jax.vjp(reference, source)[1](cotangent)[0]
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    frozen = lambda data: run(jax.lax.stop_gradient(data))
    np.testing.assert_array_equal(jax.jvp(frozen, (source,), (source,))[1], jnp.zeros(shape[0], dtype))


TRACE_COEFFICIENT_DTYPES = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
]

TRACE_DTYPE_PAIRS = [(dtype, dtype) for dtype in ("float16", "bfloat16", "float64", "complex128")]

TRACE_DTYPE_PAIRS += [(dtype, coefficient) for dtype in ("float32", "complex64") for coefficient in TRACE_COEFFICIENT_DTYPES]

@pytest.mark.parametrize("source_dtype,coefficient_dtype", TRACE_DTYPE_PAIRS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,)])
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

def test_packed_trace_mixed_storage_source_ad():
    source = jnp.ones(1, dtype=jnp.float32)
    execute = lambda data: scalar_trace(data, (jnp.float32(2),), jnp.float16)
    assert execute(source).dtype == jnp.float16
    gradient = jax.vjp(execute, source)[1](jnp.ones(1, dtype=jnp.float16))[0]
    assert gradient.dtype == source.dtype
    np.testing.assert_allclose(gradient, [2], rtol=1e-6, atol=1e-6)

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
