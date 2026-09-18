"""Dot contract: independent numerical and boundary regressions."""

from itertools import product
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, dotc, dotu
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield

def operands(left_dtype, right_dtype, batch_shape):
    left = (jnp.arange(prod(batch_shape) * 8) % 7 / 4).astype(left_dtype).reshape((*batch_shape, 8))
    right = (jnp.arange(prod(batch_shape) * 10) % 5 / 4).astype(right_dtype).reshape((*batch_shape, 10))
    if jnp.iscomplexobj(left):
        left = left + 1j * (left - 0.5)
    if jnp.iscomplexobj(right):
        right = right + 1j * (1 - right)
    return (left, right)

def numpy_dot_functions(operation, layout, dtype):
    shape, left_strides, left_offset, right_strides, right_offset = layout

    def run(left, right):
        return operation(StridedView(left, shape, left_strides, left_offset), StridedView(right, shape, right_strides, right_offset), dtype=dtype)

    def oracle(left, right):
        result_dtype = jnp.result_type(left.dtype, right.dtype) if dtype is None else jnp.dtype(dtype)
        result = np.zeros(left.shape[:-1], dtype=np.complex128)
        for coordinates in np.ndindex(shape):
            left_index = left_offset + sum((index * stride for index, stride in zip(coordinates, left_strides, strict=True)))
            right_index = right_offset + sum((index * stride for index, stride in zip(coordinates, right_strides, strict=True)))
            value = np.asarray(left)[..., left_index]
            if operation is dotc:
                value = np.conj(value)
            result += value * np.asarray(right)[..., right_index]
        if not jnp.issubdtype(result_dtype, jnp.complexfloating):
            result = result.real
        return result.astype(result_dtype)
    return (run, oracle)

LAYOUTS = [
    ((2, 3), (1, 2), 1, (-3, -1), 6),
    ((2, 3), (0, 1), 1, (1, 1), 1),
    ((0,), (1,), 0, (1,), 0),
    ((), (), 2, (), 3),
]

INEXACT_DTYPES = ('float32', 'float64', 'complex64', 'complex128')

@pytest.mark.parametrize('left_dtype,right_dtype', list(product(INEXACT_DTYPES, repeat=2)))
@pytest.mark.parametrize('dtype', [None, *INEXACT_DTYPES])
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_all_input_and_result_combinations(left_dtype, right_dtype, dtype, operation):
    left, right = operands(left_dtype, right_dtype, ())
    run, oracle = numpy_dot_functions(operation, LAYOUTS[0], dtype)
    expected_dtype = jnp.result_type(left.dtype, right.dtype) if dtype is None else jnp.dtype(dtype)
    for execute in (run, jax.jit(run)):
        result = execute(left, right)
        assert result.dtype == expected_dtype
        np.testing.assert_allclose(result, oracle(left, right), rtol=1e-06, atol=1e-06)
    text = jax.jit(run).lower(left, right).as_text()
    assert text.count('stablehlo.custom_call') == 1
    assert operation_target('dot', expected_dtype) in text
    for name in ('stablehlo.convert', 'stablehlo.gather', 'stablehlo.scatter'):
        assert name not in text

@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('batch_shape', [(2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_mixed_layouts_batches_and_vmap(layout, batch_shape, operation):
    left, right = operands('complex64', 'float64', batch_shape)
    run, oracle = numpy_dot_functions(operation, layout, 'complex128')
    np.testing.assert_allclose(jax.jit(run)(left, right), oracle(left, right), rtol=1e-13, atol=1e-13)
    mapped = jax.jit(jax.vmap(run))
    np.testing.assert_allclose(mapped(left, right), oracle(left, right), rtol=1e-13, atol=1e-13)

def test_result_dtype_does_not_preconvert_inputs():
    left = StridedView(jnp.asarray([1 + 2j], jnp.complex64), (1,), (1,), 0)
    right = StridedView(jnp.asarray([3 + 4j], jnp.complex128), (1,), (1,), 0)
    np.testing.assert_allclose(dotu(left, right, dtype=jnp.float64), -5, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dotc(left, right, dtype=jnp.float64), 11, rtol=1e-13, atol=1e-13)

@pytest.mark.parametrize('operation', [dotu, dotc])
def test_real_operand_is_not_padded_into_complex(operation):
    left = StridedView(jnp.asarray([complex(np.inf, 2)], jnp.complex64), (1,), (1,), 0)
    right = StridedView(jnp.ones(1, jnp.float64), (1,), (1,), 0)
    result = operation(left, right)
    assert jnp.isinf(result.real)
    np.testing.assert_array_equal(result.imag, -2 if operation is dotc else 2)

def values(dtype):
    result = jnp.asarray([0, 1, 2, 1], dtype=dtype)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        result = result + jnp.asarray([1j, 0j, -1j, 0j], dtype=dtype)
    return result

STORAGE_DTYPES = (
    'bool',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float16',
    'bfloat16',
    'float32',
    'float64',
    'complex64',
    'complex128',
)

@pytest.mark.parametrize('left_dtype,right_dtype', product(STORAGE_DTYPES, repeat=2))
@jax.enable_x64()
def test_all_input_pairs(left_dtype, right_dtype):
    left = values(left_dtype)
    right = values(right_dtype)
    for operation in (dotu, dotc):

        def run(first, second):
            return operation(StridedView(first, (2, 2), (-1, 1), 1), StridedView(second, (2, 2), (0, 1), 1))
        selected_left = left[jnp.asarray([[1, 2], [0, 1]])]
        selected_right = right[jnp.asarray([[1, 2], [1, 2]])]
        if operation is dotc:
            selected_left = jnp.conj(selected_left)
        dtype = jnp.result_type(left, right)
        expected = jnp.sum(selected_left * selected_right, dtype=dtype)
        lowered = jax.jit(run).lower(left, right)
        text = lowered.as_text()
        assert text.count('stablehlo.custom_call') == 1
        assert 'tensor0_stride_dot_' in text
        assert 'stablehlo.gather' not in text
        actual = lowered.compile()(left, right)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('dtype', STORAGE_DTYPES)
@jax.enable_x64()
def test_output_types_batches_and_empty(dtype):
    for batch_count, length in ((0, 4), (2, 4), (2, 0)):
        left = jnp.broadcast_to(jnp.asarray([0, 1, 2, 1], jnp.int16)[:length], (batch_count, length))
        right = jnp.broadcast_to(jnp.asarray([1, 2, 1, 0], jnp.float32)[:length], (batch_count, length))

        def run(first, second):
            return dotu(StridedView(first, (length,), (1,), 0), StridedView(second, (length,), (1,), 0), dtype=dtype)
        expected = jnp.full((batch_count,), 4 if length else 0, dtype=dtype)
        actual = jax.jit(run)(left, right)
        assert actual.dtype == jnp.dtype(dtype)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(jax.jit(jax.vmap(run))(left, right), expected)

@pytest.mark.parametrize('dtype', STORAGE_DTYPES[1:9])
@jax.enable_x64()
def test_integer_wraparound(dtype):
    info = np.iinfo(dtype)
    left = jnp.asarray([info.max, info.min, 3], dtype=dtype)
    right = jnp.asarray([2, 3, 2], dtype=dtype)
    modulus = 1 << info.bits
    expected = (int(info.max) * 2 + int(info.min) * 3 + 6) % modulus
    if info.min < 0 and expected > info.max:
        expected -= modulus
    view = lambda data: StridedView(data, (3,), (1,), 0)
    for operation in (dotu, dotc):
        actual = jax.jit(lambda first, second: operation(view(first), view(second)))(left, right)
        assert int(actual) == expected

def test_output_dtype_does_not_precast_integer_product():
    left = StridedView(jnp.asarray([100, 100], jnp.int8), (2,), (1,), 0)
    right = StridedView(jnp.asarray([2, 2], jnp.int8), (2,), (1,), 0)
    assert int(dotu(left, right, dtype=jnp.int32)) == -112

@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_low_precision_nonfinite_inputs_are_not_coefficients(dtype):
    left = StridedView(jnp.asarray([0], dtype=dtype), (1,), (1,), 0)
    right = StridedView(jnp.asarray([jnp.inf], dtype=dtype), (1,), (1,), 0)
    assert bool(jnp.isnan(dotu(left, right)))
    assert bool(jnp.isnan(dotc(left, right)))

def dot_functions(operation, layout, dtype):
    shape, left_strides, left_offset, right_strides, right_offset = layout
    left_indices = np.asarray([left_offset + sum((index * stride for index, stride in zip(coordinates, left_strides, strict=True))) for coordinates in np.ndindex(shape)], dtype=np.int32)
    right_indices = np.asarray([right_offset + sum((index * stride for index, stride in zip(coordinates, right_strides, strict=True))) for coordinates in np.ndindex(shape)], dtype=np.int32)

    def run(left, right):
        return operation(StridedView(left, shape, left_strides, left_offset), StridedView(right, shape, right_strides, right_offset), dtype=dtype)

    def reference(left, right):
        values = left[..., left_indices]
        if operation is dotc:
            values = jnp.conj(values)
        products = values * right[..., right_indices]
        result_dtype = products.dtype if dtype is None else jnp.dtype(dtype)
        accumulator_dtype = jnp.promote_types(products.dtype, result_dtype)
        result = jnp.sum(products, axis=-1, dtype=accumulator_dtype)
        if jnp.issubdtype(result_dtype, jnp.floating):
            result = jnp.real(result)
        return result.astype(result_dtype)
    return (run, reference)

@pytest.mark.parametrize('operation', [dotu, dotc])
@pytest.mark.parametrize('left_dtype,right_dtype,dtype', [('float32', 'float64', 'float64'), ('complex128', 'complex64', 'complex64')])
def test_vmap_symbolic_zero_and_native_lowering(operation, left_dtype, right_dtype, dtype):
    left, right = operands(left_dtype, right_dtype, (3,))
    run, reference = dot_functions(operation, LAYOUTS[1], dtype)
    mapped = jax.vmap(run, in_axes=(0, None))
    oracle = jax.vmap(reference, in_axes=(0, None))
    inputs = (left, right[0])
    cotangent = jnp.ones(3, dtype=dtype)
    for value, expected in zip(jax.jit(jax.vjp(mapped, *inputs)[1])(cotangent), jax.vjp(oracle, *inputs)[1](cotangent), strict=True):
        np.testing.assert_allclose(value, expected, rtol=2e-06, atol=2e-06)
    frozen = lambda first, second: mapped(jax.lax.stop_gradient(first), jax.lax.stop_gradient(second))
    np.testing.assert_array_equal(jax.jvp(frozen, inputs, inputs)[1], jnp.zeros(3, dtype=dtype))
    reverse = jax.jit(lambda first, second, value: jax.vjp(run, first, second)[1](value))
    text = reverse.lower(left[0], right[0], jnp.asarray(1, dtype=dtype)).as_text()
    assert 'tensor0_stride_accumulation_' in text
    for name in ('stablehlo.gather', 'stablehlo.scatter', 'tensor0_stride_update'):
        assert name not in text

def test_wide_input_gradient_does_not_overflow_in_narrow_cotangent_dtype():
    left = jnp.asarray([1e-30], dtype=jnp.float64)
    right = jnp.asarray([1e+20], dtype=jnp.float32)
    run, _ = dot_functions(dotu, ((1,), (1,), 0, (1,), 0), 'float32')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.float32(1e+20))[0]
    assert gradient.dtype == left.dtype
    np.testing.assert_allclose(gradient, np.asarray(right, dtype=np.float64) * np.float64(jnp.float32(1e+20)), rtol=1e-13)

@pytest.mark.parametrize('operation', [dotu, dotc])
def test_second_forward_derivative(operation):
    left, right = operands('complex64', 'complex128', ())
    run, reference = dot_functions(operation, LAYOUTS[1], 'complex128')
    tangent = lambda first, second: jax.jvp(run, (first, second), (left, right))[1]
    expected = lambda first, second: jax.jvp(reference, (first, second), (left, right))[1]
    np.testing.assert_allclose(jax.jit(lambda first, second: jax.jvp(tangent, (first, second), (left, right))[1])(left, right), jax.jvp(expected, (left, right), (left, right))[1], rtol=2e-06, atol=2e-06)

def test_mixed_dot_pullback_double_transpose():
    left, right = operands('float32', 'float64', ())
    run, _ = dot_functions(dotu, LAYOUTS[1], 'float64')
    reverse = lambda value: jax.linear_transpose(lambda values: run(values, right), left)(value)[0]
    actual = jax.jit(jax.linear_transpose(reverse, jnp.float64(1)))(left)[0]
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, run(left, right), rtol=2e-06, atol=2e-06)

CROSS = [types for types in product(INEXACT_DTYPES, repeat=3) if len({dtype.startswith('complex') for dtype in types}) > 1]

def compare(operation, left_dtype, right_dtype, dtype, batch_shape, layout):
    left, right = operands(left_dtype, right_dtype, batch_shape)
    run, reference = dot_functions(operation, layout, dtype)
    tangents = tuple((jnp.full_like(value, 1 + 2j if jnp.iscomplexobj(value) else 2) for value in (left, right)))
    for value, expected in zip(jax.jit(lambda first, second: jax.jvp(run, (first, second), tangents))(left, right), jax.jvp(reference, (left, right), tangents), strict=True):
        assert value.dtype == expected.dtype
        np.testing.assert_allclose(value, expected, rtol=2e-06, atol=2e-06)
    result = run(left, right)
    cotangent = jnp.full_like(result, 2 - 3j if jnp.iscomplexobj(result) else 3)
    expected_gradients = jax.vjp(reference, left, right)[1](cotangent)
    for actual, expected, primal in zip(jax.jit(jax.vjp(run, left, right)[1])(cotangent), expected_gradients, (left, right), strict=True):
        assert actual.dtype == primal.dtype and actual.shape == primal.shape
        np.testing.assert_allclose(actual, expected, rtol=2e-06, atol=2e-06)
    for function, primal, expected in ((lambda values: run(values, right), left, expected_gradients[0]), (lambda values: run(left, values), right, expected_gradients[1])):
        np.testing.assert_allclose(jax.jit(jax.linear_transpose(function, primal))(cotangent)[0], expected, rtol=2e-06, atol=2e-06)

@pytest.mark.parametrize('left_dtype,right_dtype,dtype', CROSS)
@pytest.mark.parametrize('operation', [dotu, dotc])
@pytest.mark.parametrize('batch_shape', [(), (2, 3)])
def test_cross_kind_type_combinations(left_dtype, right_dtype, dtype, operation, batch_shape):
    compare(operation, left_dtype, right_dtype, dtype, batch_shape, LAYOUTS[1])

@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_cross_kind_layouts_vmap_and_lowering(layout, operation):
    left, right = operands('float32', 'complex128', (3,))
    run, reference = dot_functions(operation, layout, 'complex64')
    mapped, oracle = (jax.vmap(run, in_axes=(0, None)), jax.vmap(reference, in_axes=(0, None)))
    cotangent = jnp.full(3, 2 + 3j, dtype=jnp.complex64)
    for actual, expected in zip(jax.jit(jax.vjp(mapped, left, right[0])[1])(cotangent), jax.vjp(oracle, left, right[0])[1](cotangent), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-06, atol=2e-06)
    frozen = lambda first, second: mapped(jax.lax.stop_gradient(first), jax.lax.stop_gradient(second))
    np.testing.assert_array_equal(jax.jvp(frozen, (left, right[0]), (left, right[0]))[1], jnp.zeros_like(cotangent))
    if layout[0] == (2, 3):
        text = jax.jit(jax.linear_transpose(lambda values: run(values, right[0]), left[0])).lower(cotangent[0]).as_text()
        assert 'tensor0_stride_accumulation_' in text
        assert 'stablehlo.real' not in text
        for name in ('stablehlo.gather', 'stablehlo.scatter', 'tensor0_stride_update'):
            assert name not in text

@pytest.mark.parametrize('operation', [dotu, dotc])
def test_imaginary_cotangent_contributes_to_real_input_gradient(operation):
    left = jnp.asarray([2], dtype=jnp.float32)
    right = jnp.asarray([3 + 4j], dtype=jnp.complex64)
    run, _ = dot_functions(operation, ((1,), (1,), 0, (1,), 0), 'complex64')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.complex64(1 + 2j))[0]
    np.testing.assert_array_equal(gradient, [-5])

@pytest.mark.parametrize('operation,expected', [(dotu, 3 + 4j), (dotc, 3 - 4j)])
def test_real_output_has_complex_input_gradient(operation, expected):
    left = jnp.asarray([1 + 2j], jnp.complex64)
    right = jnp.asarray([3 + 4j], jnp.complex128)
    run, _ = dot_functions(operation, ((1,), (1,), 0, (1,), 0), 'float64')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.float64(1))[0]
    np.testing.assert_array_equal(gradient, [expected])

def test_cross_kind_dot_pullback_derivative():
    left, right = operands('float32', 'complex64', ())
    run, reference = dot_functions(dotu, LAYOUTS[1], 'complex64')
    reverse = lambda value: jax.linear_transpose(lambda values: run(values, right), left)(value)[0]
    expected_reverse = lambda value: jax.vjp(lambda values: reference(values, right), left)[1](value)[0]
    for actual, expected in zip(jax.jvp(reverse, (jnp.complex64(1),), (jnp.complex64(1),)), jax.jvp(expected_reverse, (jnp.complex64(1),), (jnp.complex64(1),)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-06, atol=2e-06)

def assert_close(actual, expected):
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        if value.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(value, reference, rtol=0.002, atol=0.002)

def tangent(value):
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return np.zeros(value.shape, dtype=jax.dtypes.float0)
    return jnp.full_like(value, 1 + 1j if jnp.iscomplexobj(value) else 1)

def check(left_dtype, right_dtype, dtype, operation, batch_shape=()):
    operands = []
    for operand_dtype in (left_dtype, right_dtype):
        value = jnp.asarray([0, 1, 2], dtype=operand_dtype)
        if jnp.iscomplexobj(value):
            value = value + jnp.asarray([1j, -1j, 0j], dtype=operand_dtype)
        operands.append(jnp.broadcast_to(value, (*batch_shape, 3)))
    left, right = operands
    result_dtype = jnp.result_type(left, right) if dtype is None else jnp.dtype(dtype)

    def run(first, second):
        return operation(StridedView(first, (2, 2), (1, 1), 0), StridedView(second, (2, 2), (0, -1), 2), dtype=dtype)

    def reference(first, second):
        selected_left = first[..., jnp.asarray([[0, 1], [1, 2]])]
        selected_right = second[..., jnp.asarray([[2, 1], [2, 1]])]
        if operation is dotc:
            selected_left = jnp.conj(selected_left)
        mapped = selected_left * selected_right
        sum_dtype = jnp.promote_types(result_dtype, mapped.dtype)
        result = jnp.sum(mapped.astype(sum_dtype), axis=(-2, -1), dtype=sum_dtype)
        if jnp.iscomplexobj(result) and (not jnp.issubdtype(result_dtype, jnp.complexfloating)):
            result = result.real
        return result.astype(result_dtype)
    directions = tuple((tangent(value) for value in operands))
    actual_jvp = jax.jit(lambda first, second: jax.jvp(run, (first, second), directions))(left, right)
    assert_close(actual_jvp, jax.jvp(reference, (left, right), directions))
    output, pullback = jax.vjp(run, left, right)
    cotangent = tangent(output)
    assert_close(jax.jit(pullback)(cotangent), jax.vjp(reference, left, right)[1](cotangent))

@pytest.mark.parametrize('batch_shape', [(0,), (2,), (2, 0)])
@pytest.mark.parametrize('dtype', ['bool', 'int32', 'float16', 'complex64'])
@jax.enable_x64()
def test_batched_result_dtype(batch_shape, dtype):
    check('float16', 'float32', dtype, dotu, batch_shape)

@pytest.mark.parametrize('dtype', ['bool', 'int8', 'uint64'])
@pytest.mark.parametrize('operation', [dotu, dotc])
@jax.enable_x64()
def test_discrete_result_composed_gradient(dtype, operation):

    def loss(data):
        view = StridedView(data, (3,), (1,), 0)
        return operation(view, view, dtype=dtype).astype(jnp.float32)
    data = jnp.asarray([1, 2, 3], dtype=jnp.float32)
    gradient = jax.grad(loss)
    np.testing.assert_array_equal(jax.jit(gradient)(data), jnp.zeros_like(data))
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(gradient))(data), jnp.zeros((3, 3)))
    batch = jnp.stack((data, data * 2))
    np.testing.assert_array_equal(jax.jit(jax.vmap(gradient))(batch), jnp.zeros_like(batch))
    text = jax.jit(gradient).lower(data).as_text()
    assert 'tensor0_stride_accumulation_' not in text


def _dot_layout(records, left_size, right_size):
    return encode_layout(tuple(AffineRecord(*record) for record in records),
                         source_size=left_size, output_size=right_size)


def _raw_dot(left, right, layout, *, conjugate=0, dtype=None, shape=None):
    dtype = left.dtype if dtype is None else jnp.dtype(dtype)
    execute = jax.ffi.ffi_call(
        operation_target("dot", dtype),
        jax.ShapeDtypeStruct((left.shape[0],) if shape is None else shape, dtype),
        vmap_method="sequential",
    )
    return execute(left, right, layout=layout, conjugate_left=np.int64(conjugate))


def test_invalid_buffers_and_flags():
    layout = _dot_layout((((3,), (1,), 0, (1,), 0),), 3, 3)
    valid = jnp.ones((1, 3))
    for left, right, shape in ((jnp.ones(3), valid, (1,)), (valid, jnp.ones(3), (1,)),
                               (valid, valid, (1, 1)), (valid, valid, (2,)),
                               (jnp.ones((1, 4)), valid, (1,)), (valid, jnp.ones((2, 3)), (1,))):
        with pytest.raises(Exception, match="rank|dimensions"):
            _raw_dot(left, right, layout, shape=shape).block_until_ready()
    for left, right in ((valid.astype(jnp.float8_e4m3fn), valid), (valid, valid.astype(jnp.float8_e5m2))):
        with pytest.raises(Exception, match="dtype|type"):
            _raw_dot(left, right, layout, dtype="float32").block_until_ready()
    for conjugate in (-1, 2):
        with pytest.raises(Exception, match="conjugate_left"):
            _raw_dot(valid, valid, layout, conjugate=conjugate).block_until_ready()


def test_invalid_protocol_and_read_bounds():
    layout = _dot_layout((((2,), (1,), 0, (1,), 0),), 2, 2)
    source = jnp.ones((1, 2))
    for length in range(len(layout)):
        with pytest.raises(Exception, match="truncated|descriptor length"):
            _raw_dot(source, source, layout[:length]).block_until_ready()
    for index, value in ((0, 2), (1, -1), (2, -1), (3, -1), (3, 2**63 - 1),
                         (4, 2**63 - 1), (5, -1), (6, 2), (7, -1), (8, 2**63 - 1)):
        invalid = layout.copy()
        invalid[index] = value
        with pytest.raises(Exception):
            _raw_dot(source, source, invalid).block_until_ready()
    with pytest.raises(Exception, match="trailing"):
        _raw_dot(source, source, np.append(layout, 0)).block_until_ready()
