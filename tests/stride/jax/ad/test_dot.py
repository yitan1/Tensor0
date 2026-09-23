"""JAX AD dot contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, dotc, dotu
from tensor0._stride._jax import dot_p

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.accumulation import TWO_RECORDS
from tests.stride.support.oracles.dot import LAYOUTS, dot_functions, operands


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def accumulation_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("accumulation_enable_x64")
def test_multirecord_dot_transpose():
    source = jnp.arange(5, dtype=jnp.float32)
    records = (TWO_RECORDS[0], TWO_RECORDS[0], TWO_RECORDS[1], TWO_RECORDS[2])

    def function(left, right):
        return dot_p.bind(left, right, records=records, conjugate_left=False)
    left_gradient, right_gradient = jax.grad(function, argnums=(0, 1))(source, source)
    expected_left = np.zeros(5, np.float32)
    expected_right = np.zeros(5, np.float32)
    for record in records:
        for coordinates in np.ndindex(record.logical_shape):
            left_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
            right_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
            expected_left[left_index] += float(source[right_index])
            expected_right[right_index] += float(source[left_index])
    np.testing.assert_array_equal(left_gradient, expected_left)
    np.testing.assert_array_equal(right_gradient, expected_right)


@pytest.fixture(autouse=False)
def dot_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("dot_enable_x64")
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


@pytest.mark.usefixtures("dot_enable_x64")
def test_wide_input_gradient_does_not_overflow_in_narrow_cotangent_dtype():
    left = jnp.asarray([1e-30], dtype=jnp.float64)
    right = jnp.asarray([1e+20], dtype=jnp.float32)
    run, _ = dot_functions(dotu, ((1,), (1,), 0, (1,), 0), 'float32')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.float32(1e+20))[0]
    assert gradient.dtype == left.dtype
    np.testing.assert_allclose(gradient, np.asarray(right, dtype=np.float64) * np.float64(jnp.float32(1e+20)), rtol=1e-13)


@pytest.mark.usefixtures("dot_enable_x64")
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_second_forward_derivative(operation):
    left, right = operands('complex64', 'complex128', ())
    run, reference = dot_functions(operation, LAYOUTS[1], 'complex128')
    tangent = lambda first, second: jax.jvp(run, (first, second), (left, right))[1]
    expected = lambda first, second: jax.jvp(reference, (first, second), (left, right))[1]
    np.testing.assert_allclose(jax.jit(lambda first, second: jax.jvp(tangent, (first, second), (left, right))[1])(left, right), jax.jvp(expected, (left, right), (left, right))[1], rtol=2e-06, atol=2e-06)


@pytest.mark.usefixtures("dot_enable_x64")
def test_mixed_dot_pullback_double_transpose():
    left, right = operands('float32', 'float64', ())
    run, _ = dot_functions(dotu, LAYOUTS[1], 'float64')
    reverse = lambda value: jax.linear_transpose(lambda values: run(values, right), left)(value)[0]
    actual = jax.jit(jax.linear_transpose(reverse, jnp.float64(1)))(left)[0]
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, run(left, right), rtol=2e-06, atol=2e-06)


# Each mixed real/complex pattern keeps narrow, wide and one mixed-precision
# representative. AD intentionally does not cross every precision combination.
CROSS = [
    ('float32', 'float32', 'complex64'),
    ('float32', 'complex64', 'float32'),
    ('float32', 'complex64', 'complex64'),
    ('float32', 'complex64', 'complex128'),
    ('float32', 'complex128', 'float64'),
    ('float64', 'float32', 'complex128'),
    ('float64', 'float64', 'complex128'),
    ('float64', 'complex128', 'float64'),
    ('float64', 'complex128', 'complex128'),
    ('complex64', 'float32', 'float32'),
    ('complex64', 'float32', 'complex64'),
    ('complex64', 'complex64', 'float32'),
    ('complex64', 'complex128', 'float32'),
    ('complex128', 'float32', 'float64'),
    ('complex128', 'float64', 'float64'),
    ('complex128', 'float64', 'complex64'),
    ('complex128', 'float64', 'complex128'),
    ('complex128', 'complex128', 'float64'),
]


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


# Both conjugation modes retain these batched mixed-precision representatives,
# covering every mixed real/complex kind pattern.
DOT_BATCH_TYPES = [
    ('complex128', 'float32', 'float64'), ('float32', 'complex128', 'float64'),
    ('float64', 'float32', 'complex128'), ('complex64', 'complex128', 'float32'),
    ('complex128', 'float64', 'complex64'), ('float32', 'complex64', 'complex128'),
]


@pytest.mark.usefixtures("dot_enable_x64")
@pytest.mark.parametrize('left_dtype,right_dtype,dtype,batch_shape',
                         [(*types, ()) for types in CROSS]
                         + [(*types, (2, 3)) for types in DOT_BATCH_TYPES])
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_cross_kind_type_combinations(left_dtype, right_dtype, dtype, operation, batch_shape):
    compare(operation, left_dtype, right_dtype, dtype, batch_shape, LAYOUTS[1])


@pytest.mark.usefixtures("dot_enable_x64")
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


@pytest.mark.usefixtures("dot_enable_x64")
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_imaginary_cotangent_contributes_to_real_input_gradient(operation):
    left = jnp.asarray([2], dtype=jnp.float32)
    right = jnp.asarray([3 + 4j], dtype=jnp.complex64)
    run, _ = dot_functions(operation, ((1,), (1,), 0, (1,), 0), 'complex64')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.complex64(1 + 2j))[0]
    np.testing.assert_array_equal(gradient, [-5])


@pytest.mark.usefixtures("dot_enable_x64")
@pytest.mark.parametrize('operation,expected', [(dotu, 3 + 4j), (dotc, 3 - 4j)])
def test_real_output_has_complex_input_gradient(operation, expected):
    left = jnp.asarray([1 + 2j], jnp.complex64)
    right = jnp.asarray([3 + 4j], jnp.complex128)
    run, _ = dot_functions(operation, ((1,), (1,), 0, (1,), 0), 'float64')
    gradient = jax.jit(jax.vjp(run, left, right)[1])(jnp.float64(1))[0]
    np.testing.assert_array_equal(gradient, [expected])


@pytest.mark.usefixtures("dot_enable_x64")
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


@pytest.mark.usefixtures("dot_enable_x64")
@pytest.mark.parametrize('batch_shape', [(0,), (2,), (2, 0)])
@pytest.mark.parametrize('dtype', ['bool', 'int32', 'float16', 'complex64'])
@jax.enable_x64()
def test_batched_result_dtype(batch_shape, dtype):
    check('float16', 'float32', dtype, dotu, batch_shape)


@pytest.mark.usefixtures("dot_enable_x64")
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
