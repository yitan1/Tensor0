"""Coefficient contract: independent numerical and boundary regressions."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, dotc, dotu, materialize, reduce_sum
from tensor0._stride._jax import accumulation_p, reduction_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available
from .test_dot_contract import dot_functions
from .test_update_contract import (
    LAYOUTS,
    PAIRS as MIXED_PAIRS,
    compare,
    update_functions,
)
from tests.stride.test_reduction_ad import (
    compare_derivatives,
    functions as reduction_ad_functions,
    values,
)


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield

INTEGERS = ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')

COEFFICIENT_PAIRS = (
    ('float16', 'float16'),
    ('bfloat16', 'bfloat16'),
    ('float32', 'complex64'),
    ('complex64', 'float16'),
    ('float64', 'complex128'),
    ('complex128', 'float64'),
)

def assert_close(actual, expected):
    for value, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.dtype == target.dtype and value.shape == target.shape
        np.testing.assert_allclose(value, target, rtol=0.002, atol=0.002)

@pytest.mark.parametrize('source_dtype', INTEGERS)
@pytest.mark.parametrize('coefficient_dtype,dtype', COEFFICIENT_PAIRS)
@pytest.mark.parametrize('operation', ['accumulation', 'reduction'])
@pytest.mark.parametrize('batch_shape', [(), (2,), (0,), (2, 0)])
def test_coefficient_derivatives(source_dtype, coefficient_dtype, dtype, operation, batch_shape):
    source = jnp.broadcast_to(jnp.asarray([0, 1, 2, 3, 1], dtype=source_dtype), (*batch_shape, 5))
    arguments = (jnp.asarray(2 + 1j if coefficient_dtype.startswith('complex') else 2, dtype=coefficient_dtype), jnp.full(batch_shape, 0.5, dtype=dtype))
    execute, oracle = reduction_ad_functions(operation, dtype)
    run = lambda *factors: execute(source, *factors)
    reference = lambda *factors: oracle(source, *factors)
    directions = tuple((jnp.ones_like(value) for value in arguments))
    assert_close(jax.jit(lambda *values: jax.jvp(run, values, directions))(*arguments), jax.jvp(reference, arguments, directions))
    output, pullback = jax.vjp(run, *arguments)
    cotangent = jnp.full_like(output, 1 + 1j if jnp.iscomplexobj(output) else 1)
    expected = jax.vjp(reference, *arguments)[1](cotangent)
    assert_close(jax.jit(pullback)(cotangent), expected)
    assert_close(jax.jit(jax.linear_transpose(run, *arguments))(cotangent), expected)
    gradients = jax.jit(jax.vjp(execute, source, *arguments)[1])(cotangent)
    assert gradients[0].dtype == jax.dtypes.float0 and gradients[0].shape == source.shape
    assert_close(gradients[1:], expected)

@pytest.mark.parametrize('operation', ['accumulation', 'reduction'])
@pytest.mark.parametrize('factor', [0, 1, 2])
def test_higher_derivatives_and_lowering(operation, factor):
    source = jnp.asarray([1, 2, 3, 1, 2], dtype=jnp.int32)
    run, reference = reduction_ad_functions(operation, 'float32')
    fixed = jnp.float32(0.5)

    def loss(function, value):
        result = function(source, value, fixed)
        return jnp.sum(result * result)
    value = jnp.float32(factor)
    gradient = jax.grad(lambda coefficient: loss(run, coefficient))
    oracle = jax.grad(lambda coefficient: loss(reference, coefficient))
    assert_close(jax.jit(jax.jvp, static_argnums=0)(gradient, (value,), (jnp.float32(1),)), jax.jvp(oracle, (value,), (jnp.float32(1),)))
    assert_close(jax.jit(jax.vmap(gradient))(jnp.asarray([0.0, 1.0, 2.0])), jax.vmap(oracle)(jnp.asarray([0.0, 1.0, 2.0])))
    linear = lambda coefficient: run(source, coefficient, fixed)
    reverse = jax.jit(jax.vjp(linear, value)[1])
    text = reverse.lower(jnp.ones(6, dtype=jnp.float32)).as_text()
    assert 'tensor0_stride_dot_' in text
    assert 'stablehlo.gather' not in text and 'stablehlo.scatter' not in text

@pytest.mark.parametrize('operation', [dotu, dotc])
@pytest.mark.parametrize('source_dtype', INTEGERS)
def test_dot_pullback_with_integer_constant(operation, source_dtype):
    source = jnp.asarray([0, 1, 2], dtype=source_dtype)
    data = jnp.asarray([1, 2, 3], dtype=jnp.float16)
    view = lambda value: StridedView(value, (3,), (1,), 0)
    for constant_left in (False, True):
        run = (lambda value: operation(view(source), view(value))) if constant_left else lambda value: operation(view(value), view(source))
        reverse = jax.vjp(run, data)[1]
        coefficient = jnp.float16(1)
        expected = source.astype(data.dtype)
        value, tangent = jax.jit(lambda cotangent: jax.jvp(reverse, (cotangent,), (cotangent,)))(coefficient)
        assert_close(value[0], expected)
        assert_close(tangent[0], expected)
        actual = jax.jit(jax.linear_transpose(reverse, coefficient))((data,))[0]
        assert_close(actual, jnp.sum(data * expected))


DTYPES = ('float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128')

COMBINATIONS = [types for types in product(DTYPES, repeat=3) if any((dtype in ('float16', 'bfloat16') for dtype in types))]


@pytest.mark.parametrize('operation,source_dtype,coefficient_dtype,dtype', [(operation, *types) for operation in ('accumulation', 'reduction', 'update') for types in COMBINATIONS if operation != 'update' or types[0] == types[2] or (types[0], types[2]) in MIXED_PAIRS])
def test_shared_coefficient_rules(source_dtype, coefficient_dtype, dtype, operation):
    first = jnp.asarray(2, dtype=coefficient_dtype)
    second = jnp.asarray(0.5, dtype=dtype)
    if operation == 'update':
        run, reference = update_functions(LAYOUTS[1])
        compare(run, reference, (values((6,), source_dtype), values((10,), dtype), first, second))
    else:
        run, reference = reduction_ad_functions(operation, dtype)
        compare_derivatives(run, reference, (values((5,), source_dtype), first, second))

@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
@pytest.mark.parametrize('batch_shape', [(2,), (0,), (2, 0)])
@pytest.mark.parametrize('form', ['scalar', 'one', 'batch'])
def test_update_batch_coefficients(dtype, batch_shape, form):
    shape = {'scalar': (), 'one': (1,), 'batch': batch_shape}[form]
    run, reference = update_functions(LAYOUTS[1])
    arguments = (values((*batch_shape, 6), dtype), values((*batch_shape, 10), dtype), jnp.full(shape, 2, dtype=dtype), jnp.full(shape, 0.5, dtype=dtype))
    compare(run, reference, arguments)


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_pullback_lowering_and_vmap(dtype):
    data = values((5,), dtype)
    run, reference = dot_functions(dotu, ((2, 2), (1, 1), 0, (0, -1), 3), dtype)
    reverse = jax.vjp(lambda source: run(source, data), data)[1]
    oracle = jax.vjp(lambda source: reference(source, data), data)[1]
    cotangents = jnp.asarray([0, 1, 2], dtype=dtype)
    actual = jax.jit(jax.vmap(reverse))(cotangents)[0]
    expected = jax.vmap(oracle)(cotangents)[0]
    np.testing.assert_array_equal(actual, expected)
    second_transpose = jax.jit(jax.linear_transpose(reverse, cotangents[0]))
    text = second_transpose.lower((data,)).as_text()
    assert 'tensor0_stride_dot_' in text
    assert 'stablehlo.gather' not in text
    assert 'stablehlo.scatter' not in text

@pytest.mark.parametrize('operation', ['accumulation', 'reduction', 'update'])
def test_wide_coefficient_gradient_range(operation):
    source = jnp.full(5, 1e+20, dtype=jnp.float32)
    factor = jnp.float64(1e-30)
    if operation == 'update':
        execute, _ = update_functions((AffineRecord((1,), (1,), 0, (1,), 1),))
        run = lambda coefficient: execute(source, jnp.zeros(3, jnp.float32), coefficient, jnp.int32(0))
        count = 1
    else:
        execute, _ = reduction_ad_functions(operation, 'float32')
        run = lambda coefficient: execute(source, coefficient, jnp.float32(0))
        count = 4
    result, pullback = jax.vjp(run, factor)
    gradient = jax.jit(pullback)(jnp.full_like(result, 1e+20))[0]
    assert gradient.dtype == factor.dtype
    np.testing.assert_allclose(gradient, count * np.float64(jnp.float32(1e+20)) ** 2, rtol=1e-13)

INEXACT = ('float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128')

DISCRETE = ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')

def check(run, arguments):
    expected = run(*arguments)
    primal, tangent = jax.jit(lambda *values: jax.jvp(run, values, tuple((jnp.ones_like(value) for value in values))))(*arguments)
    np.testing.assert_array_equal(primal, expected)
    assert primal.dtype == expected.dtype
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == primal.shape
    _, pullback = jax.vjp(run, *arguments)
    cotangent = np.zeros(primal.shape, dtype=jax.dtypes.float0)
    for gradient, argument in zip(jax.jit(pullback)(cotangent), arguments, strict=True):
        assert gradient.shape == argument.shape and gradient.dtype == argument.dtype
        np.testing.assert_array_equal(gradient, jnp.zeros_like(argument))

@pytest.mark.parametrize('source_dtype', INEXACT)
@pytest.mark.parametrize('dtype', DISCRETE)
@pytest.mark.parametrize('operation', ['copy', 'reduce', 'empty_axes'])
def test_public_discrete_results(source_dtype, dtype, operation):
    source = jnp.asarray([1, 2, 3], dtype=source_dtype)
    if jnp.iscomplexobj(source):
        source = source + 1j

    def run(data):
        view = StridedView(data, (2, 2), (1, -1), 1)
        if operation == 'copy':
            return materialize(view, dtype=dtype)
        return reduce_sum(view, axes=() if operation == 'empty_axes' else None, dtype=dtype)
    check(run, (source,))

@pytest.mark.parametrize('operation', ['reduction', 'accumulation'])
@pytest.mark.parametrize('dtype', DISCRETE)
@pytest.mark.parametrize('batch_shape', [(), (2,), (0,), (2, 0)])
def test_scaled_discrete_results(operation, dtype, batch_shape):
    records = (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1), AffineRecord((0,), (1,), 3, (1,), 3))
    source = jnp.broadcast_to(jnp.asarray([1, 2, 3], dtype=jnp.float32), (*batch_shape, 3))
    factor = jnp.full(batch_shape, 2, dtype=jnp.float16)

    def run(data, coefficient):
        parameters = dict(records=records, coefficient_records=(0,), output_size=4, dtype=jnp.dtype(dtype))
        if operation == 'reduction':
            return reduction_p.bind(data, coefficient, output_shapes=((2, 1), (1,)), reduction_axes=((False, True), (True,)), **parameters)
        return accumulation_p.bind(data, coefficient, **parameters)
    check(run, (source, factor))

@pytest.mark.parametrize('dtype', ['bool', 'int32'])
@pytest.mark.parametrize('operation', ['copy', 'reduce', 'empty_axes'])
def test_composed_derivatives(dtype, operation):

    def loss(data):
        view = StridedView(data, (3,), (1,), 0)
        result = materialize(view, dtype=dtype) if operation == 'copy' else reduce_sum(view, axes=() if operation == 'empty_axes' else None, dtype=dtype)
        return jnp.sum(result.astype(jnp.float32))
    data = jnp.asarray([1, 2, 3], dtype=jnp.float32)
    gradient = jax.grad(loss)
    np.testing.assert_array_equal(jax.jit(gradient)(data), jnp.zeros_like(data))
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(gradient))(data), jnp.zeros((3, 3)))
    for count in (0, 2):
        batch = jnp.broadcast_to(data, (count, 3))
        np.testing.assert_array_equal(jax.jit(jax.vmap(gradient))(batch), jnp.zeros_like(batch))
    assert 'tensor0_stride_accumulation_' not in jax.jit(gradient).lower(data).as_text()

@pytest.mark.parametrize('dtype', DISCRETE)
def test_integer_inputs_keep_float0_tangents(dtype):
    source = jnp.asarray([0, 1, 1], dtype=dtype)
    direction = np.zeros(source.shape, dtype=jax.dtypes.float0)
    for operation in (materialize, reduce_sum):
        run = lambda data: operation(StridedView(data, (3,), (1,), 0))
        result, tangent = jax.jvp(run, (source,), (direction,))
        assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape
        gradient = jax.vjp(run, source)[1](np.zeros(result.shape, dtype=jax.dtypes.float0))[0]
        assert gradient.dtype == jax.dtypes.float0 and gradient.shape == source.shape
