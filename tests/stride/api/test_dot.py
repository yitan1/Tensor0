"""Public Dot types, layouts and arithmetic semantics."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, dotc, dotu
from tensor0._stride._ffi._registration import operation_target

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.dot import INEXACT_DTYPES, LAYOUTS, operands


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


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


@pytest.mark.usefixtures("enable_x64")
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('batch_shape', [(2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('operation', [dotu, dotc])
def test_mixed_layouts_batches_and_vmap(layout, batch_shape, operation):
    left, right = operands('complex64', 'float64', batch_shape)
    run, oracle = numpy_dot_functions(operation, layout, 'complex128')
    np.testing.assert_allclose(jax.jit(run)(left, right), oracle(left, right), rtol=1e-13, atol=1e-13)
    mapped = jax.jit(jax.vmap(run))
    np.testing.assert_allclose(mapped(left, right), oracle(left, right), rtol=1e-13, atol=1e-13)


@pytest.mark.usefixtures("enable_x64")
def test_result_dtype_does_not_preconvert_inputs():
    left = StridedView(jnp.asarray([1 + 2j], jnp.complex64), (1,), (1,), 0)
    right = StridedView(jnp.asarray([3 + 4j], jnp.complex128), (1,), (1,), 0)
    np.testing.assert_allclose(dotu(left, right, dtype=jnp.float64), -5, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dotc(left, right, dtype=jnp.float64), 11, rtol=1e-13, atol=1e-13)


@pytest.mark.usefixtures("enable_x64")
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


# Identity paths and both directions of semantic promotion/conversion anchors.
DOT_MIXED_PAIRS = {
    ('bool', 'int8'),
    ('bool', 'uint64'),
    ('bool', 'float16'),
    ('bool', 'bfloat16'),
    ('bool', 'complex64'),
    ('int8', 'uint8'),
    ('int16', 'uint16'),
    ('int32', 'uint32'),
    ('int64', 'uint64'),
    ('int8', 'int64'),
    ('uint8', 'uint64'),
    ('int64', 'uint32'),
    ('float16', 'bfloat16'),
    ('float16', 'float32'),
    ('bfloat16', 'float32'),
    ('float32', 'float64'),
    ('int64', 'float16'),
    ('uint64', 'bfloat16'),
    ('int32', 'float32'),
    ('complex64', 'complex128'),
    ('float32', 'complex64'),
    ('float64', 'complex64'),
    ('float16', 'complex64'),
    ('bfloat16', 'complex64'),
    ('int64', 'complex64'),
    ('float64', 'complex128'),
}


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('left_dtype,right_dtype', [
    (left, right) for left in STORAGE_DTYPES for right in STORAGE_DTYPES
    if left == right or (left, right) in DOT_MIXED_PAIRS
    or (right, left) in DOT_MIXED_PAIRS
])
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


@pytest.mark.usefixtures("enable_x64")
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


@pytest.mark.usefixtures("enable_x64")
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


@pytest.mark.usefixtures("enable_x64")
def test_output_dtype_does_not_precast_integer_product():
    left = StridedView(jnp.asarray([100, 100], jnp.int8), (2,), (1,), 0)
    right = StridedView(jnp.asarray([2, 2], jnp.int8), (2,), (1,), 0)
    assert int(dotu(left, right, dtype=jnp.int32)) == -112


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_low_precision_nonfinite_inputs_are_not_coefficients(dtype):
    left = StridedView(jnp.asarray([0], dtype=dtype), (1,), (1,), 0)
    right = StridedView(jnp.asarray([jnp.inf], dtype=dtype), (1,), (1,), 0)
    assert bool(jnp.isnan(dotu(left, right)))
    assert bool(jnp.isnan(dotc(left, right)))
