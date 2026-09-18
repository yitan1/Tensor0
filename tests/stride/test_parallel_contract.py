"""Parallel contract: independent numerical and boundary regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_copy, execute_dot
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


@pytest.fixture(autouse=True)
def worker_limit():
    from tensor0 import _native

    previous = _native._stride_native_worker_limit()
    _native._set_stride_native_worker_limit(4)
    try:
        yield
    finally:
        _native._set_stride_native_worker_limit(previous)

def source_values(source):
    return source.reshape(COLUMNS, ROWS)[:, ::-1].T.reshape(-1)

ROWS, COLUMNS = 513, 257
SINGLE_SIZE = ROWS * COLUMNS

OUTPUT_SIZE = 2 * SINGLE_SIZE + 4

SINGLE_RECORDS = (
    AffineRecord((ROWS, COLUMNS), (-1, ROWS), ROWS - 1, (2 * COLUMNS, 2), 2),
    AffineRecord((), (), 0, (), 1),
)

SINGLE_LAYOUT = encode_layout(SINGLE_RECORDS, source_size=SINGLE_SIZE, output_size=OUTPUT_SIZE)

def update(source, base, alpha, beta, *, alias=False, layout=SINGLE_LAYOUT):
    operation = jax.ffi.ffi_call(operation_target('update', base.dtype), jax.ShapeDtypeStruct(base.shape, base.dtype), input_output_aliases={1: 0} if alias else {}, vmap_method='sequential')
    return operation(source, base, alpha, beta, layout=layout)

@pytest.mark.parametrize('source_dtype,result_dtype', [('float16', 'float32'), ('float32', 'float32'), ('float32', 'complex64'), ('complex64', 'float32'), ('float64', 'complex128'), ('int32', 'int32')])
@pytest.mark.parametrize('alias', [False, True])
@jax.enable_x64()
def test_single_batch_copy_and_update(source_dtype, result_dtype, alias):
    source = (jnp.arange(SINGLE_SIZE).reshape(1, SINGLE_SIZE) % 17).astype(source_dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 2j
    original = np.array(source)
    values = source_values(original)
    if not jnp.issubdtype(jnp.dtype(result_dtype), jnp.complexfloating):
        values = values.real
    expected_copy = np.zeros((1, OUTPUT_SIZE), dtype=result_dtype)
    expected_copy[0, 2:2 * SINGLE_SIZE + 2:2] = values
    expected_copy[0, 1] = original[0, 0] if np.iscomplexobj(expected_copy) else original[0, 0].real
    copied = jax.jit(lambda data: execute_copy(data, layout=SINGLE_LAYOUT, output_size=OUTPUT_SIZE, dtype=result_dtype))(source)
    np.testing.assert_array_equal(copied, expected_copy)
    base = jnp.full((1, OUTPUT_SIZE), 3, dtype=result_dtype)
    operation = jax.jit(lambda source, base, alpha, beta: update(source, base, alpha, beta, alias=alias))
    for alpha, beta in [(0, 0), (0, 3), (1, 0), (1, 1), (2, 3)]:
        expected = np.array(base)
        expected[0, 2:2 * SINGLE_SIZE + 2:2] = alpha * values + beta * 3
        expected[0, 1] = alpha * expected_copy[0, 1] + beta * 3
        np.testing.assert_array_equal(operation(source, base, jnp.float32(alpha), jnp.int32(beta)), expected)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(base, np.full(base.shape, 3))

def test_single_batch_same_storage_donation():
    layout = encode_layout((AffineRecord((SINGLE_SIZE,), (-2,), 2 * SINGLE_SIZE, (-2,), 2 * SINGLE_SIZE),), source_size=OUTPUT_SIZE, output_size=OUTPUT_SIZE)
    values = jnp.full((1, OUTPUT_SIZE), 3, jnp.float32)
    operation = jax.jit(lambda data: update(data, data, jnp.float32(2), jnp.int32(1), alias=True, layout=layout), donate_argnums=(0,))
    expected = np.full((1, OUTPUT_SIZE), 3, np.float32)
    expected[0, 2:2 * SINGLE_SIZE + 2:2] = 9
    np.testing.assert_array_equal(operation(values), expected)
    assert values.is_deleted()

UPDATE_RECORDS = (AffineRecord((2048,), (0,), 3, (2,), 1), AffineRecord((1024,), (-1,), 4095, (3,), 5000))

def update_reference(source, base, alpha, beta, records=UPDATE_RECORDS):
    result = np.array(base, copy=True)
    for record in records:
        positions = np.arange(record.logical_shape[0])
        source_indices = record.source_offset + positions * record.source_strides[0]
        destination_indices = record.destination_offset + positions * record.destination_strides[0]
        for batch in range(len(base)):
            left = alpha if alpha.ndim == 0 else alpha[batch]
            right = beta if beta.ndim == 0 else beta[batch]
            values = left * source[batch, source_indices] + right * base[batch, destination_indices]
            if not np.iscomplexobj(result):
                values = np.real(values)
            result[batch, destination_indices] = values
    return result

update_batch_threads_BATCHES = 17

UPDATE_SIZE = 8192

def operation(source, base, alpha, beta, *, alias, records=UPDATE_RECORDS):
    layout = encode_layout(records, source_size=UPDATE_SIZE, output_size=UPDATE_SIZE)
    execute = jax.ffi.ffi_call(operation_target('update', base.dtype), jax.ShapeDtypeStruct(base.shape, base.dtype), input_output_aliases={1: 0} if alias else {}, vmap_method='sequential')
    return execute(source, base, alpha, beta, layout=layout)

@pytest.mark.parametrize('source_dtype,result_dtype', [('float16', 'float32'), ('float32', 'float32'), ('float32', 'complex64'), ('complex64', 'float32'), ('float64', 'complex128'), ('int32', 'int32')])
@pytest.mark.parametrize('alias', [False, True])
@pytest.mark.parametrize('shared', [False, True])
@jax.enable_x64()
def test_large_batches(source_dtype, result_dtype, alias, shared):
    source = (jnp.arange(update_batch_threads_BATCHES * UPDATE_SIZE).reshape(update_batch_threads_BATCHES, UPDATE_SIZE) % 7).astype(source_dtype)
    base = jnp.full((update_batch_threads_BATCHES, UPDATE_SIZE), 5, dtype=result_dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 2j
    alpha = jnp.float32(2) if shared else (jnp.arange(update_batch_threads_BATCHES) % 3).astype(jnp.float32)
    beta = jnp.int32(3) if shared else ((jnp.arange(update_batch_threads_BATCHES) + 1) % 3).astype(jnp.int32)
    original_source, original_base = (np.array(source), np.array(base))
    expected = update_reference(original_source, original_base, np.array(alpha), np.array(beta))
    execute = jax.jit(lambda source, base, alpha, beta: operation(source, base, alpha, beta, alias=alias))
    np.testing.assert_array_equal(execute(source, base, alpha, beta), expected)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(base, original_base)

dot_batch_threads_BATCHES = 17

LEFT_SIZE = 8192

RIGHT_SIZE = 12288

def inputs(left_dtype, right_dtype):
    left = (jnp.arange(dot_batch_threads_BATCHES * LEFT_SIZE).reshape(dot_batch_threads_BATCHES, LEFT_SIZE) % 5).astype(left_dtype)
    right = (jnp.arange(dot_batch_threads_BATCHES * RIGHT_SIZE).reshape(dot_batch_threads_BATCHES, RIGHT_SIZE) % 3).astype(right_dtype)
    if jnp.issubdtype(left.dtype, jnp.complexfloating):
        left = left + 1j
    if jnp.issubdtype(right.dtype, jnp.complexfloating):
        right = right - 2j
    return (left, right)

DOT_RECORDS = (AffineRecord((2048,), (0,), 7, (-2,), 8191), AffineRecord((8, 4), (1, 1), 20, (4, 1), 5))

def dot_reference(left, right, conjugate, dtype):
    result = np.zeros(left.shape[:-1], dtype=dtype)
    for record in DOT_RECORDS:
        coordinates = np.asarray(list(np.ndindex(record.logical_shape)))
        left_indices = record.source_offset + coordinates @ np.asarray(record.source_strides)
        right_indices = record.destination_offset + coordinates @ np.asarray(record.destination_strides)
        source = left[..., left_indices]
        if conjugate:
            source = np.conj(source)
        contribution = np.sum(source * right[..., right_indices], axis=-1)
        if not np.issubdtype(result.dtype, np.complexfloating):
            contribution = np.real(contribution)
        result = (result + contribution).astype(dtype)
    return result

DOT_LAYOUT = encode_layout(DOT_RECORDS, source_size=LEFT_SIZE, output_size=RIGHT_SIZE)

@pytest.mark.parametrize('left_dtype,right_dtype,result_dtype', [('float16', 'float32', 'float32'), ('float32', 'float64', 'float32'), ('complex64', 'float32', 'complex64'), ('float32', 'complex64', 'complex64'), ('complex128', 'float64', 'float32'), ('complex64', 'complex128', 'complex128'), ('int32', 'int32', 'int32')])
@pytest.mark.parametrize('conjugate', [False, True])
@jax.enable_x64()
def test_parallel_multirecord_dot(left_dtype, right_dtype, result_dtype, conjugate):
    left, right = inputs(left_dtype, right_dtype)
    original_left, original_right = (np.array(left), np.array(right))
    execute = jax.jit(lambda left, right: execute_dot(left, right, layout=DOT_LAYOUT, conjugate_left=conjugate, dtype=result_dtype))
    result = execute(left, right)
    np.testing.assert_allclose(result, dot_reference(original_left, original_right, conjugate, result_dtype), rtol=2e-06, atol=2e-06)
    serial = np.stack([np.asarray(execute(left[batch], right[batch])) for batch in range(dot_batch_threads_BATCHES)])
    np.testing.assert_array_equal(result, serial)
    np.testing.assert_array_equal(left, original_left)
    np.testing.assert_array_equal(right, original_right)

def test_parallel_record_order_and_nonfinite_values():
    size = LEFT_SIZE
    records = tuple((AffineRecord((), (), index, (), index) for index in range(3)))
    layout = encode_layout(records, source_size=size, output_size=size)
    left = jnp.zeros((dot_batch_threads_BATCHES, size), jnp.float32).at[:, :3].set(jnp.array([2 ** 26, 1, -2 ** 26]))
    right = jnp.ones_like(left)
    execute = jax.jit(lambda left, right: execute_dot(left, right, layout=layout, conjugate_left=False))
    np.testing.assert_array_equal(execute(left, right), np.zeros(dot_batch_threads_BATCHES, np.float32))
    left = left.at[:, 0].set(0)
    right = right.at[:, 0].set(jnp.inf)
    assert np.isnan(np.asarray(execute(left, right))).all()

@pytest.mark.parametrize('batches,size,empty_records', [(0, LEFT_SIZE, False), (dot_batch_threads_BATCHES, 0, False), (dot_batch_threads_BATCHES, LEFT_SIZE, True)])
def test_empty_batches_and_zero_results(batches, size, empty_records):
    records = () if empty_records else (AffineRecord((size,), (1,), 0, (1,), 0),)
    layout = encode_layout(records, source_size=size, output_size=size)
    values = jnp.ones((batches, size), jnp.float32)
    result = jax.jit(lambda values: execute_dot(values, values, layout=layout, conjugate_left=True))(values)
    np.testing.assert_array_equal(result, np.zeros(batches, np.float32))
