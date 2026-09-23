"""Native scheduling, concurrency and execution ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensor0._stride as stride_api
from tensor0._stride import (
    StridedView,
    dotu,
    enable_threads,
    get_num_threads,
    scale,
    set_num_threads,
)
from tensor0._stride._ffi._calls import execute_copy, execute_dot
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.affine import (
    MIXED,
    assert_close as affine_assert_close,
    broadcast_record,
    native,
    reference,
    restore_threads as affine_restore_threads,
    values as affine_values,
)
from tests.stride.support.oracles.complex_kernels import (
    assert_close as complex_kernels_assert_close,
    assert_native,
    mapped as complex_kernels_mapped,
    values as complex_kernels_values,
)
from tests.stride.support.oracles.copy import reference_map as copy_reference_map
from tests.stride.support.oracles.generic import (
    assert_close as generic_assert_close,
    blocked_record,
    complex_values,
    mapped as generic_mapped,
    reference_map as generic_reference_map,
    thread_limit,
)
from tests.stride.support.oracles.raw_update import reference_update


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,factor", MIXED)
def test_large_mixed_rank2_forward_and_reverse_under_thread_limits(source_dtype, result_dtype, factor):
    rows = columns = 512
    record = AffineRecord((rows, columns), (1, rows), 0, (columns, 1), 0)
    source = affine_values(source_dtype, rows * columns)
    execute = lambda value: native(value, record, factor, result_dtype)
    oracle = lambda value: reference(value, record, factor, result_dtype)
    compiled = jax.jit(execute)
    expected = oracle(source)
    cotangent = affine_values(result_dtype, rows * columns)
    pullback = jax.jit(jax.vjp(execute, source)[1])
    expected_vjp = jax.vjp(oracle, source)[1](cotangent)[0]
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            affine_assert_close(compiled(source), expected)
            affine_assert_close(pullback(cotangent)[0], expected_vjp)
    finally:
        affine_restore_threads(previous)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_large_broadcast_under_thread_limits():
    record, size = broadcast_record(shape=(4096, 512), broadcast_axes=(1,), source_signs=(1,))
    source = jnp.arange(size, dtype=jnp.float32)
    compiled = jax.jit(lambda value: native(value, record, 1, "float32"))
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            np.testing.assert_array_equal(compiled(source).reshape(4096, 512),
                                           jnp.broadcast_to(source[:, None], (4096, 512)))
    finally:
        affine_restore_threads(previous)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("workers", [1, 4])
def test_native_dot_multidimensional_domain_with_thread_limits(workers) -> None:
    shape = (512, 512)
    element_count = shape[0] * shape[1]
    left_data = (jnp.arange(element_count, dtype=jnp.float32) % 17 - 8) * 0.125
    right_data = (jnp.arange(element_count, dtype=jnp.float32) % 13 - 6) * 0.25
    left = StridedView(left_data, shape, (1, shape[0]), 0)
    right = StridedView(right_data, shape, (shape[1], 1), 0)

    original = stride_api.get_num_threads()
    stride_api.set_num_threads(workers)
    try:
        actual = jax.jit(dotu)(left, right)
        actual.block_until_ready()
    finally:
        if original is None:
            stride_api.enable_threads()
        else:
            stride_api.set_num_threads(original)

    expected = np.sum(np.asarray(left_data).reshape(shape).T * np.asarray(right_data).reshape(shape),
                      dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("workers", [1, 4])
def test_native_multirecord_dot_with_thread_limits(workers) -> None:
    record_size = 65_536
    layout = encode_layout(
        (
            AffineRecord((record_size,), (1,), 0, (1,), 0),
            AffineRecord(
                (record_size,),
                (-1,),
                2 * record_size - 1,
                (-1,),
                2 * record_size - 1,
            ),
        ),
        output_size=2 * record_size,
        source_size=2 * record_size,
    )
    left = jnp.linspace(-1, 2, 2 * record_size, dtype=jnp.float32)
    right = jnp.linspace(3, -2, 2 * record_size, dtype=jnp.float32)

    original = stride_api.get_num_threads()
    stride_api.set_num_threads(workers)
    try:
        actual = jax.jit(
            lambda lhs, rhs: execute_dot(lhs, rhs, layout=layout, conjugate_left=False)
        )(left, right)
        actual.block_until_ready()
    finally:
        if original is None:
            stride_api.enable_threads()
        else:
            stride_api.set_num_threads(original)

    expected = jnp.sum(left * right)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16, jnp.complex64, jnp.int32])
@pytest.mark.parametrize("layout_kind", ["compact", "transpose", "partial"])
def test_large_scaled_layouts_match_single_worker_and_concurrent_execution(dtype, layout_kind):
    rows, columns = 512, 513
    size = rows * columns
    offset = 32 if layout_kind == "partial" else 0
    output_size = size + 2 * offset
    record = (AffineRecord((size,), (1,), 0, (1,), 0) if layout_kind == "compact" else
              AffineRecord((rows, columns), (1, rows), 0, (columns, 1), offset))
    source = (jnp.arange(size, dtype=jnp.int32) % 64 - 32).astype(dtype)
    coefficient = jnp.asarray(-1, dtype=jnp.int32)
    operation = jax.jit(lambda value: accumulation_p.bind(value, coefficient, records=(record,),
                         coefficient_records=(0,), output_size=output_size, dtype=value.dtype)).lower(source).compile()
    previous = get_num_threads()
    try:
        for workers in (1, 4):
            set_num_threads(workers)
            actual = operation(source)
            np.testing.assert_array_equal(actual, copy_reference_map(np.asarray(source), (record,), (-1,), output_size))
        with ThreadPoolExecutor(max_workers=4) as executor:
            values = [source + jnp.asarray(index, dtype=dtype) for index in range(4)]
            outputs = list(executor.map(operation, values))
        for value, actual in zip(values, outputs, strict=True):
            np.testing.assert_array_equal(actual, copy_reference_map(np.asarray(value), (record,), (-1,), output_size))
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("shape", [(16, 16, 16, 64), (8, 8, 8, 8, 64), (8, 8, 8, 8, 8, 8),
                                  (4, 4, 4, 4, 4, 4, 64), (4, 4, 4, 4, 4, 4, 4, 16)])
def test_large_high_rank_blocking_under_thread_limits(shape):
    record = blocked_record(shape)
    source = jnp.arange(prod(shape), dtype=jnp.float32) - 3
    operation = jax.jit(lambda value: generic_mapped(value, record, jnp.float32(-1.25), value.dtype, source.size))
    expected = generic_reference_map(source, record, jnp.float32(-1.25), source.dtype, source.size)
    for limit in (1, 4):
        with thread_limit(limit):
            generic_assert_close(operation(source), expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("record,size,dtype,factor", [
    (AffineRecord((65536, 4), (8, 2), 0, (4, 1), 0), 524287, jnp.float32, .75),
    (AffineRecord((1024, 512), (1, 1024), 0, (512, 1), 0), 524288, jnp.float32, 1.25),
    (AffineRecord((524291,), (1,), 0, (1,), 0), 524291, jnp.complex64, .75 - .5j),
])
def test_large_ranges_under_thread_limits(record, size, dtype, factor):
    raw = np.resize(np.asarray([0., -0., np.inf, -np.inf, np.nan, np.finfo(np.float32).max]), size)
    source = jnp.asarray(raw, dtype=dtype) if dtype == jnp.float32 else complex_values(size)
    coefficient = jnp.asarray(factor, dtype=dtype)
    output_size = prod(record.logical_shape)
    operation = jax.jit(lambda value: generic_mapped(value, record, coefficient, dtype, output_size))
    expected = generic_reference_map(source, record, coefficient, dtype, output_size)
    for limit in (1, 4):
        with thread_limit(limit):
            generic_assert_close(operation(source), expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_large_half_pullback_under_thread_limits():
    size = 1 << 20
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    operation = lambda value: generic_mapped(value, record, jnp.float32(-1.25), value.dtype, size)
    pullback = jax.jit(jax.vjp(operation, source)[1])
    cotangent = jnp.ones_like(source)
    for limit in (1, 4):
        with thread_limit(limit):
            generic_assert_close(pullback(cotangent)[0], jnp.full_like(source, -1.25))


def complex_kernel_restore_threads(previous):
    if previous is None:
        enable_threads()
    else:
        set_num_threads(previous)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_large_mixed_compact_mapping_under_thread_limits():
    size = (1 << 19) + 3
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    source = jnp.arange(size, dtype=jnp.float32) / 1024 - 2
    operation = jax.jit(lambda value: complex_kernels_mapped(value, record, .75 - .5j))
    expected = source * jnp.complex64(.75 - .5j)
    assert_native(operation.lower(source))
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            complex_kernels_assert_close(operation(source), expected)
    finally:
        complex_kernel_restore_threads(previous)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_selected_scale_batched_factors_under_thread_limits():
    size = 262147
    base = jnp.stack(tuple(complex_kernels_values(size) + index for index in range(3)))
    factors = jnp.asarray([.75 - .5j, -1.25 + .25j, 2 + .125j], dtype=jnp.complex64)
    operation = jax.jit(lambda data, alpha: scale(StridedView(data, (size,), (1,), 0), alpha).data)
    expected = base * factors[:, None]
    assert_native(operation.lower(base, factors))
    previous = get_num_threads()
    try:
        for limit in (1, 8):
            set_num_threads(limit)
            complex_kernels_assert_close(operation(base, factors), expected)
    finally:
        complex_kernel_restore_threads(previous)


@pytest.fixture(autouse=False)
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('batches,size,empty_records', [(0, LEFT_SIZE, False), (dot_batch_threads_BATCHES, 0, False), (dot_batch_threads_BATCHES, LEFT_SIZE, True)])
def test_empty_batches_and_zero_results(batches, size, empty_records):
    records = () if empty_records else (AffineRecord((size,), (1,), 0, (1,), 0),)
    layout = encode_layout(records, source_size=size, output_size=size)
    values = jnp.ones((batches, size), jnp.float32)
    result = jax.jit(lambda values: execute_dot(values, values, layout=layout, conjugate_left=True))(values)
    np.testing.assert_array_equal(result, np.zeros(batches, np.float32))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [1., -.75])
@pytest.mark.parametrize("shape,batches,workers", [((8, 8, 8, 8), 3, 1), ((32, 16, 16, 16), 1, 4)])
def test_rank4_prepared_layout_batches_and_worker_limits(factor, shape, batches, workers):
    outer, rows, inner, columns = shape
    size = int(np.prod(shape))
    record = AffineRecord(shape, (inner * columns, outer * inner * columns, 1, inner), 0,
                          (rows * inner * columns, inner * columns, columns, 1), 0)
    source = jnp.arange(batches * size, dtype=jnp.float32).reshape(batches, size)
    operation = jax.jit(lambda value, coefficient: accumulation_p.bind(
        value, coefficient, records=(record,), coefficient_records=(0,), output_size=size, dtype=np.dtype(jnp.float32),
    ))
    previous = get_num_threads()
    try:
        set_num_threads(workers)
        actual = operation(source, jnp.float32(factor))
        actual.block_until_ready()
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)
    expected = np.asarray(source).reshape(batches, rows, outer, columns, inner).transpose(0, 2, 1, 4, 3)
    np.testing.assert_array_equal(actual, expected.reshape(batches, size) * factor)


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("layout_kind", ["transpose", "negative", "broadcast"])
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_layout_updates_with_thread_limits(beta, layout_kind):
    rows, columns = 512, 512
    size = rows * columns
    if layout_kind == "transpose":
        record = AffineRecord((rows, columns), (1, rows), 0, (columns, 1), 0)
        source = jnp.resize(jnp.array([0., -0., jnp.inf, -jnp.inf, jnp.nan, 1.5, -2.]), (size,))
    elif layout_kind == "negative":
        record = AffineRecord((rows, columns), (-columns, 1), (rows - 1) * columns, (1, rows), 0)
        source = (jnp.arange(size, dtype=jnp.float32) % 17) * .125
    else:
        record = AffineRecord((rows, columns), (1, 0), 0, (columns, 1), 0)
        source = jnp.linspace(-2, 3, rows)
    base = jnp.linspace(-3, 2, size)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(1.25), jnp.int32(beta), records=(record,)))
    expected = reference_update(source, base, (record,), 1.25, beta)
    original = get_num_threads()
    try:
        for workers in (1, 4):
            set_num_threads(workers)
            np.testing.assert_allclose(operation(base, source), expected, rtol=2e-6, atol=1e-6)
    finally:
        enable_threads() if original is None else set_num_threads(original)
