"""Native batch, single-output and independent-output reduction scheduling."""

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import (
    StridedView,
    dotu,
    enable_threads,
    get_num_threads,
    reduce_sum,
    set_num_threads,
)
from tensor0._stride._ffi._calls import execute_accumulation, execute_dot, execute_reduction
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.reduction import addresses, assert_close


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("kind", ["row", "gapped", "outputs", "batch", "integer"])
def test_long_reduction_worker_limits_concurrency_and_ad(kind):
    if kind == "gapped":
        shape, strides, source_size, axes = (257, 263), (527, 2), 1 + 256 * 527 + 262 * 2, (0, 1)
    elif kind in ("outputs", "batch"):
        shape, strides, source_size, axes = (8, 8192), (8192, 1), 65536, (1,)
    else:
        shape, strides, source_size, axes = (65536,), (1,), 65536, (0,)
    dtype = jnp.int32 if kind == "integer" else jnp.float32
    source = (jnp.arange(source_size, dtype=jnp.int32) % 16).astype(dtype)
    if kind == "batch":
        source = jnp.stack((source, source + 1))
    native = lambda value: reduce_sum(StridedView(value, shape, strides, 0), axes, dtype=dtype)
    indices = addresses(shape, strides, 0)
    reference = lambda value: jnp.sum(value[..., indices], axis=tuple(value.ndim - 1 + axis for axis in axes), dtype=dtype)
    compiled = jax.jit(native).lower(source).compile()
    previous = get_num_threads()
    try:
        for workers in (1, 4):
            set_num_threads(workers)
            assert_close(compiled(source), reference(source))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda shift: compiled(source + shift), range(4)))
        for shift, actual in enumerate(results):
            assert_close(actual, reference(source + shift))
        if kind in ("row", "gapped"):
            cotangent = jnp.zeros_like(compiled(source))
            pullback = lambda cot: jax.vjp(native, source)[1](cot)[0]
            actual = jax.jit(lambda cot: jax.linear_transpose(pullback, cotangent)(cot)[0])(jnp.ones_like(source))
            assert_close(actual, reference(jnp.ones_like(source)))
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)


@pytest.fixture(autouse=False)
def restore_worker_limit():
    original = get_num_threads()
    set_num_threads(4)
    try:
        yield
    finally:
        if original is None:
            enable_threads()
        else:
            set_num_threads(original)


def _encode_thread_layout(accumulation, records, source_size, output_size):
    if accumulation:
        return encode_layout(records, source_size=source_size, output_size=output_size)
    axes = tuple(tuple(stride == 0 for stride in record.destination_strides) for record in records)
    return encode_reduction_layout(
        tuple(AffineRecord(record.logical_shape, record.source_strides, record.source_offset,
                           tuple(stride or 1 for stride in record.destination_strides),
                           record.destination_offset) for record in records),
        output_shapes=tuple(tuple(1 if reduced else extent for extent, reduced in zip(record.logical_shape, flags))
                            for record, flags in zip(records, axes)),
        reduction_axes=axes, source_size=source_size, output_size=output_size,
    )


BATCH_BATCHES = 17


BATCH_SIZE = 8192


def _batch_layouts(accumulation):
    records = (
        AffineRecord((4,), (-1,), 7, (0,), 3),
        AffineRecord((4,), (1,), 11, (2,), 9),
        AffineRecord((2, 3), (0, 1), 30, (1, 1) if accumulation else (1, 0), 3),
    )
    return records, _encode_thread_layout(accumulation, records, BATCH_SIZE, BATCH_SIZE)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("dtype,factor_dtype", [
    ("float16", "float32"), ("float32", "float64"),
    ("complex64", "float32"), ("float64", "complex128"), ("int32", "int32"),
])
@pytest.mark.parametrize("batch_shape", [(BATCH_BATCHES,), (1, BATCH_BATCHES)])
@jax.enable_x64()
def test_batch_parallel_coefficients(accumulation, dtype, factor_dtype, batch_shape):
    records, layout = _batch_layouts(accumulation)
    source = (jnp.arange(BATCH_BATCHES * BATCH_SIZE).reshape(*batch_shape, BATCH_SIZE) % 5).astype(dtype)
    factors = (jnp.arange(BATCH_BATCHES).reshape(batch_shape) % 5 - 1).astype(factor_dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 1j
    if jnp.issubdtype(factors.dtype, jnp.complexfloating):
        factors = factors + 1j
    result_dtype = jnp.result_type(source, factors)
    operation = execute_accumulation if accumulation else execute_reduction
    execute = jax.jit(lambda data, factor: operation(
        data, (factor, jnp.asarray([2], dtype=jnp.int32)), coefficient_records=(0, 2),
        layout=layout, output_size=BATCH_SIZE, dtype=result_dtype,
    ))
    original = np.asarray(source).reshape(BATCH_BATCHES, BATCH_SIZE)
    expected = np.zeros((BATCH_BATCHES, BATCH_SIZE), dtype=result_dtype)
    flat_factors = np.asarray(factors).reshape(BATCH_BATCHES)
    for record_index, record in enumerate(records):
        coefficient = flat_factors if record_index == 0 else (2 if record_index == 2 else 1)
        for coordinate in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum(index * stride for index, stride in
                                                       zip(coordinate, record.source_strides))
            target = record.destination_offset + sum(index * stride for index, stride in
                                                     zip(coordinate, record.destination_strides))
            expected[:, target] += original[:, source_index] * coefficient
    result = execute(source, factors)
    np.testing.assert_allclose(np.asarray(result).reshape(BATCH_BATCHES, BATCH_SIZE), expected, rtol=2e-6, atol=2e-6)
    serial = np.stack([np.asarray(execute(source.reshape(BATCH_BATCHES, BATCH_SIZE)[batch],
                                         factors.reshape(BATCH_BATCHES)[batch])) for batch in range(BATCH_BATCHES)])
    np.testing.assert_array_equal(np.asarray(result).reshape(BATCH_BATCHES, BATCH_SIZE), serial)
    np.testing.assert_array_equal(np.asarray(source).reshape(BATCH_BATCHES, BATCH_SIZE), original)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("batches,source_size,output_size", [(0, BATCH_SIZE, BATCH_SIZE), (17, 0, BATCH_SIZE), (17, BATCH_SIZE, 0)])
def test_batch_empty_initialization(accumulation, batches, source_size, output_size):
    layout = (encode_layout((), source_size=source_size, output_size=output_size) if accumulation else
              encode_reduction_layout((), output_shapes=(), reduction_axes=(), source_size=source_size, output_size=output_size))
    operation = execute_accumulation if accumulation else execute_reduction
    result = jax.jit(lambda source: operation(source, layout=layout, output_size=output_size))(
        jnp.ones((batches, source_size), dtype=jnp.float32))
    np.testing.assert_array_equal(result, np.zeros((batches, output_size), dtype=np.float32))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
def test_batch_concurrent_prepared_reuse(accumulation):
    _, layout = _batch_layouts(accumulation)
    operation = execute_accumulation if accumulation else execute_reduction
    execute = jax.jit(lambda source, factor: operation(
        source, (factor,), coefficient_records=(0,), layout=layout, output_size=BATCH_SIZE))
    source = jnp.ones((BATCH_BATCHES, BATCH_SIZE), dtype=jnp.float32)
    factors = jnp.arange(BATCH_BATCHES, dtype=jnp.float32)
    compiled = execute.lower(source, factors).compile()
    expected = [np.asarray(compiled(source, factors + offset)) for offset in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda offset: np.asarray(compiled(source, factors + offset)), range(8)))
    for result, reference in zip(results, expected):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_batch_parallel_record_writeback_order():
    records = tuple(AffineRecord((), (), index, (), 2) for index in range(3))
    layout = encode_layout(records, source_size=BATCH_SIZE, output_size=BATCH_SIZE)
    source = jnp.zeros((BATCH_BATCHES, BATCH_SIZE), dtype=jnp.float32).at[:, :3].set(
        jnp.asarray([2**26, 1, -(2**26)], dtype=jnp.float32))
    result = jax.jit(lambda source: execute_accumulation(source, layout=layout, output_size=BATCH_SIZE))(source)
    np.testing.assert_array_equal(result, np.zeros((BATCH_BATCHES, BATCH_SIZE), dtype=np.float32))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_batch_parallel_zero_coefficient_skips_nonfinite_input():
    record = AffineRecord((3,), (1,), 0, (0,), 2)
    layout = encode_layout((record,), source_size=BATCH_SIZE, output_size=BATCH_SIZE)
    source = jnp.zeros((BATCH_BATCHES, BATCH_SIZE), dtype=jnp.float32).at[:, :3].set(
        jnp.asarray([jnp.inf, jnp.nan, -jnp.inf]))
    factors = jnp.arange(BATCH_BATCHES, dtype=jnp.float32) % 3
    result = jax.jit(lambda source, factor: execute_accumulation(
        source, (factor,), coefficient_records=(0,), layout=layout, output_size=BATCH_SIZE))(source, factors)
    expected = np.zeros((BATCH_BATCHES, BATCH_SIZE), dtype=np.float32)
    expected[np.asarray(factors) != 0, 2] = np.nan
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_batch_public_parallel_reduction_ad():
    source = (jnp.arange(BATCH_BATCHES * BATCH_SIZE).reshape(BATCH_BATCHES, BATCH_SIZE) % 5).astype(jnp.float32)
    indices = 4095 - 2 * np.arange(2048)
    operation = lambda data: reduce_sum(StridedView(data, (2048,), (-2,), 4095))
    reference = lambda data: jnp.sum(data[:, indices], axis=-1)
    tangent = jnp.ones_like(source)
    actual = jax.jit(lambda data, tangent: jax.jvp(operation, (data,), (tangent,)))(source, tangent)
    expected = jax.jvp(reference, (source,), (tangent,))
    for value, oracle in zip(actual, expected):
        np.testing.assert_allclose(value, oracle, rtol=2e-6, atol=2e-6)
    actual_gradient = jax.jit(jax.grad(lambda data: operation(data).sum()))(source)
    expected_gradient = jax.grad(lambda data: reference(data).sum())(source)
    np.testing.assert_array_equal(actual_gradient, expected_gradient)


SINGLE_SIZE = 200003


SINGLE_OUTPUT_SIZE = 7


SINGLE_RECORDS = (AffineRecord((SINGLE_SIZE,), (-1,), SINGLE_SIZE - 1, (0,), 3),
           AffineRecord((11,), (0,), 2, (0,), 3))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("source_dtype,result_dtype", [
    ("float32", "float32"), ("float64", "float32"), ("float16", "float64"),
    ("complex64", "complex64"), ("complex128", "float64"), ("float64", "complex128"),
])
@pytest.mark.parametrize("factor", [0., 1., 1.5])
@jax.enable_x64()
def test_single_output_reduction(accumulation, source_dtype, result_dtype, factor):
    source = ((jnp.arange(SINGLE_SIZE) % 7 - 3) * 0.125).astype(source_dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 0.25j
    original = np.array(source)
    layout = _encode_thread_layout(accumulation, SINGLE_RECORDS, SINGLE_SIZE, SINGLE_OUTPUT_SIZE)
    operation = execute_accumulation if accumulation else execute_reduction
    execute = jax.jit(lambda values, coefficient: operation(
        values, (coefficient,), coefficient_records=(0,), layout=layout,
        output_size=SINGLE_OUTPUT_SIZE, dtype=result_dtype))
    actual = execute(source, jnp.float64(factor))
    contribution = original.astype(np.complex128 if np.iscomplexobj(original) else np.float64)
    total = contribution.sum() * factor + contribution[2] * 11
    if not np.issubdtype(np.dtype(result_dtype), np.complexfloating):
        total = total.real
    expected = np.zeros(SINGLE_OUTPUT_SIZE, dtype=result_dtype)
    expected[3] = total
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(source, original)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("left_dtype,right_dtype,result_dtype", [
    ("float32", "float32", "float32"), ("float32", "float64", "float32"),
    ("complex64", "float32", "complex64"), ("float32", "complex128", "complex128"),
    ("complex128", "complex64", "float64"),
])
@jax.enable_x64()
def test_single_batch_dot(left_dtype, right_dtype, result_dtype, conjugate):
    records = (AffineRecord((SINGLE_SIZE,), (-1,), SINGLE_SIZE - 1, (1,), 0),
               AffineRecord((3,), (0,), 1, (-1,), 5))
    layout = encode_layout(records, source_size=SINGLE_SIZE, output_size=SINGLE_SIZE)
    left = ((jnp.arange(SINGLE_SIZE) % 7 - 3) * 0.125).astype(left_dtype)
    right = ((jnp.arange(SINGLE_SIZE) % 3 - 1) * 0.5).astype(right_dtype)
    if jnp.issubdtype(left.dtype, jnp.complexfloating):
        left = left + 0.25j
    if jnp.issubdtype(right.dtype, jnp.complexfloating):
        right = right + 0.125j
    original_left, original_right = np.array(left), np.array(right)
    mapped_left = np.conj(original_left) if conjugate else original_left
    expected = np.sum(mapped_left[::-1] * original_right, dtype=np.complex128)
    expected += np.sum(mapped_left[1] * original_right[5:2:-1], dtype=np.complex128)
    if not np.issubdtype(np.dtype(result_dtype), np.complexfloating):
        expected = expected.real
    actual = jax.jit(lambda left, right: execute_dot(
        left, right, layout=layout, conjugate_left=conjugate, dtype=result_dtype))(left, right)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(left, original_left)
    np.testing.assert_array_equal(right, original_right)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
def test_single_zero_record_does_not_read_or_erase_contributions(accumulation):
    records = (AffineRecord((SINGLE_SIZE,), (0,), 0, (0,), 3),
               AffineRecord((SINGLE_SIZE,), (0,), 1, (0,), 3))
    layout = _encode_thread_layout(accumulation, records, SINGLE_SIZE, SINGLE_OUTPUT_SIZE)
    source = jnp.zeros(SINGLE_SIZE, jnp.float32).at[0].set(jnp.nan).at[1].set(0.125)
    operation = execute_accumulation if accumulation else execute_reduction
    result = operation(source, (jnp.float32(0),), coefficient_records=(0,),
                       layout=layout, output_size=SINGLE_OUTPUT_SIZE)
    expected = np.zeros(SINGLE_OUTPUT_SIZE, np.float32)
    expected[3] = SINGLE_SIZE * 0.125
    np.testing.assert_array_equal(result, expected)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_single_general_overlap_falls_back_without_losing_contributions():
    record = AffineRecord((2, SINGLE_SIZE // 2), (1, 2), 0, (1, 1), 0)
    layout = encode_layout((record,), source_size=SINGLE_SIZE, output_size=SINGLE_SIZE)
    result = execute_accumulation(jnp.ones(SINGLE_SIZE, jnp.float32), layout=layout, output_size=SINGLE_SIZE)
    expected = np.zeros(SINGLE_SIZE, np.float32)
    expected[:SINGLE_SIZE // 2] += 1
    expected[1:SINGLE_SIZE // 2 + 1] += 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_single_partial_rounding_within_numerical_tolerance():
    random = np.random.default_rng(37)
    left = random.normal(size=SINGLE_SIZE).astype(np.float32) * 0.01
    right = random.normal(size=SINGLE_SIZE).astype(np.float32) * 0.01
    record = AffineRecord((SINGLE_SIZE,), (1,), 0, (1,), 0)
    layout = encode_layout((record,), source_size=SINGLE_SIZE, output_size=SINGLE_SIZE)
    result = execute_dot(jnp.asarray(left), jnp.asarray(right), layout=layout, conjugate_left=False)
    expected = np.sum(left.astype(np.float64) * right.astype(np.float64))
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=2e-6)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_single_dot_zero_input_is_not_a_coefficient_shortcut():
    record = AffineRecord((SINGLE_SIZE,), (0,), 0, (0,), 0)
    layout = encode_layout((record,), source_size=1, output_size=1)
    result = execute_dot(jnp.asarray([0.], jnp.float32), jnp.asarray([jnp.inf], jnp.float32),
                         layout=layout, conjugate_left=False)
    assert np.isnan(np.asarray(result))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_single_concurrent_prepared_reuse_and_public_ad():
    source = jnp.full(SINGLE_SIZE, 0.125, jnp.float32)
    layout = _encode_thread_layout(False, SINGLE_RECORDS, SINGLE_SIZE, SINGLE_OUTPUT_SIZE)
    operation = jax.jit(lambda values, factor: execute_reduction(
        values, (factor,), coefficient_records=(0,), layout=layout, output_size=SINGLE_OUTPUT_SIZE))
    compiled = operation.lower(source, jnp.float32(2)).compile()

    def run(factor):
        expected = np.zeros(SINGLE_OUTPUT_SIZE, np.float32)
        expected[3] = (SINGLE_SIZE * factor + 11) * 0.125
        np.testing.assert_array_equal(compiled(source, jnp.float32(factor)), expected)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(run, range(4)))
    def summed(values):
        return reduce_sum(StridedView(values, (SINGLE_SIZE,), (1,), 0))
    np.testing.assert_array_equal(jax.jit(jax.grad(summed))(source), np.ones(SINGLE_SIZE, np.float32))
    def dotted(values):
        return dotu(StridedView(values, (SINGLE_SIZE,), (1,), 0), StridedView(source, (SINGLE_SIZE,), (1,), 0))
    np.testing.assert_array_equal(jax.jit(jax.grad(dotted))(source), np.full(SINGLE_SIZE, 0.125, np.float32))


OUTPUTS_ROWS, OUTPUTS_COLUMNS = 513, 257


OUTPUTS_SIZE = OUTPUTS_ROWS * OUTPUTS_COLUMNS


OUTPUTS_OUTPUT_SIZE = 4 * OUTPUTS_ROWS + 8


OUTPUTS_DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
          "uint64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128")


OUTPUTS_RECORDS = (
    AffineRecord((OUTPUTS_ROWS, OUTPUTS_COLUMNS), (1, OUTPUTS_ROWS), 0, (-2, 0), 2 * OUTPUTS_ROWS),
    AffineRecord((OUTPUTS_ROWS, OUTPUTS_COLUMNS), (1, OUTPUTS_ROWS), 0, (2, 0), 2 * OUTPUTS_ROWS + 5),
    AffineRecord((OUTPUTS_COLUMNS, OUTPUTS_ROWS), (OUTPUTS_ROWS, 0), 0, (0, -2), 2 * OUTPUTS_ROWS),
)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("dtype", OUTPUTS_DTYPES)
@pytest.mark.parametrize("accumulation", [False, True])
@jax.enable_x64()
def test_output_partitioning_preserves_batched_dtype_contract(dtype, accumulation):
    source = (jnp.arange(OUTPUTS_SIZE) % 3).astype(dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 0.25j
    original = np.array(source)
    operation = execute_accumulation if accumulation else execute_reduction
    layout = _encode_thread_layout(accumulation, OUTPUTS_RECORDS, OUTPUTS_SIZE, OUTPUTS_OUTPUT_SIZE)
    execute = jax.jit(lambda data: operation(data, layout=layout, output_size=OUTPUTS_OUTPUT_SIZE))
    batched = np.asarray(execute(jnp.stack((source, source))))[0]
    actual = execute(source)
    np.testing.assert_array_equal(actual, batched)
    selected = np.zeros(OUTPUTS_OUTPUT_SIZE, dtype=bool)
    selected[2:2 * OUTPUTS_ROWS + 1:2] = True
    selected[2 * OUTPUTS_ROWS + 5:4 * OUTPUTS_ROWS + 4:2] = True
    np.testing.assert_array_equal(np.asarray(actual)[~selected], 0)
    np.testing.assert_array_equal(source, original)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("source_dtype,result_dtype,factor_dtype", [
    ("float64", "float32", "float64"), ("float16", "float32", "complex64"),
    ("complex64", "complex128", "float32"), ("float64", "int32", "float64"),
])
@jax.enable_x64()
def test_outputs_mixed_coefficients_and_original_record_indices(accumulation, source_dtype, result_dtype, factor_dtype):
    source = ((jnp.arange(OUTPUTS_SIZE) % 7 - 3) * 0.125).astype(source_dtype)
    if jnp.issubdtype(source.dtype, jnp.complexfloating):
        source = source + 0.125j
    operation = execute_accumulation if accumulation else execute_reduction
    layout = _encode_thread_layout(accumulation, OUTPUTS_RECORDS, OUTPUTS_SIZE, OUTPUTS_OUTPUT_SIZE)
    execute = jax.jit(lambda data, coefficient: operation(
        data, (coefficient, jnp.int32(3)), coefficient_records=(0, 2),
        layout=layout, output_size=OUTPUTS_OUTPUT_SIZE, dtype=result_dtype))
    for factor in (0, 1, 1.5):
        factor = jnp.asarray(factor, dtype=factor_dtype)
        expected = np.asarray(execute(jnp.stack((source, source)), factor))[0]
        np.testing.assert_array_equal(execute(source, factor), expected)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_outputs_record_order_and_zero_coefficient():
    rows = 40001
    records = tuple(AffineRecord((rows,), (0,), index, (1,), 0) for index in range(4))
    layout = encode_layout(records, source_size=4, output_size=rows)
    source = jnp.asarray([2**26, 1, -(2**26), jnp.nan], jnp.float32)
    actual = execute_accumulation(source, (jnp.float32(0),), coefficient_records=(3,),
                                  layout=layout, output_size=rows)
    np.testing.assert_array_equal(actual, np.zeros(rows, np.float32))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("overlapping", [False, True])
def test_outputs_interleaved_or_overlapping_output_maps_remain_legal(overlapping):
    rows = 40001
    records = (AffineRecord((rows,), (0,), 0, (2,), 0),
               AffineRecord((rows,), (0,), 1, (2,), 2 if overlapping else 1))
    output_size = 2 * rows + 2
    layout = encode_layout(records, source_size=2, output_size=output_size)
    result = execute_accumulation(jnp.asarray([2., 3.], jnp.float32), layout=layout, output_size=output_size)
    expected = np.zeros(output_size, np.float32)
    expected[:2 * rows:2] += 2
    start = 2 if overlapping else 1
    expected[start:start + 2 * rows:2] += 3
    np.testing.assert_array_equal(result, expected)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_outputs_concurrent_prepared_reuse_and_empty_record():
    records = (OUTPUTS_RECORDS[0], AffineRecord((0,), (1,), 0, (1,), 0), *OUTPUTS_RECORDS[1:])
    layout = _encode_thread_layout(True, records, OUTPUTS_SIZE, OUTPUTS_OUTPUT_SIZE)
    source = jnp.full(OUTPUTS_SIZE, 0.25, jnp.float32)
    execute = jax.jit(lambda data, coefficient: execute_accumulation(
        data, (coefficient,), coefficient_records=(3,), layout=layout, output_size=OUTPUTS_OUTPUT_SIZE))
    compiled = execute.lower(source, jnp.float32(2)).compile()

    def run(factor):
        expected = np.zeros(OUTPUTS_OUTPUT_SIZE, np.float32)
        expected[2:2 * OUTPUTS_ROWS + 1:2] = (1 + factor) * OUTPUTS_COLUMNS * 0.25
        expected[2 * OUTPUTS_ROWS + 5:4 * OUTPUTS_ROWS + 4:2] = OUTPUTS_COLUMNS * 0.25
        np.testing.assert_array_equal(compiled(source, jnp.float32(factor)), expected)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(run, range(8)))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_multioutput_public_jvp_vjp():
    source = jnp.ones(OUTPUTS_SIZE, jnp.float32)

    def operation(data):
        return reduce_sum(StridedView(data, (OUTPUTS_ROWS, OUTPUTS_COLUMNS), (OUTPUTS_COLUMNS, 1), 0), axes=(1,))

    primal, tangent = jax.jit(lambda data: jax.jvp(operation, (data,), (jnp.ones_like(data),)))(source)
    np.testing.assert_array_equal(primal, np.full(OUTPUTS_ROWS, OUTPUTS_COLUMNS, np.float32))
    np.testing.assert_array_equal(tangent, np.full(OUTPUTS_ROWS, OUTPUTS_COLUMNS, np.float32))
    cotangent = jnp.arange(OUTPUTS_ROWS, dtype=jnp.float32) % 3
    derivative = jax.jit(lambda data, cotangent: jax.vjp(operation, data)[1](cotangent)[0])(source, cotangent)
    np.testing.assert_array_equal(derivative, np.repeat(np.asarray(cotangent), OUTPUTS_COLUMNS))
