"""Native Reduction initialization, record carry and execution."""

from itertools import permutations, product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, reduce_sum
from tensor0._stride._ffi._calls import execute_reduction
from tensor0._stride._ffi._descriptor import encode_reduction_layout
from tensor0._stride._jax import accumulation_p, reduction_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.ffi import REDUCTION_FIBER, _encode_reduction_records, reduction_layout
from tests.stride.support.oracles.generic import assert_close as generic_assert_close, thread_limit
from tests.stride.support.oracles.reduction import (
    addresses,
    assert_close as reduction_assert_close,
    reference_sum,
)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_overlapping_reduction_records_initialize_once():
    count, width = 32768, 4
    size = count * width
    records = tuple(AffineRecord((count, width), (width, 1), offset, (1, 0), 0) for offset in (0, size))
    source = jnp.arange(2 * size, dtype=jnp.float32) - 5
    operation = jax.jit(lambda value: accumulation_p.bind(value, jnp.float32(2), jnp.float32(-3),
        records=records, coefficient_records=(0, 1), output_size=count, dtype=value.dtype))
    expected = 2 * source[:size].reshape(count, width).sum(axis=1) - 3 * source[size:].reshape(count, width).sum(axis=1)
    for limit in (1, 4):
        with thread_limit(limit):
            generic_assert_close(operation(source), expected)


SIGNED_CASES = [(order, signs, reduced) for order in permutations(range(2))
                for signs in product((-1, 1), repeat=2) for reduced in ((True, False), (False, True), (True, True))]


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("order,signs,axes", SIGNED_CASES)
def test_signed_permuted_reduction_layouts(order, signs, axes):
    shape = (2, 3)
    absolute = [0, 0]
    stride = 1
    for axis in order:
        absolute[axis] = stride
        stride *= shape[axis]
    strides = tuple(sign * value for sign, value in zip(signs, absolute, strict=True))
    offset = sum((size - 1) * step for size, step in zip(shape, strides, strict=True) if step < 0) * -1
    output_shape = tuple(1 if reduced else size for reduced, size in zip(axes, shape, strict=True))
    output_strides = (-output_shape[1], -1)
    output_offset = int(np.prod(output_shape))
    record = AffineRecord(shape, strides, offset, output_strides, output_offset)
    parameters = dict(records=(record,), output_shapes=(output_shape,), reduction_axes=(axes,),
                      output_size=output_offset + 2, dtype=np.dtype(jnp.float32))
    source = jnp.arange(6, dtype=jnp.float32)
    actual = jax.jit(lambda value: reduction_p.bind(value, **parameters))(source)
    reduction_assert_close(actual, reference_sum(source, (None,), **parameters))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                                   jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64])
def test_integer_reduction_promotion_and_storage_conversion(dtype):
    with jax.enable_x64():
        source = jnp.arange(12, dtype=jnp.int32).astype(dtype)
        if dtype == jnp.bool_:
            source = jnp.arange(12) % 2 == 0
        factor = jnp.asarray(True if dtype == jnp.bool_ else 3 if jnp.issubdtype(dtype, jnp.unsignedinteger) else -3, dtype=dtype)
        parameters = dict(records=(AffineRecord((4, 3), (3, 1), 0, (3, 1), 0),),
                          output_shapes=((1, 3),), reduction_axes=((True, False),), output_size=3, dtype=np.dtype(dtype))
        actual = jax.jit(lambda value: reduction_p.bind(value, factor, coefficient_records=(0,), **parameters))(source)
        expected = (source.reshape(4, 3) * factor).sum(axis=0, dtype=dtype)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("shape,output_shape,axes,source_size", [
    ((2, 0), (2, 1), (False, True), 0), ((0, 3), (0, 1), (False, True), 0),
    ((2, 3), (2, 1), (False, True), 6),
])
@pytest.mark.parametrize("batch_count", [0, 2])
def test_empty_reduction_and_partial_output_initialize_once(shape, output_shape, axes, source_size, batch_count):
    parameters = dict(records=(AffineRecord(shape, (3, 1), 0, (2, 1), 1),),
                      output_shapes=(output_shape,), reduction_axes=(axes,), output_size=6, dtype=np.dtype(jnp.float32))
    source = jnp.ones((batch_count, source_size), dtype=jnp.float32)
    actual = jax.jit(lambda value: reduction_p.bind(value, **parameters))(source)
    reduction_assert_close(actual, reference_sum(source, (None,), **parameters))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_record_coefficients_keep_types_and_bind_independently(dtype):
    with jax.enable_x64():
        records = tuple(AffineRecord((3, 128), (128, 1), index * 384, (1, 1), 0) for index in range(3))
        parameters = dict(records=records, output_shapes=((3, 1),) * 3, reduction_axes=((False, True),) * 3,
                          output_size=3, dtype=np.dtype(jnp.float32))
        source = jnp.stack((jnp.ones(1152, dtype=dtype), jnp.full(1152, 2, dtype=dtype)))
        compiled = jax.jit(lambda values, first, second: reduction_p.bind(
            values, first, second, coefficient_records=(1, 2), **parameters))
        for first, second in ((.75, -.25), (2., -.5)):
            actual = compiled(source, jnp.float32(first), jnp.float64(second))
            expected = np.broadcast_to(np.asarray([[1.], [2.]], dtype=np.float32) * 128 * (1 + first + second), (2, 3))
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_each_record_uses_its_own_batch_coefficients():
    records = (AffineRecord((2, 3), (3, 1), 0, (1, 1), 0), AffineRecord((2, 3), (3, 1), 6, (1, 1), 0))
    parameters = dict(records=records, output_shapes=((2, 1),) * 2, reduction_axes=((False, True),) * 2,
                      output_size=2, dtype=np.dtype(jnp.float32))
    source = jnp.arange(36, dtype=jnp.float32).reshape(3, 12)
    first, second = jnp.asarray([0., 1., .5]), jnp.asarray([1., 0., -2.])
    actual = jax.jit(lambda values, alpha, beta: reduction_p.bind(
        values, alpha, beta, coefficient_records=(0, 1), **parameters))(source, first, second)
    expected = source[:, :6].reshape(3, 2, 3).sum(axis=-1) * first[:, None]
    expected += source[:, 6:].reshape(3, 2, 3).sum(axis=-1) * second[:, None]
    reduction_assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("order,expected", [((0, 1, 2, 3), 3.), ((0, 2, 1, 3), 4.)])
def test_record_order_sets_cross_record_rounding_boundary(order, expected):
    records = tuple(AffineRecord((1,), (1,), index, (1,), 0) for index in order)
    source = jnp.asarray([2**24, 1, -(2**24), 3], dtype=jnp.float32)
    actual = jax.jit(lambda value: reduction_p.bind(
        value, records=records, output_shapes=((1,),) * 4, reduction_axes=((True,),) * 4,
        output_size=1, dtype=value.dtype))(source)
    np.testing.assert_array_equal(actual, [expected])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [0., 1., -.75])
def test_nonfinite_coefficient_branch_does_not_clear_other_records(factor):
    records = (AffineRecord((2,), (1,), 0, (1,), 1), AffineRecord((2,), (1,), 2, (1,), 1))
    parameters = dict(records=records, output_shapes=((1,),) * 2, reduction_axes=((True,),) * 2,
                      output_size=3, dtype=np.dtype(jnp.float32))
    source = jnp.asarray([2., 3., np.inf, np.nan])
    actual = jax.jit(lambda value, coefficient: reduction_p.bind(value, coefficient, coefficient_records=(1,), **parameters))(
        source, jnp.float32(factor))
    np.testing.assert_array_equal(actual[jnp.asarray([0, 2])], 0)
    if factor == 0:
        assert actual[1] == 5
    else:
        assert np.isnan(actual[1])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("output_count", [1, 3, 9, 17])
def test_reduction_output_rows_and_tails_finite_accuracy(dtype, output_count):
    rng = np.random.default_rng(20260917)
    source = rng.standard_normal((127, output_count)).astype(np.float32) / 16
    if dtype == jnp.complex64:
        source = source + 1j * source[::-1]
    storage = jnp.asarray(source.ravel(), dtype=dtype)
    actual = jax.jit(lambda value: reduce_sum(StridedView(value, source.shape, (output_count, 1), 0), (0,)))(storage)
    expected = jnp.asarray(source.astype(np.complex128 if dtype == jnp.complex64 else np.float64).sum(axis=0), dtype=dtype)
    reduction_assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("special", [np.inf, np.nan])
def test_long_reduction_preserves_nonfinite_contributions(special):
    source = jnp.ones(65536, dtype=jnp.float32).at[32767].set(special)
    actual = jax.jit(lambda value: reduce_sum(StridedView(value, (65536,), (1,), 0)))(source)
    if np.isnan(special):
        assert np.isnan(actual)
    else:
        assert np.isposinf(actual)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_reduction_rank_above_eight_is_not_rejected():
    shape, strides = (2,) * 9, tuple(3**axis for axis in range(8, -1, -1))
    source = jnp.arange(sum(strides) + 1, dtype=jnp.float32) % 8
    actual = jax.jit(lambda value: reduce_sum(StridedView(value, shape, strides, 0), (0, 2, 4, 6, 8)))(source)
    expected = source[addresses(shape, strides, 0)].sum(axis=(0, 2, 4, 6, 8))
    reduction_assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_reduction_batches_zero_record_preserves_prior_contributions_per_batch():
    records = tuple(dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index, output_offset=1)
                    for index in range(3))
    layout = _encode_reduction_records(records, source_size=3, output_size=3)
    source = jnp.asarray([[7, np.nan, 3], [7, 2, np.inf], [7, 2, 3]], dtype=jnp.float32)
    result = execute_reduction(
        source, (jnp.asarray([0, 1, 2], dtype=jnp.float16), jnp.asarray([1, 0, 2], dtype=jnp.int32)),
        coefficient_records=(1, 2), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, [[0, 10, 0], [0, 9, 0], [0, 17, 0]])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("batch_shape", [(3,), (0,), (2, 0)])
def test_reduction_batches_empty_reduction_with_per_batch_coefficients(batch_shape):
    record = dict(REDUCTION_FIBER, source_shape=(0,), output_offset=2)
    layout = _encode_reduction_records((record,), source_size=0, output_size=3)
    result = execute_reduction(jnp.zeros((*batch_shape, 0)), (jnp.full(batch_shape, np.nan),),
                               coefficient_records=(0,), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, np.zeros((*batch_shape, 3)))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_reduction_batches_per_batch_fused_rounding_and_reused_coefficient_buffers():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    source = jnp.asarray([[2048, 1, -2048]] * 3, dtype=jnp.float16)
    execute = jax.jit(lambda data, factors: execute_reduction(
        data, (factors,), coefficient_records=(0,), layout=layout, output_size=1))
    for factors, expected in (([0, 1, 1.5], [0, 0, 1.5]), ([1.5, 0, 1], [1.5, 0, 0])):
        np.testing.assert_array_equal(execute(source, jnp.asarray(factors, dtype=jnp.float32)),
                                      np.asarray(expected)[:, None])


requires_native = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


@requires_native
@pytest.mark.parametrize("empty", [False, True])
def test_reduction_initializes_output_independently_of_input_size(empty):
    count = 0 if empty else 12
    source = jnp.arange(count, dtype=jnp.float32)
    layout = reduction_layout(source_shape=(0, 3) if empty else (4, 3), source_size=count)
    actual = execute_reduction(source, layout=layout, output_size=3)
    np.testing.assert_array_equal(actual, np.zeros(3) if empty else np.arange(12).reshape(4, 3).sum(axis=0))


@requires_native
def test_multirecord_reduction_uses_one_initialization_without_mode_flags():
    merged = encode_reduction_layout(
        tuple(AffineRecord((4, 3), (3, 1), offset, (3, 1), 0) for offset in (0, 12)),
        output_shapes=((1, 3),) * 2, reduction_axes=((True, False),) * 2,
        source_size=24, output_size=3,
    )
    assert merged.view("<u8")[3] == 2
    source = jnp.arange(24, dtype=jnp.float32)
    np.testing.assert_array_equal(execute_reduction(source, layout=merged, output_size=3),
                                  np.arange(24).reshape(8, 3).sum(axis=0))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_record_continuation_and_local_writeback():
    with jax.enable_x64():
        records = (dict(REDUCTION_FIBER, source_shape=(1,)), dict(REDUCTION_FIBER, source_shape=(2,), source_offset=1))
        layout = _encode_reduction_records(records, source_size=3, output_size=1)
        source = jnp.asarray([1, 1e20, -1e20], dtype=jnp.float32)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1), [0])
        source = jnp.asarray([2**24, 1, -(2**24)], dtype=jnp.float64)
        local = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
        separate = _encode_reduction_records(tuple(
            dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index) for index in range(3)
        ), source_size=3, output_size=1)
        np.testing.assert_array_equal(execute_reduction(source, layout=local, output_size=1, dtype="float32"), [1])
        np.testing.assert_array_equal(execute_reduction(source, layout=separate, output_size=1, dtype="float32"), [0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_shared_planner_order_counterexample():
    record = dict(source_shape=(3, 3, 2), source_strides=(2, 1, 3), source_offset=0, output_shape=(1, 1, 1), output_strides=(1, 1, 1), output_offset=0, reduction_axes=(True, True, True))
    layout = _encode_reduction_records((record,), source_size=10, output_size=1)
    source = jnp.asarray([0, 0, 1e20, -1e20, 1, 0, 0, 0, 0, 0], dtype=jnp.float32)
    np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1), [1])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_nonzero_destination_stride_writes_back_each_sum():
    with jax.enable_x64():
        record = dict(source_shape=(2,), source_strides=(1,), source_offset=0, output_shape=(2,), output_strides=(1,), output_offset=0, reduction_axes=(False,))
        records = (record, dict(record, source_offset=2))
        layout = _encode_reduction_records(records, source_size=4, output_size=2)
        source = jnp.asarray([-1, -1, 1 + 2**-24, 1 + 2**-24], dtype=jnp.float64)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=2, dtype="float32"),
                                      [np.float32(2**-24)] * 2)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_broadcast_and_negative_output_stride():
    record = dict(source_shape=(2, 3), source_strides=(1, 0), source_offset=0, output_shape=(2, 1), output_strides=(-1, 0), output_offset=2, reduction_axes=(False, True))
    layout = _encode_reduction_records((record,), source_size=2, output_size=4)
    np.testing.assert_array_equal(execute_reduction(jnp.asarray([1, 2], dtype=jnp.int32),
                                                   layout=layout, output_size=4), [0, 6, 3, 0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("batch_shape", [(), (2,), (0,), (2, 0)])
def test_empty_input_preserves_real_output_initialization(batch_shape):
    record = dict(REDUCTION_FIBER, source_shape=(0,), output_offset=2, output_strides=(2**63 - 1,))
    layout = _encode_reduction_records((record,), source_size=0, output_size=3)
    source = jnp.zeros((*batch_shape, 0), dtype=jnp.float32)
    result = execute_reduction(source, (jnp.float32(np.nan),), coefficient_records=(0,), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, np.zeros((*batch_shape, 3)))
    no_records = _encode_reduction_records((), source_size=0, output_size=3)
    np.testing.assert_array_equal(execute_reduction(source, layout=no_records, output_size=3), result)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_zero_output_and_large_unsigned_extent():
    empty = dict(source_shape=(0, 2), source_strides=(2, 1), source_offset=0, output_shape=(0, 1), output_strides=(1, 1), output_offset=0, reduction_axes=(False, True))
    layout = _encode_reduction_records((empty,), source_size=0, output_size=0)
    assert execute_reduction(jnp.zeros(0), layout=layout, output_size=0).shape == (0,)
    huge = dict(REDUCTION_FIBER, source_shape=(2**63,), source_strides=(0,), output_strides=(-(2**63),))
    layout = _encode_reduction_records((huge,), source_size=1, output_size=1)
    assert execute_reduction(jnp.zeros((0, 1)), layout=layout, output_size=1).shape == (0, 1)
