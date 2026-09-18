from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product
from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, enable_threads, get_num_threads, reduce_sum, set_num_threads
from tensor0._stride import _jax
from tensor0._stride._ffi._calls import execute_reduction
from tensor0._stride._ffi._descriptor import encode_reduction_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import copy_p, reduction_p
from tensor0._stride._layout import AffineRecord, contiguous_strides
from tensor0._stride._tensor_ops import _strided_tensortrace

from ._support import native_available
from .test_reduction_ad import scalar_trace


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


def addresses(shape, strides, offset):
    indices = np.full(shape, offset, dtype=np.int64)
    for axis, (size, stride) in enumerate(zip(shape, strides, strict=True)):
        broadcast_shape = (1,) * axis + (size,) + (1,) * (len(shape) - axis - 1)
        indices += np.arange(size).reshape(broadcast_shape) * stride
    return indices


def reference_sum(source, factors, *, records, output_shapes, reduction_axes, output_size, dtype):
    result = jnp.zeros((*source.shape[:-1], output_size), dtype=dtype)
    for record, shape, axes, factor in zip(records, output_shapes, reduction_axes, factors, strict=True):
        values = source[..., addresses(record.logical_shape, record.source_strides, record.source_offset)]
        if factor is not None:
            values = values * factor
        summed = jnp.sum(values, axis=tuple(source.ndim - 1 + axis for axis, reduced in enumerate(axes) if reduced), keepdims=True)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            summed = jnp.real(summed)
        destinations = addresses(shape, record.destination_strides, record.destination_offset).ravel()
        result = result.at[..., destinations].add(summed.astype(dtype).reshape(*source.shape[:-1], destinations.size))
    return result


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if jnp.issubdtype(actual.dtype, jnp.integer) or actual.dtype == jnp.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = .008 if actual.dtype == jnp.bfloat16 else .002 if actual.dtype == jnp.float16 else 2e-5
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex128), np.asarray(expected).astype(np.complex128),
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("strides,offset,source_size", [((3, 1), 0, 9), ((1, 1), 0, 5),
                                                       ((1, 0), 0, 3), ((-1, 0), 2, 3)])
@pytest.mark.parametrize("axes", [(0,), (1,), (0, 1)])
def test_public_reduction_repeated_reads_forward_jvp_vjp(strides, offset, source_size, axes):
    source = jnp.arange(source_size, dtype=jnp.float32) / 4
    indices = addresses((3, 3), strides, offset)
    native = lambda value: reduce_sum(StridedView(value, (3, 3), strides, offset), axes)
    reference = lambda value: jnp.sum(value[indices], axis=axes)
    actual = jax.jit(native)(source)
    assert_close(actual, reference(source))
    cotangent = jnp.ones_like(actual) * .5
    direction = jnp.ones_like(source) * .25
    for result, wanted in zip(jax.jvp(native, (source,), (direction,)), jax.jvp(reference, (source,), (direction,)), strict=True):
        assert_close(result, wanted)
    assert_close(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])


SIGNED_CASES = [(order, signs, reduced) for order in permutations(range(2))
                for signs in product((-1, 1), repeat=2) for reduced in ((True, False), (False, True), (True, True))]


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
    assert_close(actual, reference_sum(source, (None,), **parameters))


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32), (jnp.bfloat16, jnp.float32),
    (jnp.float16, jnp.float16), (jnp.bfloat16, jnp.bfloat16),
    (jnp.float32, jnp.float32), (jnp.float64, jnp.float64), (jnp.complex64, jnp.complex64),
    (jnp.complex128, jnp.complex128), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32)])
def test_mixed_reduction_forward_coefficient_ad_batch_and_nested_transpose(source_dtype, result_dtype):
    with jax.enable_x64():
        parameters = dict(records=(AffineRecord((4, 3), (3, 1), 0, (3, 1), 0),),
                          output_shapes=((1, 3),), reduction_axes=((True, False),), output_size=3, dtype=np.dtype(result_dtype))
        source = (jnp.arange(12, dtype=jnp.float32) / 8 - .5).astype(source_dtype)
        if jnp.issubdtype(source_dtype, jnp.complexfloating):
            source = source * (1 + .5j)
        factor = jnp.asarray(.75 - .5j if jnp.issubdtype(result_dtype, jnp.complexfloating) else -.75,
                             dtype=result_dtype)
        native = lambda value, coefficient: reduction_p.bind(value, coefficient, coefficient_records=(0,), **parameters)
        reference = lambda value, coefficient: reference_sum(value, (coefficient,), **parameters)
        cotangent = jnp.ones(3, dtype=result_dtype) * .5
        directions = (jnp.ones_like(source) * .25, jnp.ones_like(factor) * .5)
        for function in (lambda operation: jax.jvp(operation, (source, factor), directions),
                         lambda operation: jax.vjp(operation, source, factor)[1](cotangent)):
            actual, expected = function(native), function(reference)
            for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                assert_close(result, wanted)
        sources = jnp.stack((source, source * 2))
        assert_close(jax.jit(jax.vmap(native, in_axes=(0, None)))(sources, factor),
                     jax.vmap(reference, in_axes=(0, None))(sources, factor))
        pullback = lambda cot: jax.vjp(lambda value: native(value, factor), source)[1](cot)[0]
        expected_pullback = lambda cot: jax.vjp(lambda value: reference(value, factor), source)[1](cot)[0]
        assert_close(jax.jit(lambda cot: jax.linear_transpose(pullback, cotangent)(cot)[0])(jnp.ones_like(source)),
                     jax.linear_transpose(expected_pullback, cotangent)(jnp.ones_like(source))[0])


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
    assert_close(actual, reference_sum(source, (None,), **parameters))


@pytest.mark.parametrize("axes", [(False, False), (False, True)])
def test_overlapping_records_different_fibers_and_output_selections(axes):
    reduced = axes[1]
    records = (AffineRecord((4, 2 if reduced else 1), (2, 1), 0, (-2, 1), 7),
               AffineRecord((4, 3 if reduced else 1), (3, 1), 8, (-2, 1), 7),
               AffineRecord((2, 1), (1, 1), 20, (2, 1), 2))
    parameters = dict(records=records, output_shapes=((4, 1), (4, 1), (2, 1)),
                      reduction_axes=(axes, axes, axes), output_size=10, dtype=np.dtype(jnp.float32))
    factors = (jnp.float32(1.25), jnp.float32(-.5), jnp.float32(2))
    native = lambda value: reduction_p.bind(value, *factors, coefficient_records=(0, 1, 2), **parameters)
    reference = lambda value: reference_sum(value, factors, **parameters)
    source = jnp.arange(22, dtype=jnp.float32) / 4
    compiled = jax.jit(native).lower(source).compile()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda shift: compiled(source + shift), range(4)))
    for shift, actual in enumerate(outputs):
        assert_close(actual, reference(source + shift))
    cotangent = jnp.linspace(-1, 1, 10)
    assert_close(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])
    assert_close(jax.jit(jax.vmap(native))(jnp.stack((source, source * 2))),
                 jax.vmap(reference)(jnp.stack((source, source * 2))))


def test_many_small_output_groups_share_one_native_call():
    records = tuple(AffineRecord((1, 4), (4, 1), (2 * group + term) * 4, (1, 1), group)
                    for group in range(1024) for term in range(2))
    source = jnp.ones(8192, dtype=jnp.float32)
    parameters = dict(records=records, output_shapes=((1, 1),) * len(records),
                      reduction_axes=((False, True),) * len(records), output_size=1024, dtype=np.dtype(jnp.float32))
    compiled = jax.jit(lambda value: reduction_p.bind(value, **parameters))
    np.testing.assert_array_equal(compiled(source), np.full(1024, 8, dtype=np.float32))
    assert compiled.lower(source).as_text().count("custom_call") == 1


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
    assert_close(actual, expected)


@pytest.mark.parametrize("order,expected", [((0, 1, 2, 3), 3.), ((0, 2, 1, 3), 4.)])
def test_record_order_sets_cross_record_rounding_boundary(order, expected):
    records = tuple(AffineRecord((1,), (1,), index, (1,), 0) for index in order)
    source = jnp.asarray([2**24, 1, -(2**24), 3], dtype=jnp.float32)
    actual = jax.jit(lambda value: reduction_p.bind(
        value, records=records, output_shapes=((1,),) * 4, reduction_axes=((True,),) * 4,
        output_size=1, dtype=value.dtype))(source)
    np.testing.assert_array_equal(actual, [expected])


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
    assert_close(actual, expected)


@pytest.mark.parametrize("special", [np.inf, np.nan])
def test_long_reduction_preserves_nonfinite_contributions(special):
    source = jnp.ones(65536, dtype=jnp.float32).at[32767].set(special)
    actual = jax.jit(lambda value: reduce_sum(StridedView(value, (65536,), (1,), 0)))(source)
    if np.isnan(special):
        assert np.isnan(actual)
    else:
        assert np.isposinf(actual)


def test_reduction_rank_above_eight_is_not_rejected():
    shape, strides = (2,) * 9, tuple(3**axis for axis in range(8, -1, -1))
    source = jnp.arange(sum(strides) + 1, dtype=jnp.float32) % 8
    actual = jax.jit(lambda value: reduce_sum(StridedView(value, shape, strides, 0), (0, 2, 4, 6, 8)))(source)
    expected = source[addresses(shape, strides, 0)].sum(axis=(0, 2, 4, 6, 8))
    assert_close(actual, expected)


def test_broadcast_map_transpose_accumulates_and_supports_nested_ad():
    record = AffineRecord((3, 4), (0, -1), 3, (-1, 3), 2)
    source = jnp.arange(4, dtype=jnp.float32)
    native = lambda value: copy_p.bind(value, records=(record,), output_size=12, dtype=value.dtype)
    reference = lambda value: jnp.zeros(12).at[addresses((3, 4), (-1, 3), 2)].set(value[addresses((3, 4), (0, -1), 3)])
    native_pullback = lambda cot: jax.vjp(native, source)[1](cot)[0]
    reference_pullback = lambda cot: jax.vjp(reference, source)[1](cot)[0]
    cotangent = jnp.arange(36, dtype=jnp.float32).reshape(3, 12) / 4
    assert_close(jax.jit(jax.vmap(native_pullback))(cotangent), jax.vmap(reference_pullback)(cotangent))
    assert_close(jax.linear_transpose(native_pullback, cotangent[0])(jnp.ones_like(source))[0],
                 jax.linear_transpose(reference_pullback, cotangent[0])(jnp.ones_like(source))[0])


@pytest.mark.parametrize("invalid,message", [("version", "unsupported native reduction layout version"),
    ("axes", "axis flag is not boolean"), ("output_shape", "shapes do not match axis roles"),
    ("dtype", "dtype|element type")])
def test_reduction_ffi_rejects_invalid_protocol_or_typed_output(invalid, message):
    layout = _encode_reduction_records((dict(source_shape=(2, 3), source_strides=(3, 1), source_offset=0, output_shape=(2, 1), output_strides=(1, 1), output_offset=0, reduction_axes=(False, True)),), source_size=6, output_size=2)
    words = layout.view("<u8")
    if invalid == "version":
        words[0] = 99
    elif invalid == "axes":
        words[-1] = 2
    elif invalid == "output_shape":
        words[12] = 3
    dtype = jnp.complex64 if invalid == "dtype" else jnp.float32
    call = jax.ffi.ffi_call(operation_target("reduction", np.dtype(jnp.float32)), jax.ShapeDtypeStruct((1, 2), dtype))
    with pytest.raises(Exception, match=message):
        call(jnp.ones((1, 6), dtype=jnp.float32), layout=layout,
             coefficient_records=np.asarray([], dtype=np.int64)).block_until_ready()


def test_reduction_non_cpu_lowering_fails_closed():
    operation = jax.jit(lambda value: reduce_sum(StridedView(value, (2, 3), (3, 1), 0)))
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="tpu|TPU"):
        operation.trace(jax.ShapeDtypeStruct((6,), jnp.float32)).lower(lowering_platforms=("tpu",))


REDUCTION_DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
)


REDUCTION_FIBER = dict(source_shape=(3,), source_strides=(1,), source_offset=0, output_shape=(1,), output_strides=(1,), output_offset=0, reduction_axes=(True,))



def _encode_reduction_records(records, *, source_size, output_size):
    """Encode dictionary-based reduction fixtures in one batch."""
    return encode_reduction_layout(
        tuple(AffineRecord(record["source_shape"], record["source_strides"], record["source_offset"],
                           record["output_strides"], record["output_offset"]) for record in records),
        output_shapes=tuple(record["output_shape"] for record in records),
        reduction_axes=tuple(record["reduction_axes"] for record in records),
        source_size=source_size, output_size=output_size,
    )


def _raw_reduction_call(source, layout, *, shape=(1, 1), dtype="float32", coefficients=(), indices=(), aliases=None):
    execute = jax.ffi.ffi_call(
        operation_target("reduction", jnp.dtype(dtype)), jax.ShapeDtypeStruct(shape, dtype),
        input_output_aliases=aliases,
    )
    return execute(source, *coefficients, layout=layout, coefficient_records=np.asarray(indices, dtype=np.int64))



@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("axes", [None, (), (0,), (1,), (-1,), (1, 0)])
@pytest.mark.parametrize("shape,strides,offset,storage_size", [
    ((2, 3), (3, 1), 0, 6),
    ((2, 3), (1, 2), 1, 8),
    ((2, 3), (-3, -1), 6, 8),
    ((2, 3), (0, 1), 1, 4),
    ((0, 3), (3, 1), 0, 0),
    ((2, 0), (1, 1), 0, 0),
])
def test_public_sum_layout_axes_and_batches(batch_shape, axes, shape, strides, offset, storage_size):
    source = jnp.arange(prod(batch_shape) * storage_size, dtype=jnp.float32).reshape(
        (*batch_shape, storage_size))
    view = StridedView(source, shape, strides, offset)
    indices = np.asarray([
        offset + sum(coordinate * stride for coordinate, stride in zip(index, strides))
        for index in np.ndindex(shape)
    ], dtype=np.int64).reshape(shape)
    selected = np.asarray(source)[..., indices]
    logical_axes = tuple(range(len(shape))) if axes is None else tuple(axis % len(shape) for axis in axes)
    expected = selected.sum(axis=tuple(len(batch_shape) + axis for axis in logical_axes))
    for execute in (lambda bound: reduce_sum(bound, axes), jax.jit(lambda bound: reduce_sum(bound, axes))):
        actual = execute(view)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
])
def test_public_sum_default_dtype_resolution(dtype, x64):
    with jax.enable_x64(x64):
        concrete = jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        source = jnp.asarray([0, 1, 2, 3], dtype=concrete)
        actual = reduce_sum(StridedView.from_dense(source, (2, 2)))
        expected = jnp.sum(source)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("axes", [None, ()])
def test_public_sum_scalar_view(axes):
    source = jnp.asarray([[5, 7, 9], [11, 13, 15]], dtype=jnp.int8)
    actual = reduce_sum(StridedView(source, (), (), 1), axes)
    np.testing.assert_array_equal(actual, [7, 13])
    assert actual.dtype == jnp.sum(source[0, 1]).dtype



def test_public_sum_real_output_layout(monkeypatch):
    encode = _jax.encode_reduction_layout
    captured = []

    def capture(records, **fields):
        captured.append(dict(records=records, **fields))
        return encode(records, **fields)

    monkeypatch.setattr(_jax, "encode_reduction_layout", capture)
    view = StridedView.from_dense(jnp.arange(24, dtype=jnp.float32), (2, 3, 4))
    actual = reduce_sum(view, (1,))
    np.testing.assert_array_equal(actual, np.arange(24).reshape(2, 3, 4).sum(axis=1))
    assert len(captured) == 1
    assert captured[0]["output_shapes"] == ((2, 1, 4),)
    assert captured[0]["records"][0].destination_strides == (4, 4, 1)
    assert captured[0]["reduction_axes"] == ((False, True, False),)



def test_public_sum_dtype_does_not_preconvert_source():
    view = StridedView.from_dense(jnp.asarray([1.75, 1.75]), (2,))
    np.testing.assert_array_equal(reduce_sum(view, dtype=jnp.int32), 3)
    np.testing.assert_array_equal(jnp.sum(view.data, dtype=jnp.int32), 2)



def test_public_sum_low_precision_uses_native_accumulation():
    view = StridedView.from_dense(jnp.asarray([2048, 1, -2048], dtype=jnp.float16), (3,))
    np.testing.assert_array_equal(reduce_sum(view), 0)
    np.testing.assert_array_equal(jnp.sum(view.data), 1)
    np.testing.assert_array_equal(reduce_sum(view, dtype=jnp.float32), 1)



def test_public_sum_empty_axes_do_not_add_zero():
    view = StridedView.from_dense(jnp.asarray([-0.0], dtype=jnp.float32), (1,))
    assert np.signbit(np.asarray(reduce_sum(view, ())))[0]



def test_public_sum_explicit_dtype_respects_x64():
    with jax.enable_x64(False):
        view = StridedView.from_dense(jnp.asarray([1, 2], dtype=jnp.float32), (2,))
        assert reduce_sum(view, dtype=jnp.float64).dtype == jnp.float32



def test_public_sum_high_rank_singleton_layout():
    view = StridedView(jnp.asarray([7], dtype=jnp.int32), (1,) * 12, ((1 << 63) - 1,) * 12, 0)
    np.testing.assert_array_equal(reduce_sum(view), 7)



def test_public_sum_vmap_and_native_lowering():
    execute = lambda source: reduce_sum(StridedView.from_dense(source, (2, 3)), (1,))
    source = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3)
    actual = jax.jit(jax.vmap(execute))(source)
    np.testing.assert_array_equal(actual, np.asarray(source).sum(axis=2))
    lowered = str(jax.jit(execute).lower(source[0]).compiler_ir())
    assert "tensor0_stride_reduction_f32_cpu_v1" in lowered



@pytest.mark.parametrize("axes,error", [((True,), TypeError), ((0.0,), TypeError),
                                       ((2,), ValueError), ((-3,), ValueError),
                                       ((1, -1), ValueError)])
def test_public_sum_invalid_axes(axes, error):
    view = StridedView.from_dense(jnp.ones((2, 3)), (2, 3))
    with pytest.raises(error):
        reduce_sum(view, axes)



def test_public_sum_unsupported_calls_fail_explicitly():
    with pytest.raises(TypeError, match="StridedView"):
        reduce_sum(jnp.ones(3))



def test_public_sum_low_precision_complex_result_ad():
    primal, tangent = jax.jvp(
        lambda source: reduce_sum(StridedView.from_dense(source, (3,)), dtype="complex64"),
        (jnp.ones(3, dtype=jnp.float16),), (jnp.ones(3, dtype=jnp.float16),))
    assert primal.dtype == tangent.dtype == jnp.complex64
    np.testing.assert_array_equal(primal, 3)
    np.testing.assert_array_equal(tangent, 3)



@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("shared", ["scalar", "singleton", "batch"])
def test_reduction_batches_coefficient_shapes_and_types(dtype, batch_shape, shared):
    with jax.enable_x64():
        source = jnp.broadcast_to(jnp.asarray([1, 2, 3], dtype=jnp.float32), (*batch_shape, 3))
        factors = jnp.asarray(np.arange(prod(batch_shape)).reshape(batch_shape) % 3, dtype=dtype)
        if shared != "batch":
            factors = jnp.asarray(1 if shared == "scalar" else [2], dtype=dtype)
        layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
        execute = jax.jit(lambda data, coefficient: execute_reduction(
            data, (coefficient,), coefficient_records=(0,), layout=layout, output_size=1))
        result = execute(source, factors)
        expected_factor = np.asarray(factors).reshape(()) if shared != "batch" else np.asarray(factors)
        expected = np.broadcast_to(6 * expected_factor, batch_shape)[..., None]
        np.testing.assert_array_equal(result, expected)
        text = execute.lower(source, factors).as_text()
        assert text.count("stablehlo.custom_call @tensor0_stride_reduction_f32_cpu_v1") == 1



def test_reduction_batches_zero_record_preserves_prior_contributions_per_batch():
    records = tuple(dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index, output_offset=1)
                    for index in range(3))
    layout = _encode_reduction_records(records, source_size=3, output_size=3)
    source = jnp.asarray([[7, np.nan, 3], [7, 2, np.inf], [7, 2, 3]], dtype=jnp.float32)
    result = execute_reduction(
        source, (jnp.asarray([0, 1, 2], dtype=jnp.float16), jnp.asarray([1, 0, 2], dtype=jnp.int32)),
        coefficient_records=(1, 2), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, [[0, 10, 0], [0, 9, 0], [0, 17, 0]])



@pytest.mark.parametrize("batch_shape", [(3,), (0,), (2, 0)])
def test_reduction_batches_empty_reduction_with_per_batch_coefficients(batch_shape):
    record = dict(REDUCTION_FIBER, source_shape=(0,), output_offset=2)
    layout = _encode_reduction_records((record,), source_size=0, output_size=3)
    result = execute_reduction(jnp.zeros((*batch_shape, 0)), (jnp.full(batch_shape, np.nan),),
                               coefficient_records=(0,), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, np.zeros((*batch_shape, 3)))



def test_reduction_batches_invalid_coefficient_batch_shapes():
    layout = _encode_reduction_records((dict(REDUCTION_FIBER, source_shape=(1,)),), source_size=1, output_size=1)
    source = jnp.ones((3, 1))
    for coefficient in (jnp.ones(2), jnp.ones((3, 1)), jnp.ones(0)):
        with pytest.raises(Exception, match="batch count"):
            _raw_reduction_call(source, layout, shape=(3, 1), coefficients=(coefficient,), indices=(0,)).block_until_ready()



@pytest.mark.parametrize("outer_size", [0, 3])
def test_reduction_batches_outer_vmap_with_shared_storage_batch_coefficients_and_ad(outer_size):
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    source = jnp.ones((outer_size, 3, 3), dtype=jnp.float32)

    def execute(data):
        return reduction_p.bind(
            data, factors, coefficient_records=(0,),
            records=(AffineRecord((3,), (1,), 0, (1,), 0),),
            output_shapes=((1,),), reduction_axes=((True,),), output_size=1, dtype=data.dtype)

    mapped = jax.vmap(execute)
    result, tangent = jax.jit(lambda data: jax.jvp(mapped, (data,), (data,)))(source)
    expected = np.broadcast_to(np.asarray([[0], [3], [6]]), (outer_size, 3, 1))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(tangent, expected)
    gradient = jax.jit(jax.grad(lambda data: mapped(data).sum()))(source)
    np.testing.assert_array_equal(gradient, np.broadcast_to(np.asarray(factors)[:, None], source.shape))



def test_reduction_batches_per_batch_fused_rounding_and_reused_coefficient_buffers():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    source = jnp.asarray([[2048, 1, -2048]] * 3, dtype=jnp.float16)
    execute = jax.jit(lambda data, factors: execute_reduction(
        data, (factors,), coefficient_records=(0,), layout=layout, output_size=1))
    for factors, expected in (([0, 1, 1.5], [0, 0, 1.5]), ([1.5, 0, 1], [1.5, 0, 0])):
        np.testing.assert_array_equal(execute(source, jnp.asarray(factors, dtype=jnp.float32)),
                                      np.asarray(expected)[:, None])



REDUCTION_FLOATS = ("float16", "bfloat16", "float32", "float64")



REDUCTION_MIXED_PAIRS = [(source, result) for source in REDUCTION_FLOATS for result in REDUCTION_FLOATS if source != result] + [
    ("complex64", "complex128"), ("complex128", "complex64")]



@pytest.mark.parametrize("source_dtype,result_dtype", REDUCTION_MIXED_PAIRS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("shape,strides,offset,axes", [((2, 3), (3, -1), 2, (1,)),
    ((2, 3), (0, 1), 1, None), ((2, 2), (1, 1), 0, (0,)), ((2, 0), (1, 1), 0, (1,))])
def test_mixed_reduction_reduce_sum_mixed_source_ad(source_dtype, result_dtype, batch_shape, shape, strides, offset, axes):
    with jax.enable_x64():
        source = (jnp.arange(prod(batch_shape) * 6, dtype=jnp.float32) % 11 / 8).astype(source_dtype).reshape((*batch_shape, 6))
        if source_dtype.startswith("complex"):
            source = source * (1 + .5j)
        indices = np.asarray([offset + sum(index * stride for index, stride in zip(coordinate, strides))
                              for coordinate in np.ndindex(shape)], dtype=np.int32).reshape(shape)
        logical_axes = tuple(range(len(shape))) if axes is None else axes
        reference_axes = tuple(len(batch_shape) + axis for axis in logical_axes)
        run = lambda values: reduce_sum(StridedView(values, shape, strides, offset), axes, dtype=result_dtype)
        oracle = lambda values: jnp.sum(values[..., indices], axis=reference_axes, dtype=result_dtype)
        tolerance = 8 * max(float(jnp.finfo(source_dtype).eps), float(jnp.finfo(result_dtype).eps))
        tangent = jnp.ones_like(source)
        primal, direction = jax.jit(lambda values: jax.jvp(run, (values,), (tangent,)))(source)
        expected_primal, expected_direction = jax.jvp(oracle, (source,), (tangent,))
        for actual, expected in ((primal, expected_primal), (direction, expected_direction)):
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        cotangent = jnp.full(primal.shape, 1 + .5j if result_dtype.startswith("complex") else 1, dtype=result_dtype)
        expected = jax.vjp(oracle, source)[1](cotangent)[0]
        for reverse in (jax.vjp(run, source)[1], jax.linear_transpose(run, source)):
            actual = jax.jit(reverse)(cotangent)[0]
            assert actual.dtype == source.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)



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



def test_mixed_reduction_empty_input_with_nonempty_output_and_zero_tangent():
    source = jnp.zeros(0, dtype=jnp.float16)
    run = lambda values: reduce_sum(StridedView(values, (2, 0), (1, 1), 0), (1,), dtype=jnp.float32)
    np.testing.assert_array_equal(run(source), [0, 0])
    gradient = jax.vjp(run, source)[1](jnp.asarray([jnp.nan, jnp.inf]))[0]
    assert gradient.shape == (0,) and gradient.dtype == source.dtype
    _, tangent = jax.jvp(lambda values: run(jax.lax.stop_gradient(values)), (source,), (source,))
    np.testing.assert_array_equal(tangent, [0, 0])



def test_mixed_reduction_outer_vmap_and_nested_derivatives():
    source = (jnp.arange(10, dtype=jnp.float32) / 4).astype(jnp.float16).reshape(2, 5)
    run = jax.vmap(lambda values: reduce_sum(StridedView(values, (2, 3), (1, 1), 0), (1,), dtype=jnp.float32))
    oracle = lambda values: jnp.sum(values[..., jnp.asarray([[0, 1, 2], [1, 2, 3]])], axis=-1, dtype=jnp.float32)
    cotangent = jnp.ones((2, 2), dtype=jnp.float32)
    reverse = lambda values: jax.linear_transpose(run, source)(values)[0]
    np.testing.assert_allclose(jax.jit(reverse)(cotangent), jax.vjp(oracle, source)[1](cotangent)[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(jax.jit(jax.linear_transpose(reverse, cotangent))(jnp.ones_like(source))[0],
                               oracle(jnp.ones_like(source)), rtol=1e-3, atol=1e-3)
    actual = jax.jacfwd(jax.grad(lambda values: jnp.sum(run(values)**2)))(source)
    expected = jax.jacfwd(jax.grad(lambda values: jnp.sum(oracle(values)**2)))(source)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)



def test_mixed_reduction_reverse_product_range_and_coefficient_ad_boundary():
    with jax.enable_x64():
        source = jnp.asarray([.001], dtype=jnp.float64)
        record = AffineRecord((1,), (1,), 0, (1,), 0)
        def execute(values, factor):
            return reduction_p.bind(values, factor, records=(record,), output_shapes=((1,),),
                reduction_axes=((True,),), coefficient_records=(0,), output_size=1, dtype=jnp.dtype("float16"))
        run = lambda values: execute(values, jnp.float16(1000))
        actual = jax.jit(jax.vjp(run, source)[1])(jnp.asarray([1000], dtype=jnp.float16))[0]
        assert actual.dtype == source.dtype
        np.testing.assert_allclose(actual, [1_000_000], rtol=1e-14, atol=0)
        _, tangent = jax.jvp(lambda factor: execute(source, factor), (jnp.float16(2),), (jnp.float16(1),))
        np.testing.assert_array_equal(tangent, source.astype(jnp.float16))



def test_mixed_reduction_batch_coefficients_and_zero_one_branches():
    source = jnp.ones((3, 4), dtype=jnp.float16)
    record = AffineRecord((2, 2), (1, 1), 0, (1, 1), 1)
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    def execute(values, coefficients):
        return reduction_p.bind(values, coefficients, records=(record,), output_shapes=((2, 1),),
            reduction_axes=((False, True),), coefficient_records=(0,), output_size=4, dtype=jnp.dtype("float32"))
    run = lambda values: execute(values, factors)
    cotangent = jnp.asarray([[jnp.nan, 1, 2, jnp.nan]] * 3, dtype=jnp.float32)
    actual = jax.jit(jax.vjp(run, source)[1])(cotangent)[0]
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, [[0, 0, 0, 0], [1, 3, 2, 0], [2, 6, 4, 0]], rtol=1e-3, atol=1e-3)
    source_rows = jnp.stack((source, source * 2))
    factor_rows = jnp.stack((factors, factors))
    mapped = jax.vmap(execute)
    expected = jnp.stack((run(source), run(source * 2)))
    np.testing.assert_allclose(jax.jit(mapped)(source_rows, factor_rows), expected, rtol=1e-3, atol=1e-3)
    _, tangent = jax.jvp(run, (source,), (jnp.full_like(source, jnp.inf),))
    np.testing.assert_array_equal(tangent[0], [0, 0, 0, 0])
    gradient = jax.linear_transpose(run, source)(jnp.full((3, 4), jnp.inf, dtype=jnp.float32))[0]
    np.testing.assert_array_equal(gradient[0], [0, 0, 0, 0])
    assert jnp.all(jnp.isposinf(gradient[1, :3]))
    assert gradient[1, 3] == 0



@pytest.mark.parametrize("source_dtype,result_dtype", [("float32", "int32")])
def test_mixed_reduction_discrete_result_has_zero_tangent(source_dtype, result_dtype):
    source = jnp.ones(2, dtype=source_dtype)
    result, tangent = jax.jvp(
        lambda values: reduce_sum(StridedView(values, (2,), (1,), 0), dtype=result_dtype),
        (source,), (source,))
    np.testing.assert_array_equal(result, 2)
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape


def test_packed_trace_multi_record_protocol_assembly():
    records = (AffineRecord((2,), (1,), 0, (1,), 1), AffineRecord((), (), 3, (), 2))
    layout = encode_reduction_layout(
        records, output_shapes=((1,), ()), reduction_axes=((True,), ()),
        source_size=4, output_size=3,
    )
    expected = [1, 4, 3, 2, 1, 0, 1, 2, 1, 1, 1, 1, 0, 3, 2]
    assert layout.tobytes() == b"".join(word.to_bytes(8, "little") for word in expected)
    empty = encode_reduction_layout(
        (), output_shapes=(), reduction_axes=(), source_size=4, output_size=3,
    )
    np.testing.assert_array_equal(empty.view("<u8"), [1, 4, 3, 0])


@pytest.mark.parametrize("output_shapes,axes", [
    (((1,),), ((True,), ())),
    (((1,), ()), ((True,),)),
    (((1,), (), ()), ((True,), ())),
    (((1,), ()), ((True,), (), ())),
])
def test_reduction_encoder_rejects_mismatched_record_counts(output_shapes, axes):
    records = (AffineRecord((2,), (1,), 0, (1,), 1), AffineRecord((), (), 3, (), 2))
    with pytest.raises(ValueError, match="zip"):
        encode_reduction_layout(records, output_shapes=output_shapes, reduction_axes=axes,
                                source_size=4, output_size=3)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0)])
def test_packed_trace_real_trace_subblocks_coefficients_and_batches(batch_shape, monkeypatch):
    inputs = (
        SimpleNamespace(sizes=(2, 2, 2), strides=(4, 2, 1), offset=0),
        SimpleNamespace(sizes=(2, 3, 3), strides=(9, 3, 1), offset=8),
    )
    outputs = (SimpleNamespace(sizes=(2,), strides=(1,), offset=0),
               SimpleNamespace(sizes=(2,), strides=(1,), offset=1))
    source = jnp.arange(prod(batch_shape) * 26, dtype=jnp.float32).reshape((*batch_shape, 26))
    captured = []
    encode = _jax.encode_reduction_layout

    def capture(records, **fields):
        captured.append(dict(records=records, **fields))
        return encode(records, **fields)

    monkeypatch.setattr(_jax, "encode_reduction_layout", capture)

    def execute(data, half, integer):
        return _strided_tensortrace(
            data, destination_size=4, source_subblocks=inputs, destination_subblocks=outputs,
            entries=((0, 0, None), (1, 1, half), (0, 1, integer)), result_dtype=jnp.float32,
            permutation=(), num_open_out=1, num_open_in=0, trace_count=1)

    compiled = jax.jit(execute)
    first = np.trace(np.asarray(source)[..., :8].reshape((*batch_shape, 2, 2, 2)), axis1=-2, axis2=-1)
    second = np.trace(np.asarray(source)[..., 8:].reshape((*batch_shape, 2, 3, 3)), axis1=-2, axis2=-1)
    for half in (.5, 1.5):
        expected = np.zeros((*batch_shape, 4), dtype=np.float32)
        expected[..., :2] += first
        expected[..., 1:3] += half * second + 2 * first
        np.testing.assert_array_equal(compiled(source, jnp.float32(half), jnp.int32(2)), expected)
    assert len(captured) == 1
    assert captured[0] == dict(
        records=(AffineRecord((2, 2), (4, 3), 0, (1, 1), 0),
                 AffineRecord((2, 3), (9, 4), 8, (1, 1), 1),
                 AffineRecord((2, 2), (4, 3), 0, (1, 1), 1)),
        output_shapes=((2, 1),) * 3, reduction_axes=((False, True),) * 3,
        source_size=26, output_size=4,
    )
    assert "tensor0_stride_reduction_f32_cpu_v1" in compiled.lower(
        source, jnp.float32(.5), jnp.int32(2)).as_text()

@pytest.mark.parametrize("permutation", [(), (5, 2, 0, 4, 1, 3)])
def test_packed_trace_two_pairs_with_open_input_and_output(permutation):
    canonical_shape = (2, 2, 3, 2, 2, 3)
    shape = canonical_shape if not permutation else tuple(
        canonical_shape[permutation.index(axis)] for axis in range(6))
    dense = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    canonical = dense if not permutation else dense.transpose(permutation)
    expected = np.einsum("oabjab->oj", canonical)
    source = jnp.asarray(dense.reshape(-1))
    actual = _strided_tensortrace(
        source, destination_size=6,
        source_subblocks=(SimpleNamespace(sizes=shape, strides=contiguous_strides(shape), offset=0),),
        destination_subblocks=(SimpleNamespace(sizes=(2, 2), strides=(2, 1), offset=1),),
        entries=((0, 0, 1),), result_dtype=jnp.float32, permutation=permutation,
        num_open_out=1, num_open_in=1, trace_count=2)
    np.testing.assert_array_equal(actual, np.concatenate(([0], expected.reshape(-1), [0])))

@pytest.mark.parametrize("strides,offset,storage_size", [((-2, -1), 3, 4), ((0, 0), 0, 1)])
def test_packed_trace_signed_and_broadcast_trace(strides, offset, storage_size):
    source = jnp.arange(1, storage_size + 1, dtype=jnp.float32)
    actual = _strided_tensortrace(
        source, destination_size=3,
        source_subblocks=(SimpleNamespace(sizes=(2, 2), strides=strides, offset=offset),),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=1),),
        entries=((0, 0, 2),), result_dtype=jnp.float32, permutation=(),
        num_open_out=0, num_open_in=0, trace_count=1)
    expected = 2 * (source[offset] + source[offset + sum(strides)])
    np.testing.assert_array_equal(actual, [0, expected, 0])

@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,)])
@pytest.mark.parametrize("entries", [(), ((0, 0, None),)])
def test_packed_trace_empty_input_and_entries(batch_shape, entries):
    actual = _strided_tensortrace(
        jnp.zeros((*batch_shape, 0)), destination_size=3,
        source_subblocks=(SimpleNamespace(sizes=(0, 0), strides=(1, 1), offset=0),),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=2),),
        entries=iter(entries), result_dtype=jnp.float32, permutation=(),
        num_open_out=0, num_open_in=0, trace_count=1)
    np.testing.assert_array_equal(actual, np.zeros((*batch_shape, 3)))

def test_packed_trace_numeric_contract_and_coefficient_types():
    with jax.enable_x64():
        source = jnp.asarray([16777217], dtype=jnp.int32)
        np.testing.assert_array_equal(scalar_trace(source, (None,), jnp.int64), [16777217])
        np.testing.assert_array_equal(scalar_trace(source, (jnp.float32(1),), jnp.int64), [16777217])
        source = jnp.asarray([-1, 1], dtype=jnp.float32)
        np.testing.assert_array_equal(scalar_trace(source, (None, jnp.float64(1 + 2**-24)), jnp.float32), [2**-24])
        source = jnp.asarray([1e30, np.inf], dtype=jnp.float32)
        np.testing.assert_array_equal(scalar_trace(source, (None, 1e-46), jnp.float32), source[:1])
        source = jnp.asarray([1, 1e20, -1e20], dtype=jnp.float32)
        np.testing.assert_array_equal(scalar_trace(source, (None,) * 3, jnp.float32), [0])

def test_packed_trace_vmap_and_ad_boundary():
    execute = lambda source, factor: scalar_trace(source, (factor,), jnp.float32)
    np.testing.assert_array_equal(jax.jit(jax.vmap(execute))(
        jnp.asarray([[2.], [3.]]), jnp.asarray([3., 4.])), [[6], [12]])
    np.testing.assert_array_equal(
        jax.grad(lambda source: execute(source, jnp.float32(2)).sum())(jnp.ones(1)), [2])
    np.testing.assert_array_equal(jax.grad(lambda factor: execute(jnp.ones(1), factor).sum())(jnp.float32(2)), 1)

def test_packed_trace_zero_output_storage():
    actual = _strided_tensortrace(
        jnp.zeros((2, 0)), destination_size=0,
        source_subblocks=(SimpleNamespace(sizes=(0, 2, 2), strides=(4, 2, 1), offset=0),),
        destination_subblocks=(SimpleNamespace(sizes=(0,), strides=(1,), offset=0),),
        entries=((0, 0, jnp.float32(2)),), result_dtype=jnp.float32, permutation=(),
        num_open_out=1, num_open_in=0, trace_count=1)
    assert actual.shape == (2, 0)

def test_packed_trace_empty_record_keeps_original_coefficient_index():
    actual = _strided_tensortrace(
        jnp.asarray([2], dtype=jnp.int32), destination_size=1,
        source_subblocks=(SimpleNamespace(sizes=(0, 0), strides=(1, 1), offset=0),
                          SimpleNamespace(sizes=(1, 1), strides=(1, 1), offset=0)),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
        entries=((0, 0, None), (1, 0, jnp.int32(3))), result_dtype=jnp.int32,
        permutation=(), num_open_out=0, num_open_in=0, trace_count=1)
    np.testing.assert_array_equal(actual, [6])

@pytest.mark.parametrize("change,message", [
    ({"trace_count": -1}, "counts"), ({"permutation": (0, 0)}, "permutation"),
    ({"entries": ((-1, 0, None),)}, "index"), ({"entries": ((0, 1, None),)}, "index"),
    ({"entries": ((0, 0, jnp.ones(1)),)}, "scalars"),
    ({"source_subblocks": (SimpleNamespace(sizes=(2, 3), strides=(3, 1), offset=0),)}, "sizes"),
    ({"source_subblocks": (SimpleNamespace(sizes=(2,), strides=(1,), offset=0),)}, "rank"),
    ({"destination_subblocks": (SimpleNamespace(sizes=(1,), strides=(1,), offset=0),)}, "shape"),
    ({"source_subblocks": (SimpleNamespace(sizes=(2, 2), strides=(2**63 - 1,) * 2, offset=0),)}, "strides"),
])
def test_packed_trace_invalid_trace_layout_and_coefficients(change, message):
    arguments = dict(destination_size=1,
                     source_subblocks=(SimpleNamespace(sizes=(2, 2), strides=(2, 1), offset=0),),
                     destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
                     entries=((0, 0, None),), result_dtype=jnp.float32, permutation=(),
                     num_open_out=0, num_open_in=0, trace_count=1)
    with pytest.raises(ValueError, match=message):
        _strided_tensortrace(jnp.ones(4), **(arguments | change))
