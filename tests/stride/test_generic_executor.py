"""General execution contracts, independent of specialized kernel selection."""

from contextlib import contextmanager
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, enable_threads, get_num_threads, scale, set_num_threads
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord, contiguous_strides

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
MIXED = [(jnp.float16, jnp.float32, -1.25), (jnp.float32, jnp.complex64, 1.25 - .75j),
         (jnp.complex64, jnp.float32, -.75), (jnp.float64, jnp.complex128, .5 + .75j),
         (jnp.complex128, jnp.float64, -.75)]
FAMILIES = [(jnp.bool_, True), (jnp.int8, -3), (jnp.int16, -3), (jnp.int32, -3), (jnp.int64, -3),
            (jnp.uint8, 3), (jnp.uint16, 3), (jnp.uint32, 3), (jnp.uint64, 3),
            (jnp.float16, -1.25), (jnp.bfloat16, -1.25), (jnp.float32, -1.25), (jnp.float64, -1.25),
            (jnp.complex64, 1.25 - .75j), (jnp.complex128, 1.25 - .75j)]
PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
              AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))


@contextmanager
def thread_limit(limit):
    previous = get_num_threads()
    set_num_threads(limit)
    try:
        yield
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)


def blocked_record(shape):
    strides = contiguous_strides(shape)
    source_strides = (-strides[0], *strides[1:]) if shape else ()
    offset = (shape[0] - 1) * strides[0] if shape else 0
    return AffineRecord(shape, source_strides, offset, tuple(prod(shape[:axis]) for axis in range(len(shape))), 0)


def addresses(record):
    source = np.full(record.logical_shape, record.source_offset, dtype=np.int64)
    destination = np.full(record.logical_shape, record.destination_offset, dtype=np.int64)
    for axis, extent in enumerate(record.logical_shape):
        coordinate = np.arange(extent).reshape((1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1))
        source += coordinate * record.source_strides[axis]
        destination += coordinate * record.destination_strides[axis]
    return source.ravel(), destination.ravel()


def dtype_values(dtype, size):
    raw = np.arange(size) % 17
    if dtype == jnp.bool_:
        return jnp.asarray(raw % 3 != 0)
    result = jnp.asarray(raw, dtype=dtype)
    return result + 1j * result[::-1] if jnp.issubdtype(dtype, jnp.complexfloating) else result


def complex_values(size):
    components = [0., -0., 1.25, -1.25, .125, 1e10, -1e10, 1e-10]
    return jnp.asarray(np.resize(np.asarray([complex(real, imaginary) for real in components
                                             for imaginary in components], dtype=np.complex64), size))


def cast(value, dtype):
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
        value = jnp.real(value)
    return value.astype(dtype)


def mapped(source, record, coefficient, dtype, output_size):
    base = jnp.zeros((*source.shape[:-1], output_size), dtype=dtype)
    return update_p.bind(source, base, coefficient, jnp.int32(0), records=(record,))


def reference_map(source, record, coefficient, dtype, output_size):
    source_indices, destinations = addresses(record)
    selected = source[..., source_indices] * coefficient
    return jnp.zeros((*source.shape[:-1], output_size), dtype=dtype).at[..., destinations].set(cast(selected, dtype))


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if jnp.issubdtype(actual.dtype, jnp.integer) or actual.dtype == jnp.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 2e-3 if actual.dtype in (jnp.float16, jnp.bfloat16) else 2e-6
        for component in (jnp.real, jnp.imag):
            np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                       np.asarray(component(expected)).astype(np.float64),
                                       rtol=tolerance, atol=1e-7, equal_nan=True)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("rank", range(10))
def test_effective_ranks_including_scalar_and_beyond_eight(rank):
    shape = (2,) * rank
    record = blocked_record(shape)
    source = jnp.arange(prod(shape), dtype=jnp.float32) - 3
    operation = jax.jit(lambda value: mapped(value, record, jnp.float32(1.25), value.dtype, source.size))
    assert_close(operation(source), reference_map(source, record, jnp.float32(1.25), source.dtype, source.size))
    assert_native(operation.lower(source))


@pytest.mark.parametrize("rank", [3, 8])
def test_batched_high_rank_walkers(rank):
    record = blocked_record((2,) * rank)
    source = jnp.arange(4 * 2**rank, dtype=jnp.float32).reshape(4, -1) / 8
    operation = lambda value: mapped(value, record, jnp.float32(.75), value.dtype, value.shape[-1])
    expected = reference_map(source, record, jnp.float32(.75), source.dtype, source.shape[-1])
    for function in (operation, jax.vmap(operation)):
        assert_close(jax.jit(function)(source), expected)


@pytest.mark.parametrize("batched", [False, True])
def test_multirecord_mapping_with_independent_coefficients(batched):
    source = jnp.arange(64 if batched else 16, dtype=jnp.float32)
    if batched:
        source = source.reshape(4, 16)
    factors = (jnp.float32(1.25), jnp.float32(-.5))
    operation = lambda value: accumulation_p.bind(value, *factors, records=PARTITIONS,
        coefficient_records=(0, 1), output_size=16, dtype=value.dtype)
    expected = sum(reference_map(source, record, factor, source.dtype, 16)
                   for record, factor in zip(PARTITIONS, factors, strict=True))
    assert_close(jax.jit(operation)(source), expected)
    assert_native(jax.jit(operation).lower(source))
    if batched:
        assert_close(jax.jit(jax.vmap(operation))(source), expected)


@pytest.mark.parametrize("shape", [(16, 16, 16, 64), (8, 8, 8, 8, 64), (8, 8, 8, 8, 8, 8),
                                  (4, 4, 4, 4, 4, 4, 64), (4, 4, 4, 4, 4, 4, 4, 16)])
def test_large_high_rank_blocking_under_thread_limits(shape):
    record = blocked_record(shape)
    source = jnp.arange(prod(shape), dtype=jnp.float32) - 3
    operation = jax.jit(lambda value: mapped(value, record, jnp.float32(-1.25), value.dtype, source.size))
    expected = reference_map(source, record, jnp.float32(-1.25), source.dtype, source.size)
    for limit in (1, 4):
        with thread_limit(limit):
            assert_close(operation(source), expected)


@pytest.mark.parametrize("dtype,factor", FAMILIES)
@pytest.mark.parametrize("shape", [(17,), (8, 7, 4)])
def test_scalar_families_on_contiguous_and_strided_layouts(dtype, factor, shape):
    with jax.enable_x64():
        record = blocked_record(shape)
        source = dtype_values(dtype, prod(shape))
        coefficient = jnp.asarray(factor, dtype=dtype)
        actual = jax.jit(lambda value: mapped(value, record, coefficient, dtype, source.size))(source)
        assert_close(actual, reference_map(source, record, coefficient, dtype, source.size))


@pytest.mark.parametrize("source_dtype,result_dtype,factor", MIXED)
@pytest.mark.parametrize("shape", [(35,), (5, 7), (8, 7, 4)])
def test_mixed_mapping_batch_jvp_vjp(source_dtype, result_dtype, factor, shape):
    with jax.enable_x64():
        record = blocked_record(shape)
        source = dtype_values(source_dtype, prod(shape))
        coefficient = jnp.asarray(factor, dtype=jnp.complex128 if isinstance(factor, complex) else jnp.float64)
        operation = lambda value: mapped(value, record, coefficient, result_dtype, source.size)
        reference = lambda value: reference_map(value, record, coefficient, result_dtype, source.size)
        direction = jnp.ones_like(source) * .25
        cotangent = dtype_values(result_dtype, source.size) * .5
        actual = jax.jit(lambda value, tangent, cot: (
            jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
        expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
        for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert_close(result, wanted)
        batch = jnp.stack((source, source * .5, -source))
        assert_close(jax.jit(jax.vmap(operation))(batch), reference(batch))


@pytest.mark.parametrize("record,size", [
    (AffineRecord((2, 4, 3, 8), (32, 1, 64, 4), 0, (96, 24, 8, 1), 0), 192),
    (AffineRecord((8, 8, 8, 8), (64, 512, 1, 8), 0, (512, 64, 8, 1), 0), 4096),
])
def test_rank_four_tiled_and_two_pair_layouts(record, size):
    source = jnp.arange(size, dtype=jnp.float32) / 4
    actual = jax.jit(lambda value: mapped(value, record, jnp.float32(-.75), value.dtype, size))(source)
    assert_close(actual, reference_map(source, record, jnp.float32(-.75), source.dtype, size))


@pytest.mark.parametrize("beta", [0., 1., -.25])
@pytest.mark.parametrize("record,size", [(AffineRecord((), (), 0, (), 0), 1),
                                        (blocked_record((1024, 64, 4)), 262144)])
def test_assign_accumulate_and_update_use_original_base(beta, record, size):
    source = dtype_values(jnp.float32, size)
    base = source + 11
    source_indices, destinations = addresses(record)
    expected = base.at[destinations].set(-1.25 * source[source_indices] + beta * base[destinations])
    operation = jax.jit(lambda old, value: update_p.bind(value, old, jnp.float32(-1.25),
                                                        jnp.float32(beta), records=(record,)))
    for limit in (1, 4):
        with thread_limit(limit):
            assert_close(operation(base, source), expected)


HALF_BITS = np.asarray([0, 0x8000, 0x3C00, 0xBC00, 0x4100, 0xC100, 0x3000, 1, 0x3FF,
                       0x7800, 0x3555, 0xB555, 0x4000, 0xC000, 0x400, 0x8400, 0x3800,
                       0x7C00, 0xFC00, 0x7E00, 0x7BFF], dtype=np.uint16)
HALF_LAYOUTS = [
    (AffineRecord((257,), (1,), 0, (1,), 0), 257),
    (AffineRecord((128, 257), (512, 1), 0, (257, 1), 0), 65536),
    (AffineRecord((65, 129), (1, 65), 0, (129, 1), 0), 8385),
    (AffineRecord((65, 129), (129, 1), 0, (1, 65), 0), 8385),
    (AffineRecord((512, 512), (1, 512), 0, (512, 1), 0), 262144),
    (AffineRecord((512, 512), (512, 1), 0, (1, 512), 0), 262144),
]


@pytest.mark.parametrize("record,source_size", HALF_LAYOUTS)
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("factor", [1., -1.25])
def test_half_tails_pitched_rows_and_permutations(record, source_size, dtype, factor):
    source = jnp.asarray(np.resize(HALF_BITS, source_size).view(np.float16))
    output_size = prod(record.logical_shape)
    coefficient = jnp.float32(factor)
    operation = jax.jit(lambda value: mapped(value, record, coefficient, dtype, output_size))
    assert_close(operation(source), reference_map(source, record, coefficient, dtype, output_size))
    if factor == 1 and dtype == jnp.float16:
        source_indices, destinations = addresses(record)
        expected = np.zeros(output_size, dtype=np.uint16)
        expected[destinations] = np.asarray(source).view(np.uint16)[source_indices]
        actual = jax.jit(lambda value: copy_p.bind(value, records=(record,), output_size=output_size, dtype=dtype))(source)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("record,size", [(HALF_LAYOUTS[0][0], 257),
                                        (AffineRecord((64, 128), (1, 64), 0, (128, 1), 0), 8192)])
def test_half_forward_and_reverse_with_batching(dtype, record, size):
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    operation = lambda value: mapped(value, record, jnp.float32(-1.25), dtype, size)
    reference = lambda value: reference_map(value, record, jnp.float32(-1.25), dtype, size)
    direction, cotangent = source[::-1], jnp.linspace(-2, 3, size, dtype=dtype)
    for result, wanted in zip(jax.jvp(operation, (source,), (direction,)),
                              jax.jvp(reference, (source,), (direction,)), strict=True):
        assert_close(result, wanted)
    assert_close(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])
    batch = jnp.stack((source, source * .5))
    assert_close(jax.jit(jax.vmap(operation))(batch), reference(batch))


@pytest.mark.parametrize("factor", [complex(0., -0.), .125 + 1j, 2.5 + 3.5j, -1.25 + .75j])
@pytest.mark.parametrize("selected", [False, True])
def test_complex_signed_rows_and_selected_scaling(factor, selected):
    source = complex_values(31)
    source_strides, offset = (-11, 1), 22
    coefficient = jnp.complex64(factor)
    if selected:
        actual = jax.jit(lambda data: scale(StridedView(data, (3, 9), source_strides, offset), coefficient).data)(source)
        record = AffineRecord((3, 9), source_strides, offset, source_strides, offset)
        _, destinations = addresses(record)
        expected = source.at[destinations].set(source[destinations] * coefficient)
    else:
        record = AffineRecord((3, 9), source_strides, offset, (9, 1), 0)
        actual = jax.jit(lambda data: mapped(data, record, coefficient, data.dtype, 27))(source)
        expected = reference_map(source, record, coefficient, source.dtype, 27)
    assert_close(actual, expected)


@pytest.mark.parametrize("shape,strides,offset,size", [((), (), 0, 1), ((128, 257), (512, 1), 0, 65536)])
def test_selected_scale_preserves_gaps_with_and_without_donation(shape, strides, offset, size):
    source = complex_values(size)
    coefficient = jnp.complex64(-1.25 + .75j)
    record = AffineRecord(shape, strides, offset, strides, offset)
    _, destinations = addresses(record)
    expected = source.at[destinations].set(source[destinations] * coefficient)
    operation = lambda value: scale(StridedView(value, shape, strides, offset), coefficient).data
    assert_close(jax.jit(operation)(source), expected)
    assert_close(jax.jit(operation, donate_argnums=(0,))(jnp.array(source)), expected)


@pytest.mark.parametrize("complex_output", [False, True])
def test_broadcast_pullback_accumulates_repeated_reads(complex_output):
    record = AffineRecord((2, 3), (0, -1), 2, (3, 1), 0)
    source = jnp.asarray([1., -2., 3.], dtype=jnp.float32)
    dtype = jnp.complex64 if complex_output else jnp.float32
    coefficient = jnp.asarray(1.25 - .5j if complex_output else 1.25, dtype=dtype)
    operation = lambda value: mapped(value, record, coefficient, dtype, 6)
    reference = lambda value: reference_map(value, record, coefficient, dtype, 6)
    cotangent = dtype_values(dtype, 6)
    assert_close(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])


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
            assert_close(operation(source), expected)


def test_mixed_repeated_source_reduction_differentials():
    count, width = 1024, 16
    record = AffineRecord((count, width), (1, 0), 0, (1, 0), 0)
    coefficient = jnp.complex64(1.25 - .5j)
    source = jnp.linspace(-2, 3, count, dtype=jnp.float32)
    operation = lambda value: accumulation_p.bind(value, coefficient, records=(record,),
        coefficient_records=(0,), output_size=count, dtype=jnp.dtype(jnp.complex64))
    reference = lambda value: value * coefficient * width
    direction = source * -.25 + 1
    cotangent = dtype_values(jnp.complex64, count) / 8
    actual = jax.jit(lambda value, tangent, cot: (
        jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
    expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(result, wanted)


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
    operation = jax.jit(lambda value: mapped(value, record, coefficient, dtype, output_size))
    expected = reference_map(source, record, coefficient, dtype, output_size)
    for limit in (1, 4):
        with thread_limit(limit):
            assert_close(operation(source), expected)


def test_large_half_pullback_under_thread_limits():
    size = 1 << 20
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    operation = lambda value: mapped(value, record, jnp.float32(-1.25), value.dtype, size)
    pullback = jax.jit(jax.vjp(operation, source)[1])
    cotangent = jnp.ones_like(source)
    for limit in (1, 4):
        with thread_limit(limit):
            assert_close(pullback(cotangent)[0], jnp.full_like(source, -1.25))


@pytest.mark.parametrize("source_dtype,result_dtype,factor", [
    (jnp.float16, jnp.float32, -1.25), (jnp.complex64, jnp.complex64, 1),
    (jnp.complex64, jnp.float32, -1.25 + .75j), (jnp.float32, jnp.complex64, -.75),
])
def test_mixed_contiguous_pullback_finite_and_tiny_values(source_dtype, result_dtype, factor):
    size = 1025
    source = jnp.ones(size, dtype=source_dtype)
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if isinstance(factor, complex) else jnp.float32)
    operation = lambda value: mapped(value, record, coefficient, result_dtype, size)
    reference = lambda value: reference_map(value, record, coefficient, result_dtype, size)
    raw = np.resize(np.asarray([0., -0., np.nextafter(np.float32(0), np.float32(1)), 1., -1., 1.25, -.125],
                                dtype=np.float32), size)
    cotangent = jnp.asarray(raw, dtype=result_dtype)
    if jnp.issubdtype(result_dtype, jnp.complexfloating):
        cotangent = cotangent + .5j * cotangent[::-1]
    actual = jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent)
    assert_close(actual, jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.parametrize("mergeable", [False, True])
@pytest.mark.parametrize("complex_output", [False, True])
def test_rank_nine_direct_and_vmap_without_fixed_rank_limit(mergeable, complex_output):
    shape = (2,) * 9
    source_strides = tuple((2 if mergeable else 3) ** (8 - axis) for axis in range(9))
    record = AffineRecord(shape, source_strides, 0, contiguous_strides(shape), 0)
    size = sum(source_strides) + 1
    source = (jnp.arange(3 * size, dtype=jnp.float32) % 127).reshape(3, size) / 8
    dtype = jnp.complex64 if complex_output else jnp.float32
    coefficient = jnp.asarray(1 + .5j if complex_output else 1, dtype=dtype)
    operation = lambda value: mapped(value, record, coefficient, dtype, 512)
    expected = reference_map(source, record, coefficient, dtype, 512)
    for function in (operation, jax.vmap(operation)):
        lowered = jax.jit(function).lower(source)
        assert_native(lowered)
        assert_close(lowered.compile()(source), expected)


@pytest.mark.parametrize("source_dtype,result_dtype", [
    (jnp.float16, jnp.float32), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32),
])
def test_mixed_multirecord_batch_derivatives(source_dtype, result_dtype):
    raw = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    source = (raw * (1 + .25j) if source_dtype == jnp.complex64 else raw).astype(source_dtype)
    coefficients = (jnp.float32(1.25), jnp.float32(-.5))
    operation = lambda value: accumulation_p.bind(value, *coefficients, records=PARTITIONS,
        coefficient_records=(0, 1), output_size=16, dtype=jnp.dtype(result_dtype))

    def reference(value):
        blocks = value.reshape(*value.shape[:-1], 4, 4)
        selected = jnp.concatenate((blocks[..., 2:] * coefficients[1], blocks[..., :2] * coefficients[0]), axis=-1)
        return cast(selected.reshape(value.shape), result_dtype)

    direction = source / 8
    cotangent = jnp.ones(source.shape, dtype=result_dtype)
    if result_dtype == jnp.complex64:
        cotangent = cotangent * jnp.complex64(1 + .25j)
    actual = jax.jit(lambda value, tangent, cot: (
        jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
    expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(result, wanted)
    assert_native(jax.jit(operation).lower(source))


def test_large_storage_mixed_pullback_lowering_has_no_address_arrays():
    record = AffineRecord((4,), (1,), 0, (1,), 0)
    source = jax.ShapeDtypeStruct((100000000,), jnp.float16)
    cotangent = jax.ShapeDtypeStruct((4,), jnp.float32)
    operation = lambda value: copy_p.bind(value, records=(record,), output_size=4, dtype=jnp.dtype(jnp.float32))
    pullback = jax.jit(lambda value, cot: jax.vjp(operation, value)[1](cot)[0])
    assert_native(pullback.lower(source, cotangent))
