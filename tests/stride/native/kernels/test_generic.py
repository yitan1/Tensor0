"""Native generic executors, layouts and scalar families."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord, contiguous_strides

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.generic import (
    HALF_LAYOUTS,
    PARTITIONS,
    addresses,
    assert_close,
    assert_native,
    blocked_record,
    complex_values,
    dtype_values,
    mapped,
    reference_map,
    thread_limit,
)


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')


FAMILIES = [(jnp.bool_, True), (jnp.int8, -3), (jnp.int16, -3), (jnp.int32, -3), (jnp.int64, -3),
            (jnp.uint8, 3), (jnp.uint16, 3), (jnp.uint32, 3), (jnp.uint64, 3),
            (jnp.float16, -1.25), (jnp.bfloat16, -1.25), (jnp.float32, -1.25), (jnp.float64, -1.25),
            (jnp.complex64, 1.25 - .75j), (jnp.complex128, 1.25 - .75j)]


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


@pytest.mark.parametrize("dtype,factor", FAMILIES)
@pytest.mark.parametrize("shape", [(17,), (8, 7, 4)])
def test_scalar_families_on_contiguous_and_strided_layouts(dtype, factor, shape):
    with jax.enable_x64():
        record = blocked_record(shape)
        source = dtype_values(dtype, prod(shape))
        coefficient = jnp.asarray(factor, dtype=dtype)
        actual = jax.jit(lambda value: mapped(value, record, coefficient, dtype, source.size))(source)
        assert_close(actual, reference_map(source, record, coefficient, dtype, source.size))


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
