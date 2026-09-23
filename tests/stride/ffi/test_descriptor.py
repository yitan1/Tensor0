"""Descriptor encoding and producer protocol assembly."""

from dataclasses import fields
from itertools import permutations, product
from math import prod

import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.ffi import REDUCTION_FIBER, _encode_reduction_records, reduction_layout
from tests.stride.support.oracles.addresses import addresses
from tests.stride.support.oracles.affine import native, signed_record


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_signed_affine_uses_address_only_descriptor():
    record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1))
    layout = encode_layout((record,), source_size=12, output_size=12)
    np.testing.assert_array_equal(layout, [1, 12, 12, 1, 2, 9, 0, 4, 3, -3, 1, 1, 4])


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_broadcast_requires_no_separate_axis_metadata():
    record = AffineRecord((2, 3), (0, 1), 0, (3, 1), 0)
    layout = encode_layout((record,), source_size=3, output_size=6)
    np.testing.assert_array_equal(layout, [1, 3, 6, 1, 2, 0, 0, 2, 3, 0, 1, 3, 1])
    source = jnp.arange(3, dtype=jnp.float32)
    np.testing.assert_array_equal(native(source, record, 1, "float32"), jnp.tile(source, 2))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


def compact_strides(shape, order):
    strides = [0] * len(shape)
    stride = 1
    for axis in order:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def test_noncompact_records_encode_only_addresses():
    records = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2), AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))
    layout = encode_layout(records, source_size=16, output_size=16)
    assert tuple(field.name for field in fields(AffineRecord)) == (
        "logical_shape", "source_strides", "source_offset", "destination_strides", "destination_offset")
    np.testing.assert_array_equal(layout, [1, 16, 16, 2, 2, 0, 2, 4, 2, 4, 1, 4, 1,
                                         2, 2, 0, 4, 2, 4, 1, 4, 1])
    assert layout.dtype == np.int64 and layout.nbytes == 8 * layout.size
    assert layout.nbytes < 1024


def test_single_record_encoding_is_deterministic_without_array_identity():
    first = AffineRecord((6,), (1,), 0, (1,), 0)
    second = AffineRecord((6,), (1,), 0, (1,), 0)
    assert first == second and hash(first) == hash(second)
    first_layout = encode_layout((first,), source_size=6, output_size=6)
    second_layout = encode_layout((second,), source_size=6, output_size=6)
    np.testing.assert_array_equal(first_layout, [1, 6, 6, 1, 1, 0, 0, 6, 1, 1])
    assert first_layout.tobytes() == second_layout.tobytes()


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (2, 3, 2), (2, 2, 2, 2)])
def test_compact_axis_orders_preserve_addresses_in_encoding(shape):
    rank, size = len(shape), prod(shape)
    for source_order, destination_order in product(permutations(range(rank)), repeat=2):
        source_strides = compact_strides(shape, source_order)
        destination_strides = compact_strides(shape, destination_order)
        record = AffineRecord(shape, source_strides, 0, destination_strides, 0)
        layout = encode_layout((record,), source_size=size, output_size=size)
        np.testing.assert_array_equal(layout[7:7 + rank], shape)
        np.testing.assert_array_equal(layout[7 + rank:7 + 2 * rank], source_strides)
        np.testing.assert_array_equal(layout[7 + 2 * rank:], destination_strides)
        for strides in (source_strides, destination_strides):
            np.testing.assert_array_equal(np.sort(addresses(shape, strides, 0)), np.arange(size))


def test_descriptor_size_depends_on_rank_not_element_count():
    small = encode_layout((AffineRecord((2, 3), (3, 1), 0, (1, 2), 0),), source_size=6, output_size=6)
    large = encode_layout((AffineRecord((200, 300), (300, 1), 0, (1, 200), 0),), source_size=60000, output_size=60000)
    assert small.size == large.size == 13
    assert small.nbytes == large.nbytes


def test_reduction_encoding_retains_real_output_shape_and_axes():
    layout = reduction_layout()
    assert layout.dtype == np.uint8
    np.testing.assert_array_equal(layout.view("<u8"), [1, 12, 3, 1, 2, 0, 0, 4, 3, 3, 1, 1, 3, 3, 1, 1, 0])


@pytest.mark.parametrize("record,size,message", [
    (AffineRecord((2,), (), 0, (1,), 0), 2, "ranks must match"),
    (AffineRecord((2,), (1,), -1, (1,), 0), 2, "nonnegative int64"),
    (AffineRecord((2,), (1 << 63,), 0, (1,), 0), 2, "strides must fit int64"),
    (AffineRecord((2,), (1,), 0, (1,), 0), 1 << 63, "nonnegative int64"),
])
def test_address_encoder_rejects_unrepresentable_fields(record, size, message):
    with pytest.raises(ValueError, match=message):
        encode_layout((record,), source_size=size, output_size=2)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("changes,source_size,output_size,words", [
    ({}, 3, 1, [1, 3, 1, 1, 1, 0, 0, 3, 1, 1, 1, 1]),
    ({"source_shape": (2, 3), "source_strides": (3, 1), "output_shape": (1, 1),
      "output_strides": (1, 1), "reduction_axes": (True, True)},
     6, 1, [1, 6, 1, 1, 2, 0, 0, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1]),
    ({"source_shape": (0,), "output_offset": 2, "output_strides": (2**63 - 1,)},
     0, 3, [1, 0, 3, 1, 1, 0, 2, 0, 1, 1, 2**63 - 1, 1]),
    ({"source_offset": 2, "source_strides": (-1,)},
     3, 1, [1, 3, 1, 1, 1, 2, 0, 3, 2**64 - 1, 1, 1, 1]),
    ({"source_strides": (0,)}, 1, 1, [1, 1, 1, 1, 1, 0, 0, 3, 0, 1, 1, 1]),
    ({"source_shape": (), "source_strides": (), "source_offset": 2,
      "output_shape": (), "output_strides": (), "output_offset": 1, "reduction_axes": ()},
     3, 2, [1, 3, 2, 1, 0, 2, 1]),
    ({"source_shape": (2, 2), "source_strides": (2, 1), "output_shape": (2, 1),
      "output_strides": (2, 2**63 - 1), "output_offset": 1, "reduction_axes": (False, True)},
     4, 5, [1, 4, 5, 1, 2, 0, 1, 2, 2, 2, 1, 2, 1, 2, 2**63 - 1, 0, 1]),
    ({"source_shape": (2**64 - 1,), "source_strides": (-(2**63),),
      "source_offset": 2**63 - 1, "output_offset": 2**63 - 1, "output_strides": (2**63 - 1,)},
     2**64 - 1, 2**64 - 1,
     [1, 2**64 - 1, 2**64 - 1, 1, 1, 2**63 - 1, 2**63 - 1, 2**64 - 1, 2**63, 1, 2**63 - 1, 1]),
])
def test_single_record_protocol_bytes_unchanged(changes, source_size, output_size, words):
    encoded = _encode_reduction_records((dict(REDUCTION_FIBER, **changes),), source_size=source_size, output_size=output_size)
    assert encoded.dtype == np.uint8
    assert encoded.tobytes() == b"".join(word.to_bytes(8, "little") for word in words)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("changes", [
    {"source_shape": (2**64,)}, {"source_strides": (-(2**63) - 1,)}, {"source_offset": -1},
    {"output_offset": 2**63}, {"reduction_axes": (1,)}, {"output_shape": ()},
])
def test_encoding_rejects_invalid_representation(changes):
    with pytest.raises(ValueError):
        _encode_reduction_records((dict(REDUCTION_FIBER, **changes),), source_size=3, output_size=1)
