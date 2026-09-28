"""Host descriptor preparation uses each operation's native validation contract."""

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord
from tests.stride.support.descriptors import OPERATIONS, address_words, reduction_words




@pytest.mark.parametrize("operation", ("accumulation", "reduction"))
@pytest.mark.parametrize("owners,fiber,capacity", [
    pytest.param(2, 255, 0, id="below_threshold"),
    pytest.param(2, 256, 2, id="at_threshold"),
    pytest.param(2, 1023, 2, id="below_chunk"),
    pytest.param(2, 1024, 2, id="at_chunk"),
    pytest.param(2, 1025, 4, id="above_threshold"),
    pytest.param(2, 4096, 8, id="four_chunks"),
    pytest.param(33, 4096, 132, id="many_owners"),
    pytest.param(1 << 20, 1024, 1 << 20, id="at_capacity_limit"),
    pytest.param((1 << 20) + 1, 1024, 0, id="above_capacity_limit"),
    pytest.param(2, 64, 0, id="short_fiber"),
    pytest.param(2, 0, 0, id="empty_fiber"),
])
def test_sum_scratch_capacity_follows_owner_fiber_plan(operation, owners, fiber, capacity):
    record = AffineRecord((owners, fiber), (max(fiber, 1), 1), 0, (1, 0), 0)
    if operation == "reduction":
        encoded = encode_reduction_layout((record,), output_shapes=((owners, 1),),
            reduction_axes=((False, True),), source_size=max(owners * fiber, 1),
            output_size=max(owners, 1)).view("<i8")
    else:
        encoded = encode_layout((record,), source_size=max(owners * fiber, 1),
                                output_size=max(owners, 1))
    prepared = _native._stride_prepare_layout(operation, tuple(map(int, encoded)))
    assert _native._stride_sum_scratch_capacity(operation, prepared) == capacity


def test_unsigned_extreme_fiber_uses_sequential_without_rejecting_layout():
    words = (1, 8, 8, 1, 1, 0, 0, -1, 0, 1, 99, 1)
    prepared = _native._stride_prepare_layout("reduction", words)
    assert _native._stride_sum_scratch_capacity("reduction", prepared) == 0


def test_prepare_reduction_canonicalizes_only_reduced_destination_strides():
    words = reduction_words()
    result = _native._stride_prepare_layout("reduction", words)
    assert result == (1, 6, 2, 1, 2, 2, 0, 3, 2, -1, 3, 1, 2, 0, 1, 1, 0)
    assert _native._stride_prepare_layout("reduction", result) == result
    assert isinstance(result, tuple)
    assert all(type(word) is int for word in result)


@pytest.mark.parametrize("operation", OPERATIONS[:-1])
def test_prepare_address_layout_fuses_contiguous_axes(operation):
    words = address_words()
    expected = (1, 6, 6, 1, 1, 0, 0, 6, 1, 1)
    assert _native._stride_prepare_layout(operation, words) == expected
    assert _native._stride_prepare_layout(operation, list(words)) == expected
    assert _native._stride_prepare_layout(operation, expected) == expected


@pytest.mark.parametrize("operation", OPERATIONS[:-1])
def test_prepare_overlap_depends_on_operation(operation):
    words = address_words(destination_strides=(1, 1))
    if operation in ("copy", "update", "accumulation"):
        with pytest.raises(ValueError, match="injective|overlap"):
            _native._stride_prepare_layout(operation, words)
    else:
        result = _native._stride_prepare_layout(operation, words)
        assert _native._stride_prepare_layout(operation, result) == result


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mutation", ["version", "truncated", "trailing", "source", "destination"])
def test_prepare_rejects_invalid_descriptors(operation, mutation):
    words = list(reduction_words() if operation == "reduction" else address_words())
    if mutation == "version":
        words[0] = 99
    elif mutation == "truncated":
        words.pop()
    elif mutation == "trailing":
        words.append(0)
    elif mutation == "source":
        words[5] = 100
    else:
        words[6] = 100
    with pytest.raises(ValueError):
        _native._stride_prepare_layout(operation, words)


@pytest.mark.parametrize("index,value,message", [
    (-1, 2, "axis flag"), (12, 3, "shapes do not match axis roles"),
])
def test_prepare_reduction_validates_axis_roles(index, value, message):
    words = list(reduction_words())
    words[index] = value
    with pytest.raises(ValueError, match=message):
        _native._stride_prepare_layout("reduction", words)


def test_prepare_reduction_preserves_unsigned_extents_with_x64_disabled():
    words = (1, 1, 1, 1, 1, 0, 0, -1, 0, 1, -(1 << 63), 1)
    with jax.enable_x64(False):
        result = _native._stride_prepare_layout("reduction", words)
    assert result == (1, 1, 1, 1, 1, 0, 0, -1, 0, 1, 0, 1)
    assert np.asarray(result, dtype=np.int64).view(np.uint64)[7] == (1 << 64) - 1


@pytest.mark.parametrize("suffix", [(), (0,), (0, -1, 0)])
def test_prepare_reduction_original_prefix_overflow_has_priority(suffix):
    words = [1, 0, 2, 1, 3, 0, 0, -1, 2, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1]
    if len(suffix) == 3:
        words[3] = 2
    with pytest.raises(ValueError, match="layout element count overflows"):
        _native._stride_prepare_layout("reduction", (*words, *suffix))


def test_prepare_unknown_operation():
    with pytest.raises(ValueError):
        _native._stride_prepare_layout("unknown", address_words())


@pytest.mark.parametrize("value", [1 << 63, -(1 << 63) - 1])
def test_prepare_requires_signed_i64_python_words(value):
    with pytest.raises(OverflowError):
        _native._stride_prepare_layout("copy", (1, value, 0, 0))


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("words", [(), (1,), (1, 0, 0), (1, 0, 0, 1)])
def test_prepare_rejects_truncated_headers_and_records(operation, words):
    with pytest.raises(ValueError):
        _native._stride_prepare_layout(operation, words)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_prepare_empty_record_list(operation):
    assert _native._stride_prepare_layout(operation, (1, 0, 0, 0)) == (1, 0, 0, 0)


def test_prepare_reduction_requires_injective_output_even_for_empty_fiber():
    words = (1, 0, 3, 1, 3, 0, 0,
             2, 2, 0, 0, 0, 0, 2, 2, 1, 1, 1, (1 << 63) - 1, 0, 0, 1)
    with pytest.raises(ValueError, match="injective|overlap"):
        _native._stride_prepare_layout("reduction", words)


@pytest.mark.parametrize("operation", OPERATIONS[:-1])
def test_prepare_preserves_singletons_empty_scalars_and_record_order(operation):
    words = (1, 6, 6, 4,
             0, 2, 3,
             3, 0, 0, 2, 1, 3, 3, -(1 << 63), 1, 3, (1 << 63) - 1, 1,
             3, 6, 6, 0, 2, 3, 0, 3, 1, 0, 3, 1,
             2, 5, 5, 2, 3, -3, -1, -3, -1)
    prepared = _native._stride_prepare_layout(operation, words)
    assert prepared[:4] == words[:4]
    assert _native._stride_prepare_layout(operation, prepared) == prepared


def test_prepare_reduction_fuses_shapes_and_roles_together():
    words = (1, 24, 6, 1, 4, 0, 0,
             2, 3, 2, 2, 12, 4, 2, 1,
             2, 3, 1, 1, 3, 1, -99, 99, 0, 0, 1, 1)
    expected = (1, 24, 6, 1, 2, 0, 0, 4, 6, 1, 4, 1, 6, 0, 1, 1, 0)
    assert _native._stride_prepare_layout("reduction", words) == expected
    assert _native._stride_prepare_layout("reduction", expected) == expected


def test_prepare_fused_extent_uses_operation_protocol_limit():
    maximum = (1 << 63) - 1
    address = (1, 1, 1, 1, 2, 0, 0, maximum, 2, 0, 0, 0, 0)
    assert _native._stride_prepare_layout("dot", address) == address
    reduction = (1, 1, 1, 1, 2, 0, 0, maximum, 2, 0, 0, 1, 1, -99, 99, 1, 1)
    expected = (1, 1, 1, 1, 1, 0, 0, -2, 0, 1, 0, 1)
    with jax.enable_x64(False):
        assert _native._stride_prepare_layout("reduction", reduction) == expected
    assert _native._stride_prepare_layout("reduction", expected) == expected


@pytest.mark.parametrize("operation", OPERATIONS[:-1])
def test_prepare_column_major_sort_and_fuse(operation):
    words = (1, 6, 6, 1, 2, 0, 0, 2, 3, 1, 2, 1, 2)
    canonical = (1, 6, 6, 1, 1, 0, 0, 6, 1, 1)
    assert _native._stride_prepare_layout(operation, words) == canonical
    assert _native._stride_prepare_layout(operation, canonical) == canonical


@pytest.mark.parametrize("operation", ("copy", "update"))
def test_prepare_recomputes_locality_after_singleton_removal(operation):
    words = (1, 9, 15, 1, 3, 2, 0, 2, 1, 3, 4, 0, -1, 2, -1, 6)
    canonical = (1, 9, 15, 1, 2, 2, 0, 2, 3, 4, -1, 2, 6)
    assert _native._stride_prepare_layout(operation, words) == canonical
    assert _native._stride_prepare_layout(operation, canonical) == canonical

@pytest.mark.parametrize("operation", OPERATIONS)
def test_owner_fiber_packer_is_a_separate_mechanical_device_encoding(operation):
    address = (1, 12, 8, 1, 2, 4, 1, 2, 3, 3, -1, 3, 1)
    reduction = (1, 12, 8, 1, 4, 4, 1, 2, 3, 1, 2,
                 6, -2, -(1 << 63), 1, 2, 1, 1, 1,
                 1, 0, 0, 0, 0, 1, 0, 1)
    semantic = reduction if operation == "reduction" else address
    with jax.enable_x64(False):
        packed = _native._stride_pack_owner_fiber(operation, semantic)
    if operation == "reduction":
        expected = (1, 12, 8, 1, 2, 2, 4, 1, 2, 1, 3, 2,
                    6, -(1 << 63), -2, 1, 1, 0, 0, 0)
    elif operation == "dot":
        expected = (1, 12, 8, 1, 0, 2, 4, 1, 2, 3, 3, -1, 3, 1)
    else:
        expected = (1, 12, 8, 1, 2, 0, 4, 1, 2, 3, 3, -1, 3, 1)
    assert packed == expected
    assert _native._stride_pack_owner_fiber(operation, semantic) == packed
    assert _native._stride_prepare_layout(operation, semantic) != packed


def test_owner_fiber_packer_checks_operation_and_original_descriptor():
    with pytest.raises(ValueError, match="unsupported owner/fiber operation"):
        _native._stride_pack_owner_fiber("unknown", address_words())
    colliding = (1, 4, 3, 1, 2, 0, 0, 2, 2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="cannot prove injective output owners"):
        _native._stride_pack_owner_fiber("accumulation", colliding)
    with pytest.raises(ValueError, match="layout element count overflows"):
        _native._stride_pack_owner_fiber("reduction", (1, 0, 2, 1, 3, 0, 0,
            -1, 2, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1))
    with pytest.raises(ValueError, match="trailing words"):
        _native._stride_pack_owner_fiber("accumulation", (*address_words(), 7))


def test_dot_scratch_capacity_uses_native_prepared_schedule():
    for count, expected in ((0, 0), (1, 0), (2, 1), (1024, 1),
                            (1025, 2), (65536, 64)):
        words = (1, 1, 1, 1, 1, 0, 0, count, 0, 0)
        assert _native._stride_dot_scratch_capacity(words) == expected
    # Record scratch is reused in stream order, not summed across records.
    records = (1, 1, 1, 2, 1, 0, 0, 1025, 0, 0, 1, 0, 0, 0, 0, 0)
    assert _native._stride_dot_scratch_capacity(records) == 2
