"""Address encoding and operation-specific layout boundaries."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from itertools import permutations, product
from math import prod
from pathlib import Path
import tomllib

import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation, execute_copy, execute_reduction, execute_update
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout, merge_reduction_layouts
from tensor0._stride._layout import AffineRecord, INT64_MAX

from ._support import native_available


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_FFI_FILE_HASHES = {
    "include/xla/ffi/api/api.h": "7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb",
    "include/xla/ffi/api/c_api.h": "85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539",
    "include/xla/ffi/api/ffi.h": "4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5",
    "LICENSE.txt": "e3d8688a2c75d4e33641cc88046a8a1593ee79822411fa45a7bd32e37f847d29",
}
requires_native = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
U1_LAYOUTS = [
    ((AffineRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
      AffineRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1)), 5),
    ((AffineRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
      AffineRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
      AffineRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072)), 4608),
]


def compact_strides(shape, order):
    strides = [0] * len(shape)
    stride = 1
    for axis in order:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def addresses(shape, strides, offset):
    return np.asarray([offset + sum(index * stride for index, stride in zip(coordinates, strides, strict=True))
                       for coordinates in product(*(range(extent) for extent in shape))], dtype=np.int64)


def reduction_layout(source_shape=(4, 3), output_shape=(1, 3), output_strides=(3, 1),
                     source_offset=0, source_size=12, output_size=3):
    return encode_reduction_layout(source_shape=source_shape, source_strides=(3, 1), source_offset=source_offset,
        output_shape=output_shape, output_strides=output_strides, output_offset=0,
        reduction_axes=(True, False), source_size=source_size, output_size=output_size)


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


def test_vendored_ffi_headers_and_license_match_frozen_hashes():
    vendor_root = _REPO_ROOT / "crates" / "tensor0-py" / "vendor" / "jaxlib-0.10.1"
    for relative_path, expected in _VENDORED_FFI_FILE_HASHES.items():
        assert hashlib.sha256((vendor_root / relative_path).read_bytes()).hexdigest() == expected
    packaged_license = _REPO_ROOT / "src" / "tensor0" / "_licenses" / "JAXLIB_FFI_LICENSE.txt"
    assert hashlib.sha256(packaged_license.read_bytes()).hexdigest() == _VENDORED_FFI_FILE_HASHES["LICENSE.txt"]


def test_declared_jax_runtime_matches_vendored_ffi_version():
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    build_script = (_REPO_ROOT / "crates" / "tensor0-py" / "build.rs").read_text()
    assert "jax==0.10.1" in dependencies
    assert "jaxlib==0.10.1" in dependencies
    assert 'const VENDORED_JAX_VERSION: &str = "0.10.1";' in build_script
    assert 'const VENDORED_JAXLIB_VERSION: &str = "0.10.1";' in build_script


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


@requires_native
def test_layout_reuse_does_not_capture_factors_or_initialization():
    layout = encode_layout((AffineRecord((2,), (1,), 0, (1,), 1),), source_size=2, output_size=4)
    before = layout.tobytes()
    source, base = jnp.asarray([2., 3.], dtype=jnp.float32), jnp.full(4, 7., dtype=jnp.float32)
    np.testing.assert_array_equal(execute_copy(source, layout=layout, output_size=4), [0, 2, 3, 0])
    for factor in (1, 2, -1):
        coefficient = jnp.float32(factor)
        mapped = execute_accumulation(source, (coefficient,), coefficient_records=(0,), layout=layout, output_size=4)
        np.testing.assert_array_equal(mapped, [0, 2 * factor, 3 * factor, 0])
        for beta in (0, 1):
            updated = execute_update(source, base, coefficient, jnp.int32(beta), layout=layout)
            np.testing.assert_array_equal(updated, [7, 2 * factor + 7 * beta, 3 * factor + 7 * beta, 7])
    assert layout.tobytes() == before


def test_reduction_encoding_retains_real_output_shape_and_axes():
    layout = reduction_layout()
    assert layout.dtype == np.uint8
    np.testing.assert_array_equal(layout.view("<u8"), [1, 12, 3, 1, 2, 0, 0, 4, 3, 3, 1, 1, 3, 3, 1, 1, 0])


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
    layouts = (reduction_layout(source_size=24), reduction_layout(source_offset=12, source_size=24))
    merged = merge_reduction_layouts(layouts, source_size=24, output_size=3)
    assert merged.view("<u8")[3] == 2
    source = jnp.arange(24, dtype=jnp.float32)
    np.testing.assert_array_equal(execute_reduction(source, layout=merged, output_size=3),
                                  np.arange(24).reshape(8, 3).sum(axis=0))


@requires_native
@pytest.mark.parametrize("output_shape,output_strides,output_size,message", [
    ((2, 3), (3, 1), 6, "shapes do not match axis roles"),
    ((1, 3), (3, 0), 3, "zero nontrivial stride"),
    ((1, 3), (3, 2), 3, "address exceeds storage"),
])
def test_reduction_validates_true_output_layout(output_shape, output_strides, output_size, message):
    layout = reduction_layout(output_shape=output_shape, output_strides=output_strides, output_size=output_size)
    with pytest.raises(Exception, match=message):
        execute_reduction(jnp.arange(12, dtype=jnp.float32), layout=layout, output_size=output_size).block_until_ready()


@requires_native
def test_empty_reduction_still_checks_nonempty_output_bounds():
    layout = reduction_layout(source_shape=(0, 3), source_size=0, output_size=2)
    with pytest.raises(Exception, match="address exceeds storage"):
        execute_reduction(jnp.empty(0, dtype=jnp.float32), layout=layout, output_size=2).block_until_ready()


@requires_native
@pytest.mark.parametrize("records,size", U1_LAYOUTS)
def test_frozen_u1_address_layouts_are_permutations_and_execute(records, size):
    source = jnp.arange(size, dtype=jnp.float32)
    expected = np.zeros(size, dtype=np.float32)
    sources, destinations = [], []
    for record in records:
        selected = addresses(record.logical_shape, record.source_strides, record.source_offset)
        target = addresses(record.logical_shape, record.destination_strides, record.destination_offset)
        sources.extend(selected)
        destinations.extend(target)
        expected[target] = np.asarray(source)[selected]
    np.testing.assert_array_equal(np.sort(sources), np.arange(size))
    np.testing.assert_array_equal(np.sort(destinations), np.arange(size))
    layout = encode_layout(records, source_size=size, output_size=size)
    np.testing.assert_array_equal(execute_copy(source, layout=layout, output_size=size), expected)


@requires_native
def test_repeated_positive_source_addresses_are_valid():
    record = AffineRecord((5, 5), (2, 2), 0, (5, 1), 0)
    source = jnp.arange(17, dtype=jnp.float32)
    layout = encode_layout((record,), source_size=17, output_size=25)
    expected = np.asarray(source)[addresses(record.logical_shape, record.source_strides, 0)]
    np.testing.assert_array_equal(execute_copy(source, layout=layout, output_size=25), expected)


@requires_native
@pytest.mark.parametrize("strides", [(2, 2), (0, 1)])
def test_map_rejects_repeated_targets_but_address_accumulation_accepts_them(strides):
    record = AffineRecord((5, 5), (5, 1), 0, strides, 0)
    source = jnp.arange(25, dtype=jnp.float32)
    layout = encode_layout((record,), source_size=25, output_size=25)
    with pytest.raises(Exception, match="injective"):
        execute_copy(source, layout=layout, output_size=25).block_until_ready()
    expected = np.zeros(25, dtype=np.float32)
    np.add.at(expected, addresses(record.logical_shape, strides, 0), np.asarray(source))
    np.testing.assert_array_equal(execute_accumulation(source, layout=layout, output_size=25), expected)


@pytest.mark.parametrize("record,size,message", [
    (AffineRecord((2,), (), 0, (1,), 0), 2, "ranks must match"),
    (AffineRecord((2,), (1,), -1, (1,), 0), 2, "nonnegative int64"),
    (AffineRecord((2,), (1 << 63,), 0, (1,), 0), 2, "strides must fit int64"),
    (AffineRecord((2,), (1,), 0, (1,), 0), 1 << 63, "nonnegative int64"),
])
def test_address_encoder_rejects_unrepresentable_fields(record, size, message):
    with pytest.raises(ValueError, match=message):
        encode_layout((record,), source_size=size, output_size=2)


@requires_native
def test_native_address_arithmetic_overflow_is_rejected_without_large_allocation():
    record = AffineRecord((2,), (1,), INT64_MAX, (1,), 0)
    layout = encode_layout((record,), source_size=INT64_MAX, output_size=2)
    with pytest.raises(Exception, match="address arithmetic overflow"):
        execute_copy(jnp.zeros(2, dtype=jnp.float32), layout=layout, output_size=2).block_until_ready()
