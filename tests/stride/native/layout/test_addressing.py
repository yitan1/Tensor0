"""Native affine addressing and layout traversal."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import copy_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.addresses import addresses
from tests.stride.support.oracles.affine import (
    assert_close,
    broadcast_record,
    native,
    reference,
    signed_record,
    values,
)
from tests.stride.support.oracles.copy import reference_map


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("permutation,source_signs,destination_fastest,destination_signs", [
    ((0, 1), (-1, 1), (1, 0), (-1, 1)),
    ((0, 1), (1, -1), (0, 1), (1, -1)),
    ((1, 0), (-1, -1), (1, 0), (1, -1)),
    ((1, 0), (1, -1), (0, 1), (-1, 1)),
])
def test_native_executes_signed_permuted_layouts(permutation, source_signs, destination_fastest, destination_signs):
    record = signed_record(logical_to_physical=permutation, source_signs=source_signs,
                           destination_fastest=destination_fastest, destination_signs=destination_signs)
    source = values("float32", 6)
    compiled = jax.jit(lambda value: native(value, record, 1.25, "float32"))
    assert_close(compiled(source), reference(source, record, 1.25, "float32"))
    text = compiled.lower(source).as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("axes,signs,fastest,destination_signs", [
    ((0,), (1,), (1, 0), (1, 1)), ((0,), (-1,), (0, 1), (-1, 1)),
    ((1,), (-1,), (1, 0), (1, -1)), ((0, 1), (), (0, 1), (-1, -1)),
])
def test_native_executes_signed_broadcast_layouts(axes, signs, fastest, destination_signs):
    record, size = broadcast_record(broadcast_axes=axes, source_signs=signs,
                                    destination_fastest=fastest, destination_signs=destination_signs)
    source = values("float32", size)
    assert_close(jax.jit(lambda value: native(value, record, -.75, "float32"))(source),
                 reference(source, record, -.75, "float32"))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("extent", [2, 3, 4, 7, 16, 64])
def test_broadcast_contiguous_row_extents(extent):
    record, size = broadcast_record(shape=(257, extent), broadcast_axes=(1,), source_signs=(1,))
    source = values("float32", size)
    assert_close(jax.jit(lambda value: native(value, record, 1.25, "float32"))(source),
                 reference(source, record, 1.25, "float32"))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("compact", [False, True])
def test_nine_dimensional_layout_no_longer_requires_rank_compression(compact):
    strides = tuple((2 if compact else 3)**axis for axis in range(8, -1, -1))
    record = AffineRecord((2,) * 9, strides, 0, tuple(2**axis for axis in range(8, -1, -1)), 0)
    source = jnp.arange(sum(strides) + 1, dtype=jnp.float32)
    actual = jax.jit(lambda value: copy_p.bind(value, records=(record,), output_size=512, dtype=value.dtype))(source)
    np.testing.assert_array_equal(actual, reference_map(np.asarray(source), (record,), (1,), 512))


requires_native = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


U1_LAYOUTS = [
    ((AffineRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
      AffineRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1)), 5),
    ((AffineRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
      AffineRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
      AffineRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072)), 4608),
]


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
