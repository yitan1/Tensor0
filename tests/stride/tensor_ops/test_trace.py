"""Packed trace layout, coefficient and boundary contracts."""

from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import _jax
from tensor0._stride._layout import AffineRecord, contiguous_strides
from tensor0._stride._tensor_ops import _strided_tensortrace

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.trace import scalar_trace


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_packed_trace_zero_output_storage():
    actual = _strided_tensortrace(
        jnp.zeros((2, 0)), destination_size=0,
        source_subblocks=(SimpleNamespace(sizes=(0, 2, 2), strides=(4, 2, 1), offset=0),),
        destination_subblocks=(SimpleNamespace(sizes=(0,), strides=(1,), offset=0),),
        entries=((0, 0, jnp.float32(2)),), result_dtype=jnp.float32, permutation=(),
        num_open_out=1, num_open_in=0, trace_count=1)
    assert actual.shape == (2, 0)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_packed_trace_empty_record_keeps_original_coefficient_index():
    actual = _strided_tensortrace(
        jnp.asarray([2], dtype=jnp.int32), destination_size=1,
        source_subblocks=(SimpleNamespace(sizes=(0, 0), strides=(1, 1), offset=0),
                          SimpleNamespace(sizes=(1, 1), strides=(1, 1), offset=0)),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
        entries=((0, 0, None), (1, 0, jnp.int32(3))), result_dtype=jnp.int32,
        permutation=(), num_open_out=0, num_open_in=0, trace_count=1)
    np.testing.assert_array_equal(actual, [6])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_trace_vector_coefficients_remain_rejected():
    source = jnp.ones((2, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="scalars shared"):
        scalar_trace(source, (jnp.ones(2),), source.dtype)
