"""Raw Dot descriptor, buffer and conjugation flag validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_dot
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available


def test_native_dot_rejects_malformed_descriptor() -> None:
    left = jnp.arange(4, dtype=jnp.float32)
    right = jnp.arange(4, dtype=jnp.float32)
    descriptor = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),),
                               source_size=4, output_size=4)
    descriptor[0] ^= 0xFF

    with pytest.raises(Exception, match="unsupported native layout version"):
        execute_dot(
            left,
            right,
            layout=descriptor,
            conjugate_left=False,
        ).block_until_ready()


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


def _dot_layout(records, left_size, right_size):
    return encode_layout(tuple(AffineRecord(*record) for record in records),
                         source_size=left_size, output_size=right_size)


def _raw_dot(left, right, layout, *, conjugate=0, dtype=None, shape=None):
    dtype = left.dtype if dtype is None else jnp.dtype(dtype)
    execute = jax.ffi.ffi_call(
        operation_target("dot", dtype),
        jax.ShapeDtypeStruct((left.shape[0],) if shape is None else shape, dtype),
        vmap_method="sequential",
    )
    return execute(left, right, layout=layout, conjugate_left=np.int64(conjugate))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_invalid_buffers_and_flags():
    layout = _dot_layout((((3,), (1,), 0, (1,), 0),), 3, 3)
    valid = jnp.ones((1, 3))
    for left, right, shape in ((jnp.ones(3), valid, (1,)), (valid, jnp.ones(3), (1,)),
                               (valid, valid, (1, 1)), (valid, valid, (2,)),
                               (jnp.ones((1, 4)), valid, (1,)), (valid, jnp.ones((2, 3)), (1,))):
        with pytest.raises(Exception, match="rank|dimensions"):
            _raw_dot(left, right, layout, shape=shape).block_until_ready()
    for left, right in ((valid.astype(jnp.float8_e4m3fn), valid), (valid, valid.astype(jnp.float8_e5m2))):
        with pytest.raises(Exception, match="dtype|type"):
            _raw_dot(left, right, layout, dtype="float32").block_until_ready()
    for conjugate in (-1, 2):
        with pytest.raises(Exception, match="conjugate_left"):
            _raw_dot(valid, valid, layout, conjugate=conjugate).block_until_ready()


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_invalid_protocol_and_read_bounds():
    layout = _dot_layout((((2,), (1,), 0, (1,), 0),), 2, 2)
    source = jnp.ones((1, 2))
    for length in range(len(layout)):
        with pytest.raises(Exception, match="truncated|descriptor length"):
            _raw_dot(source, source, layout[:length]).block_until_ready()
    for index, value in ((0, 2), (1, -1), (2, -1), (3, -1), (3, 2**63 - 1),
                         (4, 2**63 - 1), (5, -1), (6, 2), (7, -1), (8, 2**63 - 1)):
        invalid = layout.copy()
        invalid[index] = value
        with pytest.raises(Exception):
            _raw_dot(source, source, invalid).block_until_ready()
    with pytest.raises(Exception, match="trailing"):
        _raw_dot(source, source, np.append(layout, 0)).block_until_ready()
