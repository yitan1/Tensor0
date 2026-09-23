"""Address accumulation protocol, aliases and buffer validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation, execute_copy
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.layouts import OVERLAP


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
def test_map_and_axis_reduction_still_reject_overlap():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=5)
    with pytest.raises(Exception, match='injective'):
        execute_copy(jnp.ones(4), layout=layout, output_size=5).block_until_ready()
    axis_layout = encode_reduction_layout(
        (AffineRecord((2, 2), (1, 2), 0, (1, 1), 1),),
        output_shapes=((2, 2),), reduction_axes=((False, False),),
        source_size=4, output_size=5)
    with pytest.raises(Exception, match='injective'):
        jax.ffi.ffi_call(operation_target('reduction', jnp.dtype('float32')), jax.ShapeDtypeStruct((1, 5), jnp.float32))(jnp.ones((1, 4)), layout=axis_layout, coefficient_records=np.asarray([], np.int64)).block_until_ready()


def raw_call(source, layout, *, coefficients=(), indices=(), shape=(1, 5), dtype=jnp.float32, aliases=None):
    return jax.ffi.ffi_call(operation_target('accumulation', jnp.dtype(dtype)), jax.ShapeDtypeStruct(shape, dtype), vmap_method='sequential', input_output_aliases=aliases)(source, *coefficients, layout=layout, coefficient_records=np.asarray(indices, dtype=np.int64))


@pytest.mark.usefixtures("enable_x64")
def test_invalid_protocol_and_buffers():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=5)
    for length in range(layout.size):
        with pytest.raises(Exception, match='descriptor|record count|rank'):
            raw_call(jnp.ones((1, 4)), layout[:length]).block_until_ready()
    malformed = [np.append(layout, 0)]
    for index, value in ((0, 2), (1, -1), (2, 2), (3, -1), (4, -1), (5, -1), (6, 4), (7, -1)):
        words = layout.copy()
        words[index] = value
        malformed.append(words)
    for words in malformed:
        with pytest.raises(Exception, match='layout|address|descriptor'):
            raw_call(jnp.ones((1, 4)), words).block_until_ready()
    for source, shape in ((jnp.ones(4), (1, 5)), (jnp.ones((1, 3)), (1, 5)), (jnp.ones((1, 4)), (2, 5)), (jnp.ones((1, 4)), (1, 4))):
        with pytest.raises(Exception, match='rank-two|dimensions'):
            raw_call(source, layout, shape=shape).block_until_ready()
    for indices, coefficients in (((0,), ()), ((), (jnp.float32(2),)), ((1,), (jnp.float32(2),)), ((-1,), (jnp.float32(2),)), ((0, 0), (jnp.float32(2), jnp.float32(3)))):
        with pytest.raises(Exception, match='coefficient.*(count|indices)'):
            raw_call(jnp.ones((1, 4)), layout, coefficients=coefficients, indices=indices).block_until_ready()
    for coefficient in (jnp.ones(2), jnp.ones((1, 1)), jnp.float8_e4m3fn(1)):
        with pytest.raises(Exception, match='batch count|unsupported scalar dtype'):
            raw_call(jnp.ones((1, 4)), layout, coefficients=(coefficient,), indices=(0,)).block_until_ready()
    with pytest.raises(Exception, match='unsupported scalar dtype'):
        raw_call(jnp.ones((1, 4), jnp.float8_e4m3fn), layout).block_until_ready()


@pytest.mark.usefixtures("enable_x64")
def test_alias_and_python_boundaries():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=4)
    with pytest.raises(Exception, match='overlap'):
        raw_call(jnp.ones((1, 4), dtype=jnp.float32), layout, shape=(1, 4), aliases={0: 0}).block_until_ready()
    with pytest.raises(TypeError, match='JAX'):
        execute_accumulation(np.ones(4), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='storage'):
        execute_accumulation(jnp.float32(1), layout=layout, output_size=4)
    with pytest.raises(TypeError, match='explicit dtypes'):
        execute_accumulation(jnp.ones(4), (2,), coefficient_records=(0,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='batch shape'):
        execute_accumulation(jnp.ones(4), (jnp.ones(2),), coefficient_records=(0,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='nonnegative int64'):
        execute_accumulation(jnp.ones(4), coefficient_records=(-1,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='cannot be differentiated'):
        jax.grad(lambda source: execute_accumulation(source, layout=layout, output_size=4).sum())(jnp.ones(4))


@pytest.mark.usefixtures("enable_x64")
def test_unsupported_source_and_result_dtypes():
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    with pytest.raises(Exception, match='unsupported scalar dtype'):
        execute_accumulation(jnp.ones(1, dtype=jnp.float8_e4m3fn), (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=1, dtype='float32').block_until_ready()
    with pytest.raises(NotImplementedError, match='does not support'):
        execute_accumulation(jnp.ones(1), (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=1, dtype=jnp.float8_e4m3fn)
