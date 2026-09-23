"""Update buffer aliasing and mixed-storage boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.update_storage import MIXED_RECORDS, mixed_operands, reference


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("reuse_base", [False, True])
def test_same_update_target_accepts_separate_output_and_base_reuse(reuse_base):
    records = (AffineRecord((3,), (2,), 0, (2,), 1),)
    layout = encode_layout(records, source_size=8, output_size=8)
    call = jax.ffi.ffi_call(
        operation_target("update", np.dtype(jnp.float32)), jax.ShapeDtypeStruct((2, 8), jnp.float32),
        input_output_aliases={1: 0} if reuse_base else {},
    )
    operation = jax.jit(lambda source, base, alpha, beta: call(source, base, alpha, beta, layout=layout),
                        donate_argnums=(1,) if reuse_base else ())
    source = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    base = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) + 20
    original_source, original_base = np.asarray(source).copy(), np.asarray(base).copy()
    expected = original_base.copy()
    expected[:, 1:6:2] = 2 * original_source[:, :5:2] - original_base[:, 1:6:2]
    pointer = base.unsafe_buffer_pointer()
    actual = operation(source, base, jnp.float32(2), jnp.float32(-1))
    actual.block_until_ready()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original_source)
    if reuse_base:
        assert base.is_deleted() and actual.unsafe_buffer_pointer() == pointer
    else:
        np.testing.assert_array_equal(base, original_base)


@pytest.mark.parametrize("factor", [0, 2])
def test_update_rejects_source_only_alias_even_for_zero_contribution(factor):
    layout = encode_layout((AffineRecord((8,), (1,), 0, (1,), 0),), source_size=8, output_size=8)
    call = jax.ffi.ffi_call(operation_target("update", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 8), jnp.float32), input_output_aliases={0: 0})
    source = jnp.arange(8, dtype=jnp.float32).reshape(1, 8)
    base = source + 20
    with pytest.raises(Exception, match="unsupported source/result alias"):
        call(source, base, jnp.float32(factor), jnp.float32(0), layout=layout).block_until_ready()
    np.testing.assert_array_equal(source, np.arange(8).reshape(1, 8))
    np.testing.assert_array_equal(base, np.arange(8).reshape(1, 8) + 20)


@pytest.mark.parametrize("invalid", ["version", "truncated", "storage_size", "buffer_dtype"])
def test_update_ffi_validation_with_base_alias(invalid):
    layout = encode_layout((AffineRecord((8,), (1,), 0, (1,), 0),), source_size=8, output_size=8)
    dtype, shape = jnp.float32, (1, 8)
    if invalid == "version":
        layout[0] = 99
        message = "unsupported native layout version"
    elif invalid == "truncated":
        layout = layout[:-1]
        message = "layout rank exceeds descriptor length"
    elif invalid == "storage_size":
        shape = (1, 9)
        message = "buffer dimensions do not match layout"
    else:
        dtype = jnp.complex64
        message = "dtype|element type"
    call = jax.ffi.ffi_call(operation_target("update", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct(shape, dtype), input_output_aliases={1: 0})
    source = jnp.arange(shape[-1], dtype=jnp.float32).reshape(shape).astype(dtype)
    base = source + 20
    with pytest.raises(Exception, match=message):
        call(source, base, jnp.float32(2), jnp.float32(0), layout=layout).block_until_ready()
    np.testing.assert_array_equal(source, np.arange(shape[-1]).reshape(shape))
    np.testing.assert_array_equal(base, np.arange(shape[-1]).reshape(shape) + 20)


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('source_dtype,base_dtype', [('float16', 'float32'), ('float32', 'float16')])
def test_native_mixed_source_rank_and_base_alias_checks(source_dtype, base_dtype):
    layout = encode_layout(MIXED_RECORDS, source_size=5, output_size=10)
    source, base = mixed_operands(source_dtype, base_dtype, (1,))
    target = operation_target('update', base.dtype)
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(base.shape, base.dtype))
    with pytest.raises(Exception, match='rank-two'):
        call(source[0], base, jnp.int32(1), jnp.int32(1), layout=layout).block_until_ready()
    alias_call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(base.shape, base.dtype), input_output_aliases={1: 0})
    result = alias_call(source, base, jnp.int32(1), jnp.int32(1), layout=layout)
    np.testing.assert_array_equal(result, reference(source, base, 1, 1))
