"""Native address validation and representability boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation, execute_copy, execute_reduction
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord, INT64_MAX

from tests.stride.support.availability import native_available
from tests.stride.support.ffi import reduction_layout
from tests.stride.support.oracles.addresses import addresses
from tests.stride.support.oracles.affine import signed_record


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_native_rejects_signed_stride_address_overflow():
    record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1))
    layout = encode_layout((record,), source_size=12, output_size=12)
    layout[9] = -(1 << 63)
    call = jax.ffi.ffi_call(operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 12), jnp.float32))
    with pytest.raises(Exception, match="source address arithmetic overflow"):
        call(jnp.arange(12, dtype=jnp.float32).reshape(1, 12), layout=layout).block_until_ready()


requires_native = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


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


@requires_native
def test_native_address_arithmetic_overflow_is_rejected_without_large_allocation():
    record = AffineRecord((2,), (1,), INT64_MAX, (1,), 0)
    layout = encode_layout((record,), source_size=INT64_MAX, output_size=2)
    with pytest.raises(Exception, match="address arithmetic overflow"):
        execute_copy(jnp.zeros(2, dtype=jnp.float32), layout=layout, output_size=2).block_until_ready()
