"""Raw reduction FFI calls and descriptor fixtures."""

import jax
import jax.numpy as jnp
import numpy as np

from tensor0._stride._ffi._descriptor import encode_reduction_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._layout import AffineRecord


REDUCTION_FIBER = dict(source_shape=(3,), source_strides=(1,), source_offset=0, output_shape=(1,), output_strides=(1,), output_offset=0, reduction_axes=(True,))


def _encode_reduction_records(records, *, source_size, output_size):
    """Encode dictionary-based reduction fixtures in one batch."""
    return encode_reduction_layout(
        tuple(AffineRecord(record["source_shape"], record["source_strides"], record["source_offset"],
                           record["output_strides"], record["output_offset"]) for record in records),
        output_shapes=tuple(record["output_shape"] for record in records),
        reduction_axes=tuple(record["reduction_axes"] for record in records),
        source_size=source_size, output_size=output_size,
    )


def _raw_reduction_call(source, layout, *, shape=(1, 1), dtype="float32", coefficients=(), indices=(), aliases=None):
    execute = jax.ffi.ffi_call(
        operation_target("reduction", jnp.dtype(dtype)), jax.ShapeDtypeStruct(shape, dtype),
        input_output_aliases=aliases,
    )
    return execute(source, *coefficients, layout=layout, coefficient_records=np.asarray(indices, dtype=np.int64))


def reduction_layout(source_shape=(4, 3), output_shape=(1, 3), output_strides=(3, 1),
                     source_offset=0, source_size=12, output_size=3):
    return encode_reduction_layout(
        (AffineRecord(source_shape, (3, 1), source_offset, output_strides, 0),),
        output_shapes=(output_shape,), reduction_axes=((True, False),),
        source_size=source_size, output_size=output_size)
