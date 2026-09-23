"""NumPy mixed-storage Update reference and operands."""

from math import prod

import jax.numpy as jnp
import numpy as np

from tensor0._stride._layout import AffineRecord


def mixed_operands(source_dtype, base_dtype, batch_shape):
    source = (jnp.arange(prod(batch_shape) * 5) % 4 + 1).astype(source_dtype).reshape((*batch_shape, 5))
    base = (jnp.arange(prod(batch_shape) * 10) % 5 + 1).astype(base_dtype).reshape((*batch_shape, 10))
    if jnp.iscomplexobj(source):
        source = source + 2j
    if jnp.iscomplexobj(base):
        base = base - 3j
    return (source, base)


MIXED_RECORDS = (AffineRecord((2, 2), (0, -1), 3, (4, 2), 1), AffineRecord((), (), 2, (), 8))


def reference(source, base, alpha, beta, records=MIXED_RECORDS):
    expected = np.asarray(base).copy()
    for record in records:
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum((index * stride for index, stride in zip(coordinates, record.source_strides, strict=True)))
            destination_index = record.destination_offset + sum((index * stride for index, stride in zip(coordinates, record.destination_strides, strict=True)))
            source_term = 0 if alpha == 0 else np.asarray(source)[..., source_index] * alpha
            base_term = 0 if beta == 0 else np.asarray(base)[..., destination_index] * beta
            value = source_term + base_term
            if not jnp.iscomplexobj(base):
                value = np.real(value)
            expected[..., destination_index] = np.asarray(value).astype(base.dtype)
    return expected
