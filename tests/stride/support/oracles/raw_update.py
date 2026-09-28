"""Raw Update reference for beta restricted to zero or one."""

import jax.numpy as jnp
import numpy as np




def reference_update(source, base, records, alpha, beta):
    """Reference for beta=0/1 and records without overlapping destination writes."""
    result = base
    for record in records:
        source_indices = np.full(record.logical_shape, record.source_offset, dtype=np.int32)
        destination_indices = np.full(record.logical_shape, record.destination_offset, dtype=np.int32)
        for axis, extent in enumerate(record.logical_shape):
            shape = (1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1)
            coordinates = np.arange(extent, dtype=np.int32).reshape(shape)
            source_indices += coordinates * record.source_strides[axis]
            destination_indices += coordinates * record.destination_strides[axis]
        source_indices = source_indices.reshape(-1)
        destination_indices = destination_indices.reshape(-1)
        term = source[..., source_indices] if alpha == 1 else alpha * source[..., source_indices]
        if beta:
            term = term + base[..., destination_indices]
        if not jnp.iscomplexobj(base):
            term = jnp.real(term)
        result = result.at[..., destination_indices].set(term.astype(base.dtype))
    return result
