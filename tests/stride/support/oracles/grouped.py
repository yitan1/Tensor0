"""Independent grouped-transform reference."""

from math import prod

import jax.numpy as jnp
import numpy as np


def indices(view):
    return [view.offset + sum((index * stride for index, stride in zip(coordinate, view.strides))) for coordinate in np.ndindex(tuple(view.sizes))]


def oracle(source, source_layout, destination_layout, dtype, permutation, groups):
    result = jnp.zeros((*source.shape[:-1], destination_layout.total_dim), dtype=dtype)
    for group in groups:
        source_shape = tuple(source_layout.subblockstructure[group.src_indices[0]].sizes)
        rows = jnp.stack([source[..., jnp.asarray(indices(source_layout.subblockstructure[index]), dtype=jnp.int32)] for index in group.src_indices], axis=-2)
        matrix = jnp.asarray(group.transform, dtype=dtype)
        if matrix.shape == (1, 1):
            factor = np.asarray(group.transform).reshape(())
            mapped = jnp.zeros_like(rows) if factor == 0 else rows if factor == 1 else matrix.reshape(()) * rows
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
                mapped = jnp.real(mapped)
            mapped = mapped.astype(dtype)
        else:
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
                rows = jnp.real(rows)
            mapped = matrix @ rows.astype(dtype)
        batch_shape = source.shape[:-1]
        for row, destination_index in enumerate(group.dst_indices):
            shaped = mapped[..., row, :].reshape((*batch_shape, *source_shape))
            axes = (*range(len(batch_shape)), *(len(batch_shape) + axis for axis in permutation))
            values = jnp.transpose(shaped, axes).reshape((*batch_shape, prod(source_shape)))
            result = result.at[..., jnp.asarray(indices(destination_layout.subblockstructure[destination_index]), dtype=jnp.int32)].set(values)
    return result
