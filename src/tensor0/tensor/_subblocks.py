from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .. import _native


def subblock_indices(subblock: _native.SubblockStructure) -> Array:
    return strided_indices(
        tuple(subblock.sizes),
        tuple(subblock.strides),
        subblock.offset,
    )


def gather_subblock(
    storage: object,
    subblock: _native.SubblockStructure,
    dtype: jnp.dtype | None = None,
) -> Array:
    return gather_strided(
        storage,
        tuple(subblock.sizes),
        tuple(subblock.strides),
        subblock.offset,
        dtype,
    )


def scatter_add_subblock(
    storage: Array,
    subblock: _native.SubblockStructure,
    value: Array,
) -> Array:
    return scatter_add_strided(
        storage,
        tuple(subblock.sizes),
        tuple(subblock.strides),
        subblock.offset,
        value,
    )


def gather_strided(
    storage: object,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    dtype: jnp.dtype | None = None,
) -> Array:
    data = jnp.asarray(storage)
    block = data[strided_indices(sizes, strides, offset)]
    if dtype is not None:
        block = jnp.asarray(block, dtype=dtype)
    return block.reshape(sizes)


def scatter_add_strided(
    storage: Array,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    value: Array,
) -> Array:
    return storage.at[strided_indices(sizes, strides, offset)].add(value.reshape(-1))


def strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> Array:
    indices = jnp.zeros(sizes, dtype=jnp.result_type(offset))
    for axis, (size, stride) in enumerate(zip(sizes, strides)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices = indices + jnp.arange(size).reshape(shape) * stride
    return (indices + offset).reshape(-1)
