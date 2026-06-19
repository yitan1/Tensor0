from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import (
    get_blockstructure,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..structure.sector_dict import SectorDict


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


def pack_complete_blocks(
    space: _native.HomSpace,
    blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
    *,
    dtype: DTypeLike | None,
) -> Array:
    block_arrays = SectorDict(blocks)
    degeneracystructure = get_degeneracystructure(space)
    blockstructures = get_blockstructure(space)

    flat_blocks: list[Array] = []
    for coupled, block in blockstructures.items():
        if coupled not in block_arrays:
            raise ValueError(f"missing data for block sector {coupled}")

        expected_shape = (block.row_dim, block.col_dim)
        value = jnp.asarray(block_arrays[coupled], dtype=dtype)
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"block sector {coupled} has shape {actual_shape}; "
                f"expected {expected_shape}",
            )
        flat_blocks.append(jnp.reshape(value, (-1,)))

    for coupled, value in block_arrays.items():
        if coupled in blockstructures:
            continue
        array = jnp.asarray(value, dtype=dtype)
        if array.size != 0:
            raise ValueError(f"unexpected block sector {coupled}")

    if flat_blocks:
        return jnp.concatenate(tuple(flat_blocks), axis=0)
    return jnp.zeros((degeneracystructure.total_dim,), dtype=dtype)


def pack_blocks(
    space: _native.HomSpace,
    blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
    *,
    dtype: DTypeLike | None,
) -> Array:
    block_arrays = SectorDict(blocks)
    degeneracystructure = get_degeneracystructure(space)
    blockstructures = get_blockstructure(space)

    flat_blocks: list[Array] = []
    for coupled, block in blockstructures.items():
        expected_shape = (block.row_dim, block.col_dim)
        value = block_arrays.get(coupled)
        if value is None:
            value = jnp.zeros(expected_shape, dtype=dtype)
        else:
            value = jnp.asarray(value, dtype=dtype)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"block {coupled} has shape {actual_shape}; "
                    f"expected {expected_shape}",
                )
        flat_blocks.append(jnp.reshape(value, (-1,)))

    if flat_blocks:
        return jnp.concatenate(tuple(flat_blocks), axis=0)
    return jnp.zeros((degeneracystructure.total_dim,), dtype=dtype)


def normalize_fusiontree_pair_key(
    key: object,
) -> tuple[_native.FusionTree, _native.FusionTree]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise TypeError("TensorMap indices must be a pair of FusionTree objects")
    row_tree, col_tree = key
    if not isinstance(row_tree, _native.FusionTree) or not isinstance(
        col_tree,
        _native.FusionTree,
    ):
        raise TypeError("TensorMap indices must be a pair of FusionTree objects")
    return row_tree, col_tree


def find_subblock_structure(
    space: _native.HomSpace,
    row_tree: _native.FusionTree,
    col_tree: _native.FusionTree,
) -> _native.SubblockStructure:
    key = (row_tree.static_key, col_tree.static_key)
    sectorstructure = get_sectorstructure(space)
    degeneracystructure = get_degeneracystructure(space)
    for pair, subblock in zip(
        sectorstructure.fusiontree_pairs,
        degeneracystructure.subblockstructure,
    ):
        if (pair[0].static_key, pair[1].static_key) == key:
            return subblock
    raise KeyError(key)


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
