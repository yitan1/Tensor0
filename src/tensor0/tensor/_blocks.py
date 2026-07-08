from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import math
from typing import Any

import jax
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


_StridedIndicesCacheKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    int,
    str,
    str,
]

_STRIDED_INDICES_CACHE_MAXSIZE = 1_024
_STRIDED_INDICES_CACHE: OrderedDict[_StridedIndicesCacheKey, Array] = OrderedDict()


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


def _is_contiguous_strided(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
) -> bool:
    if len(sizes) != len(strides):
        return False

    expected_stride = 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size != 1 and stride != expected_stride:
            return False
        expected_stride *= size

    return True


def gather_strided(
    storage: object,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    dtype: jnp.dtype | None = None,
) -> Array:
    data = jnp.asarray(storage)
    if _is_contiguous_strided(sizes, strides) and not _is_traced_array(offset):
        size = math.prod(sizes)
        block = data[offset : offset + size]
    else:
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
    if _is_contiguous_strided(sizes, strides) and not _is_traced_array(offset):
        size = math.prod(sizes)
        return storage.at[offset : offset + size].add(value.reshape(-1))
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


def _strided_indices_cache_key(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    index_dtype: jnp.dtype,
) -> _StridedIndicesCacheKey:
    backend = jax.default_backend()
    return (tuple(sizes), tuple(strides), int(offset), str(jnp.dtype(index_dtype)), backend)


def _clear_strided_indices_cache_for_tests() -> None:
    _STRIDED_INDICES_CACHE.clear()


def _is_traced_array(value: object) -> bool:
    return isinstance(value, jax.core.Tracer)


def _build_strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    index_dtype: jnp.dtype,
) -> Array:
    indices = jnp.zeros(sizes, dtype=index_dtype)
    for axis, (size, stride) in enumerate(zip(sizes, strides)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices = indices + jnp.arange(size, dtype=index_dtype).reshape(shape) * stride
    return (indices + offset).reshape(-1)


def strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> Array:
    index_dtype = jnp.result_type(offset)
    if _is_traced_array(offset):
        return _build_strided_indices(sizes, strides, offset, index_dtype)

    key = _strided_indices_cache_key(sizes, strides, offset, index_dtype)
    cached = _STRIDED_INDICES_CACHE.get(key)
    if cached is not None:
        _STRIDED_INDICES_CACHE.move_to_end(key)
        return cached

    indices = _build_strided_indices(sizes, strides, offset, index_dtype)
    if _is_traced_array(indices):
        return indices

    indices.block_until_ready()
    _STRIDED_INDICES_CACHE[key] = indices
    _STRIDED_INDICES_CACHE.move_to_end(key)
    while len(_STRIDED_INDICES_CACHE) > _STRIDED_INDICES_CACHE_MAXSIZE:
        _STRIDED_INDICES_CACHE.popitem(last=False)
    return indices
