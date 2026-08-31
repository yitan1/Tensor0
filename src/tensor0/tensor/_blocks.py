from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import _blockstructure_items, _find_blockstructure
from ..structure.sector_dict import SectorDict
from ..structure.spaces import storage_dim


def pack_complete_blocks(
    space: _native.HomSpace,
    blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
    *,
    dtype: DTypeLike | None,
) -> Array:
    block_arrays = SectorDict(blocks)
    if space.sector_spec == _native.Trivial:
        total_dim = storage_dim(space)
        if total_dim == 0:
            for coupled, value in block_arrays.items():
                if jnp.asarray(value, dtype=dtype).size != 0:
                    raise ValueError(f"unexpected block sector {coupled}")
            return jnp.zeros((0,), dtype=dtype)

        coupled = ()
        if coupled not in block_arrays:
            raise ValueError(f"missing data for block sector {coupled}")

        dims = space.dims
        expected_shape = (
            math.prod(dims[: space.numout]),
            math.prod(dims[space.numout :]),
        )
        value = jnp.asarray(block_arrays[coupled], dtype=dtype)
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"block sector {coupled} has shape {actual_shape}; "
                f"expected {expected_shape}",
            )
        for extra_coupled, extra_value in block_arrays.items():
            if extra_coupled != coupled and jnp.asarray(
                extra_value,
                dtype=dtype,
            ).size != 0:
                raise ValueError(f"unexpected block sector {extra_coupled}")
        return value.reshape((total_dim,))

    flat_blocks: list[Array] = []
    for coupled, block in _blockstructure_items(space):
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
        if _find_blockstructure(space, coupled, suppress_invalid=True) is not None:
            continue
        array = jnp.asarray(value, dtype=dtype)
        if array.size != 0:
            raise ValueError(f"unexpected block sector {coupled}")

    if flat_blocks:
        return jnp.concatenate(tuple(flat_blocks), axis=0)
    return jnp.zeros((0,), dtype=dtype)


def pack_blocks(
    space: _native.HomSpace,
    blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
    *,
    dtype: DTypeLike | None,
) -> Array:
    block_arrays = SectorDict(blocks)
    if space.sector_spec == _native.Trivial:
        total_dim = storage_dim(space)
        if total_dim == 0:
            return jnp.zeros((0,), dtype=dtype)

        dims = space.dims
        expected_shape = (
            math.prod(dims[: space.numout]),
            math.prod(dims[space.numout :]),
        )
        value = block_arrays.get(())
        if value is None:
            value = jnp.zeros(expected_shape, dtype=dtype)
        else:
            value = jnp.asarray(value, dtype=dtype)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"block {()} has shape {actual_shape}; "
                    f"expected {expected_shape}",
                )
        return value.reshape((total_dim,))

    flat_blocks: list[Array] = []
    for coupled, block in _blockstructure_items(space):
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
    return jnp.zeros((0,), dtype=dtype)


def _packed_sector_values(
    space: _native.ElementarySpace,
    sector_arrays: Mapping[tuple[int, ...], Array],
    *,
    dtype: DTypeLike | None,
) -> Array:
    flat_blocks: list[Array] = []
    for sector, dim in space.sectors:
        value = sector_arrays[sector]
        actual_shape = tuple(getattr(value, "shape", ()))
        expected_shape = (dim,)
        if actual_shape != expected_shape:
            raise ValueError(
                f"sector vector block {sector} has shape {actual_shape}; "
                f"expected {expected_shape}",
            )
        flat_blocks.append(jnp.reshape(value, (-1,)))

    if flat_blocks:
        return jnp.concatenate(tuple(flat_blocks), axis=0)
    return jnp.zeros((0,), dtype=dtype)


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
