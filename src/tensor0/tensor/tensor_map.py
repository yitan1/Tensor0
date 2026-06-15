from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jax import Array
from jax import tree_util as _tree_util
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import (
    get_blockstructure,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..structure.sector_dict import SectorDict
from ..structure.spaces import _normalize_sector_key, hom
from ._subblocks import gather_subblock as _gather_subblock
from .storage import VectorStorage, _validate_vector_storage_data


@dataclass(frozen=True, eq=False, init=False)
class TensorMap:
    space: _native.HomSpace
    storage: VectorStorage

    def __init__(self, space: _native.HomSpace, storage: object) -> None:
        if not isinstance(space, _native.HomSpace):
            raise TypeError("TensorMap requires a HomSpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        degeneracystructure = get_degeneracystructure(space)
        _validate_vector_storage_data(vector_storage.data, degeneracystructure.total_dim)

        object.__setattr__(self, "space", space)
        object.__setattr__(self, "storage", vector_storage)

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        key = _normalize_sector_key(coupled)
        try:
            block = get_blockstructure(self.space)[key]
        except KeyError:
            raise KeyError(key) from None
        return self.storage.data[block.start : block.stop].reshape(
            (block.row_dim, block.col_dim),
        )

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        return tuple(
            (
                coupled,
                self.storage.data[block.start : block.stop].reshape(
                    (block.row_dim, block.col_dim),
                ),
            )
            for coupled, block in get_blockstructure(self.space).items()
        )

    def subblock(self, row_tree: _native.FusionTree, col_tree: _native.FusionTree) -> Array:
        if not isinstance(row_tree, _native.FusionTree) or not isinstance(
            col_tree,
            _native.FusionTree,
        ):
            raise TypeError("subblock() requires a pair of FusionTree objects")
        return self._subblock(row_tree, col_tree)

    def _subblock(self, row_tree: _native.FusionTree, col_tree: _native.FusionTree) -> Array:
        subblock = _subblock_for_fusiontrees(self.space, row_tree, col_tree)
        return _gather_subblock(self.storage.data, subblock)

    def subblocks(
        self,
    ) -> tuple[tuple[tuple[_native.FusionTree, _native.FusionTree], Array], ...]:
        sectorstructure = get_sectorstructure(self.space)
        degeneracystructure = get_degeneracystructure(self.space)

        return tuple(
            (
                pair,
                _gather_subblock(self.storage.data, subblock),
            )
            for pair, subblock in zip(
                sectorstructure.fusiontree_pairs,
                degeneracystructure.subblockstructure,
            )
        )

    def __getitem__(self, key: object) -> Array:
        row_tree, col_tree = _normalize_fusiontree_pair_key(key)
        return self._subblock(row_tree, col_tree)

    def permute(self, p: tuple[tuple[int, ...], tuple[int, ...]]) -> TensorMap:
        from ..transforms import permute

        return permute(self, p)

    def braid(
        self,
        p: tuple[tuple[int, ...], tuple[int, ...]],
        levels: tuple[int, ...],
    ) -> TensorMap:
        from ..transforms import braid

        return braid(self, p, levels)

    def transpose(
        self,
        p: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> TensorMap:
        from ..transforms import transpose

        return transpose(self, p)

    def repartition(self, nout: int, nin: int | None = None) -> TensorMap:
        from ..transforms import repartition

        return repartition(self, nout, nin)

    def __matmul__(self, other: object) -> TensorMap:
        if not isinstance(other, TensorMap):
            return NotImplemented

        if self.space.domain != other.space.codomain:
            raise ValueError(
                "TensorMap spaces are not composable: "
                "left domain must equal right codomain",
            )

        result_space = hom(self.space.codomain, other.space.domain)
        dtype = jnp.result_type(self.storage.data, other.storage.data)
        left_blocks = SectorDict(self.blocks())
        right_blocks = SectorDict(other.blocks())
        result_blocks: list[tuple[tuple[int, ...], Array]] = []

        for coupled in get_sectorstructure(result_space).blocksectors:
            left = left_blocks.get(coupled)
            right = right_blocks.get(coupled)
            if left is not None and right is not None:
                result_blocks.append((coupled, left @ right))

        return TensorMap(
            result_space,
            _packed_vector_from_blocks(
                result_space,
                SectorDict(result_blocks),
                dtype=dtype,
            ),
        )


def _packed_vector_from_blocks(
    space: _native.HomSpace,
    block_arrays: Mapping[tuple[int, ...], Array],
    *,
    dtype: DTypeLike | None,
) -> Array:
    degeneracystructure = get_degeneracystructure(space)
    blockstructures = get_blockstructure(space)

    flat_blocks: list[Array] = []
    for coupled, block in blockstructures.items():
        expected_shape = (block.row_dim, block.col_dim)
        value = block_arrays.get(coupled)
        if value is None:
            value = jnp.zeros(expected_shape, dtype=dtype)
        else:
            actual_shape = tuple(getattr(value, "shape", ()))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"block {coupled} has shape {actual_shape}; "
                    f"expected {expected_shape}",
                )
        flat_blocks.append(jnp.reshape(value, (-1,)))

    if flat_blocks:
        return jnp.concatenate(tuple(flat_blocks), axis=0)
    return jnp.zeros((degeneracystructure.total_dim,), dtype=dtype)


def _normalize_fusiontree_pair_key(
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


def _subblock_for_fusiontrees(
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


def _tensormap_flatten(tensor: TensorMap) -> tuple[tuple[object, ...], _native.HomSpace]:
    return (tensor.storage.data,), tensor.space


def _tensormap_unflatten(
    aux_data: _native.HomSpace,
    children: tuple[object, ...],
) -> TensorMap:
    (data,) = children
    return TensorMap(aux_data, data)


_tree_util.register_pytree_node(
    TensorMap,
    _tensormap_flatten,
    _tensormap_unflatten,
)
