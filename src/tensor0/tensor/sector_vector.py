from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jax import Array
from jax import tree_util as _tree_util
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.spaces import _normalize_sector_key, hom
from .storage import VectorStorage, _validate_vector_storage_data


@dataclass(frozen=True, eq=False, init=False)
class SectorVector:
    space: _native.ElementarySpace
    storage: VectorStorage

    def __init__(self, space: _native.ElementarySpace, storage: object) -> None:
        if not isinstance(space, _native.ElementarySpace):
            raise TypeError("SectorVector requires an ElementarySpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        _validate_vector_storage_data(vector_storage.data, _space_dim(space))

        object.__setattr__(self, "space", space)
        object.__setattr__(self, "storage", vector_storage)

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        key = _normalize_sector_key(coupled)
        offset = 0
        for sector, dim in self.space.sectors:
            next_offset = offset + dim
            if sector == key:
                return self.storage.data[offset:next_offset]
            offset = next_offset
        raise KeyError(key)

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        result: list[tuple[tuple[int, ...], Array]] = []
        offset = 0
        for sector, dim in self.space.sectors:
            next_offset = offset + dim
            result.append((sector, self.storage.data[offset:next_offset]))
            offset = next_offset
        return tuple(result)

    def to_diagonal(self):
        from .tensor_map import TensorMap, _packed_vector_from_blocks

        diagonal_space = hom((self.space,), (self.space,))
        block_arrays = {sector: jnp.diag(values) for sector, values in self.blocks()}
        dtype = jnp.asarray(self.storage.data).dtype
        return TensorMap(
            diagonal_space,
            _packed_vector_from_blocks(diagonal_space, block_arrays, dtype=dtype),
        )


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


def _space_dim(space: _native.ElementarySpace) -> int:
    return sum(dim for _sector, dim in space.sectors)


def _sectorvector_flatten(
    vector: SectorVector,
) -> tuple[tuple[object, ...], _native.ElementarySpace]:
    return (vector.storage.data,), vector.space


def _sectorvector_unflatten(
    aux_data: _native.ElementarySpace,
    children: tuple[object, ...],
) -> SectorVector:
    (data,) = children
    return SectorVector(aux_data, data)


_tree_util.register_pytree_node(
    SectorVector,
    _sectorvector_flatten,
    _sectorvector_unflatten,
)
