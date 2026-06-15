from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jax import Array
from jax import tree_util as _tree_util
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.sector_dict import SectorDict
from ..structure.spaces import _normalize_sector_key, space
from ._sector_index import _sector_slices_for_space
from .storage import VectorStorage, _validate_vector_storage_data


@dataclass(frozen=True, eq=False, init=False)
class SectorVector:
    sector_type: _native.SectorSpec
    structure: SectorDict[slice]
    storage: VectorStorage

    def __init__(self, space: _native.ElementarySpace, storage: object) -> None:
        if not isinstance(space, _native.ElementarySpace):
            raise TypeError("SectorVector requires an ElementarySpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        _validate_vector_storage_data(vector_storage.data, _space_dim(space))

        object.__setattr__(self, "sector_type", space.sector_spec)
        object.__setattr__(self, "structure", _sector_slices_for_space(space))
        object.__setattr__(self, "storage", vector_storage)

    @property
    def sectors(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        return tuple(
            (sector, sector_slice.stop - sector_slice.start)
            for sector, sector_slice in self.structure.items()
        )

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        key = _normalize_sector_key(coupled)
        try:
            sector_slice = self.structure[key]
        except KeyError:
            raise KeyError(key) from None
        return self.storage.data[sector_slice]

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        return tuple(
            (sector, self.storage.data[sector_slice])
            for sector, sector_slice in self.structure.items()
        )

    def to_diagonal(self):
        from .diagonal import DiagonalTensorMap

        domain = space(self.sector_type, dict(self.sectors))
        return DiagonalTensorMap(domain, self.storage)


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
) -> tuple[
    tuple[object, ...],
    tuple[_native.SectorSpec, tuple[tuple[tuple[int, ...], int], ...]],
]:
    return (vector.storage.data,), (vector.sector_type, vector.sectors)


def _sectorvector_unflatten(
    aux_data: tuple[_native.SectorSpec, tuple[tuple[tuple[int, ...], int], ...]],
    children: tuple[object, ...],
) -> SectorVector:
    sector_type, sectors = aux_data
    (data,) = children
    return SectorVector(_native.make_space(sector_type, sectors, False), data)


_tree_util.register_pytree_node(
    SectorVector,
    _sectorvector_flatten,
    _sectorvector_unflatten,
)
