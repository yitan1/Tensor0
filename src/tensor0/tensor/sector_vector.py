from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from jax import Array
from jax import tree_util as _tree_util
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import _sector_slices_for_space
from ..structure.sector_dict import SectorDict
from ..structure.spaces import _normalize_sector_key, reduced_dim, space as _space
from .storage import (
    VectorStorage,
    _pytree_vector_storage,
    _require_unbatched_storage,
    _validate_vector_storage_data,
)


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

        _validate_vector_storage_data(vector_storage.data, reduced_dim(space))

        object.__setattr__(self, "sector_type", space.sector_spec)
        object.__setattr__(self, "structure", _sector_slices_for_space(space))
        object.__setattr__(self, "storage", vector_storage)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading pytree batch axes, separate from the tensor's space."""
        return tuple(self.storage.data.shape[:-1])

    @property
    def sectors(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        return tuple(
            (sector, sector_slice.stop - sector_slice.start)
            for sector, sector_slice in self.structure.items()
        )

    @property
    def blocksectors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.structure)

    def keys(self) -> tuple[tuple[int, ...], ...]:
        return self.blocksectors

    def values(self) -> Iterator[Array]:
        _require_unbatched_storage(self.storage.data, "values()")
        return (self.storage.data[sector_slice] for sector_slice in self.structure.values())

    def pairs(self) -> Iterator[tuple[tuple[int, ...], Array]]:
        _require_unbatched_storage(self.storage.data, "pairs()")
        return (
            (sector, self.storage.data[sector_slice])
            for sector, sector_slice in self.structure.items()
        )

    def hasblock(self, coupled: object) -> bool:
        return coupled in self.structure

    def get(self, coupled: object, default: object = None) -> Array | object:
        _require_unbatched_storage(self.storage.data, "get()")
        key = _normalize_sector_key(coupled)
        sector_slice = self.structure.get(key)
        if sector_slice is None:
            return default
        return self.storage.data[sector_slice]

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        _require_unbatched_storage(self.storage.data, "block()")
        key = _normalize_sector_key(coupled)
        try:
            sector_slice = self.structure[key]
        except KeyError:
            raise KeyError(key) from None
        return self.storage.data[sector_slice]

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        return tuple(self.pairs())

    def to_diagonal(self):
        _require_unbatched_storage(self.storage.data, "to_diagonal()")
        from .diagonal import DiagonalTensorMap

        domain = self._visible_space()
        return DiagonalTensorMap(domain, self.storage)

    def copy(self) -> SectorVector:
        _require_unbatched_storage(self.storage.data, "copy()")
        return SectorVector(
            self._visible_space(),
            jnp.array(self.storage.data, copy=True),
        )

    def similar(
        self,
        dtype: DTypeLike | None = None,
        *,
        space: _native.ElementarySpace | None = None,
    ) -> SectorVector:
        _require_unbatched_storage(self.storage.data, "similar()")
        target_space = self._visible_space() if space is None else space
        target_dtype = self.storage.data.dtype if dtype is None else dtype
        return SectorVector(
            target_space,
            jnp.empty((reduced_dim(target_space),), dtype=target_dtype),
        )

    def _visible_space(self) -> _native.ElementarySpace:
        return _space(self.sector_type, dict(self.sectors))


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
    space = _native.make_space(sector_type, sectors, False)
    vector = object.__new__(SectorVector)
    object.__setattr__(vector, "sector_type", sector_type)
    object.__setattr__(vector, "structure", _sector_slices_for_space(space))
    object.__setattr__(
        vector, "storage", _pytree_vector_storage(data, reduced_dim(space)),
    )
    return vector


_tree_util.register_pytree_node(
    SectorVector,
    _sectorvector_flatten,
    _sectorvector_unflatten,
)
