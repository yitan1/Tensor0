from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

from .. import _native
from .sector_dict import SectorDict

# Sector structures depend only on visible sector labels; degeneracy structures also
# depend on degeneracy dimensions.
_CacheValue = TypeVar("_CacheValue")
_CacheKey = TypeVar("_CacheKey")
_Key = TypeVar("_Key")
_Value = TypeVar("_Value")
_LAYOUT_CACHE_MAXSIZE = 10_000
_sectorstructure_cache: OrderedDict[
    object,
    _native.SectorStructure,
] = OrderedDict()
_degeneracystructure_cache: OrderedDict[
    _native.HomSpace,
    _native.DegeneracyStructure,
] = OrderedDict()
_blockstructure_cache: OrderedDict[
    _native.HomSpace,
    _IndexedMapping[tuple[int, ...], _native.BlockStructure],
] = OrderedDict()
_sector_slice_cache: OrderedDict[
    tuple[object, ...],
    SectorDict[slice],
] = OrderedDict()


@dataclass(frozen=True, init=False)
class _IndexedMapping(Mapping[_Key, _Value], Generic[_Key, _Value]):
    _keys: tuple[_Key, ...]
    _values: tuple[_Value, ...]
    _index_of: Callable[[object], int | None]

    def __init__(
        self,
        keys: tuple[_Key, ...],
        values: tuple[_Value, ...],
        index_of: Callable[[object], int | None],
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("IndexedMapping requires keys and values with the same length")

        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_index_of", index_of)

    def __getitem__(self, key: object) -> _Value:
        index = self._index_of(key)
        if index is None or index < 0 or index >= len(self._values):
            raise KeyError(key)
        return self._values[index]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        try:
            return self._index_of(key) is not None
        except (TypeError, ValueError):
            return False

    def items(self) -> Iterator[tuple[_Key, _Value]]:  # type: ignore[override]
        return zip(self._keys, self._values, strict=True)

    def values(self) -> Iterator[_Value]:  # type: ignore[override]
        return iter(self._values)


def get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_sectorstructure() requires a HomSpace")
    return _get_sectorstructure(space)


def get_degeneracystructure(space: _native.HomSpace) -> _native.DegeneracyStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_degeneracystructure() requires a HomSpace")

    cached = _cache_get(_degeneracystructure_cache, space)
    if cached is not None:
        return cached

    sectorstructure = _get_sectorstructure(space)
    degeneracystructure = _native._build_degeneracystructure_from_sectorstructure(
        space,
        sectorstructure,
    )
    return _cache_set(_degeneracystructure_cache, space, degeneracystructure)


def get_blockstructure(
    space: _native.HomSpace,
) -> _IndexedMapping[tuple[int, ...], _native.BlockStructure]:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_blockstructure() requires a HomSpace")

    cached = _cache_get(_blockstructure_cache, space)
    if cached is not None:
        return cached

    sectorstructure = _get_sectorstructure(space)
    degeneracystructure = get_degeneracystructure(space)
    return _cache_set(
        _blockstructure_cache,
        space,
        _IndexedMapping(
            sectorstructure.blocksectors,
            degeneracystructure.blockstructure,
            sectorstructure.blocksector_index,
        ),
    )


def _sector_slices_for_space(
    space: _native.ElementarySpace,
) -> SectorDict[slice]:
    key = ("sector-slices", space.sector_spec.static_key, space.sectors)
    cached = _cache_get(_sector_slice_cache, key)
    if cached is not None:
        return cached

    offset = 0
    items: list[tuple[tuple[int, ...], slice]] = []
    for sector, dim in space.sectors:
        next_offset = offset + dim
        items.append((sector, slice(offset, next_offset)))
        offset = next_offset

    return _cache_set(_sector_slice_cache, key, SectorDict(items))


def _clear_layout_caches_for_tests() -> None:
    _sectorstructure_cache.clear()
    _degeneracystructure_cache.clear()
    _blockstructure_cache.clear()
    _sector_slice_cache.clear()


def _get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    key = _sectorstructure_key(space)
    cached = _cache_get(_sectorstructure_cache, key)
    if cached is not None:
        return cached

    sectorstructure = _native.build_sectorstructure(space)
    return _cache_set(_sectorstructure_cache, key, sectorstructure)


def _cache_get(
    cache: OrderedDict[_CacheKey, _CacheValue],
    key: _CacheKey,
) -> _CacheValue | None:
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
    return cached


def _cache_set(
    cache: OrderedDict[_CacheKey, _CacheValue],
    key: _CacheKey,
    value: _CacheValue,
) -> _CacheValue:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _LAYOUT_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return value


def _sectorstructure_key(space: _native.HomSpace) -> object:
    """Return the native key shared by sector-layout and transform caches."""
    return space._sector_key
