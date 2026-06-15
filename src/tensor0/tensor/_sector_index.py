from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

from .. import _native
from ..structure.sector_dict import SectorDict

_CacheValue = TypeVar("_CacheValue")
_SECTOR_INDEX_CACHE_MAXSIZE = 10_000
_sector_slice_cache: OrderedDict[
    tuple[object, ...],
    SectorDict[slice],
] = OrderedDict()


def _sector_slices_for_space(space: _native.ElementarySpace) -> SectorDict[slice]:
    if not isinstance(space, _native.ElementarySpace):
        raise TypeError("_sector_slices_for_space() requires an ElementarySpace")

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


def _clear_sector_index_caches_for_tests() -> None:
    _sector_slice_cache.clear()


def _cache_get(
    cache: OrderedDict[tuple[object, ...], _CacheValue],
    key: tuple[object, ...],
) -> _CacheValue | None:
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
    return cached


def _cache_set(
    cache: OrderedDict[tuple[object, ...], _CacheValue],
    key: tuple[object, ...],
    value: _CacheValue,
) -> _CacheValue:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _SECTOR_INDEX_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return value
