from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

from .. import _native

# Sector structures depend only on visible sector labels; degeneracy structures also
# depend on degeneracy dimensions.
_CacheValue = TypeVar("_CacheValue")
_LAYOUT_CACHE_MAXSIZE = 10_000
_sectorstructure_cache: OrderedDict[
    tuple[object, ...],
    _native.SectorStructure,
] = OrderedDict()
_degeneracystructure_cache: OrderedDict[
    tuple[object, ...],
    _native.DegeneracyStructure,
] = OrderedDict()


def get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_sectorstructure() requires a HomSpace")
    return _get_sectorstructure(space)


def get_degeneracystructure(space: _native.HomSpace) -> _native.DegeneracyStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_degeneracystructure() requires a HomSpace")

    key = _degeneracystructure_key(space)
    cached = _cache_get(_degeneracystructure_cache, key)
    if cached is not None:
        return cached

    sectorstructure = _get_sectorstructure(space)
    degeneracystructure = _native._build_degeneracystructure_from_sectorstructure(
        space,
        sectorstructure,
    )
    return _cache_set(_degeneracystructure_cache, key, degeneracystructure)


def _clear_layout_caches_for_tests() -> None:
    _sectorstructure_cache.clear()
    _degeneracystructure_cache.clear()


def _get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    key = _sectorstructure_key(space)
    cached = _cache_get(_sectorstructure_cache, key)
    if cached is not None:
        return cached

    sectorstructure = _native.build_sectorstructure(space)
    return _cache_set(_sectorstructure_cache, key, sectorstructure)


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
    while len(cache) > _LAYOUT_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return value


def _sectorstructure_key(space: _native.HomSpace) -> tuple[object, ...]:
    return (
        "hom-sectorstructure",
        tuple(_factor_sectorstructure_key(factor) for factor in space.codomain),
        tuple(_factor_sectorstructure_key(factor) for factor in space.domain),
    )


def _factor_sectorstructure_key(factor: _native.ElementarySpace) -> tuple[object, ...]:
    return (
        "space-sectorstructure",
        factor.sector_spec.static_key,
        tuple(tuple(sector) for sector, _dim in factor.sectors),
        factor.is_dual,
    )


def _degeneracystructure_key(space: _native.HomSpace) -> tuple[object, ...]:
    return ("hom-degeneracystructure", space.static_key)
