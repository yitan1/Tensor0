from __future__ import annotations

from .. import _native

# Sector structures depend only on visible sector labels; degeneracy structures also
# depend on degeneracy dimensions.
_sectorstructure_cache: dict[tuple[object, ...], _native.SectorStructure] = {}
_degeneracystructure_cache: dict[tuple[object, ...], _native.DegeneracyStructure] = {}


def get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_sectorstructure() requires a HomSpace")
    return _get_sectorstructure(space)


def get_degeneracystructure(space: _native.HomSpace) -> _native.DegeneracyStructure:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("get_degeneracystructure() requires a HomSpace")

    key = _degeneracystructure_key(space)
    cached = _degeneracystructure_cache.get(key)
    if cached is not None:
        return cached

    sectorstructure = _get_sectorstructure(space)
    degeneracystructure = _native._build_degeneracystructure_from_sectorstructure(
        space,
        sectorstructure,
    )
    _degeneracystructure_cache[key] = degeneracystructure
    return degeneracystructure


def _clear_layout_caches_for_tests() -> None:
    _sectorstructure_cache.clear()
    _degeneracystructure_cache.clear()


def _get_sectorstructure(space: _native.HomSpace) -> _native.SectorStructure:
    key = _sectorstructure_key(space)
    cached = _sectorstructure_cache.get(key)
    if cached is not None:
        return cached

    sectorstructure = _native.build_sectorstructure(space)
    _sectorstructure_cache[key] = sectorstructure
    return sectorstructure


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
