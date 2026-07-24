from __future__ import annotations

import tensor0.structure.layout as layout_module
from tensor0 import HomSpace, SU2Irrep, U1Irrep, hom, space
from tensor0.structure import get_degeneracystructure, get_sectorstructure

from .._specs import (
    FULL,
    QUICK,
    PreparedOperation,
    ScenarioSpec,
    WorkloadSpec,
    cold_execution,
    eager_execution,
)

def _clear_layout_caches() -> None:
    layout_module._clear_layout_caches_for_tests()


def _u1_two_factor_hom() -> HomSpace:
    a = space(U1Irrep, {0: 2, -1: 5, 1: 3})
    b = space(U1Irrep, {0: 7, -1: 13, 1: 11})
    c = space(U1Irrep, {0: 19, -1: 17, 1: 23})
    d = space(U1Irrep, {0: 31, -1: 29, 1: 37})
    return hom((a, b), (c, d))


def _su2_four_half_hom() -> HomSpace:
    half = space(SU2Irrep, {1: 1})
    return hom((half, half, half, half), ())


def _layout_prepared(target: HomSpace) -> PreparedOperation:
    def run() -> object:
        return get_sectorstructure(target), get_degeneracystructure(target)

    return run, ()


def _layout_workload(
    base_id: str,
    description: str,
    target: HomSpace,
) -> WorkloadSpec:
    return WorkloadSpec(
        base_id,
        "diagnostics",
        description,
        "none",
        "small",
        lambda: _layout_prepared(target),
        cache_clear=_clear_layout_caches,
    )


_COLD_LAYOUT = cold_execution(
    id_suffix="cold",
    execution="metadata",
    cache_policy="cold_layout",
)
_CACHED_LAYOUT = eager_execution(
    id_suffix="cached",
    execution="metadata",
    cache_policy="cached_layout",
)
_U1_LAYOUT = _layout_workload(
    "layout.u1_two_factor",
    "U1 two-factor sector and degeneracy structure construction",
    _u1_two_factor_hom(),
)
_SU2_LAYOUT = _layout_workload(
    "layout.su2_four_half",
    "SU2 four spin-half sector and degeneracy structure construction",
    _su2_four_half_hom(),
)

LAYOUT_CACHE_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        _U1_LAYOUT,
        _COLD_LAYOUT,
        QUICK,
    ),
    ScenarioSpec(
        _SU2_LAYOUT,
        _COLD_LAYOUT,
        QUICK,
    ),
    ScenarioSpec(
        _U1_LAYOUT,
        _CACHED_LAYOUT,
        FULL,
    ),
    ScenarioSpec(
        _SU2_LAYOUT,
        _CACHED_LAYOUT,
        FULL,
    ),
)
