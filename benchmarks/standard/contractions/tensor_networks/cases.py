from __future__ import annotations

from functools import partial
from pathlib import Path
import tomllib
from typing import NamedTuple, cast

from tensor0 import (
    ElementarySpace,
    TensorMap,
    hom,
    ncon,
)

from ..._inputs import generate_space, random_tensor
from ..._specs import (
    EAGER,
    EXPLICIT,
    JIT_CACHED,
    JIT_COMPILE,
    QUICK,
    PreparedOperation,
    ScenarioSpec,
    WorkloadSpec,
)


class TensorNetworkCase(NamedTuple):
    topology: str
    dtype: str
    sector: str
    dimensions: tuple[int, ...]
    sigmas: tuple[float, ...] | None


class _TopologyDefinition(NamedTuple):
    operand_positions: tuple[int, ...]
    labels: tuple[tuple[int, ...], ...]
    conjugate: tuple[bool, ...]
    order: tuple[int, ...]


_PARAMETERS_PATH = Path(__file__).with_name("params.toml")
with _PARAMETERS_PATH.open("rb") as stream:
    _PARAMETERS = tomllib.load(stream)

_DTYPES = {"float64", "complex128"}
_SECTORS = {"trivial", "z2", "u1", "su2"}
_QUICK_CASES = {
    ("mpo", "trivial", (10, 4, 3)),
    ("pepo", "trivial", (3, 2, 2, 50)),
    ("mera", "trivial", (2,)),
}

_MPO_DEFINITION = _TopologyDefinition(
    operand_positions=(2, 0, 1, 0, 3),
    labels=(
        (4, 2, 1),
        (1, 3, 6),
        (2, 5, 3, 7),
        (4, 5, 8),
        (6, 7, 8),
    ),
    conjugate=(False, False, False, True, False),
    order=tuple(range(1, 9)),
)
_PEPO_DEFINITION = _TopologyDefinition(
    operand_positions=(2, 5, 0, 1, 0, 4, 3),
    labels=(
        (18, 7, 4, 2, 1),
        (1, 3, 6, 9, 10),
        (2, 17, 5, 3, 11),
        (4, 16, 8, 5, 6, 12),
        (7, 15, 8, 9, 13),
        (10, 11, 12, 13, 14),
        (14, 15, 16, 17, 18),
    ),
    conjugate=(False, False, False, False, True, False, False),
    order=tuple(range(1, 19)),
)
_MERA_DEFINITION = _TopologyDefinition(
    operand_positions=(3, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 1),
    labels=(
        (9, 3, 4, 5, 1, 2),
        (1, 2, 7, 12),
        (3, 4, 11, 13),
        (8, 5, 15, 6),
        (6, 7, 19),
        (8, 9, 17, 10),
        (10, 11, 22),
        (12, 14, 20),
        (13, 14, 23),
        (18, 19, 20, 21, 22, 23),
        (16, 15, 18),
        (16, 17, 21),
    ),
    conjugate=(
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
    ),
    # Preserve the explicit parenthesization of the MERA workload.
    order=(
        1,
        2,
        3,
        4,
        6,
        5,
        7,
        10,
        8,
        9,
        11,
        14,
        20,
        23,
        12,
        13,
        19,
        22,
        15,
        18,
        16,
        17,
        21,
    ),
)
_TOPOLOGIES = {
    "mpo": _MPO_DEFINITION,
    "pepo": _PEPO_DEFINITION,
    "mera": _MERA_DEFINITION,
}


def _spaces(case: TensorNetworkCase) -> tuple[ElementarySpace, ...]:
    sigmas = case.sigmas or (None,) * len(case.dimensions)
    return tuple(
        generate_space(case.sector, dimension, sigma)
        for dimension, sigma in zip(case.dimensions, sigmas, strict=True)
    )


def _mpo_arguments(case: TensorNetworkCase) -> tuple[TensorMap, ...]:
    mps, mpo, phys = _spaces(case)
    return (
        random_tensor(
            hom((mps, phys, mps.dual()), ()),
            dtype=case.dtype,
            seed=101,
        ),
        random_tensor(
            hom((mpo, phys, phys.dual(), mpo.dual()), ()),
            dtype=case.dtype,
            seed=102,
        ),
        random_tensor(
            hom((mps, mpo.dual(), mps.dual()), ()),
            dtype=case.dtype,
            seed=103,
        ),
        random_tensor(
            hom((mps, mpo, mps.dual()), ()),
            dtype=case.dtype,
            seed=104,
        ),
    )


def _pepo_arguments(case: TensorNetworkCase) -> tuple[TensorMap, ...]:
    peps, pepo, phys, env = _spaces(case)
    return (
        random_tensor(
            hom((peps, peps, phys, peps.dual(), peps.dual()), ()),
            dtype=case.dtype,
            seed=201,
        ),
        random_tensor(
            hom(
                (
                    pepo,
                    pepo,
                    phys,
                    phys.dual(),
                    pepo.dual(),
                    pepo.dual(),
                ),
                (),
            ),
            dtype=case.dtype,
            seed=202,
        ),
        random_tensor(
            hom((env, peps, pepo.dual(), peps.dual(), env.dual()), ()),
            dtype=case.dtype,
            seed=203,
        ),
        random_tensor(
            hom((env, peps, pepo.dual(), peps.dual(), env.dual()), ()),
            dtype=case.dtype,
            seed=204,
        ),
        random_tensor(
            hom((env, peps, pepo, peps.dual(), env.dual()), ()),
            dtype=case.dtype,
            seed=205,
        ),
        random_tensor(
            hom((env, peps, pepo, peps.dual(), env.dual()), ()),
            dtype=case.dtype,
            seed=206,
        ),
    )


def _mera_arguments(case: TensorNetworkCase) -> tuple[TensorMap, ...]:
    (factor,) = _spaces(case)
    return (
        random_tensor(
            hom((factor, factor, factor.dual(), factor.dual()), ()),
            dtype=case.dtype,
            seed=301,
        ),
        random_tensor(
            hom((factor, factor, factor.dual()), ()),
            dtype=case.dtype,
            seed=302,
        ),
        random_tensor(
            hom(
                (
                    factor,
                    factor,
                    factor,
                    factor.dual(),
                    factor.dual(),
                    factor.dual(),
                ),
                (),
            ),
            dtype=case.dtype,
            seed=303,
        ),
        random_tensor(
            hom(
                (
                    factor,
                    factor,
                    factor,
                    factor.dual(),
                    factor.dual(),
                    factor.dual(),
                ),
                (),
            ),
            dtype=case.dtype,
            seed=304,
        ),
    )


def _network_arguments(case: TensorNetworkCase) -> tuple[TensorMap, ...]:
    if case.topology == "mpo":
        return _mpo_arguments(case)
    if case.topology == "pepo":
        return _pepo_arguments(case)
    if case.topology == "mera":
        return _mera_arguments(case)
    raise ValueError(f"unsupported tensor-network topology: {case.topology}")


def _evaluate_network(
    arguments: tuple[TensorMap, ...],
    definition: _TopologyDefinition,
    order: tuple[int, ...],
) -> TensorMap:
    operands = tuple(arguments[position] for position in definition.operand_positions)
    return ncon(
        operands,
        definition.labels,
        conjugate=definition.conjugate,
        order=order,
    )


def _network_prepared(case: TensorNetworkCase) -> PreparedOperation:
    arguments = _network_arguments(case)
    definition = _TOPOLOGIES[case.topology]

    def run(*values: TensorMap) -> TensorMap:
        return _evaluate_network(tuple(values), definition, definition.order)

    return run, arguments


def _parameter_entries() -> tuple[dict[str, object], ...]:
    raw_entries = _PARAMETERS.get("workload")
    if not isinstance(raw_entries, list):
        raise ValueError("missing tensor-network workload table")
    entries = cast(list[object], raw_entries)
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("invalid tensor-network workload table")
    return tuple(cast(dict[str, object], entry) for entry in entries)


def _dimension_cases(topology: str, value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{topology} dimensions must be a list")
    if not all(isinstance(item, list) for item in value):
        raise ValueError(f"{topology} dimensions must contain dimension lists")
    cases: list[tuple[int, ...]] = []
    for item in value:
        raw_dimensions = cast(list[object], item)
        if not all(isinstance(dimension, int) for dimension in raw_dimensions):
            raise ValueError(f"{topology} dimensions must contain integers")
        cases.append(tuple(cast(list[int], raw_dimensions)))
    return tuple(cases)


def _network_cases() -> tuple[TensorNetworkCase, ...]:
    cases: list[TensorNetworkCase] = []
    expected_arity = {"mpo": 3, "pepo": 4, "mera": 1}
    for entry in _parameter_entries():
        topology = entry.get("topology")
        raw_dtypes = entry.get("dtypes")
        sector = entry.get("sector")
        if not isinstance(topology, str) or topology not in expected_arity:
            raise ValueError(f"unsupported tensor-network topology: {topology}")
        if not isinstance(raw_dtypes, list) or not all(
            isinstance(value, str) and value in _DTYPES
            for value in raw_dtypes
        ):
            raise ValueError(
                f"{topology} dtypes must use Tensor0 dtype names"
            )
        if not isinstance(sector, str) or sector not in _SECTORS:
            raise ValueError(f"unsupported benchmark sector: {sector}")
        dimensions = _dimension_cases(topology, entry.get("dimensions"))
        raw_sigmas = entry.get("sigmas")
        sigmas = None
        if raw_sigmas is not None:
            if not isinstance(raw_sigmas, list):
                raise ValueError(f"{topology} sigmas must be a list")
            sigmas = tuple(float(value) for value in raw_sigmas)
            if len(sigmas) != expected_arity[topology]:
                raise ValueError(f"{topology} sigmas have the wrong arity")
        for dtype in cast(list[str], raw_dtypes):
            for dimension_case in dimensions:
                if len(dimension_case) != expected_arity[topology]:
                    raise ValueError(f"{topology} dimensions have the wrong arity")
                if any(dimension < 1 for dimension in dimension_case):
                    raise ValueError(f"{topology} dimensions must be positive")
                cases.append(
                    TensorNetworkCase(
                        topology,
                        dtype,
                        sector,
                        dimension_case,
                        sigmas,
                    )
                )
    return tuple(cases)


def _dimension_label(dimensions: tuple[int, ...]) -> str:
    return "d" + "x".join(str(dimension) for dimension in dimensions)


def _workload(case: TensorNetworkCase) -> WorkloadSpec:
    size_label = _dimension_label(case.dimensions)
    return WorkloadSpec(
        base_id=(
            f"tensor_networks.{case.topology}.{case.sector}."
            f"{case.dtype}.{size_label}"
        ),
        group="tensor_networks",
        description=(
            f"Tensor0 {case.topology.upper()} topology; sector={case.sector}, "
            f"dimensions={case.dimensions}"
        ),
        dtype=case.dtype,
        size_label=size_label,
        factory=partial(_network_prepared, case),
    )


TENSOR_NETWORK_CASES = _network_cases()


def _case_scenario_specs(
    case: TensorNetworkCase,
) -> tuple[ScenarioSpec, ...]:
    workload = _workload(case)
    is_quick = (
        case.topology,
        case.sector,
        case.dimensions,
    ) in _QUICK_CASES
    specs = [
        ScenarioSpec(
            workload,
            EAGER,
            QUICK if is_quick else EXPLICIT,
        )
    ]
    if is_quick:
        specs.extend(
            (
                ScenarioSpec(workload, JIT_COMPILE, EXPLICIT),
                ScenarioSpec(workload, JIT_CACHED, EXPLICIT),
            )
        )
    return tuple(specs)


TENSOR_NETWORK_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = tuple(
    spec
    for case in TENSOR_NETWORK_CASES
    for spec in _case_scenario_specs(case)
)


def tensor_network_metadata() -> dict[str, object]:
    topology_counts = {
        topology: sum(
            case.topology == topology for case in TENSOR_NETWORK_CASES
        )
        for topology in _TOPOLOGIES
    }
    sector_counts = {
        sector: sum(case.sector == sector for case in TENSOR_NETWORK_CASES)
        for sector in _SECTORS
    }
    return {
        "workload_contract": {
            "topologies": "fixed MPO, PEPO, and MERA index graphs",
            "parameters": "fixed dtype, sector, dimension, and sigma matrix",
            "setup": "space and random tensor construction outside timing",
            "execution": "Tensor0 functional ncon with a recorded expression tree",
            "result_boundary": "rank_zero_tensormap",
        },
        "case_count": len(TENSOR_NETWORK_CASES),
        "topology_counts": topology_counts,
        "sector_counts": sector_counts,
        "topology_definitions": {
            name: {
                "operand_positions": definition.operand_positions,
                "labels": definition.labels,
                "conjugate": definition.conjugate,
                "order": definition.order,
            }
            for name, definition in _TOPOLOGIES.items()
        },
        "cases": [
            {
                "topology": case.topology,
                "dtype": case.dtype,
                "sector": case.sector,
                "dimensions": case.dimensions,
                "sigmas": case.sigmas,
            }
            for case in TENSOR_NETWORK_CASES
        ],
        "extensions": {
            "jit": "small Trivial MPO/PEPO/MERA cases are explicit-only",
        },
    }
