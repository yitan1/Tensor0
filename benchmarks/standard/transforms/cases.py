from __future__ import annotations

from functools import partial
from math import prod
from pathlib import Path
import tomllib
from typing import NamedTuple, cast

import tensor0.operations.transforms as transforms_module
from tensor0 import (
    ComplexSpace,
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    hom,
    permute,
    repartition,
    space,
    twist,
)

from .._inputs import dense_tensor, generate_space, packed_tensor, random_tensor
from .._specs import (
    EAGER,
    EXPLICIT,
    FULL,
    JIT_CACHED,
    JIT_COMPILE,
    QUICK,
    PreparedOperation,
    ScenarioProfile,
    ScenarioSpec,
    WorkloadSpec,
    cold_execution,
    eager_execution,
)

class TransformCase(NamedTuple):
    operation: str
    dtype: str
    sector: str
    dimensions: tuple[int, ...]
    sigmas: tuple[float, ...] | None
    permutation: tuple[tuple[int, ...], tuple[int, ...]]
    logical_input_mib: int | None


_PARAMETERS_PATH = Path(__file__).with_name("params.toml")
with _PARAMETERS_PATH.open("rb") as stream:
    _PARAMETERS = tomllib.load(stream)

_DTYPE_BYTES = {"float64": 8, "complex128": 16}
_SECTORS = {"trivial", "z2"}


def _entries() -> tuple[dict[str, object], ...]:
    raw_entries = _PARAMETERS.get("workload")
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, dict) for entry in raw_entries
    ):
        raise ValueError("invalid transform workload parameters")
    return tuple(cast(dict[str, object], entry) for entry in raw_entries)


def _dimensions(raw: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not all(isinstance(item, list) for item in raw):
        raise ValueError("transform dimensions must contain dimension lists")
    cases: list[tuple[int, ...]] = []
    for item in raw:
        raw_dimensions = cast(list[object], item)
        if not all(isinstance(value, int) and value > 0 for value in raw_dimensions):
            raise ValueError("transform dimensions must be positive integers")
        cases.append(tuple(cast(list[int], raw_dimensions)))
    return tuple(cases)


def _permutations(
    raw: object,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if not isinstance(raw, list):
        raise ValueError("transform permutations must be a list")
    permutations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(
                "each transform permutation must contain codomain and domain"
            )
        raw_codomain, raw_domain = item
        if not isinstance(raw_codomain, list) or not isinstance(raw_domain, list):
            raise ValueError("transform permutation groups must be lists")
        if not all(
            isinstance(index, int) and index >= 1
            for index in (*raw_codomain, *raw_domain)
        ):
            raise ValueError("transform indices must be positive integers")
        permutations.append(
            (
                tuple(cast(list[int], raw_codomain)),
                tuple(cast(list[int], raw_domain)),
            )
        )
    return tuple(permutations)


def _sigmas(raw: object, *, arity: int) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("transform sigmas must be a list")
    sigmas = tuple(float(value) for value in raw)
    if len(sigmas) != arity:
        raise ValueError("transform sigmas have the wrong arity")
    return sigmas


def _logical_input_mib(raw: object) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or raw < 1:
        raise ValueError("logical_input_mib must be a positive integer")
    return raw


def _validate_memory_target(
    *,
    dtype: str,
    dimensions: tuple[int, ...],
    logical_input_mib: int | None,
) -> None:
    if logical_input_mib is None:
        return
    logical_bytes = prod(dimensions) * _DTYPE_BYTES[dtype]
    expected_bytes = logical_input_mib * 1024 * 1024
    if logical_bytes != expected_bytes:
        raise ValueError(
            "transform dimensions do not match logical_input_mib: "
            f"{dimensions} uses {logical_bytes} bytes, expected {expected_bytes}"
        )


def _cases() -> tuple[TransformCase, ...]:
    cases: list[TransformCase] = []
    for entry in _entries():
        operation = entry.get("operation")
        raw_dtypes = entry.get("dtypes")
        sector = entry.get("sector")
        if operation != "permute":
            raise ValueError(f"unsupported transform operation: {operation}")
        if not isinstance(raw_dtypes, list) or not all(
            isinstance(value, str) and value in _DTYPE_BYTES
            for value in raw_dtypes
        ):
            raise ValueError("transform dtypes must use Tensor0 dtype names")
        if not isinstance(sector, str) or sector not in _SECTORS:
            raise ValueError(f"unsupported transform sector: {sector}")
        target_mib = _logical_input_mib(entry.get("logical_input_mib"))
        for dtype in cast(list[str], raw_dtypes):
            for dimensions in _dimensions(entry.get("dimensions")):
                sigmas = _sigmas(entry.get("sigmas"), arity=len(dimensions))
                _validate_memory_target(
                    dtype=dtype,
                    dimensions=dimensions,
                    logical_input_mib=target_mib,
                )
                for permutation in _permutations(entry.get("permutations")):
                    flat = permutation[0] + permutation[1]
                    if sorted(flat) != list(range(1, len(dimensions) + 1)):
                        raise ValueError(
                            "transform permutation must cover each dimension once"
                        )
                    cases.append(
                        TransformCase(
                            "permute",
                            dtype,
                            sector,
                            dimensions,
                            sigmas,
                            permutation,
                            target_mib,
                        )
                    )
    return tuple(cases)


def _prepared(case: TransformCase) -> PreparedOperation:
    sigmas = case.sigmas or (None,) * len(case.dimensions)
    spaces = tuple(
        generate_space(case.sector, dimension, sigma)
        for dimension, sigma in zip(case.dimensions, sigmas, strict=True)
    )
    p_codomain, p_domain = case.permutation
    source_codomain = tuple(spaces[index - 1] for index in p_codomain)
    source_domain = tuple(spaces[index - 1] for index in p_domain)
    tensor = random_tensor(
        hom(source_codomain, source_domain),
        dtype=case.dtype,
        seed=601,
    )
    zero_based = (
        tuple(index - 1 for index in p_codomain),
        tuple(index - 1 for index in p_domain),
    )

    def run(value: TensorMap) -> TensorMap:
        return permute(value, zero_based)

    return run, (tensor,)


def _trivial_rank4_permute_prepared(
    *,
    dtype: str,
    size_label: str,
) -> PreparedOperation:
    dimensions = {
        "small": (4, 3, 5, 2),
        "medium": (12, 8, 10, 6),
    }[size_label]
    a_dim, b_dim, c_dim, d_dim = dimensions
    tensor = dense_tensor(
        hom(
            (ComplexSpace(a_dim), ComplexSpace(b_dim)),
            (ComplexSpace(c_dim), ComplexSpace(d_dim)),
        ),
        dtype=dtype,
    )

    def run(value: TensorMap) -> object:
        return permute(value, ((1, 0), (3, 2)))

    return run, (tensor,)


def _dimension_label(dimensions: tuple[int, ...]) -> str:
    return "d" + "x".join(str(value) for value in dimensions)


def _permutation_label(
    permutation: tuple[tuple[int, ...], tuple[int, ...]],
) -> str:
    left = "x".join(str(index) for index in permutation[0]) or "empty"
    right = "x".join(str(index) for index in permutation[1]) or "empty"
    return f"p{left}_to_{right}"


def _workload(case: TransformCase) -> WorkloadSpec:
    size_label = _dimension_label(case.dimensions)
    return WorkloadSpec(
        base_id=(
            f"transforms.permute.{case.sector}.{case.dtype}."
            f"{size_label}.{_permutation_label(case.permutation)}"
        ),
        group="transforms",
        description=(
            "Tensor0 functional permute workload; "
            f"sector={case.sector}, dimensions={case.dimensions}"
        ),
        dtype=case.dtype,
        size_label=size_label,
        factory=partial(_prepared, case),
    )


TRANSFORM_CASES = _cases()
TRANSFORM_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = tuple(
    ScenarioSpec(_workload(case), EAGER, EXPLICIT)
    for case in TRANSFORM_CASES
)

_SMOKE_CASE = TransformCase(
    "permute",
    "float64",
    "trivial",
    (64, 48),
    None,
    ((2, 1), ()),
    None,
)
TRANSFORM_SMOKE_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(_workload(_SMOKE_CASE), EAGER, QUICK),
)


def _clear_braider_cache() -> None:
    transforms_module._TREE_BRAIDER_CACHE.clear()


def _clear_transposer_cache() -> None:
    transforms_module._TREE_TRANSPOSER_CACHE.clear()


def _u1_permute_prepared(*, size_label: str) -> PreparedOperation:
    if size_label == "small":
        v = space(U1Irrep, {0: 2, 1: 1})
        w = space(U1Irrep, {0: 1, 1: 2})
        x = space(U1Irrep, {1: 1})
    elif size_label == "large":
        v = space(U1Irrep, {0: 16, 1: 16})
        w = space(U1Irrep, {0: 12, 1: 12})
        x = space(U1Irrep, {0: 8, 1: 8})
    else:
        raise ValueError(f"unsupported U1 permute size: {size_label}")
    tensor = packed_tensor(hom((v, w), (x,)))
    permutation = ((1,), (0, 2))

    def run(value: TensorMap) -> object:
        return permute(value, permutation)

    return run, (tensor,)


def _u1_repartition_prepared() -> PreparedOperation:
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 5})
    tensor = packed_tensor(hom((v,), (w, x)))

    def run(value: TensorMap) -> object:
        return repartition(value, 2)

    return run, (tensor,)


def _su2_permute_prepared() -> PreparedOperation:
    half = space(SU2Irrep, {1: 1})
    tensor = packed_tensor(hom((half, half, half), (half,)))
    permutation = ((1, 2), (0, 3))

    def run(value: TensorMap) -> object:
        return permute(value, permutation)

    return run, (tensor,)


def _su2_mixed_spin_permute_prepared() -> PreparedOperation:
    half = space(SU2Irrep, {1: 3})
    one = space(SU2Irrep, {2: 2})
    tensor = packed_tensor(hom((half, one), (half, one)))

    def run(value: TensorMap) -> object:
        return permute(value, ((1, 0), (3, 2)))

    return run, (tensor,)


def _u1_identity_twist_prepared() -> PreparedOperation:
    factor = space(U1Irrep, {0: 8, 1: 8})
    tensor = packed_tensor(hom((factor,), (factor,)))

    def run(value: TensorMap) -> object:
        return twist(value, 0)

    return run, (tensor,)


def _fermion_parity_twist_prepared() -> PreparedOperation:
    factor = space(FermionParity, {0: 16, 1: 16})
    tensor = packed_tensor(hom((factor,), (factor,)))

    def run(value: TensorMap) -> object:
        return twist(value, 0)

    return run, (tensor,)


_TRANSFORM_COLD = cold_execution(cache_policy="cold_transform")
_TRANSFORM_CACHED = eager_execution(
    id_suffix="cached",
    cache_policy="cached_transform",
)
_U1_PERMUTE_SMALL = WorkloadSpec(
    "transforms.permute.u1.float64.small",
    "transforms",
    "U1 permute with explicit tree-braider cache state",
    "float64",
    "small",
    partial(_u1_permute_prepared, size_label="small"),
    cache_clear=_clear_braider_cache,
)
_U1_REPARTITION_SMALL = WorkloadSpec(
    "transforms.repartition.u1.float64.small",
    "transforms",
    "U1 repartition with explicit tree-transposer cache state",
    "float64",
    "small",
    _u1_repartition_prepared,
    cache_clear=_clear_transposer_cache,
)
_SU2_PERMUTE_SMALL = WorkloadSpec(
    "transforms.permute.su2.float64.small",
    "transforms",
    "SU2 permute with explicit generic transformer cache state",
    "float64",
    "small",
    _su2_permute_prepared,
    cache_clear=_clear_braider_cache,
)
_SU2_MIXED_SPIN_PERMUTE = WorkloadSpec(
    "protect.su2.permute.float64.small",
    "transforms",
    "SU2 mixed-spin rank-4 permutation coverage",
    "float64",
    "small",
    _su2_mixed_spin_permute_prepared,
)
_U1_PERMUTE_LARGE = WorkloadSpec(
    "transforms.permute.u1.float64.large",
    "transforms",
    "Large U1 permute with explicit tree-braider cache state",
    "float64",
    "large",
    partial(_u1_permute_prepared, size_label="large"),
    cache_clear=_clear_braider_cache,
)
_U1_IDENTITY_TWIST = WorkloadSpec(
    "transforms.twist.u1.identity",
    "transforms",
    "U1 identity twist",
    "float64",
    "small",
    _u1_identity_twist_prepared,
)
_FERMION_PARITY_TWIST = WorkloadSpec(
    "transforms.twist.fermion_parity.nontrivial",
    "transforms",
    "FermionParity nontrivial twist",
    "float64",
    "small",
    _fermion_parity_twist_prepared,
)
_TRIVIAL_RANK4_PERMUTE_SMALL = WorkloadSpec(
    "permute.trivial.rank4.float64.small",
    "transforms",
    "Trivial rank-4 nonidentity permutation with shape (4, 3, 5, 2)",
    "float64",
    "small",
    partial(
        _trivial_rank4_permute_prepared,
        dtype="float64",
        size_label="small",
    ),
)
_TRIVIAL_RANK4_PERMUTE_MEDIUM = WorkloadSpec(
    "permute.trivial.rank4.complex128.medium",
    "transforms",
    "Trivial rank-4 nonidentity permutation with shape (12, 8, 10, 6)",
    "complex128",
    "medium",
    partial(
        _trivial_rank4_permute_prepared,
        dtype="complex128",
        size_label="medium",
    ),
)

TRANSFORM_EXTENSION_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    tuple(
        ScenarioSpec(
            workload,
            execution,
            cast(ScenarioProfile, profile),
        )
        for workload, execution, profile in (
            (_U1_PERMUTE_SMALL, _TRANSFORM_COLD, QUICK),
            (_U1_PERMUTE_SMALL, _TRANSFORM_CACHED, FULL),
            (_U1_REPARTITION_SMALL, _TRANSFORM_COLD, QUICK),
            (_U1_REPARTITION_SMALL, _TRANSFORM_CACHED, FULL),
            (_SU2_PERMUTE_SMALL, _TRANSFORM_COLD, QUICK),
            (_SU2_PERMUTE_SMALL, _TRANSFORM_CACHED, FULL),
            (_U1_PERMUTE_LARGE, _TRANSFORM_COLD, FULL),
            (_U1_PERMUTE_LARGE, _TRANSFORM_CACHED, FULL),
            (_U1_IDENTITY_TWIST, EAGER, QUICK),
            (_FERMION_PARITY_TWIST, EAGER, QUICK),
            (_SU2_MIXED_SPIN_PERMUTE, EAGER, QUICK),
        )
    )
    + (
        ScenarioSpec(_TRIVIAL_RANK4_PERMUTE_SMALL, EAGER, QUICK),
        ScenarioSpec(_TRIVIAL_RANK4_PERMUTE_SMALL, JIT_COMPILE, FULL),
        ScenarioSpec(_TRIVIAL_RANK4_PERMUTE_SMALL, JIT_CACHED, FULL),
        ScenarioSpec(_TRIVIAL_RANK4_PERMUTE_MEDIUM, EAGER, FULL),
    )
)


def transform_metadata() -> dict[str, object]:
    targets = sorted(
        {
            case.logical_input_mib
            for case in TRANSFORM_CASES
            if case.logical_input_mib is not None
        }
    )
    return {
        "case_count": len(TRANSFORM_CASES),
        "extension_scenario_count": len(TRANSFORM_EXTENSION_SCENARIO_SPECS),
        "operation_contract": {
            "permute": "functional operation; output allocation is inside timing",
            "twist": "functional transform; result allocation is inside timing",
        },
        "memory_contract": {
            "logical_input_mib": targets,
            "minimum_live_dense_mib": [2 * value for value in targets],
            "boundary": "input remains live while a distinct output is allocated",
        },
        "cases": [case._asdict() for case in TRANSFORM_CASES],
        "smoke_case_count": len(TRANSFORM_SMOKE_SCENARIO_SPECS),
    }
