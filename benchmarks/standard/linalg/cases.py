from __future__ import annotations

from functools import partial
from pathlib import Path
import tomllib
from typing import NamedTuple, cast

import jax.numpy as jnp

from tensor0 import (
    ComplexSpace,
    FermionParity,
    SectorType,
    TensorMap,
    U1Irrep,
    hom,
    space,
    svd_compact,
)

from .._inputs import dense_tensor, generate_space, packed_tensor, random_tensor
from .._specs import (
    EAGER,
    FULL,
    GRADIENT_CACHED,
    JIT_CACHED,
    JIT_COMPILE,
    QUICK,
    PreparedOperation,
    ScenarioSpec,
    WorkloadSpec,
)

class LinalgCase(NamedTuple):
    operation: str
    dtype: str
    sector: str
    dimensions: tuple[int, ...]
    sigmas: tuple[float, ...] | None


class BlockLinalgCase(NamedTuple):
    operation: str
    dtype: str
    sector: str
    sector_type: SectorType
    size_label: str
    spaces: tuple[tuple[tuple[int, int], ...], ...]


_PARAMETERS_PATH = Path(__file__).with_name("params.toml")
with _PARAMETERS_PATH.open("rb") as stream:
    _PARAMETERS = tomllib.load(stream)

_DTYPES = {"float64", "complex128"}
_SECTORS = {"trivial", "z2"}
_BLOCK_SECTORS = {
    "u1": U1Irrep,
    "fermion_parity": FermionParity,
}
_EXPECTED_ARITY = {"mul": 3, "svd": 2}


def _entries(table: str) -> tuple[dict[str, object], ...]:
    raw_entries = _PARAMETERS.get(table)
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, dict) for entry in raw_entries
    ):
        raise ValueError(f"invalid linalg benchmark table: {table}")
    return tuple(cast(dict[str, object], entry) for entry in raw_entries)


def _dimensions(operation: str, raw: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not all(isinstance(item, list) for item in raw):
        raise ValueError(f"{operation} dimensions must contain dimension lists")
    cases: list[tuple[int, ...]] = []
    for item in raw:
        raw_dimensions = cast(list[object], item)
        if not all(isinstance(value, int) and value > 0 for value in raw_dimensions):
            raise ValueError(f"{operation} dimensions must be positive integers")
        dimensions = tuple(cast(list[int], raw_dimensions))
        if len(dimensions) != _EXPECTED_ARITY[operation]:
            raise ValueError(f"{operation} dimensions have the wrong arity")
        cases.append(dimensions)
    return tuple(cases)


def _sigmas(operation: str, raw: object) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{operation} sigmas must be a list")
    sigmas = tuple(float(value) for value in raw)
    if len(sigmas) != _EXPECTED_ARITY[operation]:
        raise ValueError(f"{operation} sigmas have the wrong arity")
    return sigmas


def _cases() -> tuple[LinalgCase, ...]:
    cases: list[LinalgCase] = []
    for entry in _entries("workload"):
        operation = entry.get("operation")
        raw_dtypes = entry.get("dtypes")
        sector = entry.get("sector")
        if operation not in _EXPECTED_ARITY:
            raise ValueError(f"unsupported linalg operation: {operation}")
        operation = cast(str, operation)
        if not isinstance(raw_dtypes, list) or not all(
            isinstance(value, str) and value in _DTYPES
            for value in raw_dtypes
        ):
            raise ValueError(f"{operation} dtypes must use Tensor0 dtype names")
        if not isinstance(sector, str) or sector not in _SECTORS:
            raise ValueError(f"unsupported {operation} sector: {sector}")
        sigmas = _sigmas(operation, entry.get("sigmas"))
        for dtype in cast(list[str], raw_dtypes):
            for dimensions in _dimensions(operation, entry.get("dimensions")):
                cases.append(
                    LinalgCase(
                        operation,
                        dtype,
                        sector,
                        dimensions,
                        sigmas,
                    )
                )
    return tuple(cases)


def _spaces(case: LinalgCase):
    sigmas = case.sigmas or (None,) * len(case.dimensions)
    return tuple(
        generate_space(case.sector, dimension, sigma)
        for dimension, sigma in zip(case.dimensions, sigmas, strict=True)
    )


def _mul_prepared(case: LinalgCase) -> PreparedOperation:
    left_space, middle_space, right_space = _spaces(case)
    left = random_tensor(
        hom((left_space,), (middle_space,)),
        dtype=case.dtype,
        seed=501,
    )
    right = random_tensor(
        hom((middle_space,), (right_space,)),
        dtype=case.dtype,
        seed=502,
    )

    def run(left_value: TensorMap, right_value: TensorMap) -> TensorMap:
        return left_value @ right_value

    return run, (left, right)


def _trivial_composition_prepared(
    *,
    dtype: str,
    size_label: str,
) -> PreparedOperation:
    dimensions = {
        "small": (16, 12, 14),
        "medium": (64, 48, 56),
    }[size_label]
    output_dim, middle_dim, input_dim = dimensions
    middle = ComplexSpace(middle_dim)
    left = dense_tensor(
        hom((ComplexSpace(output_dim),), (middle,)),
        dtype=dtype,
    )
    right = dense_tensor(
        hom((middle,), (ComplexSpace(input_dim),)),
        dtype=dtype,
    )

    def run(left_value: TensorMap, right_value: TensorMap) -> TensorMap:
        return left_value @ right_value

    return run, (left, right)


def _svd_prepared(case: LinalgCase) -> PreparedOperation:
    left_space, right_space = _spaces(case)
    tensor = random_tensor(
        hom((left_space,), (right_space,)),
        dtype=case.dtype,
        seed=503,
    )
    return svd_compact, (tensor,)


def _prepared(case: LinalgCase) -> PreparedOperation:
    if case.operation == "mul":
        return _mul_prepared(case)
    return _svd_prepared(case)


def _dimension_label(dimensions: tuple[int, ...]) -> str:
    return "d" + "x".join(str(value) for value in dimensions)


def _workload(case: LinalgCase) -> WorkloadSpec:
    size_label = _dimension_label(case.dimensions)
    mapping = "TensorMap @" if case.operation == "mul" else "svd_compact"
    return WorkloadSpec(
        base_id=(
            f"linalg.{case.operation}.{case.sector}.{case.dtype}.{size_label}"
        ),
        group="linalg",
        description=(
            f"Tensor0 {case.operation} workload via {mapping}; "
            f"sector={case.sector}, dimensions={case.dimensions}"
        ),
        dtype=case.dtype,
        size_label=size_label,
        factory=partial(_prepared, case),
    )


def _standard_profile(case: LinalgCase):
    if (
        case.sector == "trivial"
        and case.dtype == "float64"
        and all(dimension == 2 for dimension in case.dimensions)
    ):
        return QUICK
    return FULL


def _block_space(raw: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise ValueError("blockwise sector dimensions must be a list")
    dimensions: list[tuple[int, int]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, int) for value in item)
            or cast(list[int], item)[1] < 1
        ):
            raise ValueError("invalid blockwise sector dimensions")
        charge, degeneracy = cast(list[int], item)
        dimensions.append((charge, degeneracy))
    return tuple(dimensions)


def _block_cases() -> tuple[BlockLinalgCase, ...]:
    cases: list[BlockLinalgCase] = []
    for entry in _entries("block_workload"):
        operation = entry.get("operation")
        dtype = entry.get("dtype")
        sector = entry.get("sector")
        size_label = entry.get("size")
        if operation not in _EXPECTED_ARITY:
            raise ValueError(f"unsupported blockwise operation: {operation}")
        operation = cast(str, operation)
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError(f"unsupported blockwise dtype: {dtype}")
        if not isinstance(sector, str) or sector not in _BLOCK_SECTORS:
            raise ValueError(f"unsupported blockwise sector: {sector}")
        if not isinstance(size_label, str):
            raise ValueError("blockwise size must be a string")
        names = (
            ("left", "middle", "right")
            if operation == "mul"
            else ("left", "right")
        )
        cases.append(
            BlockLinalgCase(
                operation,
                dtype,
                sector,
                _BLOCK_SECTORS[sector],
                size_label,
                tuple(_block_space(entry.get(name)) for name in names),
            )
        )
    return tuple(cases)


def _block_prepared(case: BlockLinalgCase) -> PreparedOperation:
    spaces = tuple(
        space(case.sector_type, dict(dimensions))
        for dimensions in case.spaces
    )
    if case.operation == "mul":
        left_space, middle_space, right_space = spaces
        left_hom = hom((left_space,), (middle_space,))
        right_hom = hom((middle_space,), (right_space,))
        left = packed_tensor(left_hom, dtype=case.dtype, scale=0.1)
        right = packed_tensor(right_hom, dtype=case.dtype, scale=0.1)

        def run(left_value: TensorMap, right_value: TensorMap) -> TensorMap:
            return left_value @ right_value

        return run, (left, right)

    left_space, right_space = spaces
    target = hom((left_space,), (right_space,))
    tensor = packed_tensor(target, dtype=case.dtype, scale=0.1)
    return svd_compact, (tensor,)


def _block_gradient_prepared(case: BlockLinalgCase) -> PreparedOperation:
    _, arguments = _block_prepared(case)
    left = cast(TensorMap, arguments[0])
    right = cast(TensorMap, arguments[1])

    def loss(candidate: TensorMap) -> object:
        composed = candidate @ right
        return jnp.sum(composed.storage.data * composed.storage.data)

    return loss, (left,)


def _block_workload(case: BlockLinalgCase) -> WorkloadSpec:
    return WorkloadSpec(
        base_id=(
            f"linalg.{case.operation}.{case.sector}.{case.dtype}."
            f"{case.size_label}"
        ),
        group="linalg",
        description=(
            f"Focused blockwise {case.operation} workload; "
            f"sector={case.sector}, size={case.size_label}"
        ),
        dtype=case.dtype,
        size_label=case.size_label,
        factory=partial(_block_prepared, case),
        gradient_factory=(
            partial(_block_gradient_prepared, case)
            if case.operation == "mul"
            else None
        ),
    )


def _block_profile(case: BlockLinalgCase):
    if (
        case.operation in {"mul", "svd"}
        and case.sector == "u1"
        and case.dtype == "float64"
        and case.size_label == "small"
    ):
        return QUICK
    return FULL


LINALG_CASES = _cases()
LINALG_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = tuple(
    ScenarioSpec(_workload(case), EAGER, _standard_profile(case))
    for case in LINALG_CASES
)

BLOCK_LINALG_CASES = _block_cases()
_BLOCK_WORKLOADS = tuple(
    (case, _block_workload(case))
    for case in BLOCK_LINALG_CASES
)
_BLOCK_EAGER_SPECS = tuple(
    ScenarioSpec(workload, EAGER, _block_profile(case))
    for case, workload in _BLOCK_WORKLOADS
)
_JAX_MUL_WORKLOAD = next(
    workload
    for case, workload in _BLOCK_WORKLOADS
    if (
        case.operation == "mul"
        and case.sector == "u1"
        and case.dtype == "float64"
        and case.size_label == "small"
    )
)
_TRIVIAL_COMPOSITION_SMALL = WorkloadSpec(
    "composition.trivial.float64.small",
    "linalg",
    "Trivial matrix composition with dimensions (16, 12, 14)",
    "float64",
    "small",
    partial(
        _trivial_composition_prepared,
        dtype="float64",
        size_label="small",
    ),
)
_TRIVIAL_COMPOSITION_MEDIUM = WorkloadSpec(
    "composition.trivial.complex128.medium",
    "linalg",
    "Trivial matrix composition with dimensions (64, 48, 56)",
    "complex128",
    "medium",
    partial(
        _trivial_composition_prepared,
        dtype="complex128",
        size_label="medium",
    ),
)
LINALG_EXTENSION_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    _BLOCK_EAGER_SPECS
    + (
        ScenarioSpec(_JAX_MUL_WORKLOAD, JIT_COMPILE, QUICK),
        ScenarioSpec(_JAX_MUL_WORKLOAD, JIT_CACHED, QUICK),
        ScenarioSpec(_JAX_MUL_WORKLOAD, GRADIENT_CACHED, QUICK),
        ScenarioSpec(_TRIVIAL_COMPOSITION_SMALL, EAGER, QUICK),
        ScenarioSpec(_TRIVIAL_COMPOSITION_SMALL, JIT_COMPILE, FULL),
        ScenarioSpec(_TRIVIAL_COMPOSITION_SMALL, JIT_CACHED, FULL),
        ScenarioSpec(_TRIVIAL_COMPOSITION_MEDIUM, EAGER, FULL),
    )
)


def linalg_metadata() -> dict[str, object]:
    return {
        "case_count": len(LINALG_CASES),
        "block_case_count": len(BLOCK_LINALG_CASES),
        "extension_scenario_count": len(LINALG_EXTENSION_SCENARIO_SPECS),
        "operation_counts": {
            operation: sum(
                case.operation == operation for case in LINALG_CASES
            )
            for operation in ("mul", "svd")
        },
        "operation_contract": {
            "@": (
                "functional TensorMap composition; output allocation is inside timing"
            ),
            "svd_compact": (
                "functional factorization; result allocation is inside timing"
            ),
        },
        "cases": [case._asdict() for case in LINALG_CASES],
    }
