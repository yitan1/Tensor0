"""Versioned workload and result protocol for cross-backend benchmarks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import json
from math import ceil, factorial, isfinite, log
from pathlib import Path
import re
import tomllib
from typing import Any


SCHEMA_VERSION = 1
MEASUREMENT_KIND = "steady_state"
SUPPORTED_DTYPES = {"float64", "complex128"}
SUPPORTED_SECTORS = {"trivial", "z2", "u1", "su2"}
DATA_POLICIES = {"uniform_v1", "fusion_tree_v1"}
VALIDATION_MODES = {"cross_backend", "structural_only"}
BACKEND_NAMES = ("tensor0", "tensorkit")
SPACE_POLICIES = {
    "trivial_v1",
    "z2_equal_v1",
    "u1_poisson_v1",
}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ProtocolError(ValueError):
    """Raised when a workload or backend message violates the protocol."""


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be an object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolError(f"{context} must be a list")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProtocolError(f"{context} must be an integer >= {minimum}")
    return value


def _string_list(value: object, context: str) -> list[str]:
    values = _list(value, context)
    if not all(isinstance(item, str) and item for item in values):
        raise ProtocolError(f"{context} must contain non-empty strings")
    return [item for item in values if isinstance(item, str)]


def _int_list(value: object, context: str) -> list[int]:
    values = _list(value, context)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ProtocolError(f"{context} must contain integers")
    return [
        item
        for item in values
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _bool_list(value: object, context: str) -> list[bool]:
    values = _list(value, context)
    if not all(isinstance(item, bool) for item in values):
        raise ProtocolError(f"{context} must contain booleans")
    return [item for item in values if isinstance(item, bool)]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def workload_hash(workload: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in workload.items()
        if key != "workload_hash"
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _normalized_topology(name: str, raw: object) -> dict[str, Any]:
    topology = _object(raw, f"topology.{name}")
    space_names = _string_list(
        topology.get("space_names"),
        f"topology.{name}.space_names",
    )
    if len(space_names) != len(set(space_names)):
        raise ProtocolError(f"topology.{name}.space_names contains duplicates")

    tensors: list[dict[str, Any]] = []
    tensor_names: set[str] = set()
    for index, raw_tensor in enumerate(
        _list(topology.get("tensor"), f"topology.{name}.tensor")
    ):
        context = f"topology.{name}.tensor[{index}]"
        tensor = _object(raw_tensor, context)
        tensor_name = _string(tensor.get("name"), f"{context}.name")
        if tensor_name in tensor_names:
            raise ProtocolError(f"{context}.name duplicates {tensor_name!r}")
        tensor_names.add(tensor_name)
        tensor_spaces = _string_list(
            tensor.get("spaces"),
            f"{context}.spaces",
        )
        duals = _bool_list(tensor.get("duals"), f"{context}.duals")
        if len(tensor_spaces) != len(duals):
            raise ProtocolError(f"{context}.spaces and duals differ in length")
        unknown_spaces = sorted(set(tensor_spaces) - set(space_names))
        if unknown_spaces:
            raise ProtocolError(
                f"{context}.spaces contains unknown names: "
                + ", ".join(unknown_spaces)
            )
        scale = tensor.get("scale")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise ProtocolError(f"{context}.scale must be numeric")
        if not isfinite(float(scale)):
            raise ProtocolError(f"{context}.scale must be finite")
        tensors.append(
            {
                "name": tensor_name,
                "spaces": tensor_spaces,
                "duals": duals,
                "scale": float(scale),
            }
        )

    operands = _string_list(
        topology.get("operands"),
        f"topology.{name}.operands",
    )
    labels = [
        _int_list(value, f"topology.{name}.labels[{index}]")
        for index, value in enumerate(
            _list(topology.get("labels"), f"topology.{name}.labels")
        )
    ]
    conjugate = _bool_list(
        topology.get("conjugate"),
        f"topology.{name}.conjugate",
    )
    if not (len(operands) == len(labels) == len(conjugate)):
        raise ProtocolError(
            f"topology.{name} operands, labels, and conjugate differ in length"
        )
    unknown_operands = sorted(set(operands) - tensor_names)
    if unknown_operands:
        raise ProtocolError(
            f"topology.{name}.operands contains unknown tensors: "
            + ", ".join(unknown_operands)
        )

    label_counts = Counter(label for item in labels for label in item)
    invalid_labels = sorted(
        label for label, count in label_counts.items()
        if label < 1 or count != 2
    )
    if invalid_labels:
        raise ProtocolError(
            f"topology.{name} contraction labels must be positive and paired: "
            + ", ".join(str(label) for label in invalid_labels)
        )
    order = _int_list(topology.get("order"), f"topology.{name}.order")
    if len(order) != len(set(order)) or set(order) != set(label_counts):
        raise ProtocolError(
            f"topology.{name}.order must contain each contraction label once"
        )
    return {
        "name": name,
        "space_names": space_names,
        "tensors": tensors,
        "operands": operands,
        "labels": labels,
        "conjugate": conjugate,
        "order": order,
    }


def _space_dimension(sector: str, sectors: list[list[int]]) -> int:
    if sector == "su2":
        return sum((label + 1) * degeneracy for label, degeneracy in sectors)
    return sum(degeneracy for _, degeneracy in sectors)


def _normalized_space(
    raw: object,
    *,
    sector: str,
    context: str,
) -> dict[str, Any]:
    space = _object(raw, context)
    name = _string(space.get("name"), f"{context}.name")
    sectors: list[list[int]] = []
    labels: set[int] = set()
    for index, raw_sector in enumerate(
        _list(space.get("sectors"), f"{context}.sectors")
    ):
        pair = _int_list(raw_sector, f"{context}.sectors[{index}]")
        if len(pair) != 2:
            raise ProtocolError(
                f"{context}.sectors[{index}] must be [label, degeneracy]"
            )
        label, degeneracy = pair
        if degeneracy < 1:
            raise ProtocolError(
                f"{context}.sectors[{index}] degeneracy must be positive"
            )
        if label in labels:
            raise ProtocolError(f"{context}.sectors contains duplicate label {label}")
        if sector in {"trivial", "z2", "su2"} and label < 0:
            raise ProtocolError(f"{context}.sectors contains a negative label")
        if sector == "trivial" and label != 0:
            raise ProtocolError(f"{context}.sectors uses a nontrivial label")
        if sector == "z2" and label not in {0, 1}:
            raise ProtocolError(f"{context}.sectors uses an invalid Z2 label")
        labels.add(label)
        sectors.append([label, degeneracy])
    return {
        "name": name,
        "sectors": sectors,
        "dimension": _space_dimension(sector, sectors),
    }


def _space_sectors(policy: str, dimension: int) -> list[list[int]]:
    if policy == "trivial_v1":
        return [[0, dimension]]
    if policy == "z2_equal_v1":
        even = ceil(0.5 * dimension)
        return [[0, even], [1, dimension - even]]
    if policy == "u1_poisson_v1":
        probability = 0.25
        rate = log((1 / probability + 1) / 2)
        neutral = ceil(probability * dimension)
        if (dimension - neutral) % 2 == 1:
            neutral = 2 if neutral == 1 else neutral - 1
        sectors = [[0, neutral]]
        remaining = dimension - neutral
        charge = 1
        while remaining > 0:
            degeneracy = ceil(
                probability
                * rate**charge
                / factorial(charge)
                * dimension
            )
            sectors.extend(
                ([-charge, degeneracy], [charge, degeneracy])
            )
            remaining -= 2 * degeneracy
            charge += 1
        return sorted(sectors)
    raise ProtocolError(f"unsupported space policy: {policy}")


def _expanded_matrix_cases(
    document: Mapping[str, object],
    *,
    path: Path,
    topologies: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, raw_matrix in enumerate(
        _list(document.get("case_matrix", []), f"{path}: case_matrix")
    ):
        context = f"{path}: case_matrix[{index}]"
        matrix = _object(raw_matrix, context)
        topology_name = _string(
            matrix.get("topology"),
            f"{context}.topology",
        )
        try:
            topology = topologies[topology_name]
        except KeyError:
            raise ProtocolError(
                f"{context}.topology refers to unknown topology "
                f"{topology_name!r}"
            ) from None
        sector = _string(matrix.get("sector"), f"{context}.sector")
        dtype = _string(matrix.get("dtype"), f"{context}.dtype")
        data_policy = _string(matrix.get("data"), f"{context}.data")
        validation = _string(
            matrix.get("validation"),
            f"{context}.validation",
        )
        space_policy = _string(
            matrix.get("space_policy"),
            f"{context}.space_policy",
        )
        if space_policy not in SPACE_POLICIES:
            raise ProtocolError(f"{context}.space_policy is unsupported")
        expected_space_policy = {
            "trivial": "trivial_v1",
            "z2": "z2_equal_v1",
            "u1": "u1_poisson_v1",
        }.get(sector)
        if space_policy != expected_space_policy:
            raise ProtocolError(
                f"{context}.space_policy does not match sector {sector!r}"
            )

        for case_index, raw_dimensions in enumerate(
            _list(matrix.get("dimensions"), f"{context}.dimensions")
        ):
            case_context = f"{context}.dimensions[{case_index}]"
            dimensions = _int_list(raw_dimensions, case_context)
            if (
                len(dimensions) != len(topology["space_names"])
                or any(dimension < 1 for dimension in dimensions)
            ):
                raise ProtocolError(
                    f"{case_context} must contain one positive dimension "
                    "per topology space"
                )
            suffix = "x".join(str(dimension) for dimension in dimensions)
            cases.append(
                {
                    "id": (
                        f"tensor_networks.{topology_name}.{sector}."
                        f"{dtype}.d{suffix}"
                    ),
                    "topology": topology_name,
                    "sector": sector,
                    "dtype": dtype,
                    "data": data_policy,
                    "dimensions": dimensions,
                    "validation": validation,
                    "_matrix_case": True,
                    "spaces": [
                        {
                            "name": name,
                            "sectors": _space_sectors(
                                space_policy,
                                dimension,
                            ),
                        }
                        for name, dimension in zip(
                            topology["space_names"],
                            dimensions,
                            strict=True,
                        )
                    ],
                }
            )
    return cases


def load_workloads(
    directory: Path | None = None,
) -> dict[str, dict[str, Any]]:
    root = directory or Path(__file__).with_name("workloads")
    paths = sorted(root.glob("*.toml"))
    if not paths:
        raise ProtocolError(f"no workload files found in {root}")

    workloads: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError(
                f"{path} must use schema_version {SCHEMA_VERSION}"
            )
        raw_topologies = _object(document.get("topology"), f"{path}: topology")
        topologies = {
            name: _normalized_topology(name, raw)
            for name, raw in raw_topologies.items()
        }
        raw_cases = [
            *_expanded_matrix_cases(
                document,
                path=path,
                topologies=topologies,
            ),
            *_list(document.get("case", []), f"{path}: case"),
        ]
        for index, raw_case in enumerate(raw_cases):
            context = f"{path}: case[{index}]"
            case = _object(raw_case, context)
            workload_id = _string(case.get("id"), f"{context}.id")
            if not _IDENTIFIER.fullmatch(workload_id):
                raise ProtocolError(f"{context}.id is not a valid identifier")
            if not set(workload_id.split(".")).isdisjoint(BACKEND_NAMES):
                raise ProtocolError(f"{context}.id must not contain a backend name")
            if workload_id in workloads:
                raise ProtocolError(f"duplicate workload id {workload_id!r}")
            topology_name = _string(
                case.get("topology"),
                f"{context}.topology",
            )
            try:
                topology = topologies[topology_name]
            except KeyError:
                raise ProtocolError(
                    f"{context}.topology refers to unknown topology {topology_name!r}"
                ) from None
            sector = _string(case.get("sector"), f"{context}.sector")
            if sector not in SUPPORTED_SECTORS:
                raise ProtocolError(f"{context}.sector is unsupported")
            dtype = _string(case.get("dtype"), f"{context}.dtype")
            if dtype not in SUPPORTED_DTYPES:
                raise ProtocolError(f"{context}.dtype is unsupported")
            data_policy = _string(case.get("data"), f"{context}.data")
            if data_policy not in DATA_POLICIES:
                raise ProtocolError(f"{context}.data is unsupported")
            if (sector == "su2") != (data_policy == "fusion_tree_v1"):
                raise ProtocolError(
                    f"{context}.data must be fusion_tree_v1 exactly for SU2"
                )
            validation = _string(
                case.get("validation"),
                f"{context}.validation",
            )
            if validation not in VALIDATION_MODES:
                raise ProtocolError(f"{context}.validation is unsupported")
            dimensions = _int_list(
                case.get("dimensions"),
                f"{context}.dimensions",
            )
            if any(dimension < 1 for dimension in dimensions):
                raise ProtocolError(f"{context}.dimensions must be positive")
            spaces = [
                _normalized_space(
                    raw_space,
                    sector=sector,
                    context=f"{context}.spaces[{space_index}]",
                )
                for space_index, raw_space in enumerate(
                    _list(case.get("spaces"), f"{context}.spaces")
                )
            ]
            space_names = [space["name"] for space in spaces]
            if space_names != topology["space_names"]:
                raise ProtocolError(
                    f"{context}.spaces must follow topology space_names order"
                )
            actual_dimensions = [space["dimension"] for space in spaces]
            if case.get("_matrix_case") is True:
                dimensions_match = len(dimensions) == len(actual_dimensions)
                mismatch_detail = "one nominal dimension per space"
            else:
                dimensions_match = dimensions == actual_dimensions
                mismatch_detail = f"explicit space dimensions {actual_dimensions}"
            if not dimensions_match:
                raise ProtocolError(
                    f"{context}.dimensions {dimensions} do not match "
                    f"{mismatch_detail}"
                )
            workload = {
                "id": workload_id,
                "kind": "tensor_network",
                "topology": topology,
                "sector": sector,
                "dtype": dtype,
                "data": data_policy,
                "dimensions": dimensions,
                "spaces": spaces,
                "validation": validation,
                "source": str(path.relative_to(root.parent.parent)),
            }
            workload["workload_hash"] = workload_hash(workload)
            workloads[workload_id] = workload
    return workloads


def load_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    source = path or Path(__file__).with_name("profiles.toml")
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(
            f"{source} must use schema_version {SCHEMA_VERSION}"
        )
    raw_profiles = _object(document.get("profile"), f"{source}: profile")
    profiles: dict[str, dict[str, Any]] = {}
    for name, raw_profile in raw_profiles.items():
        context = f"{source}: profile.{name}"
        profile = _object(raw_profile, context)
        workloads = _string_list(
            profile.get("workloads"),
            f"{context}.workloads",
        )
        backends = _string_list(
            profile.get("backends"),
            f"{context}.backends",
        )
        if len(workloads) != len(set(workloads)):
            raise ProtocolError(f"{context}.workloads contains duplicates")
        if len(backends) != len(set(backends)):
            raise ProtocolError(f"{context}.backends contains duplicates")
        baseline = _string(profile.get("baseline"), f"{context}.baseline")
        if baseline not in backends:
            raise ProtocolError(f"{context}.baseline must be one of its backends")
        profiles[name] = {
            "description": _string(
                profile.get("description"),
                f"{context}.description",
            ),
            "workloads": workloads,
            "backends": backends,
            "baseline": baseline,
            "warmup": _integer(
                profile.get("warmup"),
                f"{context}.warmup",
            ),
            "repeat": _integer(
                profile.get("repeat"),
                f"{context}.repeat",
                minimum=1,
            ),
            "rounds": _integer(
                profile.get("rounds"),
                f"{context}.rounds",
                minimum=1,
            ),
        }
    return profiles


def build_request(
    *,
    run_id: str,
    round_index: int,
    warmup: int,
    repeat: int,
    workloads: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "message_type": "benchmark_request",
        "run_id": run_id,
        "round": round_index,
        "measurement": {
            "kind": MEASUREMENT_KIND,
            "warmup": warmup,
            "repeat": repeat,
        },
        "workloads": list(workloads),
    }
    validate_request(request)
    return request


def validate_request(payload: object) -> dict[str, Any]:
    request = _object(payload, "request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("request uses an unsupported schema_version")
    if request.get("message_type") != "benchmark_request":
        raise ProtocolError("request has an invalid message_type")
    _string(request.get("run_id"), "request.run_id")
    _integer(request.get("round"), "request.round")
    measurement = _object(request.get("measurement"), "request.measurement")
    if measurement.get("kind") != MEASUREMENT_KIND:
        raise ProtocolError("request.measurement.kind must be steady_state")
    _integer(measurement.get("warmup"), "request.measurement.warmup")
    repeat = _integer(
        measurement.get("repeat"),
        "request.measurement.repeat",
        minimum=1,
    )
    workloads = _list(request.get("workloads"), "request.workloads")
    if not workloads:
        raise ProtocolError("request.workloads must not be empty")
    for index, raw_workload in enumerate(workloads):
        workload = _object(raw_workload, f"request.workloads[{index}]")
        context = f"request.workloads[{index}]"
        _string(workload.get("id"), f"{context}.id")
        sector = _string(workload.get("sector"), f"{context}.sector")
        if sector not in SUPPORTED_SECTORS:
            raise ProtocolError(f"{context}.sector is unsupported")
        data_policy = _string(workload.get("data"), f"{context}.data")
        if data_policy not in DATA_POLICIES:
            raise ProtocolError(f"{context}.data is unsupported")
        if (sector == "su2") != (data_policy == "fusion_tree_v1"):
            raise ProtocolError(
                f"{context}.data must be fusion_tree_v1 exactly for SU2"
            )
        digest = _string(
            workload.get("workload_hash"),
            f"{context}.workload_hash",
        )
        if digest != workload_hash(workload):
            raise ProtocolError(f"{context} has an invalid workload_hash")
    return request


def validate_backend_response(
    payload: object,
    *,
    expected_backend: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    response = _object(payload, "backend response")
    if response.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("backend response uses an unsupported schema_version")
    if response.get("message_type") != "benchmark_response":
        raise ProtocolError("backend response has an invalid message_type")
    if response.get("run_id") != request.get("run_id"):
        raise ProtocolError("backend response run_id does not match the request")
    if response.get("round") != request.get("round"):
        raise ProtocolError("backend response round does not match the request")
    backend = _object(response.get("backend"), "backend response.backend")
    if backend.get("name") != expected_backend:
        raise ProtocolError("backend response name does not match the command")
    _string(backend.get("version"), "backend response.backend.version")
    _string(backend.get("runtime"), "backend response.backend.runtime")
    expected_workloads = {
        workload["id"]: workload
        for workload in request["workloads"]
    }
    results = _list(response.get("results"), "backend response.results")
    if len(results) != len(expected_workloads):
        raise ProtocolError("backend response has the wrong result count")
    seen: set[str] = set()
    expected_repeat = request["measurement"]["repeat"]
    for index, raw_result in enumerate(results):
        context = f"backend response.results[{index}]"
        result = _object(raw_result, context)
        workload_id = _string(result.get("workload_id"), f"{context}.workload_id")
        if workload_id in seen or workload_id not in expected_workloads:
            raise ProtocolError(f"{context}.workload_id is duplicate or unknown")
        seen.add(workload_id)
        if (
            result.get("workload_hash")
            != expected_workloads[workload_id]["workload_hash"]
        ):
            raise ProtocolError(f"{context}.workload_hash does not match")
        status = result.get("status")
        if status not in {"ok", "error"}:
            raise ProtocolError(f"{context}.status must be ok or error")
        if status == "ok":
            samples = _list(result.get("samples_ms"), f"{context}.samples_ms")
            if len(samples) != expected_repeat or not all(
                isinstance(sample, (int, float))
                and not isinstance(sample, bool)
                and isfinite(float(sample))
                and sample > 0
                for sample in samples
            ):
                raise ProtocolError(
                    f"{context}.samples_ms must contain {expected_repeat} "
                    "finite positive samples"
                )
            output = _object(result.get("output"), f"{context}.output")
            for component in ("real", "imag"):
                value = output.get(component)
                if not isinstance(value, (int, float)) or not isfinite(float(value)):
                    raise ProtocolError(f"{context}.output.{component} must be finite")
        else:
            _string(result.get("error"), f"{context}.error")
    return response
