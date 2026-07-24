"""Execute cross-backend workload requests with Tensor0's public API."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import tensor0 as tensor0_package
from tensor0 import (
    ComplexSpace,
    ElementarySpace,
    FusionTree,
    HomSpace,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    Z2Irrep,
    from_blocks,
    hom,
    ncon,
    ones,
    scale,
    space,
)

from ..._protocol import SCHEMA_VERSION, validate_request


_DATA_MODULUS = 65_521
_TREE_MULTIPLIER = 257
_COORDINATE_MULTIPLIER = 1_009
_VALUE_MODULUS = 1_021
_VALUE_CENTER = 510
_VALUE_DIVISOR = 2_048


def _elementary_space(
    specification: dict[str, Any],
    sector: str,
) -> ElementarySpace:
    sectors = {
        int(label): int(degeneracy)
        for label, degeneracy in specification["sectors"]
    }
    if sector == "trivial":
        return ComplexSpace(specification["dimension"])
    if sector == "z2":
        return space(Z2Irrep, sectors)
    if sector == "u1":
        return space(U1Irrep, sectors)
    if sector == "su2":
        return space(SU2Irrep, sectors)
    raise ValueError(f"unsupported sector: {sector}")


def _dtype(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.dtype(jnp.float64)
    if name == "complex128":
        return jnp.dtype(jnp.complex128)
    raise ValueError(f"unsupported dtype: {name}")


def _mix_data_seed(seed: int, value: int) -> int:
    return (seed * _TREE_MULTIPLIER + value + 1) % _DATA_MODULUS


def _su2_label(sector: tuple[int, ...]) -> int:
    return sector[0]


def _mix_fusion_tree(seed: int, marker: int, tree: FusionTree) -> int:
    seed = _mix_data_seed(seed, marker)
    seed = _mix_data_seed(seed, len(tree.uncoupled))
    for sector in tree.uncoupled:
        seed = _mix_data_seed(seed, _su2_label(sector))
    seed = _mix_data_seed(seed, _su2_label(tree.coupled))
    for is_dual in tree.is_dual:
        seed = _mix_data_seed(seed, int(is_dual))
    seed = _mix_data_seed(seed, len(tree.innerlines))
    for sector in tree.innerlines:
        seed = _mix_data_seed(seed, _su2_label(sector))
    return seed


def _fusion_tree_seed(
    tensor_name: str,
    pair: tuple[FusionTree, FusionTree],
) -> int:
    seed = _mix_data_seed(17, len(tensor_name.encode("utf-8")))
    for value in tensor_name.encode("utf-8"):
        seed = _mix_data_seed(seed, value)
    seed = _mix_fusion_tree(seed, 11, pair[0])
    return _mix_fusion_tree(seed, 29, pair[1])


def _fusion_tree_values(
    tensor_name: str,
    pair: tuple[FusionTree, FusionTree],
    shape: tuple[int, ...],
    *,
    dtype: jnp.dtype,
    coefficient: complex | float,
) -> np.ndarray:
    code = np.full(
        shape,
        _fusion_tree_seed(tensor_name, pair),
        dtype=np.int64,
    )
    for axis, size in enumerate(shape):
        coordinate_shape = (1,) * axis + (size,) + (1,) * (len(shape) - axis - 1)
        coordinate = np.arange(1, size + 1, dtype=np.int64).reshape(
            coordinate_shape
        )
        code = np.remainder(
            code
            + (axis + 1) * _COORDINATE_MULTIPLIER * coordinate,
            _DATA_MODULUS,
        )
    centered = np.remainder(code, _VALUE_MODULUS) - _VALUE_CENTER
    storage_dtype = np.dtype(dtype)
    variation = centered.astype(storage_dtype) / _VALUE_DIVISOR
    return np.asarray(coefficient, dtype=storage_dtype) * (1 + variation)


def _tree_degeneracy_shape(
    factors: Any,
    tree: FusionTree,
) -> tuple[int, ...]:
    return tuple(
        dict(factor.sectors)[sector]
        for factor, sector in zip(factors, tree.uncoupled, strict=True)
    )


def _fusion_tree_tensor(
    target: HomSpace,
    *,
    tensor_name: str,
    dtype: jnp.dtype,
    coefficient: complex | float,
) -> TensorMap:
    pairs_by_sector: dict[
        tuple[int, ...],
        list[tuple[FusionTree, FusionTree]],
    ] = {}
    template = TensorMap.zeros(target, dtype=dtype)
    for pair in template.fusiontrees:
        pairs_by_sector.setdefault(pair[0].coupled, []).append(pair)

    blocks: dict[tuple[int, ...], np.ndarray] = {}
    for coupled, pairs in pairs_by_sector.items():
        row_layout: dict[FusionTree, tuple[int, int, tuple[int, ...]]] = {}
        col_layout: dict[FusionTree, tuple[int, int, tuple[int, ...]]] = {}
        row_stop = 0
        col_stop = 0
        for row_tree, col_tree in pairs:
            if row_tree not in row_layout:
                shape = _tree_degeneracy_shape(target.codomain, row_tree)
                size = math.prod(shape)
                row_layout[row_tree] = (row_stop, row_stop + size, shape)
                row_stop += size
            if col_tree not in col_layout:
                shape = _tree_degeneracy_shape(target.domain, col_tree)
                size = math.prod(shape)
                col_layout[col_tree] = (col_stop, col_stop + size, shape)
                col_stop += size

        block = np.empty((row_stop, col_stop), dtype=np.dtype(dtype))
        for pair in pairs:
            row_start, row_end, row_shape = row_layout[pair[0]]
            col_start, col_end, col_shape = col_layout[pair[1]]
            values = _fusion_tree_values(
                tensor_name,
                pair,
                row_shape + col_shape,
                dtype=dtype,
                coefficient=coefficient,
            )
            block[row_start:row_end, col_start:col_end] = values.reshape(
                (row_end - row_start, col_end - col_start)
            )
        blocks[coupled] = block
    return from_blocks(target, blocks, dtype=dtype)


def _prepared_operation(workload: dict[str, Any]):
    sector = workload["sector"]
    spaces = {
        specification["name"]: _elementary_space(specification, sector)
        for specification in workload["spaces"]
    }
    dtype = _dtype(workload["dtype"])
    tensors = {}
    for tensor_specification in workload["topology"]["tensors"]:
        legs = tuple(
            spaces[name].dual() if is_dual else spaces[name]
            for name, is_dual in zip(
                tensor_specification["spaces"],
                tensor_specification["duals"],
                strict=True,
            )
        )
        coefficient: complex | float = tensor_specification["scale"]
        if workload["dtype"] == "complex128":
            coefficient *= 1.0 + 0.125j
        target = hom(legs, ())
        if workload["data"] == "fusion_tree_v1":
            value = _fusion_tree_tensor(
                target,
                tensor_name=tensor_specification["name"],
                dtype=dtype,
                coefficient=coefficient,
            )
        else:
            value = scale(ones(target, dtype=dtype), coefficient)
        tensors[tensor_specification["name"]] = value

    topology = workload["topology"]
    arguments = tuple(tensors[name] for name in topology["operands"])
    labels = tuple(tuple(item) for item in topology["labels"])
    conjugate = tuple(topology["conjugate"])
    order = tuple(topology["order"])

    def operation(*values):
        return ncon(
            values,
            labels,
            conjugate=conjugate,
            order=order,
        ).scalar()

    return jax.jit(operation), arguments


def _synchronize(value: Any) -> None:
    jax.block_until_ready(value)


def _measure(
    workload: dict[str, Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    operation, arguments = _prepared_operation(workload)

    # The mandatory first call establishes the steady-state JIT boundary.
    result = operation(*arguments)
    _synchronize(result)
    for _ in range(warmup):
        result = operation(*arguments)
        _synchronize(result)

    samples_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = operation(*arguments)
        _synchronize(result)
        stop = time.perf_counter_ns()
        samples_ms.append((stop - start) / 1_000_000.0)

    scalar = complex(jax.device_get(result))
    return {
        "workload_id": workload["id"],
        "workload_hash": workload["workload_hash"],
        "status": "ok",
        "samples_ms": samples_ms,
        "output": {
            "real": scalar.real,
            "imag": scalar.imag,
        },
    }


def _backend_metadata() -> dict[str, Any]:
    return {
        "name": "tensor0",
        "version": getattr(tensor0_package, "__version__", "unknown"),
        "runtime": f"Python {platform.python_version()}",
        "jax": getattr(jax, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CROSS_BACKEND_THREADS",
                "OMP_DYNAMIC",
                "MKL_DYNAMIC",
            )
        },
    }


def execute(request: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    measurement = request["measurement"]
    results = []
    with jax.enable_x64(True):
        for workload in request["workloads"]:
            try:
                results.append(
                    _measure(
                        workload,
                        warmup=measurement["warmup"],
                        repeat=measurement["repeat"],
                    )
                )
            except Exception as error:
                results.append(
                    {
                        "workload_id": workload["id"],
                        "workload_hash": workload["workload_hash"],
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        return {
            "schema_version": SCHEMA_VERSION,
            "message_type": "benchmark_response",
            "run_id": request["run_id"],
            "round": request["round"],
            "backend": _backend_metadata(),
            "results": results,
        }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: tensor0 runner REQUEST.json")
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    print(json.dumps(execute(request), sort_keys=True))


if __name__ == "__main__":
    main()
