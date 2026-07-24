"""Subprocess orchestration and paired aggregation for backend comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any
import uuid

from ._protocol import (
    BACKEND_NAMES,
    SCHEMA_VERSION,
    ProtocolError,
    build_request,
    validate_backend_response,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TENSORKIT_ROOT = Path(__file__).parent / "backends" / "tensorkit"
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "CROSS_BACKEND_THREADS",
)
_XLA_THREAD_PREFIXES = (
    "--xla_cpu_multi_thread_eigen=",
    "intra_op_parallelism_threads=",
    "--intra_op_parallelism_threads=",
    "inter_op_parallelism_threads=",
    "--inter_op_parallelism_threads=",
)
_OUTPUT_RTOL = 1e-9
_OUTPUT_ATOL = 1e-9


def _backend_command(name: str) -> list[str]:
    if name == "tensor0":
        return [
            sys.executable,
            "-m",
            "benchmarks.cross_backend.backends.tensor0.runner",
        ]
    if name == "tensorkit":
        julia = os.environ.get("CROSS_BACKEND_JULIA") or shutil.which("julia")
        if julia is None:
            raise FileNotFoundError("julia executable was not found")
        return [
            julia,
            f"--project={_TENSORKIT_ROOT}",
            str(_TENSORKIT_ROOT / "runner.jl"),
        ]
    raise ProtocolError(f"unknown backend: {name}")


def _thread_environment(threads: int) -> dict[str, str]:
    value = str(threads)
    return {
        name: value
        for name in _THREAD_ENVIRONMENT_NAMES
    }


def _xla_flags(existing: str, threads: int) -> str:
    flags = [
        flag
        for flag in existing.split()
        if not flag.startswith(_XLA_THREAD_PREFIXES)
    ]
    if threads == 1:
        flags.append("--xla_cpu_multi_thread_eigen=false")
    else:
        flags.extend(
            (
                "--xla_cpu_multi_thread_eigen=true",
                f"intra_op_parallelism_threads={threads}",
                "inter_op_parallelism_threads=1",
            )
        )
    return " ".join(flags)


def _subprocess_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_thread_environment(threads))
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MKL_DYNAMIC"] = "FALSE"
    environment["XLA_FLAGS"] = _xla_flags(
        environment.get("XLA_FLAGS", ""),
        threads,
    )
    return environment


def _invoke_backend(
    name: str,
    request: dict[str, Any],
    *,
    threads: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        command = _backend_command(name)
    except (FileNotFoundError, ProtocolError) as error:
        return None, {
            "backend": name,
            "round": request["round"],
            "kind": "unavailable",
            "error": str(error),
        }

    with tempfile.TemporaryDirectory(prefix="tensor0-cross-backend-") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [*command, str(request_path)],
                cwd=_REPOSITORY_ROOT,
                env=_subprocess_environment(threads),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            return None, {
                "backend": name,
                "round": request["round"],
                "kind": "launch_error",
                "error": str(error),
            }
    if completed.returncode != 0:
        return None, {
            "backend": name,
            "round": request["round"],
            "kind": "backend_error",
            "returncode": completed.returncode,
            "error": completed.stderr[-4000:].strip()
            or completed.stdout[-4000:].strip()
            or "backend exited without diagnostics",
        }
    try:
        raw_response = json.loads(completed.stdout)
        response = validate_backend_response(
            raw_response,
            expected_backend=name,
            request=request,
        )
    except (json.JSONDecodeError, ProtocolError) as error:
        return None, {
            "backend": name,
            "round": request["round"],
            "kind": "protocol_error",
            "error": str(error),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    return response, None


def _statistics(samples: list[float]) -> dict[str, float]:
    quartiles = (
        statistics.quantiles(samples, n=4, method="inclusive")
        if len(samples) >= 2
        else None
    )
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "iqr_ms": 0.0 if quartiles is None else quartiles[2] - quartiles[0],
        "max_ms": max(samples),
    }


def _complex_output(result: dict[str, Any]) -> complex:
    output = result["output"]
    return complex(output["real"], output["imag"])


def _outputs_agree(left: complex, right: complex) -> bool:
    return math.isclose(
        left.real,
        right.real,
        rel_tol=_OUTPUT_RTOL,
        abs_tol=_OUTPUT_ATOL,
    ) and math.isclose(
        left.imag,
        right.imag,
        rel_tol=_OUTPUT_RTOL,
        abs_tol=_OUTPUT_ATOL,
    )


def _output_errors(reference: complex, candidate: complex) -> tuple[float, float]:
    absolute = abs(candidate - reference)
    scale = max(abs(reference), abs(candidate))
    relative = 0.0 if scale == 0 else absolute / scale
    return absolute, relative


def _repository_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--short"),
                cwd=_REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": "unknown"}
    return {"revision": revision, "dirty": dirty}


def _environment(threads: int) -> dict[str, Any]:
    suite_root = Path(__file__).parent
    support_files = sorted(
        path
        for path in suite_root.rglob("*")
        if path.is_file() and path.suffix in {".jl", ".json", ".py", ".toml"}
    )
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "orchestrator_python": platform.python_version(),
        "repository": _repository_state(),
        "support_files_sha256": {
            str(path.relative_to(_REPOSITORY_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in support_files
        },
        "thread_policy": {
            **_thread_environment(threads),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "XLA_FLAGS": _xla_flags("", threads),
        },
    }


def run_comparison(
    *,
    profile: str,
    workloads: Iterable[dict[str, Any]],
    backends: Iterable[str],
    baseline: str,
    warmup: int,
    repeat: int,
    rounds: int,
    threads: int = 1,
) -> dict[str, Any]:
    selected_workloads = tuple(workloads)
    selected_backends = tuple(backends)
    if not selected_workloads:
        raise ProtocolError("at least one workload must be selected")
    if not selected_backends:
        raise ProtocolError("at least one backend must be selected")
    unknown_backends = sorted(set(selected_backends) - set(BACKEND_NAMES))
    if unknown_backends:
        raise ProtocolError(
            "unknown backends: " + ", ".join(unknown_backends)
        )
    if len(selected_backends) != len(set(selected_backends)):
        raise ProtocolError("selected backends contain duplicates")
    if baseline not in selected_backends:
        raise ProtocolError("baseline must be one of the selected backends")
    if warmup < 0 or repeat < 1 or rounds < 1 or threads < 1:
        raise ProtocolError(
            "warmup, repeat, rounds, and threads are out of range"
        )

    run_id = uuid.uuid4().hex
    responses: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    backend_metadata: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for round_index in range(rounds):
        workload_order = (
            selected_workloads
            if round_index % 2 == 0
            else tuple(reversed(selected_workloads))
        )
        backend_order = (
            selected_backends
            if round_index % 2 == 0
            else tuple(reversed(selected_backends))
        )
        request = build_request(
            run_id=run_id,
            round_index=round_index,
            warmup=warmup,
            repeat=repeat,
            workloads=workload_order,
        )
        for backend_name in backend_order:
            response, failure = _invoke_backend(
                backend_name,
                request,
                threads=threads,
            )
            if failure is not None:
                failures.append(failure)
                continue
            assert response is not None
            previous_metadata = backend_metadata.setdefault(
                backend_name,
                response["backend"],
            )
            if previous_metadata != response["backend"]:
                failures.append(
                    {
                        "backend": backend_name,
                        "round": round_index,
                        "kind": "metadata_changed",
                        "error": "backend metadata changed between rounds",
                    }
                )
            for result in response["results"]:
                responses[(backend_name, result["workload_id"])].append(result)

    records: list[dict[str, Any]] = []
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for workload in selected_workloads:
        for backend_name in selected_backends:
            round_results = responses.get((backend_name, workload["id"]), [])
            successful = [
                result for result in round_results if result["status"] == "ok"
            ]
            errors = [
                result["error"]
                for result in round_results
                if result["status"] == "error"
            ]
            if not successful:
                record = {
                    "workload_id": workload["id"],
                    "workload_hash": workload["workload_hash"],
                    "backend": backend_name,
                    "status": "error",
                    "completed_rounds": 0,
                    "expected_rounds": rounds,
                    "errors": errors or ["backend produced no successful result"],
                    "validation": {
                        "mode": workload["validation"],
                        "status": "not_run",
                    },
                }
            else:
                samples = [
                    float(sample)
                    for result in successful
                    for sample in result["samples_ms"]
                ]
                outputs = [_complex_output(result) for result in successful]
                stable_output = all(
                    _outputs_agree(outputs[0], output)
                    for output in outputs[1:]
                )
                record = {
                    "workload_id": workload["id"],
                    "workload_hash": workload["workload_hash"],
                    "backend": backend_name,
                    "status": (
                        "ok"
                        if len(successful) == rounds and not errors
                        else "partial"
                    ),
                    "completed_rounds": len(successful),
                    "expected_rounds": rounds,
                    "warmup": warmup,
                    "repeat_per_round": repeat,
                    "samples_ms": samples,
                    **_statistics(samples),
                    "output": {
                        "real": outputs[0].real,
                        "imag": outputs[0].imag,
                    },
                    "errors": errors,
                    "validation": {
                        "mode": workload["validation"],
                        "status": (
                            "pending" if stable_output else "failed"
                        ),
                        "detail": (
                            "backend output was stable across rounds"
                            if stable_output
                            else "backend output changed across rounds"
                        ),
                    },
                }
            records.append(record)
            records_by_key[(backend_name, workload["id"])] = record

    pairwise: list[dict[str, Any]] = []
    for workload in selected_workloads:
        comparable = [
            records_by_key[(backend_name, workload["id"])]
            for backend_name in selected_backends
            if records_by_key[(backend_name, workload["id"])]["status"]
            in {"ok", "partial"}
        ]
        if workload["validation"] == "structural_only":
            for record in comparable:
                if record["validation"]["status"] != "failed":
                    record["validation"] = {
                        "mode": "structural_only",
                        "status": "structural_only",
                        "detail": "fusion-basis equivalence is not asserted",
                    }
        elif len(comparable) < 2:
            for record in comparable:
                if record["validation"]["status"] != "failed":
                    record["validation"] = {
                        "mode": "cross_backend",
                        "status": "not_compared",
                        "detail": "fewer than two backends completed",
                    }
        else:
            reference = records_by_key[(baseline, workload["id"])]
            if reference["status"] not in {"ok", "partial"}:
                reference = comparable[0]
            reference_output = _complex_output(reference)
            agrees = all(
                record["validation"]["status"] != "failed"
                and _outputs_agree(
                    reference_output,
                    _complex_output(record),
                )
                for record in comparable
            )
            for record in comparable:
                absolute_error, relative_error = _output_errors(
                    reference_output,
                    _complex_output(record),
                )
                record["validation"] = {
                    "mode": "cross_backend",
                    "status": "passed" if agrees else "failed",
                    "reference_backend": reference["backend"],
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "relative_tolerance": _OUTPUT_RTOL,
                    "absolute_tolerance": _OUTPUT_ATOL,
                }

        baseline_record = records_by_key[(baseline, workload["id"])]
        if baseline_record["status"] not in {"ok", "partial"}:
            continue
        for backend_name in selected_backends:
            if backend_name == baseline:
                continue
            contender = records_by_key[(backend_name, workload["id"])]
            if contender["status"] not in {"ok", "partial"}:
                continue
            pairwise.append(
                {
                    "workload_id": workload["id"],
                    "baseline": baseline,
                    "contender": backend_name,
                    "baseline_median_ms": baseline_record["median_ms"],
                    "contender_median_ms": contender["median_ms"],
                    "contender_speedup": (
                        baseline_record["median_ms"]
                        / contender["median_ms"]
                    ),
                    "validation": contender["validation"]["status"],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "suite": "cross_backend",
        "run_id": run_id,
        "config": {
            "profile": profile,
            "measurement": "steady_state",
            "workloads": [workload["id"] for workload in selected_workloads],
            "backends": list(selected_backends),
            "baseline": baseline,
            "warmup": warmup,
            "repeat": repeat,
            "rounds": rounds,
            "threads": threads,
        },
        "environment": _environment(threads),
        "backend_metadata": backend_metadata,
        "records": records,
        "pairwise": pairwise,
        "failures": failures,
    }
