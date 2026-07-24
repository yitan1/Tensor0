from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.cross_backend import _orchestrator
from benchmarks.cross_backend._protocol import (
    ProtocolError,
    build_request,
    load_profiles,
    load_workloads,
    validate_backend_response,
    validate_request,
    workload_hash,
)
from benchmarks.cross_backend.backends.tensor0 import runner as tensor0_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
CROSS_BACKEND_ROOT = REPO_ROOT / "benchmarks" / "cross_backend"
COMPARISON_DIMENSIONS = {
    ("mpo", "trivial"): [
        (10, 4, 3),
        (40, 4, 3),
        (160, 4, 3),
        (640, 4, 3),
        (2560, 4, 3),
        (100, 10, 10),
        (200, 10, 10),
        (300, 20, 20),
    ],
    ("mpo", "z2"): [
        (10, 4, 4),
        (40, 4, 4),
        (160, 4, 4),
        (640, 4, 4),
        (2560, 4, 4),
        (100, 10, 10),
        (200, 10, 10),
        (300, 20, 20),
    ],
    ("mpo", "u1"): [
        (40, 5, 3),
        (160, 5, 3),
        (640, 5, 3),
        (2560, 5, 3),
        (6120, 5, 3),
        (200, 20, 20),
        (400, 20, 20),
        (400, 40, 40),
    ],
    ("pepo", "trivial"): [
        (3, 2, 2, 50),
        (3, 3, 3, 100),
        (4, 2, 2, 50),
        (4, 3, 3, 100),
        (5, 2, 2, 50),
        (5, 2, 3, 100),
        (6, 2, 2, 50),
        (6, 3, 2, 100),
    ],
    ("pepo", "z2"): [
        (4, 2, 2, 50),
        (4, 4, 4, 100),
        (5, 2, 2, 50),
        (5, 3, 4, 100),
        (6, 2, 2, 50),
        (6, 2, 4, 100),
        (8, 2, 2, 50),
        (8, 3, 2, 100),
    ],
    ("pepo", "u1"): [
        (4, 2, 2, 100),
        (4, 4, 4, 200),
        (6, 2, 2, 100),
        (6, 3, 4, 200),
        (8, 2, 2, 100),
        (8, 2, 4, 200),
        (10, 2, 2, 50),
        (10, 3, 2, 100),
    ],
    ("mera", "trivial"): [(2,), (3,), (4,), (8,), (12,), (16,)],
    ("mera", "z2"): [(2,), (4,), (8,), (12,), (16,), (20,)],
    ("mera", "u1"): [(4,), (8,), (12,), (16,), (22,), (28,)],
}


def _plot_payload() -> dict[str, object]:
    workload_ok = "tensor_networks.mpo.trivial.float64.d1"
    workload_same_panel = "tensor_networks.mpo.trivial.float64.d2"
    workload_structural = "tensor_networks.mpo.su2.float64.d2"
    workload_unavailable = "tensor_networks.mpo.u1.float64.d3"
    workloads = [
        workload_ok,
        workload_same_panel,
        workload_structural,
        workload_unavailable,
    ]
    return {
        "config": {
            "profile": "test",
            "workloads": workloads,
            "backends": ["tensor0", "tensorkit"],
            "baseline": "tensorkit",
        },
        "records": [
            {
                "workload_id": workload_ok,
                "backend": "tensor0",
                "status": "ok",
                "median_ms": 1.0,
                "samples_ms": [0.8, 1.0, 1.2],
                "validation": {"status": "passed"},
            },
            {
                "workload_id": workload_ok,
                "backend": "tensorkit",
                "status": "ok",
                "median_ms": 2.0,
                "samples_ms": [1.8, 2.0, 2.2],
                "validation": {"status": "passed"},
            },
            {
                "workload_id": workload_structural,
                "backend": "tensor0",
                "status": "partial",
                "median_ms": 4.0,
                "samples_ms": [3.8, 4.0, 4.2],
                "validation": {"status": "structural_only"},
            },
            {
                "workload_id": workload_structural,
                "backend": "tensorkit",
                "status": "ok",
                "median_ms": 3.0,
                "samples_ms": [2.8, 3.0, 3.2],
                "validation": {"status": "structural_only"},
            },
            {
                "workload_id": workload_unavailable,
                "backend": "tensor0",
                "status": "error",
                "validation": {"status": "not_run"},
            },
            {
                "workload_id": workload_unavailable,
                "backend": "tensorkit",
                "status": "error",
                "validation": {"status": "not_run"},
            },
        ],
        "pairwise": [
            {
                "workload_id": workload_ok,
                "baseline": "tensorkit",
                "contender": "tensor0",
                "contender_speedup": 2.0,
                "validation": "passed",
            },
            {
                "workload_id": workload_structural,
                "baseline": "tensorkit",
                "contender": "tensor0",
                "contender_speedup": 0.75,
                "validation": "structural_only",
            },
        ],
    }


def test_cross_backend_workloads_are_normalized_and_backend_neutral():
    workloads = load_workloads()
    profiles = load_profiles()

    assert len(workloads) == 69
    assert {
        workload["topology"]["name"] for workload in workloads.values()
    } == {"mpo", "pepo", "mera"}
    assert {workload["sector"] for workload in workloads.values()} == {
        "trivial",
        "z2",
        "u1",
        "su2",
    }
    assert all(
        workload["data"]
        == (
            "fusion_tree_v1"
            if workload["sector"] == "su2"
            else "uniform_v1"
        )
        for workload in workloads.values()
    )
    assert all(
        workload["validation"] == "cross_backend"
        for workload in workloads.values()
    )
    assert all(
        {"tensor0", "tensorkit"}.isdisjoint(workload_id.split("."))
        for workload_id in workloads
    )
    comparison_dimensions = {
        key: [
            tuple(workload["dimensions"])
            for workload in workloads.values()
            if (
                workload["topology"]["name"],
                workload["sector"],
            )
            == key
        ]
        for key in COMPARISON_DIMENSIONS
    }
    assert comparison_dimensions == COMPARISON_DIMENSIONS
    assert all(
        len(workload["dimensions"]) == len(workload["spaces"])
        for workload in workloads.values()
    )
    assert [
        space["dimension"]
        for space in workloads[
            "tensor_networks.pepo.u1.float64.d10x3x2x100"
        ]["spaces"]
    ] == [12, 3, 2, 100]
    assert set(profiles) == {"smoke", "medium", "full"}
    assert len(profiles["medium"]["workloads"]) == 36
    assert set(profiles["medium"]["workloads"]) <= set(
        profiles["full"]["workloads"]
    )
    assert Counter(
        (
            workloads[workload_id]["topology"]["name"],
            workloads[workload_id]["sector"],
        )
        for workload_id in profiles["medium"]["workloads"]
    ) == Counter(
        {
            ("mpo", "trivial"): 4,
            ("mpo", "z2"): 4,
            ("mpo", "u1"): 4,
            ("mpo", "su2"): 1,
            ("pepo", "trivial"): 4,
            ("pepo", "z2"): 4,
            ("pepo", "u1"): 4,
            ("pepo", "su2"): 1,
            ("mera", "trivial"): 3,
            ("mera", "z2"): 3,
            ("mera", "u1"): 3,
            ("mera", "su2"): 1,
        }
    )
    assert set(profiles["full"]["workloads"]) == set(workloads)
    assert all(
        workload_id in workloads
        for profile in profiles.values()
        for workload_id in profile["workloads"]
    )


def test_cross_backend_protocol_detects_workload_drift():
    workload = next(iter(load_workloads().values()))
    request = build_request(
        run_id="test",
        round_index=0,
        warmup=0,
        repeat=1,
        workloads=(workload,),
    )

    mutated = copy.deepcopy(request)
    mutated["workloads"][0]["dimensions"][0] += 1

    with pytest.raises(ProtocolError, match="invalid workload_hash"):
        validate_request(mutated)


def test_cross_backend_protocol_rejects_wrong_su2_data_policy():
    workload = copy.deepcopy(
        load_workloads()["tensor_networks.mpo.su2.float64.d40x5x3"]
    )
    workload["data"] = "uniform_v1"
    workload["workload_hash"] = workload_hash(workload)

    with pytest.raises(
        ProtocolError,
        match="data must be fusion_tree_v1 exactly for SU2",
    ):
        build_request(
            run_id="test",
            round_index=0,
            warmup=0,
            repeat=1,
            workloads=(workload,),
        )


def test_tensor0_adapter_executes_protocol_smoke():
    workload = load_workloads()[
        "tensor_networks.mpo.trivial.float64.d10x4x3"
    ]
    request = build_request(
        run_id="tensor0-smoke",
        round_index=0,
        warmup=0,
        repeat=1,
        workloads=(workload,),
    )

    response = tensor0_runner.execute(request)
    validate_backend_response(
        response,
        expected_backend="tensor0",
        request=request,
    )

    result = response["results"][0]
    assert result["status"] == "ok"
    assert result["output"]["real"] > 0
    assert result["output"]["imag"] == 0

    invalid = copy.deepcopy(response)
    invalid["results"][0]["samples_ms"] = [0.0]
    with pytest.raises(ProtocolError, match="finite positive samples"):
        validate_backend_response(
            invalid,
            expected_backend="tensor0",
            request=request,
        )


def test_tensor0_adapter_reuses_last_timed_result_for_validation(monkeypatch):
    calls: list[int] = []

    def operation():
        calls.append(len(calls) + 1)
        return calls[-1]

    monkeypatch.setattr(
        tensor0_runner,
        "_prepared_operation",
        lambda _workload: (operation, ()),
    )
    monkeypatch.setattr(tensor0_runner, "_synchronize", lambda _value: None)

    result = tensor0_runner._measure(
        {"id": "test", "workload_hash": "0" * 64},
        warmup=2,
        repeat=3,
    )

    assert calls == [1, 2, 3, 4, 5, 6]
    assert result["output"] == {"real": 6.0, "imag": 0.0}


def test_tensor0_su2_adapter_uses_nonuniform_fusion_tree_data():
    workload = load_workloads()[
        "tensor_networks.mpo.su2.float64.d40x5x3"
    ]

    with tensor0_runner.jax.enable_x64(True):
        _operation, arguments = tensor0_runner._prepared_operation(workload)
        storage_values = [
            np.asarray(tensor.storage.data)
            for tensor in arguments
        ]

    assert all(np.unique(values).size > 1 for values in storage_values)


def test_orchestrator_alternates_rounds_and_aggregates_pairwise(monkeypatch):
    workloads = load_workloads()
    selected = (
        workloads["tensor_networks.mpo.trivial.float64.d10x4x3"],
        workloads["tensor_networks.mpo.z2.float64.d10x4x4"],
    )
    calls: list[tuple[int, str, tuple[str, ...]]] = []

    def invoke(name, request, *, threads):
        assert threads == 1
        calls.append(
            (
                request["round"],
                name,
                tuple(workload["id"] for workload in request["workloads"]),
            )
        )
        sample = 1.0 if name == "tensor0" else 2.0
        return (
            {
                "schema_version": 1,
                "message_type": "benchmark_response",
                "run_id": request["run_id"],
                "round": request["round"],
                "backend": {
                    "name": name,
                    "version": "test",
                    "runtime": "test",
                },
                "results": [
                    {
                        "workload_id": workload["id"],
                        "workload_hash": workload["workload_hash"],
                        "status": "ok",
                        "samples_ms": [sample],
                        "output": {"real": 3.0, "imag": 0.0},
                    }
                    for workload in request["workloads"]
                ],
            },
            None,
        )

    monkeypatch.setattr(_orchestrator, "_invoke_backend", invoke)
    payload = _orchestrator.run_comparison(
        profile="test",
        workloads=selected,
        backends=("tensor0", "tensorkit"),
        baseline="tensorkit",
        warmup=0,
        repeat=1,
        rounds=2,
    )

    assert [call[:2] for call in calls] == [
        (0, "tensor0"),
        (0, "tensorkit"),
        (1, "tensorkit"),
        (1, "tensor0"),
    ]
    assert calls[0][2] == tuple(workload["id"] for workload in selected)
    assert calls[2][2] == tuple(
        workload["id"] for workload in reversed(selected)
    )
    assert all(record["completed_rounds"] == 2 for record in payload["records"])
    assert all(
        record["validation"]["status"] == "passed"
        for record in payload["records"]
    )
    assert all(
        record["validation"]["absolute_error"] == 0
        and record["validation"]["relative_error"] == 0
        for record in payload["records"]
    )
    assert all(
        comparison["contender_speedup"] == 2.0
        for comparison in payload["pairwise"]
    )
    assert payload["config"]["threads"] == 1


def test_cross_backend_thread_budget_configures_each_runtime():
    single = _orchestrator._subprocess_environment(1)
    threaded = _orchestrator._subprocess_environment(8)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "JULIA_NUM_THREADS",
        "CROSS_BACKEND_THREADS",
    ):
        assert single[name] == "1"
        assert threaded[name] == "8"
    assert "--xla_cpu_multi_thread_eigen=false" in single["XLA_FLAGS"]
    assert "--xla_cpu_multi_thread_eigen=true" in threaded["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=8" in threaded["XLA_FLAGS"]
    assert "inter_op_parallelism_threads=1" in threaded["XLA_FLAGS"]


def test_cross_backend_module_entrypoint_lists_workloads():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.cross_backend",
            "--list-workloads",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    workload_ids = completed.stdout.splitlines()
    assert len(workload_ids) == 69
    assert "tensor_networks.mera.su2.float64.d4" in workload_ids


@pytest.mark.parametrize("suffix", [".svg", ".png"])
def test_cross_backend_plot_renders_comparison_and_missing_results(
    tmp_path,
    suffix,
):
    pytest.importorskip("matplotlib")
    from benchmarks.cross_backend._plot import render_plot

    output = tmp_path / f"comparison{suffix}"
    assert render_plot(_plot_payload(), output) == output

    content = output.read_bytes()
    if suffix == ".svg":
        assert b"<svg" in content[:1000]
        assert b"Cross-backend steady-state latency" in content
        assert b"Tensor0" in content
        assert b"TensorKit" in content
        assert b"not compared" in content
        assert b"structural only" in content
    else:
        assert content.startswith(b"\x89PNG\r\n\x1a\n")


def test_cross_backend_plot_rejects_unknown_output_format(tmp_path):
    from benchmarks.cross_backend._plot import render_plot

    with pytest.raises(ValueError, match="plot output must use"):
        render_plot(_plot_payload(), tmp_path / "comparison.txt")


def test_cross_backend_plot_explains_optional_dependency(monkeypatch, tmp_path):
    from benchmarks.cross_backend import __main__ as cli

    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(SystemExit, match="--group bench"):
        cli._write_plot(_plot_payload(), tmp_path / "comparison.svg")


def test_cross_backend_module_entrypoint_writes_requested_plot(
    monkeypatch,
    tmp_path,
):
    from benchmarks.cross_backend import __main__ as cli

    payload = {
        "config": {},
        "environment": {},
        "backend_metadata": {},
        "records": [
            {
                "status": "ok",
                "validation": {"status": "passed"},
            }
        ],
        "pairwise": [],
        "failures": [],
    }
    output = tmp_path / "comparison.svg"
    written: list[tuple[object, Path]] = []
    monkeypatch.setattr(cli, "run_comparison", lambda **_kwargs: payload)
    monkeypatch.setattr(cli, "render_markdown", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        cli,
        "_write_plot",
        lambda current_payload, path: written.append((current_payload, path)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["cross-backend", "--plot", str(output)],
    )

    cli.main()

    assert written == [(payload, output)]


@pytest.mark.parametrize(
    ("status", "validation_status"),
    (("partial", "passed"), ("ok", "failed")),
)
def test_cross_backend_module_entrypoint_rejects_incomplete_results(
    monkeypatch,
    status,
    validation_status,
):
    from benchmarks.cross_backend import __main__ as cli

    payload = {
        "config": {},
        "environment": {},
        "backend_metadata": {},
        "records": [
            {
                "status": status,
                "validation": {"status": validation_status},
            }
        ],
        "pairwise": [],
        "failures": [],
    }
    monkeypatch.setattr(cli, "run_comparison", lambda **_kwargs: payload)
    monkeypatch.setattr(cli, "render_markdown", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(sys, "argv", ["cross-backend", "--json"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1


def test_cross_backend_protocol_and_tensorkit_environment_are_reproducible():
    schema = json.loads(
        (CROSS_BACKEND_ROOT / "protocol.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = (
        CROSS_BACKEND_ROOT / "backends" / "tensorkit" / "Manifest.toml"
    ).read_text(encoding="utf-8")

    assert schema["$defs"]["measurement"]["properties"]["kind"] == {
        "const": "steady_state"
    }
    assert schema["$defs"]["workload"]["properties"]["data"] == {
        "enum": ["uniform_v1", "fusion_tree_v1"]
    }
    sample_schema = schema["$defs"]["result"]["properties"]["samples_ms"]["items"]
    assert sample_schema["exclusiveMinimum"] == 0
    assert "[[deps.TensorKit]]" in manifest
    assert 'version = "0.16.5"' in manifest
    assert "local/reference" not in manifest


def test_tensorkit_adapter_uses_fixed_compiled_contraction_kernels():
    source = (
        CROSS_BACKEND_ROOT / "backends" / "tensorkit" / "runner.jl"
    ).read_text(encoding="utf-8")

    assert "using TensorOperations: @tensor" in source
    assert "ncon(" not in source
    assert source.count("@tensor order=") == 3
    assert "validated_topology_name(topology)" in source


def test_cross_backend_does_not_import_standard_suite():
    sources = [
        path.read_text(encoding="utf-8")
        for path in CROSS_BACKEND_ROOT.rglob("*.py")
    ]

    assert all("benchmarks.standard" not in source for source in sources)
    assert all("standard._" not in source for source in sources)
