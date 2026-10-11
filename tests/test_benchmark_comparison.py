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


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_tensor0_adapter_native_thread_budget(monkeypatch, threads, backend):
    request = build_request(
        run_id="thread-budget",
        round_index=0,
        warmup=0,
        repeat=1,
        workloads=(next(iter(load_workloads().values())),),
    )
    monkeypatch.setenv("CROSS_BACKEND_THREADS", str(threads))
    monkeypatch.setattr(tensor0_runner.jax, "default_backend", lambda: backend)
    limits = []
    monkeypatch.setattr(tensor0_runner, "set_num_threads", limits.append)
    monkeypatch.setattr(
        tensor0_runner, "get_num_threads", lambda: limits[-1]
    )

    def measure(workload, **kwargs):
        assert limits == ([threads] if backend == "cpu" else [])
        return {"status": "ok"}

    monkeypatch.setattr(tensor0_runner, "_measure", measure)
    response = tensor0_runner.execute(request)
    assert response["backend"]["native_worker_limit"] == (
        threads if backend == "cpu" else None
    )
    assert (
        response["backend"]["thread_settings"]["CROSS_BACKEND_THREADS"]
        == str(threads)
    )


def test_tensor0_adapter_executes_protocol_smoke(monkeypatch):
    monkeypatch.setenv("CROSS_BACKEND_THREADS", "1")
    previous_limit = tensor0_runner.get_num_threads()
    from tensor0._stride import enable_threads

    def restore_limit():
        if previous_limit is None:
            enable_threads()
        else:
            tensor0_runner.set_num_threads(previous_limit)

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

    try:
        response = tensor0_runner.execute(request)
    finally:
        restore_limit()
    assert response["backend"]["native_worker_limit"] == 1
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


def test_tensorkit_command_disables_user_startup(monkeypatch):
    monkeypatch.setenv("CROSS_BACKEND_JULIA", "julia")
    command = _orchestrator._backend_command("tensorkit")
    assert command[0] == "julia"
    assert "--startup-file=no" in command


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
        assert b"Median" in content
        assert b"Arithmetic mean (all timings)" in content
        assert b"1.5" in content
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


def test_sampling_plan_protocol_and_backend_counts(monkeypatch):
    workloads = list(load_workloads().values())[:2]
    plan = {w['id']: i + 2 for i, w in enumerate(workloads)}
    request = build_request(run_id='plan', round_index=0, warmup=2,
                            repeat=7, workloads=workloads,
                            repeat_by_workload=plan)
    assert validate_request(request) is request
    observed = []
    monkeypatch.setattr(tensor0_runner.jax, 'default_backend', lambda: 'cuda')

    def measure(w, *, warmup, repeat):
        observed.append((w['id'], warmup, repeat))
        return {'workload_id': w['id'], 'workload_hash': w['workload_hash'],
                'status': 'ok', 'samples_ms': [1.] * repeat,
                'output': {'real': 3., 'imag': 0.}}

    monkeypatch.setattr(tensor0_runner, '_measure', measure)
    response = tensor0_runner.execute(request)
    validate_backend_response(response, expected_backend='tensor0', request=request)
    assert observed == [(w['id'], 2, plan[w['id']]) for w in workloads]
    invalid = copy.deepcopy(response)
    invalid['results'][0]['samples_ms'].append(1.)
    with pytest.raises(ProtocolError, match='finite positive samples'):
        validate_backend_response(invalid, expected_backend='tensor0', request=request)
    default = build_request(run_id='default', round_index=0, warmup=2,
                            repeat=7, workloads=workloads)
    assert default['measurement'] == {'kind': 'steady_state', 'warmup': 2, 'repeat': 7}
    tensor0_runner.execute(default)
    assert observed[-2:] == [(w['id'], 2, 7) for w in workloads]


@pytest.mark.parametrize('variant', ['unknown', 'missing', 'duplicate', 'zero', 'bool'])
def test_sampling_plan_rejects_invalid_entries(tmp_path, variant):
    from benchmarks.cross_backend._protocol import load_sampling_plan
    ids = list(load_workloads())[:2]
    entries = [{'workload_id': wid, 'repeat': 3} for wid in ids]
    if variant == 'unknown': entries[0]['workload_id'] = 'unknown'
    elif variant == 'missing': entries.pop()
    elif variant == 'duplicate': entries.append(entries[0])
    elif variant == 'zero': entries[0]['repeat'] = 0
    else: entries[0]['repeat'] = True
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(entries))
    with pytest.raises(ProtocolError): load_sampling_plan(path, ids)


def test_sampling_plan_round_counts_pairing_and_reversal(monkeypatch):
    from benchmarks.cross_backend._report import render_markdown
    workloads = list(load_workloads().values())[:2]
    plan = {w['id']: i + 2 for i, w in enumerate(workloads)}
    seen = []

    def invoke(name, request, *, threads):
        seen.append((name, request['round'], request['measurement']))
        # Fail a middle round: do not accidentally pair r2 with r1.
        if name == 'tensor0' and request['round'] == 1:
            return None, {'backend': name, 'round': 1, 'kind': 'test', 'error': 'missing'}
        sample = (1. if request['round'] == 0 else 4.) if name == 'tensor0' else 2.
        return {'backend': {'name': name, 'version': 'test', 'runtime': 'test'},
                'results': [{'workload_id': w['id'], 'status': 'ok',
                             'samples_ms': [sample] * plan[w['id']],
                             'output': {'real': 1., 'imag': 0.}}
                            for w in request['workloads']]}, None

    monkeypatch.setattr(_orchestrator, '_invoke_backend', invoke)
    monkeypatch.setattr(_orchestrator, '_environment', lambda _: {})
    payload = _orchestrator.run_comparison(profile='test', workloads=workloads,
                                          backends=['tensor0', 'tensorkit'],
                                          baseline='tensorkit', warmup=2, repeat=7,
                                          rounds=3, repeat_by_workload=plan)
    assert all(x[2]['repeat_by_workload'] == plan for x in seen)
    for record in payload['records']:
        count = plan[record['workload_id']]
        assert record['repeat_per_round'] == count
        assert len(record['samples_ms']) == count * record['completed_rounds']
        assert all(r['sample_count'] == count for r in record['round_statistics'])
        if record['backend'] == 'tensor0':
            assert [r['round'] for r in record['round_statistics']] == [0, 2]
            assert record['median_ms'] == 2.5
    for pair in payload['pairwise']:
        assert [r['round'] for r in pair['paired_rounds']] == [0, 2]
        assert [r['contender_speedup'] for r in pair['paired_rounds']] == [2., .5]
        assert pair['direction_inconsistent'] is True
        assert pair['contender_speedup'] == .8
    assert 'direction inconsistent' in render_markdown(payload, command='test')
    assert json.loads(json.dumps(payload))['config']['repeat_by_workload'] == plan


def test_sampling_plan_cli_retains_file_provenance(monkeypatch, tmp_path):
    import hashlib
    from benchmarks.cross_backend import __main__ as cli
    wid = next(iter(load_workloads()))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps([{'workload_id': wid, 'repeat': 15}]))
    output = tmp_path / 'output.json'
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        return {'config': {}, 'records': [], 'failures': []}
    monkeypatch.setattr(cli, 'run_comparison', run)
    monkeypatch.setattr(cli, 'render_markdown', lambda *args, **kwargs: '')
    monkeypatch.setattr(sys, 'argv', ['cross-backend', '--workload', wid,
                                    '--sampling-plan', str(path), '--repeat', '7',
                                    '--json-output', str(output)])
    cli.main()
    assert captured['repeat'] == 7
    assert captured['repeat_by_workload'] == {wid: 15}
    assert json.loads(output.read_text())['config']['sampling_plan'] == {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_sampling_plan_rejects_duplicate_selected_workloads(monkeypatch):
    workload = next(iter(load_workloads().values()))
    with pytest.raises(ProtocolError, match='duplicates'):
        build_request(run_id='duplicate', round_index=0, warmup=0,
                      repeat=1, workloads=[workload, workload])
    with pytest.raises(ProtocolError, match='duplicates'):
        _orchestrator.run_comparison(profile='test', workloads=[workload, workload],
                                    backends=['tensor0'], baseline='tensor0',
                                    warmup=0, repeat=1, rounds=1)


def test_unfiltered_mean_retains_outliers_and_weights_samples_not_rounds():
    import math

    rounds = [[1.0], [2.0, 2.0, 1000.0]]
    samples = [sample for row in rounds for sample in row]
    original = samples.copy()
    stats = _orchestrator._statistics(samples)

    assert stats['mean_ms'] == math.fsum(samples) / len(samples) == 251.25
    assert stats['mean_ms'] != sum(
        _orchestrator._statistics(row)['mean_ms'] for row in rounds
    ) / len(rounds)
    assert stats['median_ms'] == 2.0
    assert stats['max_ms'] == 1000.0
    assert samples == original  # Neither filtering nor sorting the raw data.
    assert _orchestrator._statistics([1000.0])['mean_ms'] == 1000.0


def test_orchestrator_and_report_include_unfiltered_mean_ratios(monkeypatch):
    import math
    from benchmarks.cross_backend._report import render_markdown

    workload = next(iter(load_workloads().values()))

    def invoke(name, request, *, threads):
        samples = ([1.0, 1.0, 1000.0] if request['round'] == 0
                   else [1.0, 1.0, 1.0]) if name == 'tensor0' else [2.0] * 3
        return {
            'backend': {'name': name, 'version': 'test', 'runtime': 'test'},
            'results': [{'workload_id': workload['id'], 'status': 'ok',
                         'samples_ms': samples,
                         'output': {'real': 1.0, 'imag': 0.0}}],
        }, None

    monkeypatch.setattr(_orchestrator, '_invoke_backend', invoke)
    monkeypatch.setattr(_orchestrator, '_environment', lambda _: {})
    payload = _orchestrator.run_comparison(
        profile='test', workloads=[workload],
        backends=['tensor0', 'tensorkit'], baseline='tensorkit',
        warmup=0, repeat=3, rounds=2,
    )
    for record in payload['records']:
        assert record['mean_ms'] == math.fsum(record['samples_ms']) / 6
        for row in record['round_statistics']:
            start = row['round'] * 3
            assert row['mean_ms'] == math.fsum(record['samples_ms'][start:start + 3]) / 3
    tensor0 = next(r for r in payload['records'] if r['backend'] == 'tensor0')
    assert tensor0['samples_ms'] == [1.0, 1.0, 1000.0, 1.0, 1.0, 1.0]
    pair = payload['pairwise'][0]
    assert pair['contender_speedup'] == 2.0  # Original median direction unchanged.
    assert pair['baseline_mean_ms'] == 2.0
    assert pair['contender_mean_ms'] == 167.5
    assert pair['contender_mean_ratio'] == 2.0 / 167.5 < 1
    report = render_markdown(payload, command='test')
    assert 'Contender median / mean ms | Baseline median / mean ms' in report
    assert 'Record and sample details' in report
    assert 'Median ratio | Mean ratio' in report
    assert '167.500000' in report
    assert '0.012×' in report
    assert 'unfiltered arithmetic mean latency' in report
    assert 'not independent throughput measurements' in report

    # Historical aggregate payloads can still render without derived fields.
    for record in payload['records']:
        record.pop('mean_ms')
    pair.pop('contender_mean_ratio')
    assert '0.012×' in render_markdown(payload, command='test')


def test_latency_plots_preserve_all_formal_samples_and_means(tmp_path, monkeypatch):
    import json
    import math
    import xml.etree.ElementTree as ET
    from pathlib import Path
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from benchmarks.cross_backend._plot import render_plot

    means, boxes, axes = [], [], []
    scatter, boxplot, set_yscale = Axes.scatter, Axes.boxplot, Axes.set_yscale

    def capture_mean(self, x, y, **kwargs):
        assert kwargs["marker"] == "D"
        means.extend(y)
        return scatter(self, x, y, **kwargs)

    def capture_box(self, values, **kwargs):
        assert kwargs["showfliers"] is True
        boxes.extend(values)
        return boxplot(self, values, **kwargs)

    def capture_scale(self, value, **kwargs):
        assert value == "log"
        axes.append(self)
        return set_yscale(self, value, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_mean)
    monkeypatch.setattr(Axes, "boxplot", capture_box)
    monkeypatch.setattr(Axes, "set_yscale", capture_scale)
    root = Path("benchmarks/cross_backend/results")
    for n in (1, 4, 8):
        payload = json.loads((root / f"cpu-{n}t/results.json").read_text())
        means.clear(); boxes.clear(); axes.clear()
        output = render_plot(payload, tmp_path / f"latency-{n}.svg")
        ET.parse(output)
        from collections import Counter
        assert Counter(w.split(".")[1] for w in payload["config"]["workloads"]) == {
            "mpo": 25, "pepo": 25, "mera": 19,
        }
        assert len(payload["config"]["workloads"]) == 69
        assert len(means) == len(boxes) == 138
        assert len(axes) == 12
        expected = {tuple(r["samples_ms"]): r for r in payload["records"]}
        for samples, mean in zip(boxes, means):
            record = expected[tuple(samples)]
            assert mean == record["mean_ms"]
            assert mean == math.fsum(samples) / len(samples)
        for axis in axes:
            lower, upper = axis.get_ylim()
            assert all(lower <= y <= upper for collection in axis.collections
                       for _, y in collection.get_offsets())
            assert all(lower <= y <= upper for line in axis.lines
                       for y in line.get_ydata())


@pytest.mark.parametrize("stored_mean", [False, True])
def test_latency_plot_mean_includes_outlier(tmp_path, monkeypatch, stored_mean):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from benchmarks.cross_backend._plot import render_plot
    payload = _plot_payload()
    payload["records"][0]["samples_ms"] = [1., 1., 100.]
    if stored_mean:
        payload["records"][0]["mean_ms"] = 34.
    means = []
    scatter = Axes.scatter

    def capture(self, x, y, **kwargs):
        means.extend(y)
        return scatter(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture)
    render_plot(payload, tmp_path / "means.svg")
    assert means[0] == 34.  # Not median=1 or an outlier-filtered mean.


@pytest.mark.parametrize("failure_source", ["pairwise", "record"])
def test_latency_plot_omits_failed_and_missing_backend_distributions(
    tmp_path, monkeypatch, failure_source,
):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from benchmarks.cross_backend._plot import render_plot
    payload = _plot_payload()
    payload["records"] = payload["records"][:2]
    if failure_source == "pairwise":
        payload["pairwise"][0]["validation"] = "failed"
    else:
        for record in payload["records"]:
            record["validation"]["status"] = "failed"
    means = []
    scatter = Axes.scatter

    def capture(self, x, y, **kwargs):
        means.extend(y)
        return scatter(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture)
    output = render_plot(payload, tmp_path / "failed.svg")
    assert means == []
    content = output.read_text()
    assert "validation failed" in content
    assert "unavailable" in content


def test_published_report_layout_and_combined_tables():
    from pathlib import Path
    root = Path('benchmarks/cross_backend/results')
    overview = (root / 'report.md').read_text()
    assert 'Geomean median TK/T0 | Geomean mean TK/T0' in overview
    assert 'arithmetic mean of all recorded timings' in overview
    assert 'ratio.svg' not in overview
    assert 'timeseries.svg' not in overview
    assert 'Same-version scaling' in overview
    for n in (1, 4, 8):
        text = (root / f'cpu-{n}t/report.md').read_text()
        assert text.count('| `tensor_networks.') == 69
        assert all(f'### {topology}' in text for topology in ('MPO', 'PEPO', 'MERA'))
        assert 'T0 median / mean ms | TK median / mean ms' in text
        assert 'Preparation failures occurred before formal sampling' in text
        assert 'no formal sample was omitted or replaced' in text


def test_single_backend_report_retains_latency_summary():
    from benchmarks.cross_backend._report import render_markdown
    payload = {'config': {'profile': 'test', 'measurement': 'steady_state',
                         'backends': ['tensor0'], 'baseline': 'tensor0',
                         'threads': 1, 'warmup': 0, 'repeat': 2, 'rounds': 1},
               'records': [{'workload_id': 'tensor_networks.mpo.z2.float64.d10x4x4',
                            'backend': 'tensor0', 'status': 'ok', 'median_ms': 2.,
                            'iqr_ms': 1., 'samples_ms': [1., 3.],
                            'validation': {'status': 'not_run'}}],
               'pairwise': [], 'failures': [], 'environment': {}, 'backend_metadata': {}}
    report = render_markdown(payload, command='test')
    assert '2.000000 / 2.000000' in report
    assert '| `ok` | 2 | 2 | 1.000000 | `not_run` |' in report


def test_mixed_paired_and_unpaired_report_retains_latencies_and_failures():
    from benchmarks.cross_backend._report import render_markdown
    payload = _plot_payload()
    payload['config'].update(measurement='steady_state', threads=1,
                             warmup=0, repeat=2, rounds=1)
    payload.update(environment={}, backend_metadata={})
    paired_workload = payload['pairwise'][0]['workload_id']
    payload['pairwise'] = payload['pairwise'][:1]
    paired_records = [r for r in payload['records']
                      if r['workload_id'] == paired_workload]
    unpaired = next(r.copy() for r in payload['records']
                    if r['workload_id'] != paired_workload)
    unpaired.update(status='ok', median_ms=4., samples_ms=[2., 6.],
                    validation={'status': 'not_run'})
    unpaired.pop('mean_ms', None)
    failed = {'workload_id': unpaired['workload_id'],
              'backend': next(b for b in payload['config']['backends']
                              if b != unpaired['backend']),
              'status': 'error', 'validation': {'status': 'not_run'}}
    payload['records'] = paired_records + [unpaired, failed]
    payload['failures'] = [{'backend': failed['backend'], 'round': 0,
                            'kind': 'backend_error', 'error': 'test failure'}]
    report = render_markdown(payload, command='test')
    assert f"| `{paired_workload}` |" in report
    assert f"| `{unpaired['workload_id']}` | `{unpaired['backend']}` | 4.000000 / 4.000000 |" in report
    assert f"| `{failed['workload_id']}` | `{failed['backend']}` | — |" in report
    assert '| `error` |' in report
    assert 'test failure' in report
    assert 'mean_ms' not in failed

    # An absent backend record must not hide the successful backend either.
    payload['records'].remove(failed)
    report = render_markdown(payload, command='test')
    assert f"| `{unpaired['workload_id']}` | `{unpaired['backend']}` | 4.000000 / 4.000000 |" in report
    assert 'test failure' in report
