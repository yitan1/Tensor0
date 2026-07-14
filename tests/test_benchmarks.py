from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "core.py"
CONTRACTION_BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "contractions.py"
RUNNER_MODULE = REPO_ROOT / "benchmarks" / "_runner.py"

EXPECTED_QUICK_SCENARIOS = {
    "layout.u1_two_factor.cold",
    "layout.su2_four_half.cold",
    "composition.u1.eager",
    "svd.u1_compact.eager",
    "transform.u1_permute.cold",
    "transform.u1_repartition.cold",
    "transform.su2_permute.cold",
    "jax.composition.jit_compile_and_run",
    "jax.composition.jit_cached_run",
    "jax.composition.value_and_grad_cached",
}

EXPECTED_FULL_ONLY_SCENARIOS = {
    "layout.u1_two_factor.cached",
    "layout.su2_four_half.cached",
    "composition.u1.eager.float64.medium",
    "composition.u1.eager.float64.large",
    "composition.u1.eager.complex128.small",
    "composition.u1.eager.complex128.medium",
    "composition.fermion_parity.eager.float64.small",
    "svd.u1_compact.eager.float64.medium",
    "svd.u1_compact.eager.complex128.small",
    "svd.u1_compact.eager.complex128.medium",
    "svd.fermion_parity_compact.eager.float64.small",
    "transform.u1_permute.cached",
    "transform.u1_repartition.cached",
    "transform.su2_permute.cached",
    "transform.u1_permute.large.cold",
    "transform.u1_permute.large.cached",
}

EXPECTED_EXPLICIT_ONLY_SCENARIOS = {
    "internal.strided_indices.rank2_noncontiguous",
    "internal.strided_indices.rank3_noncontiguous",
}

RESULT_FIELDS = {
    "id",
    "group",
    "description",
    "scenario_profile",
    "dtype",
    "size_label",
    "execution",
    "cache_policy",
    "warmup",
    "repeat",
    "min_ms",
    "median_ms",
    "iqr_ms",
    "max_ms",
    "times_ms",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    return imported_roots


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "tensor0_benchmark_runner_test",
        RUNNER_MODULE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_core_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CORE_BENCHMARK_SCRIPT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _run_contraction_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CONTRACTION_BENCHMARK_SCRIPT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_core_benchmark_lists_exact_historical_scenarios():
    completed = _run_core_benchmark("--list-scenarios")

    scenario_ids = completed.stdout.splitlines()

    assert len(scenario_ids) == len(set(scenario_ids))
    assert set(scenario_ids) == (
        EXPECTED_QUICK_SCENARIOS
        | EXPECTED_FULL_ONLY_SCENARIOS
        | EXPECTED_EXPLICIT_ONLY_SCENARIOS
    )


def test_core_benchmark_quick_json_selected_scenarios_run(tmp_path):
    json_output = tmp_path / "core.json"
    completed = _run_core_benchmark(
        "--quick",
        "--scenario",
        "layout.u1_two_factor.cold",
        "--scenario",
        "composition.u1.eager",
        "--json",
        "--json-output",
        str(json_output),
    )

    payload = json.loads(completed.stdout)
    file_payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert file_payload == payload
    assert payload["config"]["profile"] == "quick"
    assert payload["config"]["warmup"] == 1
    assert payload["config"]["repeat"] == 2
    assert payload["config"]["scenario_count"] == 2
    assert {result["id"] for result in payload["results"]} == {
        "layout.u1_two_factor.cold",
        "composition.u1.eager",
    }
    for result in payload["results"]:
        assert RESULT_FIELDS <= result.keys()
        assert result["warmup"] == 1
        assert result["repeat"] == 2
        assert len(result["times_ms"]) == 2
        assert result["min_ms"] >= 0.0
        assert result["median_ms"] >= 0.0
        assert result["iqr_ms"] >= 0.0
        assert result["max_ms"] >= 0.0


def test_core_benchmark_runs_explicit_only_scenario():
    completed = _run_core_benchmark(
        "--scenario",
        "internal.strided_indices.rank2_noncontiguous",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--json",
    )

    payload = json.loads(completed.stdout)

    assert payload["config"]["profile"] == "full"
    assert payload["config"]["warmup"] == 0
    assert payload["config"]["repeat"] == 1
    assert payload["config"]["scenario_count"] == 1
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert RESULT_FIELDS <= result.keys()
    assert result["id"] == "internal.strided_indices.rank2_noncontiguous"
    assert result["scenario_profile"] == "explicit-only"
    assert result["group"] == "internal"
    assert result["dtype"] == "int64"
    assert result["size_label"] == "large"
    assert result["execution"] == "eager"
    assert result["cache_policy"] == "cache_miss_builder"
    assert len(result["times_ms"]) == 1


def test_core_benchmark_writes_historical_markdown_sections(tmp_path):
    output = tmp_path / "baseline.md"

    _run_core_benchmark(
        "--quick",
        "--scenario",
        "composition.u1.eager",
        "--markdown",
        str(output),
    )

    content = output.read_text(encoding="utf-8")

    assert content.startswith("# Tensor0 Core Benchmark Baseline\n")
    assert "`uv run python benchmarks/core.py" in content
    assert "## Measurement Scope" in content
    assert "## TensorKit Reference Alignment" in content
    assert "## Results" in content
    assert "## Highest Median Scenarios" in content
    assert "## Recommendation" in content
    assert "## Deferred TensorKit Parity" in content
    assert "`composition.u1.eager`" in content
    assert "- Times ms: `" in content


def test_core_benchmark_keeps_dependency_surface_small():
    imported_roots = _imported_roots(CORE_BENCHMARK_SCRIPT)

    assert imported_roots <= {
        "_runner",
        "__future__",
        "platform",
        "typing",
        "jax",
        "tensor0",
    }
    assert {"_runner", "jax", "tensor0"} <= imported_roots


def test_common_benchmark_runner_keeps_dependency_surface_small():
    imported_roots = _imported_roots(RUNNER_MODULE)

    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "json",
        "pathlib",
        "shlex",
        "statistics",
        "time",
        "typing",
    }
    assert {"dataclasses", "statistics", "time"} <= imported_roots


def test_runner_runs_per_iteration_setup_outside_timed_region(monkeypatch):
    runner = _load_runner()
    events: list[str] = []
    ticks = iter((1_000_000, 2_000_000))

    def operation() -> None:
        events.append("operation")

    def factory():
        events.append("factory")
        return operation

    def before_each() -> None:
        events.append("setup")

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(runner.time, "perf_counter_ns", clock)
    item = runner.scenario(
        "setup-boundary",
        "test",
        "setup boundary",
        "quick",
        "none",
        "tiny",
        "eager",
        "cold",
        factory,
        before_each=before_each,
    )

    result = runner.measure_scenario(item, warmup=1, repeat=1)

    assert events == [
        "factory",
        "setup",
        "operation",
        "setup",
        "clock",
        "operation",
        "clock",
    ]
    assert result.times_ms == [1.0]


def test_runner_rejects_invalid_scenario_registries():
    runner = _load_runner()

    item = runner.scenario(
        "duplicate",
        "test",
        "duplicate id",
        "quick",
        "none",
        "tiny",
        "eager",
        "none",
        lambda: lambda: None,
    )
    with pytest.raises(ValueError, match="duplicate benchmark scenario ids"):
        runner.validate_scenarios((item, item))

    invalid_profile = runner.scenario(
        "invalid-profile",
        "test",
        "invalid profile",
        "typo",
        "none",
        "tiny",
        "eager",
        "none",
        lambda: lambda: None,
    )
    with pytest.raises(ValueError, match="invalid benchmark scenario profiles"):
        runner.validate_scenarios((invalid_profile,))


def test_contraction_benchmark_lists_scenarios():
    completed = _run_contraction_benchmark("--list-scenarios")

    scenario_ids = completed.stdout.splitlines()

    assert len(scenario_ids) == len(set(scenario_ids))
    assert set(scenario_ids) == {
        "twist.u1.identity",
        "twist.fermion.nontrivial",
        "trace.u1.partial",
        "trace.su2.partial",
        "trace.fermion.full",
        "contract.u1.partial",
        "contract.su2.fusion_basis",
        "contract.fermion.twist",
        "contract.u1.partial.complex",
        "network.named.default",
        "network.named.default.medium",
        "network.named.custom.medium",
        "network.ncon.default",
        "network.ncon.default.medium",
        "network.ncon.custom.medium",
        "network.disconnected",
        "jax.network.compile_and_run",
        "jax.network.cached",
        "jax.network.value_and_grad",
    }


def test_contraction_benchmark_selected_quick_json_runs():
    completed = _run_contraction_benchmark(
        "--quick",
        "--scenario",
        "twist.u1.identity",
        "--scenario",
        "network.named.default",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--json",
    )

    payload = json.loads(completed.stdout)

    assert payload["config"] == {
        "profile": "quick",
        "warmup": 0,
        "repeat": 1,
        "scenario_count": 2,
    }
    assert {result["id"] for result in payload["results"]} == {
        "twist.u1.identity",
        "network.named.default",
    }
    assert "public_repository" in payload["environment"]
    assert "private_repository" not in payload["environment"]
    for result in payload["results"]:
        assert RESULT_FIELDS <= result.keys()
        assert len(result["times_ms"]) == 1


def test_contraction_benchmark_custom_network_and_markdown_run(tmp_path):
    output = tmp_path / "contraction-baseline.md"
    json_output = tmp_path / "contraction-baseline.json"

    completed = _run_contraction_benchmark(
        "--scenario",
        "network.ncon.custom.medium",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--json",
        "--json-output",
        str(json_output),
        "--markdown",
        str(output),
    )
    payload = json.loads(completed.stdout)
    file_payload = json.loads(json_output.read_text(encoding="utf-8"))
    content = output.read_text(encoding="utf-8")

    assert payload["results"][0]["id"] == "network.ncon.custom.medium"
    assert file_payload == payload
    assert "## Environment" in content
    assert "## Results" in content
    assert "## Interpretation Boundary" in content
    assert "`network.ncon.custom.medium`" in content
    assert "`uv run python benchmarks/contractions.py" in content


def test_contraction_benchmark_keeps_dependency_surface_small():
    imported_roots = _imported_roots(CONTRACTION_BENCHMARK_SCRIPT)

    assert imported_roots <= {
        "_runner",
        "__future__",
        "json",
        "os",
        "pathlib",
        "platform",
        "subprocess",
        "typing",
        "jax",
        "tensor0",
    }
    assert {"_runner", "jax", "tensor0"} <= imported_roots
