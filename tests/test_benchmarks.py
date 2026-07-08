from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "v0_benchmarks.py"

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


def _run_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_benchmark_script_lists_expected_scenarios():
    completed = _run_benchmark("--list-scenarios")

    scenario_ids = set(completed.stdout.splitlines())

    assert EXPECTED_QUICK_SCENARIOS <= scenario_ids
    assert EXPECTED_FULL_ONLY_SCENARIOS <= scenario_ids
    assert EXPECTED_EXPLICIT_ONLY_SCENARIOS <= scenario_ids


def test_benchmark_script_quick_json_selected_scenarios_run():
    completed = _run_benchmark(
        "--quick",
        "--scenario",
        "layout.u1_two_factor.cold",
        "--scenario",
        "composition.u1.eager",
        "--json",
    )

    payload = json.loads(completed.stdout)

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


def test_benchmark_script_runs_explicit_only_scenario():
    completed = _run_benchmark(
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


def test_benchmark_script_writes_markdown_report(tmp_path):
    output = tmp_path / "baseline.md"

    _run_benchmark(
        "--quick",
        "--scenario",
        "composition.u1.eager",
        "--markdown",
        str(output),
    )

    content = output.read_text(encoding="utf-8")

    assert "## Measurement Scope" in content
    assert "## TensorKit Reference Alignment" in content
    assert "## Results" in content
    assert "## Highest Median Scenarios" in content
    assert "## Recommendation" in content
    assert "## Deferred TensorKit Parity" in content
    assert "`composition.u1.eager`" in content
    assert "- Times ms: `" in content


def test_benchmark_script_keeps_dependency_surface_small():
    tree = ast.parse(BENCHMARK_SCRIPT.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "json",
        "pathlib",
        "platform",
        "shlex",
        "statistics",
        "time",
        "typing",
        "jax",
        "tensor0",
    }
