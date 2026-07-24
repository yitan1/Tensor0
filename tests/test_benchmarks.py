from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_ROOT = REPO_ROOT / "benchmarks" / "standard"
RUNNER_MODULE = STANDARD_ROOT / "_runner.py"
PARAMETER_FILES = (
    STANDARD_ROOT / "linalg" / "params.toml",
    STANDARD_ROOT / "transforms" / "params.toml",
    STANDARD_ROOT
    / "contractions"
    / "api_networks"
    / "params.toml",
    STANDARD_ROOT
    / "contractions"
    / "tensor_networks"
    / "params.toml",
)

EXPECTED_TRIVIAL_PERMUTE_SCENARIOS = {
    "permute.trivial.rank4.float64.small.eager",
    "permute.trivial.rank4.float64.small.jit_compile_and_run",
    "permute.trivial.rank4.float64.small.jit_cached_run",
    "permute.trivial.rank4.complex128.medium.eager",
}
EXPECTED_TRIVIAL_PRIMITIVE_SCENARIOS = {
    "trace.trivial.partial.rank4.float64.small.eager",
    "trace.trivial.partial.rank4.float64.small.jit_compile_and_run",
    "trace.trivial.partial.rank4.float64.small.jit_cached_run",
    "trace.trivial.partial.rank4.complex128.medium.eager",
    "contract.trivial.partial.float64.small.eager",
    "contract.trivial.partial.float64.small.jit_compile_and_run",
    "contract.trivial.partial.float64.small.jit_cached_run",
    "contract.trivial.partial.complex128.medium.eager",
}
EXPECTED_TRIVIAL_COMPOSITION_SCENARIOS = {
    "composition.trivial.float64.small.eager",
    "composition.trivial.float64.small.jit_compile_and_run",
    "composition.trivial.float64.small.jit_cached_run",
    "composition.trivial.complex128.medium.eager",
}
TRANSFORM_SMOKE_SCENARIO = (
    "transforms.permute.trivial.float64.d64x48.p2x1_to_empty.eager"
)
EXPECTED_QUICK_SUITE_SCENARIOS = {
    "linalg.mul.trivial.float64.d2x2x2.eager",
    "linalg.svd.trivial.float64.d2x2.eager",
    "linalg.mul.u1.float64.small.eager",
    "linalg.svd.u1.float64.small.eager",
    "linalg.mul.u1.float64.small.jit_compile_and_run",
    "linalg.mul.u1.float64.small.jit_cached_run",
    "linalg.mul.u1.float64.small.value_and_grad_cached",
    TRANSFORM_SMOKE_SCENARIO,
    "transforms.permute.u1.float64.small.cold",
    "transforms.repartition.u1.float64.small.cold",
    "transforms.permute.su2.float64.small.cold",
    "transforms.twist.u1.identity.eager",
    "transforms.twist.fermion_parity.nontrivial.eager",
    "trace.u1.partial",
    "trace.su2.partial",
    "contract.u1.partial",
    "network.named.default",
    "network.ncon.default",
    "network.disconnected",
    "jax.network.compile_and_run",
    "jax.network.cached",
    "jax.network.value_and_grad",
    "tensor_networks.mpo.trivial.float64.d10x4x3.eager",
    "tensor_networks.pepo.trivial.float64.d3x2x2x50.eager",
    "tensor_networks.mera.trivial.float64.d2.eager",
    "layout.u1_two_factor.cold",
    "layout.su2_four_half.cold",
    "permute.trivial.rank4.float64.small.eager",
    "trace.trivial.partial.rank4.float64.small.eager",
    "contract.trivial.partial.float64.small.eager",
    "composition.trivial.float64.small.eager",
    "protect.su2.permute.float64.small.eager",
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


def _load_tensor_network_cases():
    return importlib.import_module(
        "benchmarks.standard.contractions.tensor_networks.cases"
    )


def _suite_registry() -> list[dict[str, str]]:
    script = (
        "import json; "
        "from benchmarks.standard import __main__ as suite; "
        "print(json.dumps(["
        "{'id': item.id, 'group': item.group, "
        "'profile': item.scenario_profile} "
        "for item in suite._scenarios()]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _run_benchmark_suite(
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.standard", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_common_benchmark_runner_keeps_dependency_surface_small():
    imported_roots = _imported_roots(RUNNER_MODULE)

    assert imported_roots <= sys.stdlib_module_names | {"__future__"}


def test_parameter_files_use_tensor0_native_schema():
    legacy_keys = {"T", "I", "dims", "profile"}
    legacy_values = {
        "Float64",
        "ComplexF64",
        "Trivial",
        "Z2Irrep",
        "U1Irrep",
        "SU2Irrep",
    }

    def check(value: object) -> None:
        if isinstance(value, dict):
            assert legacy_keys.isdisjoint(value)
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)
        elif isinstance(value, str):
            assert value not in legacy_values

    for path in PARAMETER_FILES:
        with path.open("rb") as stream:
            parameters = tomllib.load(stream)
        check(parameters)
        for workload in parameters.get("workload", []):
            assert {"sector", "dimensions"} <= workload.keys()
            assert "dtype" in workload or "dtypes" in workload


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
    item = runner.Scenario(
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


def test_runner_synchronizes_tensormap_storage_before_timer_stops(monkeypatch):
    runner = _load_runner()
    events: list[str] = []
    ticks = iter((1_000_000, 2_000_000))

    class Data:
        def block_until_ready(self) -> None:
            events.append("ready")

    class Storage:
        data = Data()

    class Tensor:
        storage = Storage()

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(runner.time, "perf_counter_ns", clock)
    item = runner.Scenario(
        "synchronization-boundary",
        "test",
        "synchronization boundary",
        "quick",
        "none",
        "tiny",
        "eager",
        "none",
        lambda: lambda: {"tensor": Tensor()},
    )

    result = runner.measure_scenario(item, warmup=0, repeat=1)

    assert events == ["clock", "ready", "clock"]
    assert result.times_ms == [1.0]


def test_runner_rejects_invalid_scenario_registries():
    runner = _load_runner()

    item = runner.Scenario(
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

    invalid_profile = runner.Scenario(
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


def test_runner_all_selection_includes_explicit_only_scenarios():
    runner = _load_runner()
    quick = runner.Scenario(
        "quick",
        "fast",
        "quick",
        "quick",
        "none",
        "tiny",
        "eager",
        "none",
        lambda: lambda: None,
    )
    explicit = runner.Scenario(
        "explicit",
        "slow",
        "explicit",
        "explicit-only",
        "none",
        "large",
        "eager",
        "none",
        lambda: lambda: None,
    )

    selected = runner.selected_scenarios(
        (quick, explicit),
        None,
        quick=False,
        include_explicit=True,
    )

    assert selected == (quick, explicit)

    selected_group = runner.selected_scenarios(
        (quick, explicit),
        None,
        quick=False,
        include_explicit=True,
        selected_groups=("slow",),
    )

    assert selected_group == (explicit,)
    with pytest.raises(SystemExit, match="unknown benchmark groups"):
        runner.selected_scenarios(
            (quick, explicit),
            None,
            quick=False,
            selected_groups=("missing",),
        )


def test_tensor_network_space_generator_matches_standard_distributions():
    cases = _load_tensor_network_cases()

    z2 = cases.generate_space("z2", 10, 0.5)
    u1 = cases.generate_space("u1", 40, 0.5)
    su2 = cases.generate_space("su2", 40, 2.0)

    assert z2.sectors == (((0,), 5), ((1,), 5))
    assert u1.sectors == (((0,), 32), ((1,), 5), ((-1,), 5))
    assert su2.sectors == (
        ((0,), 8),
        ((1,), 4),
        ((2,), 3),
        ((3,), 2),
        ((4,), 1),
        ((5,), 1),
    )


def test_benchmark_suite_lists_valid_registry():
    completed = _run_benchmark_suite("--list-scenarios")

    scenario_ids = completed.stdout.splitlines()
    scenario_set = set(scenario_ids)

    assert {
        "linalg.mul.z2.complex128.d8x8x8.eager",
        "transforms.permute.u1.float64.small.cold",
        "network.named.default",
        "tensor_networks.mpo.trivial.float64.d10x4x3.eager",
        "tensor_networks.pepo.u1.float64.d4x2x2x100.eager",
        "tensor_networks.mera.su2.float64.d4.eager",
        "layout.u1_two_factor.cold",
        "internal.strided_indices.rank2_noncontiguous",
    } <= scenario_set


def test_benchmark_suite_quick_profile_has_expected_domain_coverage():
    quick_ids = {
        item["id"]
        for item in _suite_registry()
        if item["profile"] == "quick"
    }

    assert quick_ids == EXPECTED_QUICK_SUITE_SCENARIOS


def test_direct_twist_scenarios_belong_to_transforms():
    groups = {
        item["id"]: item["group"]
        for item in _suite_registry()
        if "twist" in item["id"].split(".")
    }

    assert groups == {
        "transforms.twist.u1.identity.eager": "transforms",
        "transforms.twist.fermion_parity.nontrivial.eager": "transforms",
        "contract.fermion.twist": "contractions",
    }


def test_public_path_scenarios_belong_to_operation_domains():
    expected_groups = {
        **dict.fromkeys(EXPECTED_TRIVIAL_COMPOSITION_SCENARIOS, "linalg"),
        **dict.fromkeys(EXPECTED_TRIVIAL_PERMUTE_SCENARIOS, "transforms"),
        **dict.fromkeys(
            EXPECTED_TRIVIAL_PRIMITIVE_SCENARIOS,
            "contractions",
        ),
        "protect.su2.permute.float64.small.eager": "transforms",
    }
    groups = {
        item["id"]: item["group"]
        for item in _suite_registry()
        if item["id"] in expected_groups
    }

    assert groups == expected_groups


def test_benchmark_suite_selected_smoke_and_report(tmp_path):
    markdown_output = tmp_path / "benchmark-suite.md"
    scenario_ids = {
        "linalg.mul.trivial.float64.d2x2x2.eager",
        "linalg.mul.u1.float64.small.eager",
        "transforms.permute.u1.float64.small.cold",
        "network.named.default",
        "tensor_networks.mpo.trivial.float64.d10x4x3.eager",
        "layout.u1_two_factor.cold",
    }
    arguments = [
        "--quick",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--json",
        "--markdown",
        str(markdown_output),
    ]
    for scenario_id in sorted(scenario_ids):
        arguments.extend(("--scenario", scenario_id))
    completed = _run_benchmark_suite(
        *arguments,
    )
    payload = json.loads(completed.stdout)

    assert payload["config"] == {
        "profile": "quick",
        "warmup": 0,
        "repeat": 1,
        "scenario_count": len(scenario_ids),
    }
    assert {result["id"] for result in payload["results"]} == scenario_ids
    assert all(
        set(result) == RESULT_FIELDS for result in payload["results"]
    )
    assert {result["group"] for result in payload["results"]} == {
        "linalg",
        "transforms",
        "contractions",
        "tensor_networks",
        "diagnostics",
    }
    metadata = payload["environment"]["benchmark_suite"]
    assert {
        "linalg",
        "transforms",
        "contractions",
        "tensor_networks",
        "diagnostics",
    } <= metadata.keys()
    assert metadata["transforms"]["memory_contract"] == {
        "logical_input_mib": [32],
        "minimum_live_dense_mib": [64],
        "boundary": "input remains live while a distinct output is allocated",
    }
    assert (
        metadata["tensor_networks"]["workload_contract"]["result_boundary"]
        == "rank_zero_tensormap"
    )
    topology_definitions = metadata["tensor_networks"]["topology_definitions"]
    assert topology_definitions["mpo"]["operand_positions"] == [2, 0, 1, 0, 3]
    assert topology_definitions["pepo"]["labels"][0] == [18, 7, 4, 2, 1]
    assert len(topology_definitions["mera"]["order"]) == 23
    support_files = payload["environment"]["benchmark_support_files_sha256"]
    assert {
        "benchmarks/standard/_inputs.py",
        "benchmarks/standard/_runner.py",
        "benchmarks/standard/_specs.py",
        "benchmarks/standard/contractions/api_networks/cases.py",
        "benchmarks/standard/contractions/api_networks/params.toml",
        "benchmarks/standard/contractions/primitives/cases.py",
        "benchmarks/standard/diagnostics/layout_cache.py",
        "benchmarks/standard/diagnostics/strided_indices.py",
        "benchmarks/standard/linalg/cases.py",
        "benchmarks/standard/linalg/params.toml",
        "benchmarks/standard/contractions/tensor_networks/cases.py",
        "benchmarks/standard/contractions/tensor_networks/params.toml",
        "benchmarks/standard/transforms/cases.py",
        "benchmarks/standard/transforms/params.toml",
    } <= support_files.keys()
    content = markdown_output.read_text(encoding="utf-8")
    assert content.startswith("# Tensor0 Benchmark Suite\n")
    assert "## Measurement Boundary" in content
    assert "## Workload Boundary" in content
    assert "`uv run python -m benchmarks.standard --quick" in content


def test_benchmark_suite_nontrivial_network_and_jit_run():
    scenario_ids = {
        "tensor_networks.mpo.z2.float64.d10x4x4.eager",
        "tensor_networks.mpo.trivial.float64.d10x4x3.jit_compile_and_run",
        "tensor_networks.mpo.trivial.float64.d10x4x3.jit_cached_run",
    }
    arguments = ["--warmup", "0", "--repeat", "1", "--json"]
    for scenario_id in sorted(scenario_ids):
        arguments.extend(("--scenario", scenario_id))

    completed = _run_benchmark_suite(*arguments)
    payload = json.loads(completed.stdout)

    assert {result["id"] for result in payload["results"]} == scenario_ids
    assert all(
        result["group"] == "tensor_networks" for result in payload["results"]
    )
    assert all(
        result["scenario_profile"] == "explicit-only"
        for result in payload["results"]
    )
    assert {result["execution"] for result in payload["results"]} == {
        "eager",
        "jit_compile_and_run",
        "jit_cached_run",
    }


def test_benchmark_suite_domain_and_diagnostic_scenarios_run():
    scenario_ids = {
        "network.ncon.custom.medium",
        "protect.su2.permute.float64.small.eager",
        "internal.strided_indices.rank2_noncontiguous",
    }
    arguments = ["--warmup", "0", "--repeat", "1", "--json"]
    for scenario_id in sorted(scenario_ids):
        arguments.extend(("--scenario", scenario_id))

    completed = _run_benchmark_suite(*arguments)
    payload = json.loads(completed.stdout)

    assert {result["id"] for result in payload["results"]} == scenario_ids
    assert {result["group"] for result in payload["results"]} == {
        "contractions",
        "diagnostics",
        "transforms",
    }


def test_benchmark_suite_z2_complex_linalg_runs():
    scenario_ids = {
        "linalg.mul.z2.complex128.d8x8x8.eager",
        "linalg.svd.z2.complex128.d8x8.eager",
    }
    arguments = ["--warmup", "0", "--repeat", "1", "--json"]
    for scenario_id in sorted(scenario_ids):
        arguments.extend(("--scenario", scenario_id))

    completed = _run_benchmark_suite(*arguments)
    payload = json.loads(completed.stdout)

    assert {result["id"] for result in payload["results"]} == scenario_ids
    assert all(result["dtype"] == "complex128" for result in payload["results"])
