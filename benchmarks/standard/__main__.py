from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax

import tensor0

from .contractions import (
    CONTRACTION_SCENARIO_SPECS,
    contraction_metadata,
)
from .contractions.tensor_networks import (
    TENSOR_NETWORK_CASES,
    TENSOR_NETWORK_SCENARIO_SPECS,
    tensor_network_metadata,
)
from .diagnostics import (
    DIAGNOSTIC_SCENARIO_SPECS,
    diagnostic_metadata,
)
from ._runner import Scenario, run_cli as _run_cli
from ._specs import (
    ScenarioSpec,
    materialize_scenarios,
)
from .linalg import (
    LINALG_CASES,
    LINALG_EXTENSION_SCENARIO_SPECS,
    LINALG_SCENARIO_SPECS,
    linalg_metadata,
)
from .transforms import (
    TRANSFORM_CASES,
    TRANSFORM_EXTENSION_SCENARIO_SPECS,
    TRANSFORM_SCENARIO_SPECS,
    TRANSFORM_SMOKE_SCENARIO_SPECS,
    transform_metadata,
)


jax.config.update("jax_enable_x64", True)

_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    LINALG_SCENARIO_SPECS
    + LINALG_EXTENSION_SCENARIO_SPECS
    + TRANSFORM_SCENARIO_SPECS
    + TRANSFORM_SMOKE_SCENARIO_SPECS
    + TRANSFORM_EXTENSION_SCENARIO_SPECS
    + CONTRACTION_SCENARIO_SPECS
    + TENSOR_NETWORK_SCENARIO_SPECS
    + DIAGNOSTIC_SCENARIO_SPECS
)


def _scenarios() -> tuple[Scenario, ...]:
    return materialize_scenarios(_SCENARIO_SPECS)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _source_state(root: Path) -> dict[str, Any]:
    revision = _git_bytes(root, "rev-parse", "HEAD").decode().strip()
    status = _git_bytes(root, "status", "--short")
    diff = _git_bytes(root, "diff", "--binary", "--no-ext-diff", "HEAD")
    untracked = _git_bytes(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    manifest = []
    for encoded_path in untracked.split(b"\0"):
        if not encoded_path:
            continue
        relative_path = encoded_path.decode()
        path = root / relative_path
        if path.is_file():
            manifest.append(
                {"path": relative_path, "sha256": _sha256(path.read_bytes())}
            )
    return {
        "revision": revision,
        "dirty": bool(status),
        "binary_diff_sha256": _sha256(diff),
        "untracked_files": manifest,
    }


def _environment(*, registered_scenario_count: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve()
    support_files = sorted(
        path
        for path in (root / "benchmarks" / "standard").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".toml"}
        and path != script
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "jax": getattr(jax, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": True,
        "tensor0": getattr(tensor0, "__version__", "unknown"),
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "benchmark_suite": {
            "standard_case_count": (
                len(LINALG_CASES)
                + len(TRANSFORM_CASES)
                + len(TENSOR_NETWORK_CASES)
            ),
            "registered_scenario_count": registered_scenario_count,
            "linalg": linalg_metadata(),
            "transforms": transform_metadata(),
            "contractions": contraction_metadata(),
            "tensor_networks": tensor_network_metadata(),
            "diagnostics": diagnostic_metadata(),
        },
        "benchmark_support_files_sha256": {
            str(path.relative_to(root)): _sha256(path.read_bytes())
            for path in support_files
        },
        "benchmark_script_sha256": _sha256(script.read_bytes()),
        "public_repository": _source_state(root),
    }


def _render_markdown(payload: dict[str, Any], *, command: str) -> str:
    lines = [
        "# Tensor0 Benchmark Suite",
        "",
        "This suite defines Tensor0's standard benchmark workload and parameter matrix",
        "through Tensor0's public functional API. All results measure Tensor0.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Environment And Source State",
        "",
        "```json",
        json.dumps(payload["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "## Configuration",
        "",
    ]
    for key, value in payload["config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Results", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### `{result['id']}`",
                "",
                f"- Group: `{result['group']}`",
                f"- Description: {result['description']}",
                f"- Profile: `{result['scenario_profile']}`",
                f"- Dtype/size: `{result['dtype']}` / `{result['size_label']}`",
                f"- Execution/cache: `{result['execution']}` / "
                f"`{result['cache_policy']}`",
                f"- Warmup/repeat: `{result['warmup']}/{result['repeat']}`",
                f"- Min/median/IQR/max ms: `{result['min_ms']:.3f}` / "
                f"`{result['median_ms']:.3f}` / `{result['iqr_ms']:.3f}` / "
                f"`{result['max_ms']:.3f}`",
                "- Times ms: `"
                + ", ".join(f"{sample:.3f}" for sample in result["times_ms"])
                + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## Measurement Boundary",
            "",
            "- Space and seeded random-tensor construction occur outside timing.",
            (
                "- Eager and cached modes perform their declared untimed warmup "
                "before measurement; cold and raw modes do not."
            ),
            "- Cold hooks, including cache clearing, run outside the timer.",
            "- JIT compile scenarios clear JAX caches before each timed iteration.",
            (
                "- Cached JIT and gradient scenarios compile and synchronize once "
                "in the factory."
            ),
            "- Every timed result is synchronized before the clock stops.",
            (
                "- Network contractions return rank-zero TensorMaps; scalar "
                "extraction is excluded."
            ),
            "",
            "## Workload Boundary",
            "",
            (
                "- MPO, PEPO, and MERA use fixed topology, expression tree, "
                "and parameter tables."
            ),
            "- Tensor0 `@`, `svd_compact`, and `permute` are functional operations, "
            "so output allocation is timed.",
            "- Large parameter-matrix cases and private implementation diagnostics "
            "are explicit-only; `--all` opts into every registered scenario.",
            (
                "- Focused U1, SU2, and FermionParity cases complement the "
                "Trivial/Z2 standard matrix without a second benchmark entry point."
            ),
            (
                "- Results are Tensor0 regression measurements, not "
                "cross-library comparisons."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    scenarios = _scenarios()
    _run_cli(
        description="Run Tensor0's unified benchmark suite.",
        module="benchmarks.standard",
        scenarios=scenarios,
        environment=lambda: _environment(
            registered_scenario_count=len(scenarios)
        ),
        render_markdown=_render_markdown,
    )


if __name__ == "__main__":
    main()
