"""Shared benchmark scenario runner and command-line interface."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import statistics
import time
from typing import Any


Operation = Callable[[], object]

QUICK_WARMUP = 1
QUICK_REPEAT = 2
FULL_WARMUP = 2
FULL_REPEAT = 7


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str
    description: str
    scenario_profile: str
    dtype: str
    size_label: str
    execution: str
    cache_policy: str
    factory: Callable[[], Operation]
    before_each: Callable[[], None] | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    id: str
    group: str
    description: str
    scenario_profile: str
    dtype: str
    size_label: str
    execution: str
    cache_policy: str
    warmup: int
    repeat: int
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float
    times_ms: list[float]


def scenario(
    id: str,
    group: str,
    description: str,
    scenario_profile: str,
    dtype: str,
    size_label: str,
    execution: str,
    cache_policy: str,
    factory: Callable[[], Operation],
    *,
    before_each: Callable[[], None] | None = None,
) -> Scenario:
    return Scenario(
        id=id,
        group=group,
        description=description,
        scenario_profile=scenario_profile,
        dtype=dtype,
        size_label=size_label,
        execution=execution,
        cache_policy=cache_policy,
        factory=factory,
        before_each=before_each,
    )


def block_until_ready(value: Any) -> None:
    if hasattr(value, "storage") and hasattr(value.storage, "data"):
        block_until_ready(value.storage.data)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)


def selected_scenarios(
    scenarios: tuple[Scenario, ...],
    selected_ids: Iterable[str] | None,
    *,
    quick: bool,
) -> tuple[Scenario, ...]:
    validate_scenarios(scenarios)
    if selected_ids is None:
        if quick:
            return tuple(
                item for item in scenarios if item.scenario_profile == "quick"
            )
        return tuple(
            item for item in scenarios
            if item.scenario_profile != "explicit-only"
        )

    by_id = {item.id: item for item in scenarios}
    selected: list[Scenario] = []
    for scenario_id in selected_ids:
        try:
            selected.append(by_id[scenario_id])
        except KeyError:
            valid = ", ".join(sorted(by_id))
            raise SystemExit(
                f"unknown scenario {scenario_id!r}; valid scenarios: {valid}"
            ) from None
    return tuple(selected)


def validate_scenarios(scenarios: tuple[Scenario, ...]) -> None:
    id_counts = Counter(item.id for item in scenarios)
    duplicate_ids = sorted(
        scenario_id for scenario_id, count in id_counts.items() if count > 1
    )
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise ValueError(f"duplicate benchmark scenario ids: {duplicates}")

    valid_profiles = {"quick", "full-only", "explicit-only"}
    invalid_profiles = sorted(
        {
            item.scenario_profile
            for item in scenarios
            if item.scenario_profile not in valid_profiles
        }
    )
    if invalid_profiles:
        invalid = ", ".join(invalid_profiles)
        raise ValueError(f"invalid benchmark scenario profiles: {invalid}")


def measure_scenario(
    scenario: Scenario,
    *,
    warmup: int,
    repeat: int,
) -> BenchmarkResult:
    operation = scenario.factory()
    for _ in range(warmup):
        if scenario.before_each is not None:
            scenario.before_each()
        block_until_ready(operation())

    times_ms: list[float] = []
    for _ in range(repeat):
        if scenario.before_each is not None:
            scenario.before_each()
        start = time.perf_counter_ns()
        block_until_ready(operation())
        stop = time.perf_counter_ns()
        times_ms.append((stop - start) / 1_000_000.0)

    quartiles = (
        statistics.quantiles(times_ms, n=4, method="inclusive")
        if len(times_ms) >= 2
        else None
    )
    iqr_ms = 0.0 if quartiles is None else quartiles[2] - quartiles[0]
    return BenchmarkResult(
        id=scenario.id,
        group=scenario.group,
        description=scenario.description,
        scenario_profile=scenario.scenario_profile,
        dtype=scenario.dtype,
        size_label=scenario.size_label,
        execution=scenario.execution,
        cache_policy=scenario.cache_policy,
        warmup=warmup,
        repeat=repeat,
        min_ms=min(times_ms),
        median_ms=statistics.median(times_ms),
        iqr_ms=iqr_ms,
        max_ms=max(times_ms),
        times_ms=times_ms,
    )


def run_payload(
    scenarios: tuple[Scenario, ...],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    environment: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "environment": environment(),
        "config": {
            "profile": profile,
            "warmup": warmup,
            "repeat": repeat,
            "scenario_count": len(scenarios),
        },
        "results": [
            asdict(measure_scenario(item, warmup=warmup, repeat=repeat))
            for item in scenarios
        ],
    }


def run_cli(
    *,
    description: str,
    script_path: str,
    scenarios: tuple[Scenario, ...],
    environment: Callable[[], dict[str, Any]],
    render_markdown: Callable[..., str],
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick smoke timing and scenario selection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the JSON payload to stdout.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the JSON payload to this path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Write a Markdown report to this path.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run one scenario id; repeat this flag to select multiple scenarios.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print every registered scenario id and exit.",
    )
    parser.add_argument("--repeat", type=int, help="Override measured iteration count.")
    parser.add_argument("--warmup", type=int, help="Override warmup iteration count.")
    args = parser.parse_args()

    if args.list_scenarios:
        validate_scenarios(scenarios)
        for item in scenarios:
            print(item.id)
        return

    profile = "quick" if args.quick else "full"
    warmup = args.warmup if args.warmup is not None else (
        QUICK_WARMUP if args.quick else FULL_WARMUP
    )
    repeat = args.repeat if args.repeat is not None else (
        QUICK_REPEAT if args.quick else FULL_REPEAT
    )
    if warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    selected = selected_scenarios(
        scenarios,
        args.scenario,
        quick=args.quick,
    )
    payload = run_payload(
        selected,
        profile=profile,
        warmup=warmup,
        repeat=repeat,
        environment=environment,
    )

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(
                payload,
                command=_command_from_args(args, script_path=script_path),
            ),
            encoding="utf-8",
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _command_from_args(args: argparse.Namespace, *, script_path: str) -> str:
    parts = ["uv", "run", "python", script_path]
    if args.quick:
        parts.append("--quick")
    for scenario_id in args.scenario or ():
        parts.extend(("--scenario", scenario_id))
    if args.json:
        parts.append("--json")
    if args.json_output is not None:
        parts.extend(("--json-output", str(args.json_output)))
    if args.markdown is not None:
        parts.extend(("--markdown", str(args.markdown)))
    if args.repeat is not None:
        parts.extend(("--repeat", str(args.repeat)))
    if args.warmup is not None:
        parts.extend(("--warmup", str(args.warmup)))
    return " ".join(shlex.quote(part) for part in parts)
