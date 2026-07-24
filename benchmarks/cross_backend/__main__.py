"""Command-line entry point for cross-backend benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from ._orchestrator import run_comparison
from ._protocol import (
    BACKEND_NAMES,
    ProtocolError,
    load_profiles,
    load_workloads,
)
from ._report import render_markdown


def _write_plot(payload: dict[str, Any], output: Path) -> None:
    try:
        from ._plot import render_plot

        render_plot(payload, output)
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.startswith("matplotlib"):
            raise SystemExit(
                "plotting requires the benchmark dependency group; "
                "run with `uv run --group bench python -m "
                "benchmarks.cross_backend ...`"
            ) from None
        raise
    except ValueError as error:
        raise SystemExit(str(error)) from None


def _command() -> str:
    return shlex.join(
        ["uv", "run", "python", "-m", "benchmarks.cross_backend", *sys.argv[1:]]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run backend-neutral tensor-library comparison workloads."
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        help="Named profile from profiles.toml (default: smoke).",
    )
    parser.add_argument(
        "--backend",
        action="append",
        help="Override profile backends; repeat to select multiple.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        help="Override profile workloads; repeat to select multiple.",
    )
    parser.add_argument("--warmup", type=int, help="Override warmup count.")
    parser.add_argument("--repeat", type=int, help="Override samples per round.")
    parser.add_argument("--rounds", type=int, help="Override paired round count.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU thread budget for each backend process (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the result payload to stdout.",
    )
    parser.add_argument("--json-output", type=Path, help="Write JSON to this path.")
    parser.add_argument("--markdown", type=Path, help="Write Markdown to this path.")
    parser.add_argument(
        "--plot",
        type=Path,
        help="Write an SVG or PNG small-multiple latency comparison plot.",
    )
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List normalized workload ids and exit.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List supported backend adapter names and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List profile names and exit.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        workloads = load_workloads()
        profiles = load_profiles()
        if args.list_workloads:
            print(*workloads, sep="\n")
            return
        if args.list_backends:
            print(*BACKEND_NAMES, sep="\n")
            return
        if args.list_profiles:
            print(*profiles, sep="\n")
            return
        try:
            profile = profiles[args.profile]
        except KeyError:
            valid = ", ".join(sorted(profiles))
            raise ProtocolError(
                f"unknown profile {args.profile!r}; valid profiles: {valid}"
            ) from None

        workload_ids = args.workload or profile["workloads"]
        unknown_workloads = sorted(set(workload_ids) - set(workloads))
        if unknown_workloads:
            raise ProtocolError(
                "unknown workloads: " + ", ".join(unknown_workloads)
            )
        backend_names = args.backend or profile["backends"]
        baseline = (
            profile["baseline"]
            if profile["baseline"] in backend_names
            else backend_names[0]
        )
        warmup = profile["warmup"] if args.warmup is None else args.warmup
        repeat = profile["repeat"] if args.repeat is None else args.repeat
        rounds = profile["rounds"] if args.rounds is None else args.rounds
        payload = run_comparison(
            profile=args.profile,
            workloads=[workloads[workload_id] for workload_id in workload_ids],
            backends=backend_names,
            baseline=baseline,
            warmup=warmup,
            repeat=repeat,
            rounds=rounds,
            threads=args.threads,
        )
    except ProtocolError as error:
        raise SystemExit(str(error)) from None

    report = render_markdown(payload, command=_command())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
    if args.plot is not None:
        _write_plot(payload, args.plot)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif (
        args.markdown is None
        and args.json_output is None
        and args.plot is None
    ):
        print(report)

    if payload["failures"] or any(
        record["status"] != "ok"
        or record["validation"]["status"] == "failed"
        for record in payload["records"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
