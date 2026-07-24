"""Markdown rendering for cross-backend benchmark results."""

from __future__ import annotations

import json
from typing import Any


def render_markdown(payload: dict[str, Any], *, command: str) -> str:
    config = payload["config"]
    lines = [
        "# Cross-Backend Benchmark",
        "",
        "This report compares independent implementations of the same frozen, "
        "backend-neutral workloads.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Configuration",
        "",
        f"- Profile: `{config['profile']}`",
        f"- Measurement: `{config['measurement']}`",
        f"- Backends: `{', '.join(config['backends'])}`",
        f"- Baseline: `{config['baseline']}`",
        f"- CPU threads per backend: `{config['threads']}`",
        f"- Warmup/repeat/rounds: "
        f"`{config['warmup']}/{config['repeat']}/{config['rounds']}`",
        "",
        "## Results",
        "",
        "| Workload | Backend | Status | Median ms | IQR ms | Validation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in payload["records"]:
        median = (
            f"{record['median_ms']:.6f}"
            if "median_ms" in record
            else "—"
        )
        iqr = f"{record['iqr_ms']:.6f}" if "iqr_ms" in record else "—"
        lines.append(
            f"| `{record['workload_id']}` | `{record['backend']}` | "
            f"`{record['status']}` | {median} | {iqr} | "
            f"`{record['validation']['status']}` |"
        )

    if payload["pairwise"]:
        lines.extend(
            [
                "",
                "## Paired Comparison",
                "",
                "| Workload | Baseline | Contender | Contender speedup | Validation |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for comparison in payload["pairwise"]:
            speedup = (
                f"{comparison['contender_speedup']:.3f}×"
                if comparison["validation"] == "passed"
                else "—"
            )
            lines.append(
                f"| `{comparison['workload_id']}` | "
                f"`{comparison['baseline']}` | `{comparison['contender']}` | "
                f"{speedup} | "
                f"`{comparison['validation']}` |"
            )

    if payload["failures"]:
        lines.extend(["", "## Backend Failures", ""])
        for failure in payload["failures"]:
            lines.append(
                f"- `{failure['backend']}` round `{failure['round']}` "
                f"(`{failure['kind']}`): {failure['error']}"
            )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(
                {
                    "host": payload["environment"],
                    "backends": payload["backend_metadata"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Interpretation",
            "",
            "- Inputs, space construction, compilation, and warmup are outside timing.",
            (
                "- Each backend consumes the same normalized workload and "
                "contraction order."
            ),
            (
                "- Validation reuses a timed invocation's returned scalar; it "
                "does not execute the contraction again."
            ),
            "- Raw samples are retained in the JSON result; the table reports medians.",
            (
                "- `structural_only` does not assert cross-library "
                "fusion-basis equivalence."
            ),
            "",
        ]
    )
    return "\n".join(lines)
