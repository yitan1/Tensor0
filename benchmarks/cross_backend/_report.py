"""Markdown rendering for cross-backend benchmark results."""

from __future__ import annotations

import json
from typing import Any

from ._plot import _mean_ms


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
    ]
    records = {(r["workload_id"], r["backend"]): r for r in payload["records"]}

    def latency(record):
        if "median_ms" not in record:
            return "—"
        return f"{record['median_ms']:.6f} / {_mean_ms(record):.6f}"

    compared_records = set()
    topology = None
    for comparison in payload["pairwise"]:
        workload = comparison["workload_id"]
        current = workload.split(".")[1]
        if current != topology:
            topology = current
            lines.extend(["", f"### {topology.upper()}", "",
                          "| Workload | Contender median / mean ms | Baseline median / mean ms | Median ratio | Mean ratio |",
                          "| --- | ---: | ---: | ---: | ---: |"])
        baseline = records[(workload, comparison["baseline"])]
        contender = records[(workload, comparison["contender"])]
        compared_records.update(((workload, comparison["baseline"]),
                                 (workload, comparison["contender"])))
        validated = comparison["validation"] == "passed"
        median = f"{comparison['contender_speedup']:.3f}×" if validated else "—"
        mean = f"{_mean_ms(baseline) / _mean_ms(contender):.3f}×" if validated else "—"
        lines.append(f"| `{workload}` | {latency(contender)} | {latency(baseline)} | {median} | {mean} |")
    unpaired_records = [record for key, record in records.items()
                        if key not in compared_records]
    if unpaired_records:
        lines.extend(["", "| Workload | Backend | Median / mean ms |",
                      "| --- | --- | ---: |"])
        for record in unpaired_records:
            lines.append(f"| `{record['workload_id']}` | `{record['backend']}` | {latency(record)} |")
    lines.extend(["", "Baseline and contender are identified in Configuration. "
                  "Ratios are baseline/contender; only passed comparisons have ratios.",
                  "", "### Record and sample details", "",
                  "| Workload | Backend | Status | Repeat per round | Samples | IQR ms | Validation |",
                  "| --- | --- | --- | ---: | ---: | ---: | --- |"])
    for record in payload["records"]:
        iqr = f"{record['iqr_ms']:.6f}" if "iqr_ms" in record else "—"
        lines.append(f"| `{record['workload_id']}` | `{record['backend']}` | "
                     f"`{record['status']}` | {record.get('repeat_per_round', config['repeat'])} | "
                     f"{len(record.get('samples_ms', []))} | {iqr} | "
                     f"`{record['validation']['status']}` |")

    round_records = [r for r in payload["records"] if r.get("round_statistics")]
    if round_records:
        lines.extend([
            "", "## Round Stability (Descriptive)", "",
            "Round medians and paired ratios are descriptive, not significance tests. "
            "The pooled statistics above are unchanged. Missing rounds are not renumbered.", "",
            "| Workload | Backend | Round | Samples | Median ms |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for record in round_records:
            for row in record["round_statistics"]:
                lines.append(
                    f"| `{record['workload_id']}` | `{record['backend']}` | "
                    f"{row['round']} | {row['sample_count']} | {row['median_ms']:.6f} |"
                )
        lines.extend([
            "", "| Workload | Contender | Paired round ratios | Direction |",
            "| --- | --- | --- | --- |",
        ])
        for comparison in payload["pairwise"]:
            ratios = ", ".join(
                f"r{r['round']}: {r['contender_speedup']:.3f}×"
                for r in comparison.get("paired_rounds", [])
            ) or "—"
            direction = ("direction inconsistent" if comparison.get("direction_inconsistent")
                         else "no observed reversal")
            if comparison["validation"] != "passed":
                direction = "not validated"
            lines.append(
                f"| `{comparison['workload_id']}` | `{comparison['contender']}` | "
                f"{ratios} | {direction} |"
            )

    if "repeat_by_workload" in config:
        lines.extend(["", "## Explicit Sampling Plan", "", "```json",
                      json.dumps({"repeat_by_workload": config["repeat_by_workload"],
                                  "provenance": config.get("sampling_plan")},
                                 indent=2, sort_keys=True), "```"])

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
            "- Raw samples are retained in the JSON result; the table reports pooled medians "
            "and unfiltered arithmetic mean latency (sum of all recorded timings / sample count). "
            "All timings, including natural GC/scheduling tails, are included without filtering, "
            "truncation or subtraction. The mean alone does not identify the cause of a tail.",
            "- Median and mean ratios are baseline latency / contender latency "
            "(TensorKit/Tensor0 with the default baseline). These descriptive summaries "
            "are not independent throughput measurements, significance tests or causal evidence. "
            "Latency plots show median lines and unfiltered arithmetic mean diamonds.",
            "- A speedup above one favors the contender. Direction inconsistency means "
            "paired round ratios occurred on both sides of one; ties are not reversals.",
            (
                "- `structural_only` does not assert cross-library "
                "fusion-basis equivalence."
            ),
            "",
        ]
    )
    return "\n".join(lines)
