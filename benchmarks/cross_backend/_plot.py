"""Backend-neutral plot rendering for paired benchmark results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


_SUPPORTED_SUFFIXES = {".png", ".svg"}
_BACKEND_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
_SECTOR_ORDER = ("trivial", "z2", "u1", "su2")
_SECTOR_LABELS = {
    "trivial": "Trivial",
    "z2": "Z2",
    "u1": "U1",
    "su2": "SU2",
}


@dataclass(frozen=True)
class _WorkloadLabel:
    id: str
    topology: str
    sector: str
    dtype: str
    dimensions: str


def _ordered_strings(values: Iterable[object]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _workload_ids(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config", {})
    configured = config.get("workloads", []) if isinstance(config, dict) else []
    observed_records = [
        record.get("workload_id")
        for record in payload.get("records", [])
        if isinstance(record, dict)
    ]
    observed_comparisons = [
        comparison.get("workload_id")
        for comparison in payload.get("pairwise", [])
        if isinstance(comparison, dict)
    ]
    return _ordered_strings(
        [*configured, *observed_records, *observed_comparisons]
    )


def _backend_names(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config", {})
    configured = config.get("backends", []) if isinstance(config, dict) else []
    observed = [
        record.get("backend")
        for record in payload.get("records", [])
        if isinstance(record, dict)
    ]
    return _ordered_strings([*configured, *observed])


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _samples(record: dict[str, Any] | None) -> list[float]:
    if record is None or record.get("status") not in {"ok", "partial"}:
        return []
    samples = [
        sample
        for value in record.get("samples_ms", [])
        if (sample := _positive_float(value)) is not None
    ]
    if samples:
        return samples
    median = _positive_float(record.get("median_ms"))
    return [] if median is None else [median]


def _parse_workload_id(workload_id: str) -> _WorkloadLabel:
    parts = workload_id.split(".")
    if len(parts) < 5 or parts[0] != "tensor_networks":
        raise ValueError(
            "plot workload ids must follow "
            "`tensor_networks.<topology>.<sector>.<dtype>.<dimensions>`"
        )
    return _WorkloadLabel(
        id=workload_id,
        topology=parts[1],
        sector=parts[2],
        dtype=parts[3],
        dimensions=".".join(parts[4:]),
    )


def _backend_label(name: str) -> str:
    return {
        "tensor0": "Tensor0",
        "tensorkit": "TensorKit",
    }.get(name, name)


def _dimension_label(label: _WorkloadLabel) -> str:
    return label.dimensions.removeprefix("d").replace("x", "×")


def _winner_label(
    comparison: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if comparison is None:
        return "not compared", None
    if comparison.get("validation") != "passed":
        return "not validated", None
    speedup = _positive_float(comparison.get("contender_speedup"))
    baseline = comparison.get("baseline")
    contender = comparison.get("contender")
    if (
        speedup is None
        or not isinstance(baseline, str)
        or not isinstance(contender, str)
    ):
        return "not compared", None
    if speedup >= 1:
        winner = contender
        factor = speedup
    else:
        winner = baseline
        factor = 1 / speedup
    return f"{_backend_label(winner)}\n{factor:.2f}×", winner


def _row_limits(
    labels: list[_WorkloadLabel],
    backend_names: list[str],
    records: dict[tuple[str, str], dict[str, Any]],
) -> tuple[float, float] | None:
    values = [
        value
        for label in labels
        for backend in backend_names
        for value in _samples(records.get((label.id, backend)))
    ]
    if not values:
        return None
    lower = 10 ** math.floor(math.log10(min(values)))
    upper = 10 ** math.ceil(math.log10(max(values)))
    if upper <= lower:
        upper = lower * 10
    return lower, upper


def render_plot(payload: dict[str, Any], output: Path) -> Path:
    """Render grouped backend boxplots in topology-by-sector panels."""

    suffix = output.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise ValueError(
            f"plot output must use one of these suffixes: {supported}"
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    workload_ids = _workload_ids(payload)
    backend_names = _backend_names(payload)
    if not workload_ids:
        raise ValueError("plot payload contains no workloads")
    if not backend_names:
        raise ValueError("plot payload contains no backends")

    labels = [_parse_workload_id(workload_id) for workload_id in workload_ids]
    topologies = _ordered_strings(label.topology for label in labels)
    sectors_present = {label.sector for label in labels}
    sectors = [
        sector for sector in _SECTOR_ORDER if sector in sectors_present
    ]
    sectors.extend(
        sector
        for sector in _ordered_strings(label.sector for label in labels)
        if sector not in sectors
    )
    cells: dict[tuple[str, str], list[_WorkloadLabel]] = {}
    for label in labels:
        key = (label.topology, label.sector)
        cells.setdefault(key, []).append(label)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        workload_id = record.get("workload_id")
        backend = record.get("backend")
        if isinstance(workload_id, str) and isinstance(backend, str):
            records[(workload_id, backend)] = record
    comparisons: dict[str, dict[str, Any]] = {}
    for comparison in payload.get("pairwise", []):
        if not isinstance(comparison, dict):
            continue
        workload_id = comparison.get("workload_id")
        if isinstance(workload_id, str):
            comparisons[workload_id] = comparison
    colors = {
        backend: _BACKEND_COLORS[index % len(_BACKEND_COLORS)]
        for index, backend in enumerate(backend_names)
    }
    profile = (
        payload.get("config", {}).get("profile", "custom")
        if isinstance(payload.get("config"), dict)
        else "custom"
    )
    threads = (
        payload.get("config", {}).get("threads", "unknown")
        if isinstance(payload.get("config"), dict)
        else "unknown"
    )

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure, axes = plt.subplots(
            nrows=len(topologies),
            ncols=len(sectors),
            squeeze=False,
            sharey="row",
            figsize=(
                max(8.0, 4.8 * len(sectors)),
                max(4.0, 3.35 * len(topologies) + 1.0),
            ),
        )
        try:
            for row_index, topology in enumerate(topologies):
                row_labels = [
                    label for label in labels if label.topology == topology
                ]
                row_limits = _row_limits(row_labels, backend_names, records)
                visible_columns = [
                    column_index
                    for column_index, sector in enumerate(sectors)
                    if (topology, sector) in cells
                ]
                first_visible = min(visible_columns)
                for column_index, sector in enumerate(sectors):
                    axis = axes[row_index][column_index]
                    cell_labels = cells.get((topology, sector))
                    if cell_labels is None:
                        axis.set_visible(False)
                        continue

                    group_width = 0.76
                    box_width = group_width / len(backend_names)
                    backend_offsets = [
                        (index - (len(backend_names) - 1) / 2) * box_width
                        for index in range(len(backend_names))
                    ]
                    for case_index, label in enumerate(
                        cell_labels,
                        start=1,
                    ):
                        comparison = comparisons.get(label.id)
                        validation = (
                            str(comparison.get("validation", "not_run"))
                            if isinstance(comparison, dict)
                            else "not_run"
                        )
                        if validation == "structural_only":
                            axis.axvspan(
                                case_index - 0.5,
                                case_index + 0.5,
                                color="#FFF0C2",
                                alpha=0.55,
                                zorder=0,
                            )
                        elif validation == "failed":
                            axis.axvspan(
                                case_index - 0.5,
                                case_index + 0.5,
                                color="#F8CACA",
                                alpha=0.55,
                                zorder=0,
                            )

                        for backend_index, backend in enumerate(backend_names):
                            position = (
                                case_index + backend_offsets[backend_index]
                            )
                            record = records.get((label.id, backend))
                            samples = _samples(record)
                            color = colors[backend]
                            if samples:
                                axis.boxplot(
                                    [samples],
                                    positions=[position],
                                    widths=box_width * 0.72,
                                    patch_artist=True,
                                    manage_ticks=False,
                                    showfliers=True,
                                    boxprops={
                                        "facecolor": color,
                                        "edgecolor": color,
                                        "alpha": 0.58,
                                        "linewidth": 1.2,
                                    },
                                    medianprops={
                                        "color": "#111111",
                                        "linewidth": 1.5,
                                    },
                                    whiskerprops={
                                        "color": color,
                                        "linewidth": 1.1,
                                    },
                                    capprops={
                                        "color": color,
                                        "linewidth": 1.1,
                                    },
                                    flierprops={
                                        "marker": "o",
                                        "markerfacecolor": color,
                                        "markeredgecolor": color,
                                        "markersize": 2.8,
                                        "alpha": 0.45,
                                    },
                                )
                            else:
                                status = (
                                    str(record.get("status", "unavailable"))
                                    if isinstance(record, dict)
                                    else "unavailable"
                                )
                                axis.text(
                                    position,
                                    0.48,
                                    status,
                                    color=color,
                                    fontsize=6,
                                    rotation=90,
                                    ha="center",
                                    va="center",
                                    transform=axis.get_xaxis_transform(),
                                )

                        winner, winner_backend = _winner_label(
                            comparison
                            if isinstance(comparison, dict)
                            else None
                        )
                        axis.text(
                            case_index,
                            0.98,
                            winner,
                            color=(
                                colors.get(winner_backend, "#666666")
                                if winner_backend is not None
                                else "#666666"
                            ),
                            fontsize=6,
                            fontweight="bold",
                            ha="center",
                            va="top",
                            transform=axis.get_xaxis_transform(),
                            bbox={
                                "boxstyle": "round,pad=0.12",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.78,
                            },
                        )

                    validation_states = {
                        str(comparisons[label.id].get("validation", "not_run"))
                        for label in cell_labels
                        if label.id in comparisons
                    }
                    validation_note = (
                        "structural only cases are shaded yellow"
                        if "structural_only" in validation_states
                        else (
                            "failed validation is shaded red"
                            if "failed" in validation_states
                            else None
                        )
                    )
                    if validation_note is not None:
                        axis.text(
                            0.98,
                            0.03,
                            validation_note,
                            color=(
                                "#8A6500"
                                if "structural_only" in validation_states
                                else "#B00020"
                            ),
                            fontsize=7,
                            ha="right",
                            va="bottom",
                            transform=axis.transAxes,
                        )

                    axis.set_title(
                        f"{_SECTOR_LABELS.get(sector, sector)} · "
                        f"{cell_labels[0].dtype}",
                        fontsize=9,
                        pad=8,
                    )
                    axis.set_yscale("log")
                    if row_limits is not None:
                        axis.set_ylim(*row_limits)
                    axis.set_xlim(0.5, len(cell_labels) + 0.5)
                    axis.set_xticks(
                        range(1, len(cell_labels) + 1),
                        [_dimension_label(label) for label in cell_labels],
                        rotation=35,
                        ha="right",
                    )
                    axis.set_xlabel("Nominal dimensions", fontsize=8)
                    axis.tick_params(axis="x", labelsize=7)
                    axis.tick_params(axis="y", labelsize=8)
                    axis.grid(
                        axis="y",
                        which="both",
                        alpha=0.22,
                        linewidth=0.6,
                    )
                    if column_index == first_visible:
                        axis.set_ylabel(
                            f"{topology.upper()}\nLatency (ms)",
                            fontsize=9,
                        )

            figure.suptitle(
                (
                    "Cross-backend steady-state latency "
                    f"({profile}, {threads} threads)"
                ),
                fontsize=14,
                y=0.995,
            )
            figure.legend(
                handles=[
                    Patch(
                        facecolor=colors[backend],
                        edgecolor=colors[backend],
                        alpha=0.58,
                        label=_backend_label(backend),
                    )
                    for backend in backend_names
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=len(backend_names),
                frameon=False,
            )
            figure.text(
                0.5,
                0.012,
                (
                    "Each x-axis group is one dimension-matched workload; "
                    "adjacent colors are the backends being compared. "
                    "Boxes show Q1–Q3, "
                    "center lines are medians, and whiskers use 1.5× IQR. "
                    "Lower latency is faster."
                ),
                ha="center",
                fontsize=8,
                color="#444444",
            )
            figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
    return output
