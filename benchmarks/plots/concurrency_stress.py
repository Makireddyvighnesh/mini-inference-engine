"""Plot Phase 4 throughput, latency, padding, and GPU resource metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "results/stress_batch_final/concurrent_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/figures/concurrency_stress.png"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def _metric(metrics: Mapping[str, Any], key: str, percentile: str) -> float:
    value = metrics.get(key, {}).get(percentile)
    if value is None:
        raise ValueError(f"benchmark metric {key}.{percentile} is unavailable")
    return float(value)


def _summary_metric(summary: Mapping[str, Any], key: str, percentile: str) -> float:
    values = summary.get(key, {})
    if not isinstance(values, Mapping):
        raise ValueError(f"benchmark summary {key!r} is unavailable")
    value = values.get(percentile)
    if value is None:
        raise ValueError(f"benchmark summary {key}.{percentile} is unavailable")
    return float(value)


def load_stress_metrics(
    manifest_path: Path,
    *,
    workload_name: str,
) -> list[dict[str, Any]]:
    """Extract one comparable row per maximum batch size."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("manifest must contain a workloads list")

    rows: list[dict[str, Any]] = []
    for entry in workloads:
        if str(entry.get("name")) != workload_name:
            continue
        correctness = entry.get("correctness", {})
        if correctness.get("status") != "pass":
            raise ValueError(
                f"workload {workload_name!r} batch {entry.get('max_batch_size')} "
                "did not pass correctness"
            )
        summary = entry.get("summary", {})
        metrics = summary.get("metrics", {})
        scheduler = entry.get("scheduler", {})
        memory = summary.get("memory", {})
        gpu = summary.get("gpu_utilization_percent", {})
        peak_reserved = memory.get("peak_reserved_bytes")
        if peak_reserved is None:
            raise ValueError("peak reserved VRAM is unavailable")
        rows.append(
            {
                "workload": workload_name,
                "max_batch_size": int(entry["max_batch_size"]),
                "ttft_p50_ms": _metric(metrics, "ttft_ms", "p50"),
                "ttft_p95_ms": _metric(metrics, "ttft_ms", "p95"),
                "ttft_p99_ms": _metric(metrics, "ttft_ms", "p99"),
                "e2e_p50_ms": _metric(metrics, "e2e_latency_ms", "p50"),
                "e2e_p95_ms": _metric(metrics, "e2e_latency_ms", "p95"),
                "e2e_p99_ms": _metric(metrics, "e2e_latency_ms", "p99"),
                "tpot_p50_ms": _metric(metrics, "tpot_ms", "p50"),
                "tpot_p95_ms": _metric(metrics, "tpot_ms", "p95"),
                "tpot_p99_ms": _metric(metrics, "tpot_ms", "p99"),
                "tokens_per_second_p50": _summary_metric(
                    summary, "tokens_per_second", "p50"
                ),
                "requests_per_second_p50": _summary_metric(
                    summary, "requests_per_second", "p50"
                ),
                "gpu_utilization_p50": _percentile(gpu, "p50", "GPU utilization"),
                "gpu_utilization_p95": _percentile(gpu, "p95", "GPU utilization"),
                "gpu_utilization_p99": _percentile(gpu, "p99", "GPU utilization"),
                "peak_reserved_vram_gib": float(peak_reserved) / 2**30,
                "padding_waste_percent": 100.0
                * float(scheduler.get("padding_waste_ratio", 0.0)),
                "batch_count": int(scheduler.get("batch_count", 0)),
                "maximum_queue_depth": int(scheduler.get("maximum_queue_depth", 0)),
                "request_count": int(scheduler.get("maximum_queue_depth", 0)),
                "repetitions": int(summary.get("repetitions", 0)),
            }
        )
    rows.sort(key=lambda row: row["max_batch_size"])
    if len(rows) < 2:
        raise ValueError(
            f"need at least two batch sizes for workload {workload_name!r}; "
            f"found {len(rows)}"
        )
    batch_sizes = [row["max_batch_size"] for row in rows]
    if len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError("manifest contains duplicate maximum batch sizes")
    return rows


def _percentile(values: Mapping[str, Any], percentile: str, label: str) -> float:
    value = values.get(percentile)
    if value is None:
        raise ValueError(f"{label} {percentile} is unavailable")
    return float(value)


def write_stress_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval(
    axis: Any,
    x: list[int],
    middle: list[float],
    high: list[float],
    *,
    color: str,
) -> None:
    lower = [max(0.0, upper - value) for value, upper in zip(middle, high, strict=True)]
    axis.errorbar(
        x,
        high,
        yerr=[lower, [0.0] * len(lower)],
        fmt="D",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=color,
        color=color,
        capsize=5,
        linewidth=1.5,
    )


def _label_high_values(
    axis: Any,
    x: list[int],
    middle: list[float],
    values: list[float],
    *,
    suffix: str = "",
) -> None:
    for position, base, value in zip(x, middle, values, strict=True):
        close_to_base = value - base < max(100.0, 0.05 * value)
        axis.annotate(
            f"P95 {value:,.0f}{suffix}",
            (position, value),
            xytext=(14, -2) if close_to_base else (0, 7),
            textcoords="offset points",
            ha="left" if close_to_base else "center",
            color="#4B5563",
            fontsize=7,
        )


def render_stress_chart(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    source_label: str,
    workload_name: str,
) -> None:
    """Render discrete batch-size comparisons with P50/P95 uncertainty marks."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/minillm_l4_matplotlib")
    import matplotlib.pyplot as plt

    batch_sizes = [int(row["max_batch_size"]) for row in rows]
    labels = [str(value) for value in batch_sizes]
    x = list(range(len(rows)))
    ink = "#1F2937"
    blue = "#2563EB"
    gold = "#B7791F"
    orange = "#C05621"
    teal = "#0F766E"
    grid = "#D1D5DB"

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.4))
    fig.subplots_adjust(
        left=0.05,
        right=0.99,
        top=0.78,
        bottom=0.16,
        hspace=0.46,
        wspace=0.30,
    )

    def style_axis(axis: Any, title: str, ylabel: str) -> None:
        axis.set_title(title, color=ink, fontsize=11, fontweight="bold", pad=12)
        axis.set_ylabel(ylabel, color=ink, fontsize=9)
        axis.set_xlabel("Maximum static batch size", color=ink, fontsize=9)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color=grid, linewidth=0.8, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=ink, labelsize=8)

    def bars(axis: Any, values: list[float], color: str, fmt: str) -> None:
        marks = axis.bar(x, values, color=color, edgecolor=color, width=0.56)
        for mark, value in zip(marks, values, strict=True):
            axis.annotate(
                fmt.format(value),
                (mark.get_x() + mark.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                color=ink,
                fontsize=8,
            )

    axis = axes[0, 0]
    bars(axis, [float(row["tokens_per_second_p50"]) for row in rows], blue, "{:.1f}")
    style_axis(axis, "Aggregate token throughput", "Tokens/sec (P50)")

    axis = axes[0, 1]
    bars(axis, [float(row["requests_per_second_p50"]) for row in rows], teal, "{:.2f}")
    style_axis(axis, "Request throughput", "Requests/sec (P50)")

    axis = axes[0, 2]
    ttft_p50 = [float(row["ttft_p50_ms"]) for row in rows]
    ttft_p95 = [float(row["ttft_p95_ms"]) for row in rows]
    bars(axis, ttft_p50, blue, "{:.0f}")
    _interval(axis, x, ttft_p50, ttft_p95, color=gold)
    _label_high_values(axis, x, ttft_p50, ttft_p95)
    style_axis(axis, "Time to first token", "TTFT (ms)")

    axis = axes[0, 3]
    e2e_p50 = [float(row["e2e_p50_ms"]) for row in rows]
    e2e_p95 = [float(row["e2e_p95_ms"]) for row in rows]
    bars(axis, e2e_p50, blue, "{:.0f}")
    _interval(axis, x, e2e_p50, e2e_p95, color=gold)
    _label_high_values(axis, x, e2e_p50, e2e_p95)
    style_axis(axis, "End-to-end request latency", "E2E (ms)")

    axis = axes[1, 0]
    tpot_p50 = [float(row["tpot_p50_ms"]) for row in rows]
    tpot_p95 = [float(row["tpot_p95_ms"]) for row in rows]
    bars(axis, tpot_p50, blue, "{:.0f}")
    _interval(axis, x, tpot_p50, tpot_p95, color=gold)
    _label_high_values(axis, x, tpot_p50, tpot_p95)
    style_axis(axis, "Decode time per output token", "TPOT (ms/token)")

    axis = axes[1, 1]
    gpu_p50 = [float(row["gpu_utilization_p50"]) for row in rows]
    gpu_p95 = [float(row["gpu_utilization_p95"]) for row in rows]
    bars(axis, gpu_p50, teal, "{:.0f}%")
    _interval(axis, x, gpu_p50, gpu_p95, color=gold)
    style_axis(axis, "GPU compute utilization", "GPU utilization (%)")
    axis.set_ylim(0, 105)

    axis = axes[1, 2]
    bars(
        axis,
        [float(row["peak_reserved_vram_gib"]) for row in rows],
        orange,
        "{:.1f}",
    )
    style_axis(axis, "Peak reserved GPU memory", "VRAM (GiB)")

    axis = axes[1, 3]
    padding = [float(row["padding_waste_percent"]) for row in rows]
    bars(axis, padding, teal, "{:.1f}%")
    style_axis(axis, "Static-batch padding waste", "Wasted slots (%)")

    axes[0, 2].plot([], [], color=blue, linewidth=8, label="P50")
    axes[0, 2].plot(
        [],
        [],
        marker="D",
        markerfacecolor="white",
        markeredgecolor=gold,
        color=gold,
        label="P95",
    )
    axes[0, 2].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle(
        "Static batching stress test on NVIDIA L4",
        color=ink,
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        f"Qwen3-4B FP8 · {workload_name} workload · "
        f"{int(rows[0]['request_count'])} requests/trace · "
        f"{int(rows[0]['repetitions'])} measured runs · "
        "greedy decoding · all requests ready at launch",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.text(
        0.05,
        0.035,
        "Bars show P50. Diamonds/whiskers show P95 for latency panels. "
        "Higher throughput is better; lower latency, VRAM, and padding are better. "
        f"Source: {Path(source_label).name}",
        color="#6B7280",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Phase 4 batch-size stress metrics."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workload", default="mixed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = _project_path(args.manifest)
    output_path = _project_path(args.output)
    data_path = (
        _project_path(args.data_output)
        if args.data_output is not None
        else output_path.with_suffix(".csv")
    )
    rows = load_stress_metrics(manifest_path, workload_name=str(args.workload))
    write_stress_data(rows, data_path)
    render_stress_chart(
        rows,
        output_path,
        source_label=str(manifest_path),
        workload_name=str(args.workload),
    )
    print(f"Chart: {output_path}")
    print(f"Data:  {data_path}")


if __name__ == "__main__":
    main()
