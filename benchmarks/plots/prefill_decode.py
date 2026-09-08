"""Plot prefill and decode metrics over increasing prompt context lengths."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "results/manual_context_fixed/manual_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/figures/prefill_decode_context.png"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"benchmark metric {key!r} is unavailable")
    return float(value)


def load_context_metrics(
    manifest_path: Path,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Extract one chart row per context size from a benchmark manifest."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("manifest must contain a workloads list")

    rows: list[dict[str, Any]] = []
    for workload in workloads:
        if int(workload.get("batch_size", -1)) != batch_size:
            continue
        summary = workload.get("summary", {})
        metrics = summary.get("metrics", {})
        prefill = metrics.get("prefill_ms", {})
        tpot = metrics.get("tpot_ms", {})
        decode = metrics.get("decode_ms", {})
        ttft = metrics.get("ttft_ms", {})
        rows.append(
            {
                "workload": str(workload["name"]),
                "prompt_tokens": int(workload["prompt_tokens"]),
                "output_tokens": int(workload["output_tokens"]),
                "batch_size": batch_size,
                "prefill_p50_ms": _required_number(prefill, "p50"),
                "prefill_p95_ms": _required_number(prefill, "p95"),
                "prefill_p99_ms": _required_number(prefill, "p99"),
                "tpot_p50_ms": _required_number(tpot, "p50"),
                "tpot_p95_ms": _required_number(tpot, "p95"),
                "tpot_p99_ms": _required_number(tpot, "p99"),
                "decode_p50_ms": _required_number(decode, "p50"),
                "decode_p95_ms": _required_number(decode, "p95"),
                "decode_p99_ms": _required_number(decode, "p99"),
                "ttft_p50_ms": _required_number(ttft, "p50"),
                "aggregate_tps_p50": _required_number(
                    summary.get("tokens_per_second", {}), "p50"
                ),
                "repetitions": int(summary.get("repetitions", 0)),
                "correctness": str(
                    workload.get("correctness", {}).get("status", "unknown")
                ),
            }
        )
    rows.sort(key=lambda row: row["prompt_tokens"])
    if len(rows) < 2:
        raise ValueError(
            f"need at least two context sizes for batch size {batch_size}; "
            f"found {len(rows)}"
        )
    prompt_lengths = [row["prompt_tokens"] for row in rows]
    if len(prompt_lengths) != len(set(prompt_lengths)):
        raise ValueError("manifest contains duplicate context sizes")
    return rows


def write_chart_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_context_comparison(rows: Sequence[Mapping[str, Any]]) -> int:
    """Require one output length so prompt context is the changed variable."""

    output_lengths = {int(row["output_tokens"]) for row in rows}
    if len(output_lengths) != 1:
        raise ValueError(
            "context comparison requires one fixed output length; "
            f"found {sorted(output_lengths)}"
        )
    if any(str(row.get("correctness")) != "pass" for row in rows):
        raise ValueError("all plotted workloads must pass correctness")
    return next(iter(output_lengths))


def render_context_chart(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    source_label: str,
) -> None:
    """Render comparable prefill and normalized decode panels."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/minillm_l4_matplotlib")
    import matplotlib.pyplot as plt

    contexts = [int(row["prompt_tokens"]) for row in rows]
    output_tokens = validate_context_comparison(rows)
    labels = [f"{row['prompt_tokens']:,}" for row in rows]
    batch_size = int(rows[0]["batch_size"])
    repetitions = int(rows[0]["repetitions"])
    blue = "#2563EB"
    gold = "#B7791F"
    ink = "#1F2937"
    grid = "#D1D5DB"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.22, wspace=0.28)
    panels = (
        ("Prompt prefill latency", "prefill_p50_ms", "prefill_p95_ms", "Latency (ms)"),
        ("Time per output token", "tpot_p50_ms", "tpot_p95_ms", "TPOT (ms/token)"),
    )
    x = list(range(len(contexts)))
    for axis, (title, p50_key, p95_key, ylabel) in zip(axes, panels, strict=True):
        p50 = [float(row[p50_key]) for row in rows]
        p95 = [float(row[p95_key]) for row in rows]
        upper_errors = [max(0.0, high - middle) for middle, high in zip(p50, p95)]
        bars = axis.bar(
            x,
            p50,
            color=blue,
            edgecolor="#1E40AF",
            linewidth=1.0,
            width=0.58,
            label="P50 latency",
        )
        axis.errorbar(
            x,
            p95,
            yerr=[upper_errors, [0.0] * len(upper_errors)],
            fmt="D",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=gold,
            color=gold,
            capsize=5,
            linewidth=1.5,
            label="P95 latency",
        )
        axis.set_title(
            title,
            color=ink,
            fontsize=12,
            fontweight="bold",
            pad=18,
        )
        axis.set_ylabel(ylabel, color=ink)
        axis.set_xlabel("Prompt context length (tokens)", color=ink)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, max(p95) * 1.18)
        axis.grid(axis="y", color=grid, linewidth=0.8, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=ink)
        for index, (bar, value, high) in enumerate(zip(bars, p50, p95, strict=True)):
            axis.annotate(
                f"P50 {value:,.1f}\nP95 {high:,.1f}",
                (bar.get_x() + bar.get_width() / 2, high),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=ink,
                fontsize=8,
            )

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Prefill and decode metrics across context sizes",
        color=ink,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"Qwen3-4B FP8 · batch size {batch_size} · {repetitions} measured runs · "
        f"{output_tokens} output tokens · greedy decoding",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.035,
        "Only prompt context changes; output length, model, batch size, and decoding "
        "policy are fixed. "
        f"Source: {Path(source_label).name}",
        color="#6B7280",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prefill and decode metrics across prompt context sizes."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=1)
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
    rows = load_context_metrics(manifest_path, batch_size=args.batch_size)
    validate_context_comparison(rows)
    write_chart_data(rows, data_path)
    render_context_chart(rows, output_path, source_label=str(manifest_path))
    print(f"Chart: {output_path}")
    print(f"Data:  {data_path}")


if __name__ == "__main__":
    main()
