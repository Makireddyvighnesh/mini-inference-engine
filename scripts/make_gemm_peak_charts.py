"""Render GEMM peak charts from run_gemm_peak (light and dark SVG per chart).

  .conda-env/bin/python minillm_l4/scripts/make_gemm_peak_charts.py \
      minillm_l4/results/gemm_peak_20261026/gemm_peak.json minillm_l4/docs/assets/gemm_peak

Charts:
* peak_vs_datasheet - best measured TFLOPS per precision against NVIDIA's dense
  datasheet peak (track) and the clock-adjusted peak (tick);
* size_sweep        - TFLOPS vs square matrix size, one line per precision;
* model_roofline    - Qwen3-4B per-layer projections (7 GEMMs) vs tokens per
  forward, with the bandwidth/compute roofline for FP8 and BF16 weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, figure, style, title  # noqa: E402

LABELS = {"fp32": "FP32", "tf32": "TF32", "fp16": "FP16", "bf16": "BF16",
          "fp8": "FP8 (cuBLAS)", "fp8_ours": "FP8 (ours, Triton)", "int8": "INT8"}
ORDER = ["fp32", "tf32", "fp16", "bf16", "int8", "fp8", "fp8_ours"]
# Fixed categorical slot per precision (light/dark steps of the same hue).
SLOTS = {
    "light": {"fp32": "#6250d6", "tf32": "#e87ba4", "bf16": "#1baf7a", "fp8": "#2a78d6",
              "fp8_ours": "#eb6834", "int8": "#eda100", "fp16": "#008300"},
    "dark": {"fp32": "#9085e9", "tf32": "#d55181", "bf16": "#199e70", "fp8": "#3987e5",
             "fp8_ours": "#d95926", "int8": "#c98500", "fp16": "#008300"},
}
LAYER = {"q": 1, "kv": 2, "o": 1, "gate_up": 2, "down": 1}  # projections per decoder layer


def footer(fig, t, text):
    fig.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=8, color=t["muted"])


def best_square(data):
    best = {}
    for row in data["square"]:
        if row["precision"] not in best or row["tflops"] > best[row["precision"]]["tflops"]:
            best[row["precision"]] = row
    return best


def peak_vs_datasheet(data, t, theme):
    best = best_square(data)
    names = [p for p in ORDER if p in best][::-1]
    fig, ax = figure(t, 9.2, 4.6)
    style(ax, t, xgrid=True)
    for y, p in enumerate(names):
        row = best[p]
        ax.barh(y, row["datasheet_tflops"], height=0.62, color=t["grid"], zorder=1)
        ax.barh(y, row["tflops"], height=0.62, color=SLOTS[theme][p], zorder=2)
        if row.get("clock_adjusted_peak"):
            ax.plot([row["clock_adjusted_peak"]] * 2, [y - 0.36, y + 0.36], color=t["ink2"], lw=1.6, zorder=3)
        ax.text(max(row["tflops"], row["datasheet_tflops"]) + 4, y,
                f"{row['tflops']:.0f} / {row['datasheet_tflops']:.0f}  ({row['pct_of_datasheet']:.0f}%)"
                f"   {row.get('pct_of_clock_adjusted', 0):.0f}% at {row.get('sm_mhz_median', 0):.0f} MHz",
                va="center", fontsize=8.5, color=t["ink2"])
    ax.set_yticks(range(len(names)), [LABELS[p] for p in names], fontsize=9.5, color=t["ink"])
    ax.set_xlim(0, 390)
    ax.set_xlabel("Dense TFLOPS (TOPS for INT8)", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.17)
    title(fig, t, "Measured GEMM peak vs NVIDIA L4 datasheet",
          "Best square GEMM per precision. Track = datasheet dense peak (2040 MHz);"
          " tick = peak at the clock held under the 72 W cap.")
    footer(fig, t, "Measured with torch matmul / _scaled_mm / _int_mm (cuBLAS) and our Triton FP8 kernel;"
                   " do_bench median, 1.5 s per case. L4 power cap 72 W.")
    return fig


def size_sweep(data, t, theme):
    fig, ax = figure(t, 9.2, 4.8)
    style(ax, t, xgrid=False, ygrid=True)
    for p in ORDER:
        rows = sorted((r for r in data["square"] if r["precision"] == p), key=lambda r: r["m"])
        if not rows or p == "fp16":  # FP16 tracks BF16 within a few percent; named in the subtitle
            continue
        xs, ys = [r["m"] for r in rows], [r["tflops"] for r in rows]
        ax.plot(xs, ys, color=SLOTS[theme][p], lw=2, marker="o", ms=5, zorder=3)
        ax.text(xs[-1] * 1.12, ys[-1], LABELS[p], va="center", fontsize=8.5, color=t["ink2"])
    ax.set_xscale("log", base=2)
    sizes = sorted({r["m"] for r in data["square"]})
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 3.2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Matrix size M = N = K", color=t["ink2"], fontsize=9)
    ax.set_ylabel("TFLOPS", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.14)
    title(fig, t, "GEMM throughput vs matrix size",
          "Small GEMMs cannot fill 58 SMs; throughput flattens from 4096. FP16 matches BF16 (not drawn).")
    footer(fig, t, "Datasheet dense peaks: FP32 30, TF32 60, BF16/FP16 121, FP8/INT8 242.")
    return fig


def model_roofline(data, t, theme):
    fig, ax = figure(t, 9.2, 4.8)
    style(ax, t, xgrid=False, ygrid=True)
    bw = data["datasheet"]["memory_gbps"] * 1e9
    rows_set = sorted({r["m"] for r in data["model"]})
    shapes = {r["shape"]: (r["n"], r["k"]) for r in data["model"]}
    for p, wbytes, peak_key in (("bf16", 2, "bf16"), ("fp8", 1, "fp8")):
        # Roofline for one layer: weights read once, activations in, bf16 out.
        bound = []
        for m in rows_set:
            flops = sum(2 * m * n * k * LAYER[s] for s, (n, k) in shapes.items())
            nbytes = sum((n * k * wbytes + m * k * 2 + m * n * 2) * LAYER[s] for s, (n, k) in shapes.items())
            bound.append(min(data["datasheet"][peak_key] * 1e12, flops / (nbytes / bw)) / 1e12)
        ax.plot(rows_set, bound, color=SLOTS[theme][p], lw=1.2, ls="--", zorder=2)
        ax.text(rows_set[-1] * 1.12, bound[-1], f"{LABELS[p].split()[0]} roofline", va="center",
                fontsize=8, color=t["muted"])
    for p in ("bf16", "fp8", "fp8_ours"):
        xs, ys = [], []
        for m in rows_set:
            cases = [r for r in data["model"] if r["precision"] == p and r["m"] == m]
            if not cases:
                continue
            flops = sum(2 * r["m"] * r["n"] * r["k"] * LAYER[r["shape"]] for r in cases)
            ms = sum(r["ms"] * LAYER[r["shape"]] for r in cases)
            xs.append(m)
            ys.append(flops / (ms * 1e9))
        ax.plot(xs, ys, color=SLOTS[theme][p], lw=2, marker="o", ms=5, zorder=3)
        ax.text(xs[-1] * 1.12, ys[-1], LABELS[p], va="center", fontsize=8.5, color=t["ink2"])
    ax.axvspan(rows_set[0] / 1.2, 64 * 1.4, color=t["grid"], alpha=0.5, zorder=0, lw=0)
    ax.text(rows_set[0], ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1, "decode\n(memory-bound)",
            fontsize=8, color=t["muted"], va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rows_set, [f"{s:,}" for s in rows_set])
    ax.set_xlim(rows_set[0] / 1.2, rows_set[-1] * 3.2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Tokens per forward (GEMM rows M)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("TFLOPS (one decoder layer)", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.14)
    title(fig, t, "Qwen3-4B projections on the L4: roofline",
          "Solid = measured q, k, v, o, gate, up, down per layer. Dashed = roofline"
          " (300 GB/s weight reads, datasheet compute peak).")
    footer(fig, t, "Ours includes activation quantization (as the engine runs it); cuBLAS FP8 uses per-tensor scales.")
    return fig


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    data = json.loads(Path(argv[0]).read_text())
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    charts = {"peak_vs_datasheet": peak_vs_datasheet, "size_sweep": size_sweep}
    if data.get("model"):
        charts["model_roofline"] = model_roofline
    for theme, t in THEMES.items():
        for name, make in charts.items():
            fig = make(data, t, theme)
            path = out / f"{name.replace('_', '-')}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
