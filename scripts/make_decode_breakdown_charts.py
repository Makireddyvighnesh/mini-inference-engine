"""Render decode-step breakdown charts from run_decode_breakdown (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_decode_breakdown_charts.py \
      minillm_l4/results/decode_breakdown_20261026/decode_breakdown.json minillm_l4/docs/assets/decode_breakdown

Charts:
* step-breakdown - eager decode step per batch (panel) and context (bar):
  GPU time by category plus the CPU launch gap; marker = CUDA Graph step;
* bytes-vs-time  - GPU kernel time per step vs bytes the step must read,
  against the measured copy bandwidth;
* kv-attention   - paged decode attention time vs cached tokens in the batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, figure, style  # noqa: E402

# Stack order = categorical slot order (bottom to top); the launch gap is not GPU work: neutral.
STACK = (
    ("weights_fp8_projections", "Weights: FP8 projections", 0),
    ("weights_lm_head", "Weights: lm_head (BF16)", 1),
    ("kv", "KV cache: attention + write", 2),
    ("misc", "Norm, RoPE, SiLU, other", 3),
)
EXTRA = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}


def segments(case):
    g = case["gpu_ms_by_category"]
    return {
        "weights_fp8_projections": g.get("weights_fp8_projections", 0.0),
        "weights_lm_head": g.get("weights_lm_head", 0.0),
        "kv": g.get("kv_attention", 0.0) + g.get("kv_write", 0.0),
        "misc": g.get("norm_rope_silu", 0.0) + g.get("other", 0.0),
        "launch": max(case["eager_launch_gap_ms"], 0.0),
    }


def heading(fig, t, text, sub):
    fig.text(0.01, 0.975, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def footer(fig, t, text):
    fig.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=8, color=t["muted"])


def step_breakdown(data, t, theme):
    batches = sorted({c["batch"] for c in data["cases"]})
    contexts = sorted({c["context"] for c in data["cases"]})
    fig, axes = figure(t, 11, 5.0, ncols=len(batches))
    colors = EXTRA[theme]
    top = max(c["eager_step_ms"] for c in data["cases"]) * 1.12
    for ax, batch in zip(axes, batches):
        style(ax, t, xgrid=False, ygrid=True)
        cases = sorted((c for c in data["cases"] if c["batch"] == batch), key=lambda c: c["context"])
        for x, case in enumerate(cases):
            seg, bottom = segments(case), 0.0
            for key, _, slot in STACK:
                ax.bar(x, seg[key], bottom=bottom, width=0.7, color=colors[slot], edgecolor=t["surface"], lw=1)
                bottom += seg[key]
            ax.bar(x, seg["launch"], bottom=bottom, width=0.7, color=t["grid"], edgecolor=t["axis"], lw=0.8,
                   hatch="////")
            ax.plot([x - 0.38, x + 0.38], [case["graph_step_ms"]] * 2, color=t["ink"], lw=2, zorder=4)
        ax.set_xticks(range(len(cases)), [f"{c:,}" for c in contexts], fontsize=8.5)
        ax.set_ylim(0, top)
        ax.set_title(f"batch {batch}", fontsize=10, color=t["ink"], loc="left")
        ax.set_xlabel("context (tokens)", color=t["ink2"], fontsize=8.5)
        if batch != batches[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("ms per decode step", color=t["ink2"], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[slot]) for _, _, slot in STACK]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=t["grid"], edgecolor=t["axis"], hatch="////"))
    handles.append(plt.Line2D([0], [0], color=t["ink"], lw=2))
    labels = [label for _, label, _ in STACK] + ["CPU launch gap (eager)", "CUDA Graph step"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=6, frameon=False,
               fontsize=8, labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.17, wspace=0.08)
    heading(fig, t, "Where a decode step goes (Qwen3-4B FP8, one L4)",
            "Eager bars = GPU kernel time by category + CPU launch gap. Black tick = the same step with CUDA Graphs.")
    footer(fig, t, f"Kernel times from torch.profiler (difference of two runs, prefill cancels). "
                   f"Weights read per step: {data['fp8_weight_bytes'] / 1e9:.2f} GB FP8 + "
                   f"{data['lm_head_bytes'] / 1e9:.2f} GB lm_head.")
    return fig


def bytes_vs_time(data, t, theme):
    fig, ax = figure(t, 9.2, 4.8)
    style(ax, t, xgrid=True, ygrid=True)
    bw = data["copy_bandwidth_gbps"]
    batches = sorted({c["batch"] for c in data["cases"]})
    colors = EXTRA[theme]
    xs = [(c["weight_bytes"] + c["kv_bytes"]) / 1e9 for c in data["cases"]]
    lo, hi = min(xs) * 0.97, max(xs) * 1.03
    ax.plot([lo, hi], [lo / bw * 1000, hi / bw * 1000], color=t["muted"], ls="--", lw=1.2, zorder=2)
    ax.text(hi, hi / bw * 1000, f"  {bw:.0f} GB/s", va="center", fontsize=8, color=t["muted"])
    for i, batch in enumerate(batches):
        cases = sorted((c for c in data["cases"] if c["batch"] == batch), key=lambda c: c["context"])
        ax.plot([(c["weight_bytes"] + c["kv_bytes"]) / 1e9 for c in cases], [c["gpu_kernel_ms"] for c in cases],
                color=colors[i], marker="o", ms=6, lw=2, zorder=3, label=f"batch {batch}")
    ax.set_xlabel("Bytes read per step: weights + KV cache (GB)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("GPU kernel time per step (ms)", color=t["ink2"], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.8, bottom=0.13)
    heading(fig, t, "Decode is a memory-bandwidth problem",
            "Each step reads every weight once plus the batch's KV cache; GPU time tracks bytes, not batch size.")
    footer(fig, t, "Points per batch: context 128, 256, 512, 1024. Dashed: time to read those bytes at the measured copy bandwidth.")
    return fig


def kv_attention(data, t, theme):
    fig, ax = figure(t, 9.2, 4.6)
    style(ax, t, xgrid=True, ygrid=True)
    bw = data["copy_bandwidth_gbps"]
    batches = sorted({c["batch"] for c in data["cases"]})
    colors = EXTRA[theme]
    for i, batch in enumerate(batches):
        cases = sorted((c for c in data["cases"] if c["batch"] == batch), key=lambda c: c["context"])
        g = [c["gpu_ms_by_category"] for c in cases]
        ax.plot([c["batch"] * c["context"] for c in cases], [x.get("kv_attention", 0) + x.get("kv_write", 0) for x in g],
                color=colors[i], marker="o", ms=6, lw=2, zorder=3, label=f"batch {batch}")
    tokens = sorted({c["batch"] * c["context"] for c in data["cases"]})
    ax.plot(tokens, [n * data["kv_bytes_per_token"] / (bw * 1e6) for n in tokens], color=t["muted"], ls="--", lw=1.2)
    ax.text(tokens[-1], tokens[-1] * data["kv_bytes_per_token"] / (bw * 1e6), "  KV bytes at copy bandwidth",
            va="center", fontsize=8, color=t["muted"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens[::2], [f"{n:,}" for n in tokens[::2]])
    ax.set_ylim(0, None)
    ax.set_xlabel("Cached tokens read per step (batch x context)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Attention + KV write (ms per step)", color=t["ink2"], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.85, top=0.8, bottom=0.13)
    heading(fig, t, "KV-cache reads grow with batch x context",
            f"36 layers x 8 KV heads x 128 dims x K and V x 2 bytes = {data['kv_bytes_per_token'] / 1024:.0f} KiB per cached token.")
    footer(fig, t, "Paged Triton decode attention (all 36 layers) plus the per-token K/V page write.")
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
    plt.rcParams["hatch.linewidth"] = 0.6
    charts = {"step-breakdown": step_breakdown, "bytes-vs-time": bytes_vs_time, "kv-attention": kv_attention}
    for theme, t in THEMES.items():
        plt.rcParams["hatch.color"] = t["axis"]
        for name, make in charts.items():
            fig = make(data, t, theme)
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
