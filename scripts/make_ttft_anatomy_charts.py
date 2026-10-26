"""Render Q1 TTFT charts from run_ttft_anatomy (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_ttft_anatomy_charts.py \
      minillm_l4/results/ttft_anatomy_20261026/ttft_anatomy.json minillm_l4/docs/assets/ttft_anatomy

Charts:
* ttft-heatmap  - median TTFT for every burst size x prompt length;
* ttft-anatomy  - mean TTFT split into compute / wait / gaps / memory queue,
                  against a strictly serial FIFO reference;
* ttft-timeline - per-request Gantt of two bursts: what each request is doing
                  from arrival to first token.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, figure, style  # noqa: E402

PARTS = (("compute", "Own prefill steps", 0), ("gaps", "Gaps between its chunks", 1),
         ("wait", "Admitted, waiting for first chunk", 2), ("queue", "Queued for KV memory", 3))
SLOTS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}
RAMP = {"light": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "dark": ["#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]}


def ms(value: float) -> str:
    return f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.0f} ms"


def heading(fig, t, text, sub):
    fig.text(0.01, 0.975, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def footer(fig, t, text):
    fig.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=8, color=t["muted"])


def case(data, batch, length):
    return next(c for c in data["cases"] if c["batch"] == batch and c["prompt_tokens"] == length)


def heatmap(data, t, theme):
    batches = sorted({c["batch"] for c in data["cases"]})
    lengths = sorted({c["prompt_tokens"] for c in data["cases"]})
    grid = [[case(data, b, n)["ttft_p50_ms"] for n in lengths] for b in batches]
    fig, ax = figure(t, 9.2, 5.0)
    cmap = LinearSegmentedColormap.from_list("seq", RAMP[theme])
    norm = LogNorm(vmin=min(map(min, grid)), vmax=max(map(max, grid)))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    for i, b in enumerate(batches):
        for j, n in enumerate(lengths):
            c = case(data, b, n)
            value = grid[i][j]
            dark_cell = norm(value) > 0.55 if theme == "light" else norm(value) < 0.45
            ax.text(j, i, ms(value) + (" *" if c["pool_limited"] else ""), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark_cell else "#0b0b0b")
    ax.set_xticks(range(len(lengths)), [f"{n:,}" for n in lengths], color=t["ink2"], fontsize=9)
    ax.set_yticks(range(len(batches)), [str(b) for b in batches], color=t["ink2"], fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Prompt tokens (every request in the burst)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Requests arriving together", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.14)
    heading(fig, t, "Time to first token: median request in a burst",
            "All requests arrive at once; the engine has to prefill every prompt before the last one gets a token.")
    footer(fig, t, f"* burst exceeds the KV pool ({data['kv_pool_tokens']:,} tokens), so some requests queue for memory. "
                   "Mixed batching, adaptive chunks, CUDA Graphs; Qwen3-4B FP8 on one L4.")
    return fig


def anatomy(data, t, theme):
    batches = sorted({c["batch"] for c in data["cases"]})
    lengths = [n for n in (128, 512, 2048, 8192) if any(c["prompt_tokens"] == n for c in data["cases"])]
    fig, axes = figure(t, 11, 5.0, ncols=len(lengths))
    colors = SLOTS[theme]
    for ax, length in zip(axes, lengths):
        style(ax, t, xgrid=False, ygrid=True)
        single = case(data, 1, length)["ttft_mean_ms"]
        for x, b in enumerate(batches):
            c, bottom = case(data, b, length), 0.0
            for key, _, slot in PARTS:
                value = c["mean_ms"][key] + (c["mean_ms"]["emit"] if key == "compute" else 0.0)
                ax.bar(x, max(value, 0) / 1000, bottom=bottom, width=0.7, color=colors[slot],
                       edgecolor=t["surface"], lw=1)
                bottom += max(value, 0) / 1000
        ax.plot(range(len(batches)), [single * (b + 1) / 2 / 1000 for b in batches], color=t["ink"], lw=1.4, ls="--",
                marker="o", ms=3, zorder=4)
        ax.set_xticks(range(len(batches)), [str(b) for b in batches], fontsize=8.5)
        ax.set_title(f"{length:,}-token prompts", fontsize=10, color=t["ink"], loc="left")
        ax.set_xlabel("requests in the burst", color=t["ink2"], fontsize=8.5)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Mean time to first token (s)", color=t["ink2"], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[slot]) for _, _, slot in PARTS]
    handles.append(plt.Line2D([0], [0], color=t["ink"], lw=1.4, ls="--"))
    labels = [label for _, label, _ in PARTS] + ["Serial FIFO: (B+1)/2 x single-request TTFT"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=5, frameon=False, fontsize=8,
               labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.14, wspace=0.22)
    heading(fig, t, "Where time to first token goes",
            "Mean over every request in the burst. Each panel has its own scale.")
    footer(fig, t, "Below the dashed line: the engine overlaps prompts (several short prompts share one forward). "
                   "On it: prompts are compute-bound and are served one after another.")
    return fig


def timeline(data, t, theme):
    picks = [(8, 2048), (32, 8192)]
    picks = [p for p in picks if any(c["batch"] == p[0] and c["prompt_tokens"] == p[1] for c in data["cases"])]
    fig, axes = figure(t, 11, 6.4, ncols=len(picks))
    axes = axes if isinstance(axes, (list, tuple)) or hasattr(axes, "__len__") else [axes]
    colors = SLOTS[theme]
    for ax, (batch, length) in zip(axes, picks):
        style(ax, t, xgrid=True, ygrid=False)
        c = case(data, batch, length)
        rows = sorted(c["timeline"].items(), key=lambda kv: kv[1]["first_token"])
        for y, (_, tl) in enumerate(rows):
            a, adm, first = tl["arrival"] / 1000, tl["admission"] / 1000, tl["first_token"] / 1000
            start = tl["chunks"][0][0] / 1000
            if adm - a > 0.005:
                ax.barh(y, adm - a, left=a, height=0.7, color=colors[3])
            ax.barh(y, start - adm, left=adm, height=0.7, color=colors[2])
            ax.barh(y, first - start, left=start, height=0.7, color=t["grid"])  # gaps background
            for begin, end, _ in tl["chunks"]:
                ax.barh(y, (end - begin) / 1000, left=begin / 1000, height=0.7, color=colors[0], lw=0)
            ax.plot([first], [y], marker="|", ms=8, color=t["ink"], mew=1.6)
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(0, None)
        ax.set_xlabel("seconds since the burst arrived", color=t["ink2"], fontsize=8.5)
        ax.set_ylabel("requests (sorted by first token)", color=t["ink2"], fontsize=8.5)
        ax.set_title(f"{batch} x {length:,}-token prompts  (median TTFT {ms(c['ttft_p50_ms'])})", fontsize=10,
                     color=t["ink"], loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in (3, 2, 0)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=t["grid"]), plt.Line2D([0], [0], marker="|", ls="", ms=8, mew=1.6,
                                                                        color=t["ink"])]
    fig.legend(handles, ["Queued for KV memory", "Admitted, waiting", "Prefill chunk running",
                         "Other work between its chunks", "First token"],
               loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=5, frameon=False, fontsize=8,
               labelcolor=t["ink2"], handlelength=1.2)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.1, wspace=0.12)
    heading(fig, t, "A burst, request by request",
            "Each row is one request from arrival to its first token (last measured run).")
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
    for theme, t in THEMES.items():
        for name, make in (("ttft-heatmap", heatmap), ("ttft-anatomy", anatomy), ("ttft-timeline", timeline)):
            fig = make(data, t, theme)
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
