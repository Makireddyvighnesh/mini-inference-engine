"""Render README charts from a showcase run (light and dark SVG per chart).

  .conda-env/bin/python minillm_l4/scripts/make_showcase_charts.py \
      minillm_l4/results/showcase_20261008/showcase.json minillm_l4/docs/assets/showcase
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
FONT = ["system-ui", "-apple-system", "Segoe UI", "DejaVu Sans", "sans-serif"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def style(ax, t, *, xgrid=True, ygrid=False):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_color(t["ink2"])
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t, width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=t["surface"])
    return fig, axes


def title(fig, t, text, sub):
    fig.text(0.01, 0.97, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.885, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def best(values, higher_is_better):
    """Index of the best bar, so the highlight marks the winner, not a fixed row."""
    pick = max if higher_is_better else min
    return values.index(pick(values))


def hbars(ax, t, labels, values, fmt, color, highlight=None):
    y = list(range(len(labels)))[::-1]
    colors = [color if highlight is None or i == highlight else t["muted"] for i in range(len(labels))]
    ax.barh(y, values, height=0.62, color=colors, linewidth=0)
    ax.set_yticks(y, labels)
    top = max(values)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.015, yi, fmt(v), va="center", ha="left", fontsize=9, color=t["ink"])
    ax.set_xlim(0, top * 1.22)
    style(ax, t)


def save(fig, out: Path, name: str, theme: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}-{theme}.svg", facecolor=fig.get_facecolor(), bbox_inches="tight")
    if os.environ.get("PREVIEW_PNG"):
        fig.savefig(out / f"{name}-{theme}.png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    sections = data["sections"]
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    for theme, t in THEMES.items():
        if all(key in sections for key in ("single", "serving", "prefix", "longmix")):
            indexed = {key: {r["engine"]: r for r in sections[key]}
                       for key in ("single", "serving", "prefix", "longmix")}
            pause = "worst_gap_run_median_ms" if "worst_gap_run_median_ms" in indexed["longmix"]["whole"] else "worst_gap_ms"
            comparisons = (
                ("KV cache\ntime per token", "single", "no_kv_cache", "kv_cache", "tpot_p50_ms"),
                ("Continuous batching\ntime to first token", "serving", "static", "continuous_dense", "ttft_p50_ms"),
                ("Paged KV\nthroughput under load", "serving", "continuous_dense", "continuous_paged", "tokens_per_s"),
                ("CUDA Graphs\ntime per token", "single", "hf_generate", "cuda_graph", "tpot_p50_ms"),
                ("Prefix cache\ntime to first token", "prefix", "off", "on", "ttft_p50_ms"),
                ("Mixed batching + adaptive\nworst decode pause", "longmix", "whole", "adaptive", pause),
            )
            labels, gains = [], []
            for label, key, before, after, metric in comparisons:
                without, with_ = indexed[key][before][metric], indexed[key][after][metric]
                labels.append(label)
                gains.append(with_ / without if metric == "tokens_per_s" else without / with_)
            fig, ax = figure(t, 9, 4.6)
            hbars(ax, t, labels, gains, lambda v: f"{v:.1f}x", t["series"][0])
            ax.set_xlabel("gain (x): lower latency or higher throughput", color=t["ink2"], fontsize=9)
            title(fig, t, "Speedup from each technique", "Matched comparisons from one NVIDIA L4 session")
            fig.subplots_adjust(top=0.78, left=0.35)
            save(fig, out, "speedups", theme)

        if "single" in sections:
            rows = sections["single"]
            fig, ax = figure(t, 8, 3.6)
            values = [1000.0 / r["tpot_p50_ms"] for r in rows]
            hbars(ax, t, [r["label"] for r in rows], values, lambda v: f"{v:.1f} tok/s", t["series"][0],
                  highlight=best(values, True))
            ax.set_xlabel("decode tokens/s (one request, 512-token prompt)", color=t["ink2"], fontsize=9)
            title(fig, t, "Decode speed by engine", "Each engine generates 128 tokens for the same prompt")
            fig.subplots_adjust(top=0.78, left=0.3)
            save(fig, out, "decode-engines", theme)

        if "batch" in sections:
            rows = sections["batch"]
            fig, ax = figure(t, 8, 4.2)
            families = []
            for r in rows:
                if r["family"] not in families:
                    families.append(r["family"])
            ends = []
            for color, family in zip(t["series"], families):
                pts = sorted((r["batch"], r["tokens_per_s"], r["label"]) for r in rows if r["family"] == family)
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", label=pts[0][2])
                ax.scatter(xs, ys, s=42, color=color, edgecolors=t["surface"], linewidths=2, zorder=3)
                ends.append([ys[-1], ys[-1], f"{pts[0][2]}  {ys[-1]:.0f}", xs[-1]])
            # Spread end labels so near-equal lines do not overprint each other.
            gap = max(e[0] for e in ends) * 0.065
            ends.sort(key=lambda e: e[0])
            for lower, upper in zip(ends, ends[1:]):
                upper[1] = max(upper[1], lower[1] + gap)
            for _, y, text, x in ends:
                ax.text(x * 1.06, y, text, va="center", fontsize=9, color=t["ink"])
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
            ax.set_xlim(0.8, 16 * 4.5)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("requests decoded together", color=t["ink2"], fontsize=9)
            ax.set_ylabel("output tokens/s", color=t["ink2"], fontsize=9)
            style(ax, t, xgrid=False, ygrid=True)
            legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
            for text in legend.get_texts():
                text.set_color(t["ink2"])
            title(fig, t, "Throughput as batch size grows", "512-token prompts, 128 output tokens each")
            fig.subplots_adjust(top=0.8)
            save(fig, out, "batch-scaling", theme)

        for key, name, head, sub in (
            ("serving", "serving", "Serving 16 requests arriving every 150 ms",
             "128-1024-token prompts, 128 output tokens each; every policy may run all 16 at once"),
            ("longmix", "long-prompts", "Long prompts arriving while others decode",
             "6 requests every 400 ms, alternating 128 and 6144-token prompts"),
        ):
            if key not in sections:
                continue
            rows = sections[key]
            labels = [r["label"] for r in rows]
            fig, axes = figure(t, 10, 3.4, ncols=2)
            worst = "worst_gap_run_median_ms" if "worst_gap_run_median_ms" in rows[0] else "worst_gap_ms"
            second = (worst, "worst pause between tokens (ms, median of runs)", lambda v: f"{v:,.0f} ms") if key == "longmix" \
                else ("tokens_per_s", "output tokens/s", lambda v: f"{v:,.0f}")
            ttfts = [r["ttft_p50_ms"] / 1000 for r in rows]
            hbars(axes[0], t, labels, ttfts, lambda v: f"{v:.2f} s", t["series"][0], highlight=best(ttfts, False))
            axes[0].set_xlabel("median time to first token (s)", color=t["ink2"], fontsize=9)
            seconds = [r[second[0]] for r in rows]
            hbars(axes[1], t, labels, seconds, second[2], t["series"][1],
                  highlight=best(seconds, second[0] == "tokens_per_s"))
            axes[1].set_yticks([])
            axes[1].set_xlabel(second[1], color=t["ink2"], fontsize=9)
            title(fig, t, head, sub)
            fig.subplots_adjust(top=0.76, left=0.24, wspace=0.08)
            save(fig, out, name, theme)

        if "prefix" in sections:
            rows = sections["prefix"]
            fig, ax = figure(t, 8, 2.4)
            prefix_ttft = [r["ttft_p50_ms"] for r in rows]
            hbars(ax, t, [r["label"] for r in rows], prefix_ttft,
                  lambda v: f"{v:,.0f} ms", t["series"][2], highlight=best(prefix_ttft, False))
            ax.set_xlabel("median time to first token (ms)", color=t["ink2"], fontsize=9)
            title(fig, t, "Prefix caching", "8 requests sharing a 2048-token system prompt")
            fig.subplots_adjust(top=0.68, left=0.22)
            save(fig, out, "prefix-cache", theme)

        if "prefill" in sections:
            rows = sorted(sections["prefill"], key=lambda r: r["prompt_tokens"])
            xs = [r["prompt_tokens"] for r in rows]
            ys = [r["ttft_p50_ms"] for r in rows]
            fig, ax = figure(t, 8, 3.6)
            ax.plot(xs, ys, color=t["series"][0], linewidth=2, solid_capstyle="round")
            ax.scatter(xs, ys, s=42, color=t["series"][0], edgecolors=t["surface"], linewidths=2, zorder=3)
            for x, y in zip(xs, ys):
                ax.text(x, y + max(ys) * 0.05, f"{y:,.0f} ms", ha="center", fontsize=8.5, color=t["ink"])
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [f"{x:,}" for x in xs])
            ax.set_ylim(0, max(ys) * 1.2)
            ax.set_xlabel("prompt tokens", color=t["ink2"], fontsize=9)
            ax.set_ylabel("time to first token (ms)", color=t["ink2"], fontsize=9)
            style(ax, t, xgrid=False, ygrid=True)
            title(fig, t, "Prefill time by prompt length", "One request; time until its first token")
            fig.subplots_adjust(top=0.8)
            save(fig, out, "prefill-ttft", theme)
    print(f"Charts written to {out}")


if __name__ == "__main__":
    main()
