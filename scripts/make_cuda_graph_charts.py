"""Render eager/graph comparisons as light and dark SVG charts.

  .conda-env/bin/python minillm_l4/scripts/make_cuda_graph_charts.py \
      minillm_l4/results/cuda_graphs_20261008/cuda_graphs.json minillm_l4/docs/assets/cuda_graphs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm_l4.scripts.make_showcase_charts import (  # noqa: E402
    FONT, THEMES, figure, hbars, plt, save, style, title,
)

LABELS = {"eager": "Eager", "graph": "Graph", "graph_split": "Graph + split"}


def legend(ax, theme, **placement):
    result = ax.legend(frameon=False, fontsize=9, **placement)
    for text in result.get_texts():
        text.set_color(theme["ink2"])


def render(data, out: Path):
    sections = data["sections"]
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    for name, theme in THEMES.items():
        if sections.get("decode"):
            fig, ax = figure(theme, 8, 4.1)
            rows = sections["decode"]
            for index, mode in enumerate(("eager", "graph")):
                points = sorted((r["batch"], r["tpot_p50_ms"]) for r in rows
                                if r["mode"] == mode and r["prompt_tokens"] == 512 and r["tpot_p50_ms"] is not None)
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2,
                            color=theme["series"][index], label=LABELS[mode])
                long = [r for r in rows if r["mode"] == mode and r["prompt_tokens"] == 4096 and r["tpot_p50_ms"] is not None]
                if long:
                    ax.scatter([r["batch"] for r in long], [r["tpot_p50_ms"] for r in long],
                               color=theme["series"][index], marker="D", s=64,
                               label=f"{LABELS[mode]} · 4096-token prompt", zorder=4)
            batches = sorted({r["batch"] for r in rows})
            ax.set_xscale("log", base=2)
            ax.set_xticks(batches, [str(b) for b in batches])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("requests decoded together", color=theme["ink2"])
            ax.set_ylabel("TPOT p50 (ms)", color=theme["ink2"])
            style(ax, theme, xgrid=False, ygrid=True)
            # Bottom right is empty: graph TPOT stays above ~22 ms there.
            legend(ax, theme, loc="lower right", ncol=2)
            title(fig, theme, "Decode latency by batch size", "512-token prompts, 128 outputs; diamonds: 4096-token prompt at batch 8")
            fig.subplots_adjust(top=0.78)
            save(fig, out, "decode-tpot", name)

        for workload, heading in (("serving", "Serving 16 requests every 150 ms"),
                                  ("longmix", "Serving alternating short and long prompts")):
            rows = [r for r in sections.get("serving", []) if r["workload"] == workload]
            if not rows:
                continue
            rows.sort(key=lambda r: (("whole", "mixed_adaptive").index(r["policy"]),
                                    ("eager", "graph", "graph_split").index(r["mode"])))
            labels = [("Continuous" if r["policy"] == "whole" else "Mixed + adaptive") + " · " + LABELS[r["mode"]] for r in rows]
            fig, axes = figure(theme, 11, 4.3, ncols=2)
            for ax, metric, axis_label, fmt in (
                (axes[0], "tpot_p50_ms", "TPOT p50 (ms)", lambda v: f"{v:.1f} ms"),
                (axes[1], "output_tokens_per_s", "output tokens/s (end to end)", lambda v: f"{v:,.0f}"),
            ):
                valid = [(i, r) for i, r in enumerate(rows) if r[metric] is not None]
                if valid:
                    hbars(ax, theme, [labels[i] for i, _ in valid], [r[metric] for _, r in valid], fmt, theme["series"][0])
                    for patch, (_, row) in zip(ax.patches, valid):
                        patch.set_facecolor(theme["series"][("eager", "graph", "graph_split").index(row["mode"])])
                ax.set_xlabel(axis_label, color=theme["ink2"], fontsize=9)
            axes[1].set_yticks([])
            title(fig, theme, heading, "One warmup, three measured runs · eager, graph decode, and graph decode with separate prompt work")
            fig.subplots_adjust(top=0.78, left=0.27, wspace=0.13)
            save(fig, out, f"{workload}-bars", name)

        if sections.get("prefill"):
            fig, ax = figure(theme, 8, 4.1)
            rows = sections["prefill"]
            for index, mode in enumerate(("eager", "graph")):
                points = sorted((r["prompt_tokens"], r["ttft_p50_ms"]) for r in rows
                                if r["mode"] == mode and r["ttft_p50_ms"] is not None)
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2,
                            linestyle="--" if mode == "graph" else "-", color=theme["series"][index],
                            label="Graphs enabled (no decode replay)" if mode == "graph" else "Eager")
            lengths = sorted({r["prompt_tokens"] for r in rows})
            ax.set_xscale("log", base=2)
            ax.set_xticks(lengths, [f"{n:,}" for n in lengths])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("prompt tokens", color=theme["ink2"])
            ax.set_ylabel("TTFT p50 (ms)", color=theme["ink2"])
            style(ax, theme, xgrid=False, ygrid=True)
            legend(ax, theme)
            title(fig, theme, "Prefill time with graphs enabled", "One request, one output token · prefill is eager in both cases")
            fig.subplots_adjust(top=0.78)
            save(fig, out, "prefill-ttft", name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    render(json.loads(args.summary.read_text()), args.output_dir)
    print(f"Charts written to {args.output_dir}")


if __name__ == "__main__":
    main()
