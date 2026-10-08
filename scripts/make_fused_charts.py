"""Render fused-kernel comparison charts (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_fused_charts.py \
      minillm_l4/results/fused_kernels_<date>/fused_kernels.json minillm_l4/docs/assets/fused_kernels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_showcase_charts import FONT, THEMES, figure, plt, save, style, title


def render(data: dict, out: Path) -> None:
    sections = data["sections"]
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    for name, t in THEMES.items():
        prefill = sorted(sections.get("prefill", []), key=lambda r: r["prompt_tokens"])
        if prefill:
            fig, ax = figure(t, 8, 3.8)
            for index, (fused, label) in enumerate(((False, "Unfused"), (True, "Fused"))):
                rows = [r for r in prefill if r["fused"] == fused]
                xs, ys = [r["prompt_tokens"] for r in rows], [r["ttft_p50_ms"] for r in rows]
                ax.plot(xs, ys, color=t["series"][index], linewidth=2, label=label)
                ax.scatter(xs, ys, s=42, color=t["series"][index], edgecolors=t["surface"], linewidths=2, zorder=3)
            fused_rows = {r["prompt_tokens"]: r for r in prefill if r["fused"]}
            for r in prefill:
                if not r["fused"]:
                    saved = 1 - fused_rows[r["prompt_tokens"]]["ttft_p50_ms"] / r["ttft_p50_ms"]
                    ax.text(r["prompt_tokens"], r["ttft_p50_ms"] * 1.06 + 20, f"-{saved:.0%}",
                            ha="center", fontsize=8.5, color=t["ink"])
            ax.set_xscale("log", base=2)
            ticks = [r["prompt_tokens"] for r in prefill if not r["fused"]]
            ax.set_xticks(ticks, [f"{x:,}" for x in ticks])
            ax.set_ylim(0, max(r["ttft_p50_ms"] for r in prefill) * 1.18)
            ax.set_xlabel("prompt tokens", color=t["ink2"], fontsize=9)
            ax.set_ylabel("time to first token (ms)", color=t["ink2"], fontsize=9)
            style(ax, t, xgrid=False, ygrid=True)
            legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
            for text in legend.get_texts():
                text.set_color(t["ink2"])
            title(fig, t, "Prefill with fused kernels", "One request; labels show the time saved")
            fig.subplots_adjust(top=0.8)
            save(fig, out, "prefill-ttft", name)

        decode = sections.get("decode", [])
        if decode:
            fig, ax = figure(t, 8, 3.8)
            batches = sorted({r["batch"] for r in decode})
            series = ((False, False, "Eager, unfused"), (False, True, "Eager, fused"),
                      (True, False, "CUDA Graph, unfused"), (True, True, "CUDA Graph, fused"))
            width = 0.2
            for index, (graphs, fused, label) in enumerate(series):
                values = [next(r["tpot_p50_ms"] for r in decode if r["batch"] == b and r["graphs"] == graphs
                               and r["fused"] == fused) for b in batches]
                xs = [i + (index - 1.5) * (width + 0.01) for i in range(len(batches))]
                ax.bar(xs, values, width=width, color=t["series"][index], label=label, linewidth=0)
                for x, v in zip(xs, values):
                    ax.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=t["ink"])
            ax.set_xticks(range(len(batches)), [f"{b} request{'s' if b > 1 else ''}" for b in batches])
            ax.set_ylabel("time per output token (ms)", color=t["ink2"], fontsize=9)
            style(ax, t, xgrid=False, ygrid=True)
            legend = ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
            for text in legend.get_texts():
                text.set_color(t["ink2"])
            ax.set_ylim(0, max(r["tpot_p50_ms"] for r in decode) * 1.35)
            title(fig, t, "Decode with fused kernels", "512-token prompts, 128 outputs; eager and CUDA Graph")
            fig.subplots_adjust(top=0.8)
            save(fig, out, "decode-tpot", name)
    print(f"Charts written to {out}")


if __name__ == "__main__":
    render(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
