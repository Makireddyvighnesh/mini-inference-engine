"""Render the prefill pass-size chart from run_prefill_pass_size (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_prefill_pass_charts.py \
      minillm_l4/results/prefill_pass_size_20261026/prefill_pass_size.json minillm_l4/docs/assets/prefill_pass_size

One figure, three panels over the same x axis (tokens in one pass):
speed (tokens/s), pass time (everyone else waits this long), extra memory.
The shaded band is the sweet spot: at least 95% of the best speed while the
pass stays under the engine's 400 ms step target. Dashed: the 16k step cap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, figure, style  # noqa: E402

CAP = 16384
TARGETS = ((150, "150 ms target (others are generating)"), (400, "400 ms target (GPU idle)"))
SLOTS = {"light": ["#2a78d6", "#eb6834"], "dark": ["#3987e5", "#d95926"]}
MODES = (("packed_512", "Many 512-token prompts", 0), ("one_prompt", "One long prompt", 1))


def series(data, mode, key):
    rows = sorted((c for c in data["cases"] if c["mode"] == mode and c["status"] == "ok"), key=lambda c: c["tokens"])
    return [c["tokens"] for c in rows], [c[key] for c in rows]


def sweet_spot(data):
    """Smallest and largest packed pass size with >= 95% of peak speed and time <= 400 ms."""

    xs, speed = series(data, "packed_512", "tokens_per_s")
    _, times = series(data, "packed_512", "pass_ms")
    best = max(speed)
    good = [x for x, s, t in zip(xs, speed, times) if s >= 0.95 * best and t <= 400]
    return (min(good), max(good), best) if good else (None, None, best)


def chart(data, t, theme):
    colors = SLOTS[theme]
    fig, axes = figure(t, 11.5, 4.6, ncols=3)
    lo, hi, best = sweet_spot(data)
    sizes = sorted({c["tokens"] for c in data["cases"]})
    for ax in axes:
        style(ax, t, xgrid=False, ygrid=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes[::2], [f"{s // 1024}k" if s >= 1024 else str(s) for s in sizes[::2]], fontsize=8)
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
        if lo:
            ax.axvspan(lo / 1.15, hi * 1.15, color=t["grid"], alpha=0.7, lw=0, zorder=0)
        ax.axvline(CAP, color=t["ink2"], ls="--", lw=1.1, zorder=1)
        ax.set_xlabel("tokens in one pass", color=t["ink2"], fontsize=8.5)

    # 1. Speed.
    ax = axes[0]
    for mode, label, slot in MODES:
        xs, ys = series(data, mode, "tokens_per_s")
        ax.plot(xs, [y / 1000 for y in ys], color=colors[slot], marker="o", ms=5, lw=2, label=label, zorder=3)
    ax.set_ylim(0, best / 1000 * 1.25)
    ax.set_title("1. Speed: flat after ~1k tokens", fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel("thousand tokens per second", color=t["ink2"], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=t["ink2"], loc="lower right")
    ax.text(CAP / 1.1, best / 1000 * 1.18, "16k cap", ha="right", fontsize=7.5, color=t["ink2"])

    # 2. Pass time = how long everyone else waits.
    ax = axes[1]
    for mode, label, slot in MODES:
        xs, ys = series(data, mode, "pass_ms")
        ax.plot(xs, [y / 1000 for y in ys], color=colors[slot], marker="o", ms=5, lw=2, zorder=3)
    for ms, label in TARGETS:
        ax.axhline(ms / 1000, color=t["muted"], lw=1, ls=":")
        ax.text(sizes[0] / 1.3, ms / 1000 * 1.12, label, fontsize=7.5, color=t["muted"])
    ax.set_yscale("log")
    ax.set_title("2. Others wait for the whole pass", fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel("pass time (seconds)", color=t["ink2"], fontsize=8.5)
    cap_rows = [c for c in data["cases"] if c["tokens"] == CAP and c["mode"] == "packed_512" and c["status"] == "ok"]
    if cap_rows:
        y = cap_rows[0]["pass_ms"] / 1000
        ax.annotate(f"{y:.1f} s at 16k", (CAP, y), textcoords="offset points", xytext=(-8, 8), ha="right",
                    fontsize=8, color=t["ink"])

    # 3. Extra memory.
    ax = axes[2]
    for mode, label, slot in MODES:
        xs, ys = series(data, mode, "extra_memory_bytes")
        ax.plot(xs, [y / 2**30 for y in ys], color=colors[slot], marker="o", ms=5, lw=2, zorder=3)
    ax.set_ylim(0, None)
    ax.set_title("3. Extra memory grows with the pass", fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel("GiB on top of weights and KV cache", color=t["ink2"], fontsize=8.5)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.14, wspace=0.28)
    fig.text(0.01, 0.975, "How big should one prefill pass be? (Qwen3-4B FP8, one L4)", ha="left", va="top",
             fontsize=13, fontweight="bold", color=t["ink"])
    spot = f"Shaded: sweet spot, {lo:,}–{hi:,} tokens (full speed, pass under 400 ms). " if lo else ""
    fig.text(0.01, 0.905, spot + "Dashed: the engine's 16k cap per pass, a safety limit, not a speed choice.",
             ha="left", va="top", fontsize=9.5, color=t["ink2"])
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
        fig = chart(data, t, theme)
        path = out / f"prefill-pass-size-{theme}.svg"
        fig.savefig(path, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
