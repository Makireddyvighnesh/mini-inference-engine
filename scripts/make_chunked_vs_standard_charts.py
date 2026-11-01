"""Render chunked vs standard prefill charts from run_chunked_vs_standard (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_chunked_vs_standard_charts.py \
      minillm_l4/results/chunked_vs_standard_20261101/chunked_vs_standard.json minillm_l4/docs/assets/chunked_vs_standard

* chunked-timeline - one row per policy (8k long prompt): every chat token's
                     wait since the previous token, over time; the long
                     prompt's prefill window is shaded;
* chunked-summary  - per long-prompt size: the chat users' longest freeze, their
                     decode speed while the long prompt is read (last run),
                     and the long prompt's own time to first token.

For a ``--workload mixed`` result it renders instead:

* chunked-mixed-timeline - every token's wait over time, all requests, one row
                           per policy, long-prompt arrivals marked;
* chunked-mixed-summary  - token waits (p99, longest), p90 TTFT by prompt size,
                           and throughput.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, style  # noqa: E402

# (policy, label, color slot); fixed order and colors.
POLICIES = (("standard", "Standard prefill (whole prompt at once)", 1),
            ("chunked_512", "Chunked prefill (512-token chunks)", 2),
            ("adaptive", "Adaptive chunked prefill (engine default)", 0))


def case(data, policy, size):
    return next(c for c in data["cases"] if c["policy"] == policy and c["long_prompt"] == size)


def decode_speed(data, c):
    """Chat tokens per second (all users) while the long prompt is read, from the last run's timeline."""

    arrival = data["long_arrival_ms"]
    first = c["timeline_last_run"]["long"][0]
    count = sum(1 for rid, times in c["timeline_last_run"].items() if rid.startswith("chat")
                for x in times if arrival < x <= first)
    return count / ((first - arrival) / 1000)


def heading(fig, t, title, sub):
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.93, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def timeline(data, t, size):
    fig, axes = plt.subplots(len(POLICIES), 1, figsize=(11.5, 7.2), facecolor=t["surface"], sharex=True, sharey=True)
    arrival = data["long_arrival_ms"] / 1000
    top = max(case(data, p, size)["chat_gap_max_ms"] for p, _, _ in POLICIES)
    for ax, (policy, label, slot) in zip(axes, POLICIES):
        style(ax, t, xgrid=False, ygrid=True)
        c = case(data, policy, size)
        first = c["timeline_last_run"]["long"][0] / 1000
        ax.axvspan(arrival, first, color=t["grid"], alpha=0.9, lw=0, zorder=0)
        ax.text(arrival + 0.05, top * 1.02, f"long prompt being read: {first - arrival:.2f} s", fontsize=8,
                color=t["ink2"], va="bottom")
        for rid, times in c["timeline_last_run"].items():
            if not rid.startswith("chat"):
                continue
            xs = [b / 1000 for b in times[1:]]
            ys = [b - a for a, b in zip(times, times[1:])]
            ax.plot(xs, ys, ls="none", marker="o", ms=2.6, color=t["series"][slot], alpha=0.75, zorder=3)
        ax.set_title(f"{label}  ·  longest freeze {c['chat_gap_max_ms']:.0f} ms", fontsize=10, color=t["ink"],
                     loc="left")
        ax.set_ylabel("wait for token (ms)", color=t["ink2"], fontsize=8.5)
        ax.set_ylim(0, top * 1.15)
    axes[-1].set_xlabel("seconds since the chat users started", color=t["ink2"], fontsize=8.5)
    axes[-1].set_xlim(0, None)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.08, hspace=0.42)
    heading(fig, t, f"What chat users feel when a {size // 1024}k-token prompt arrives",
            f"{data['chat_users']} users are generating; each dot is one of their tokens and how long they waited "
            f"for it. Shaded: the long prompt is being read.")
    return fig


def summary(data, t):
    sizes = sorted({c["long_prompt"] for c in data["cases"]})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=t["surface"])
    panels = (("chat_gap_max_ms", "Chat users: longest freeze", "milliseconds"),
              ("decode_speed", "Chat users: decode speed while it is read", "tokens per second, all 8 users"),
              ("long_ttft_ms", "Long prompt: time to its first token", "milliseconds"))
    width = 0.26
    for ax, (key, title, ylabel) in zip(axes, panels):
        style(ax, t, xgrid=False, ygrid=True)
        value_of = (lambda c: decode_speed(data, c)) if key == "decode_speed" else (lambda c: c[key])
        top = max(value_of(c) for c in data["cases"])
        for i, (policy, _, slot) in enumerate(POLICIES):
            for x, size in enumerate(sizes):
                value = value_of(case(data, policy, size))
                pos = x + (i - 1) * width
                ax.bar(pos, value, width=width * 0.92, color=t["series"][slot], zorder=3)
                text = f"{value / 1000:.1f} s" if value >= 1000 and key != "decode_speed" else f"{value:.0f}"
                ax.text(pos, value + top * 0.015, text,
                        ha="center", va="bottom", fontsize=7.5, color=t["ink"])
        ax.set_xticks(range(len(sizes)), [f"{s // 1024}k prompt" for s in sizes], fontsize=8.5)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, fontsize=10, color=t["ink"], loc="left")
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["series"][slot]) for *_, slot in POLICIES]
    fig.legend(handles, [label for _, label, _ in POLICIES], loc="upper left", bbox_to_anchor=(0.005, 0.87),
               ncol=3, frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.8)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.74, bottom=0.11, wspace=0.28)
    fig.text(0.01, 0.975, "Chunking removes long freezes; the long prompt pays with a later first token",
             ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, f"{data['chat_users']} chat users generating; one long prompt arrives. Median of 3 runs (decode speed: last run).",
             ha="left", va="top", fontsize=9.5, color=t["ink2"])
    return fig


def mixed_case(data, policy):
    return next(c for c in data["cases"] if c["policy"] == policy)


def mixed_timeline(data, t):
    """Every token's wait over time for all requests, one row per policy; long prompts marked."""

    fig, axes = plt.subplots(len(POLICIES), 1, figsize=(11.5, 7.2), facecolor=t["surface"], sharex=True, sharey=True)
    longs = [r for r in data["requests"] if r["class"] == "long"]
    top = max(mixed_case(data, p)["gap_max_ms"] for p, _, _ in POLICIES)
    for ax, (policy, label, slot) in zip(axes, POLICIES):
        style(ax, t, xgrid=False, ygrid=True)
        c = mixed_case(data, policy)
        for r in longs:
            ax.axvline(r["arrival_ms"] / 1000, color=t["ink2"], lw=0.8, ls=":", zorder=1)
        for times in c["timeline_last_run"].values():
            xs = [b / 1000 for b in times[1:]]
            ys = [b - a for a, b in zip(times, times[1:])]
            ax.plot(xs, ys, ls="none", marker="o", ms=2.2, color=t["series"][slot], alpha=0.6, zorder=3)
        ax.set_title(f"{label}  ·  longest freeze {c['gap_max_ms']:.0f} ms  ·  "
                     f"{c['gaps_over_500ms']} waits over 0.5 s", fontsize=10, color=t["ink"], loc="left")
        ax.set_ylabel("wait for token (ms)", color=t["ink2"], fontsize=8.5)
        ax.set_ylim(0, top * 1.1)
    axes[-1].set_xlabel("seconds since the first request", color=t["ink2"], fontsize=8.5)
    axes[-1].set_xlim(0, None)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.08, hspace=0.42)
    heading(fig, t, f"Real-world mix: {len(data['requests'])} requests, short to 16k-token prompts",
            "Each dot is one token of any request and how long it waited. Dotted lines: a long (8k–16k) "
            "prompt arrives.")
    return fig


def mixed_summary(data, t):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=t["surface"])
    classes = ("short", "medium", "long")
    width = 0.26

    def grouped(ax, groups, value_of, fmt):
        top = max(value_of(mixed_case(data, p), g) for p, _, _ in POLICIES for g in groups)
        for i, (policy, _, slot) in enumerate(POLICIES):
            for x, g in enumerate(groups):
                value = value_of(mixed_case(data, policy), g)
                pos = x + (i - 1) * width
                ax.bar(pos, value, width=width * 0.92, color=t["series"][slot], zorder=3)
                ax.text(pos, value + top * 0.015, fmt(value), ha="center", va="bottom", fontsize=7.5, color=t["ink"])
        ax.set_ylim(0, top * 1.15)

    ms = lambda v: f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f}"  # noqa: E731
    ax = axes[0]
    style(ax, t, xgrid=False, ygrid=True)
    grouped(ax, ("gap_p99_ms", "gap_max_ms"), lambda c, k: c[k], ms)
    ax.set_xticks([0, 1], ["99th percentile wait", "longest freeze"], fontsize=8.5)
    ax.set_title("Waits between tokens (all requests)", fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel("milliseconds", color=t["ink2"], fontsize=8.5)

    ax = axes[1]
    style(ax, t, xgrid=False, ygrid=True)
    grouped(ax, classes, lambda c, g: c[f"ttft_{g}_p90_ms"], ms)
    labels = {"short": "short (128–1k)", "medium": "medium (1k–4k)", "long": "long (8k–16k)"}
    ax.set_xticks(range(len(classes)), [labels[g] for g in classes], fontsize=8.5)
    ax.set_title("Time to first token, slowest 10% (p90), by prompt size", fontsize=10, color=t["ink"],
                 loc="left")
    ax.set_ylabel("milliseconds", color=t["ink2"], fontsize=8.5)

    ax = axes[2]
    style(ax, t, xgrid=False, ygrid=True)
    grouped(ax, ("throughput_tok_s",), lambda c, k: c[k], lambda v: f"{v:.0f}")
    ax.set_xticks([0], ["output tokens per second"], fontsize=8.5)
    ax.set_title("Throughput (whole run)", fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel("tokens per second", color=t["ink2"], fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=t["series"][slot]) for *_, slot in POLICIES]
    fig.legend(handles, [label for _, label, _ in POLICIES], loc="upper left", bbox_to_anchor=(0.005, 0.87),
               ncol=3, frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.8)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.74, bottom=0.11, wspace=0.28)
    fig.text(0.01, 0.975, f"Real-world mix of {len(data['requests'])} requests: chunked vs standard prefill",
             ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, f"Poisson arrivals ({data['rate_per_s']}/s); prompts 60% short, 30% medium, 10% long; "
             "one measured run per policy.", ha="left", va="top", fontsize=9.5, color=t["ink2"])
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
    if data.get("workload") == "mixed":
        figures = lambda t: (("chunked-mixed-timeline", mixed_timeline(data, t)),  # noqa: E731
                             ("chunked-mixed-summary", mixed_summary(data, t)))
    else:
        sizes = sorted({c["long_prompt"] for c in data["cases"]})
        size = 8192 if 8192 in sizes else sizes[len(sizes) // 2]
        figures = lambda t: (("chunked-timeline", timeline(data, t, size)),  # noqa: E731
                             ("chunked-summary", summary(data, t)))
    for theme, t in THEMES.items():
        for name, fig in figures(t):
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
