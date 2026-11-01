"""Render the Q6 attention charts from run_attention_compare (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_attention_charts.py \
      minillm_l4/results/attention_compare_20261031/attention_compare.json minillm_l4/docs/assets/attention

Four methods, same color everywhere: naive (PyTorch ops), PyTorch SDPA (its
standard math kernel), FlashAttention (SDPA flash kernel) and our Triton kernel
(the tiled prefill kernel; the paged kernel for decode). Prefill charts also
show our first prefill kernel (one token per program) as a gray dashed line.

* attention-prefill - time vs context, and a scoreboard: time compared with
                      FlashAttention at the longest context;
* attention-memory  - extra memory vs context, against the L4's memory;
* attention-decode  - the same two panels for decode (8 requests);
* attention-packed  - several prompts in one prefill (Flash loop, Flash varlen, ours):
                      the engine's per-request loop vs one-launch kernels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, style  # noqa: E402

# (label, prefill backend, decode backend, color slot); fixed order and colors.
# Slot None = our first prefill kernel, drawn gray and dashed.
METHODS = (("Naive (plain PyTorch ops)", "naive", "naive", 1),
           ("PyTorch SDPA (standard kernel)", "sdpa_math", "sdpa_math", 2),
           ("FlashAttention", "sdpa_flash", "sdpa_flash", 0),
           ("Our Triton kernel", "triton_flash", "triton_paged", 3),
           ("Our first kernel (one token at a time)", "triton_packed", None, None))
FLASH = "sdpa_flash"


def color(t, slot):
    return t["muted"] if slot is None else t["series"][slot]


def methods(stage):
    index = 1 if stage == "prefill" else 2
    return [(m[0], m[index], m[3]) for m in METHODS if m[index] is not None]


def rows(data, stage, backend):
    return sorted((c for c in data["cases"] if c["stage"] == stage and c["backend"] == backend),
                  key=lambda c: c["tokens"])


def ok(data, stage, backend, key):
    picked = [c for c in rows(data, stage, backend) if c["status"] == "ok"]
    return [c["tokens"] for c in picked], [c[key] for c in picked]


def oom(data, stage, backend):
    return any(c["status"] == "oom" for c in rows(data, stage, backend))


def tick(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


def ms_text(ms):
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.2f} ms" if ms < 1 else f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def vs_flash(ratio):
    if abs(ratio - 1) < 0.02:
        return "same as Flash"
    if ratio < 1:
        return f"{(1 - ratio) * 100:.0f}% less time than Flash"
    return f"{ratio:.1f}× slower" if ratio < 10 else f"{ratio:.0f}× slower"


def frame(t, title, sub, stage, width=11.5, height=4.7, ratios=(1.25, 1)):
    fig, axes = plt.subplots(1, 2, figsize=(width, height), facecolor=t["surface"],
                             gridspec_kw={"width_ratios": ratios})
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])
    picked = methods(stage)
    handles = [plt.Line2D([0], [0], color=color(t, slot), lw=2.5, marker="o", ms=5,
                          ls=(0, (4, 2)) if slot is None else "-") for *_, slot in picked]
    fig.legend(handles, [label for label, *_ in picked], loc="upper left", bbox_to_anchor=(0.005, 0.87),
               ncol=len(picked), frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=2.2,
               columnspacing=1.5)
    return fig, axes


def line_panel(ax, data, t, stage, key, factor, ylabel, title):
    style(ax, t, xgrid=False, ygrid=True)
    sizes = sorted({c["tokens"] for c in data["cases"] if c["stage"] == stage})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [tick(s) for s in sizes], fontsize=8.5)
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 1.3)
    for _, backend, slot in methods(stage):
        xs, ys = ok(data, stage, backend, key)
        if not xs:
            continue
        ys = [y * factor for y in ys]
        # Our kernels sit exactly on Flash's memory line: dash them so all stay visible.
        dashed = slot is None or (backend == "triton_flash" and key == "extra_memory_bytes")
        ax.plot(xs, ys, color=color(t, slot), lw=2, ls=(0, (4, 2)) if dashed else "-", marker="o", ms=5,
                mec=t["surface"], mew=1.2, zorder=4 if dashed else 3)
        if oom(data, stage, backend):
            # Would-be value at the next size: attention work and the score table grow 4x per doubling.
            x, y = xs[-1] * 2, ys[-1] * 4
            ax.plot([xs[-1], x], [ys[-1], y], color=color(t, slot), lw=1.5, ls=":", zorder=3)
            ax.plot([x], [y], marker="X", ms=10, color=color(t, slot), mec=t["surface"], mew=1, zorder=4)
            ax.annotate("out of memory", (x, y), textcoords="offset points", xytext=(7, -3), ha="left",
                        va="center", fontsize=7.5, color=t["ink2"])
    ax.set_title(title, fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=8.5)
    ax.set_xlabel("context length (tokens)", color=t["ink2"], fontsize=8.5)


def bars(ax, t, entries, flash_ms, title):
    """Horizontal bars of time / FlashAttention time, labeled with the time and the comparison."""

    style(ax, t, xgrid=True, ygrid=False)
    longest = max(c["ms"] / flash_ms for _, _, _, c in entries if c and c["status"] == "ok")
    for y, (label, backend, slot, case) in enumerate(reversed(entries)):
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                color=t["ink"])
        if not case or case["status"] != "ok":
            ax.text(longest * 0.02, y, "out of memory", va="center", fontsize=8.5, color=t["ink2"], style="italic")
            continue
        ratio = case["ms"] / flash_ms
        ax.barh(y, ratio, height=0.62, color=color(t, slot), zorder=3)
        note = "reference" if backend == FLASH else vs_flash(ratio)
        ax.text(ratio + longest * 0.02, y, f"{ms_text(case['ms'])}  ·  {note}", va="center", fontsize=8.5,
                color=t["ink"])
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_xlim(0, longest * 1.6)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=10, color=t["ink"], loc="left")
    ax.set_xlabel("time ÷ FlashAttention time (1 = same, lower = faster)", color=t["ink2"], fontsize=8.5)


def scoreboard(ax, data, t, stage, size):
    flash = next(c["ms"] for c in rows(data, stage, FLASH) if c["tokens"] == size and c["status"] == "ok")
    entries = [(label, backend, slot, next((c for c in rows(data, stage, backend) if c["tokens"] == size), None))
               for label, backend, slot in methods(stage)]
    bars(ax, t, entries, flash, f"At {tick(size)} tokens: time compared with FlashAttention")


def longest(data, stage):
    return max(c["tokens"] for c in data["cases"] if c["stage"] == stage)


def prefill_chart(data, t):
    fig, (left, right) = frame(t, "Prefill: tiled kernels are 40–100× faster, and ours matches FlashAttention",
                               "One attention layer of Qwen3-4B (32 query heads, 8 KV heads, bf16), one prompt. "
                               "X = out of memory (shown at the would-be value).", "prefill")
    line_panel(left, data, t, "prefill", "ms", 1, "milliseconds (log scale)", "Time per layer vs prompt length")
    scoreboard(right, data, t, "prefill", longest(data, "prefill"))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.7)
    return fig


def decode_chart(data, t):
    fig, (left, right) = frame(t, "Decode: every good kernel hits the memory limit",
                               "One attention layer of Qwen3-4B, 8 requests each adding one token. Flash and our "
                               "kernel read the KV cache once; the others also copy it.", "decode")
    line_panel(left, data, t, "decode", "ms", 1, "milliseconds (log scale)", "Time per layer vs context length")
    scoreboard(right, data, t, "decode", longest(data, "decode"))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.62)
    return fig


def memory_chart(data, t):
    fig, (left, right) = frame(t, "Naive attention needs memory for a giant score table",
                               "Extra GPU memory for one prefill layer (on top of Q, K, V). FlashAttention and our "
                               "kernels only need room for the output.", "prefill")
    line_panel(left, data, t, "prefill", "extra_memory_bytes", 2**-30, "GiB (log scale)",
               "Extra memory vs prompt length")
    limit = data.get("total_bytes", 23659151360) / 2**30
    left.axhline(limit, color=t["ink2"], lw=1.2, ls="--", zorder=2)
    left.text(left.get_xlim()[0] * 1.1, limit * 1.25, f"all of the L4's memory ({limit:.0f} GiB)", fontsize=8,
              color=t["ink2"])
    left.set_ylim(None, limit * 6)
    xs, ys = ok(data, "prefill", FLASH, "extra_memory_bytes")
    left.annotate("FlashAttention and our kernels:\nthe same line (just the output)", (xs[3], ys[3] * 2**-30),
                  textcoords="offset points", xytext=(0, 26), ha="center", fontsize=8, color=t["ink2"])

    # Right: bars at 4k, the largest size where every method fits.
    style(right, t, xgrid=True, ygrid=False)
    size = 4096
    entries = [(label, slot, next(c for c in rows(data, "prefill", backend) if c["tokens"] == size))
               for label, backend, slot in methods("prefill")]
    top = max(c["extra_memory_bytes"] for *_, c in entries) / 2**30
    for y, (label, slot, case) in enumerate(reversed(entries)):
        gib = case["extra_memory_bytes"] / 2**30
        right.barh(y, gib, height=0.62, color=color(t, slot), zorder=3)
        right.text(-0.01, y, label, transform=right.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                   color=t["ink"])
        text = f"{gib:.1f} GiB" if gib >= 1 else f"{case['extra_memory_bytes'] / 2**20:.0f} MiB"
        right.text(gib + top * 0.02, y, text, va="center", fontsize=8.5, color=t["ink"])
    right.set_yticks([])
    right.set_ylim(-0.6, len(entries) - 0.4)
    right.set_xlim(0, top * 1.3)
    right.spines["left"].set_visible(False)
    right.set_title(f"At {tick(size)} tokens", fontsize=10, color=t["ink"], loc="left")
    right.set_xlabel("extra memory (GiB)", color=t["ink2"], fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.7)
    return fig


def packed_chart(data, t):
    """Several prompts in one prefill: Flash loop vs Flash varlen vs ours, per workload."""

    cases = [c for c in data["cases"] if c["stage"] == "packed"]
    labels = list(dict.fromkeys(c["label"] for c in cases))
    fig, ax = plt.subplots(figsize=(11.5, 4.8), facecolor=t["surface"])
    style(ax, t, xgrid=True, ygrid=False)
    entries = (("FlashAttention, one call per prompt (engine's SDPA path)", "sdpa_flash_loop", 0),
               ("FlashAttention varlen, one launch", "flash_varlen", 1),
               ("Our Triton kernel, one launch", "triton_flash", 3))
    entries = tuple(e for e in entries if any(c["backend"] == e[1] for c in cases))
    offsets = (0.26, 0.0, -0.26) if len(entries) == 3 else (0.19, -0.19)
    top = max(c["ms"] for c in cases)
    for group, label in enumerate(reversed(labels)):
        loop = next(c["ms"] for c in cases if c["label"] == label and c["backend"] == "sdpa_flash_loop")
        for offset, (_, backend, slot) in zip(offsets, entries):
            case = next(c for c in cases if c["label"] == label and c["backend"] == backend)
            y = group + offset
            ax.barh(y, case["ms"], height=0.24 if len(entries) == 3 else 0.34, color=t["series"][slot], zorder=3)
            note = "" if backend == "sdpa_flash_loop" else f"  ·  {(1 - case['ms'] / loop) * 100:.0f}% less than the loop"
            ax.text(case["ms"] + top * 0.01, y, ms_text(case["ms"]) + note, va="center", fontsize=8.5,
                    color=t["ink"])
        ax.text(-0.01, group, f"{label} tokens", transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9, color=t["ink"])
    ax.set_yticks([])
    ax.set_xlim(0, top * 1.55)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("attention time for the whole batch, one layer (ms)", color=t["ink2"], fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["series"][slot]) for *_, slot in entries]
    fig.legend(handles, [e[0] for e in entries], loc="upper left", bbox_to_anchor=(0.005, 0.87), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=1.2, columnspacing=1.8)
    fig.text(0.01, 0.975, "Many prompts at once: one launch beats a loop; our kernel ties FlashAttention varlen",
             ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, "Prompts prefilled together, as the engine's whole-prompt policy does. Launch costs "
             "included; the three methods are timed in turn each round.", ha="left", va="top", fontsize=9.5,
             color=t["ink2"])
    fig.subplots_adjust(left=0.12, right=0.985, top=0.76, bottom=0.12)
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
        for name, make in (("attention-prefill", prefill_chart), ("attention-memory", memory_chart),
                           ("attention-decode", decode_chart), ("attention-packed", packed_chart)):
            fig = make(data, t)
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
