"""Render the Q6 attention charts from run_attention_compare (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_attention_charts.py \
      minillm_l4/results/attention_compare_20261030b/attention_compare.json minillm_l4/docs/assets/attention

Four methods, same color everywhere: naive (PyTorch ops), PyTorch SDPA (its
standard math kernel), FlashAttention (SDPA flash kernel) and our Triton kernel.

* attention-prefill - time vs context, and a scoreboard: how many times slower
                      than FlashAttention each method is at one context;
* attention-memory  - extra memory vs context, against the L4's memory;
* attention-decode  - the same two panels for decode (8 requests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, style  # noqa: E402

# (label, prefill backend, decode backend, color slot); fixed order and colors.
METHODS = (("Naive (plain PyTorch ops)", "naive", "naive", 1),
           ("PyTorch SDPA (standard kernel)", "sdpa_math", "sdpa_math", 2),
           ("FlashAttention", "sdpa_flash", "sdpa_flash", 0),
           ("Our Triton kernel", "triton_packed", "triton_paged", 3))
FLASH = "sdpa_flash"


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


def frame(t, title, sub, width=11.5, height=4.7, ratios=(1.25, 1)):
    fig, axes = plt.subplots(1, 2, figsize=(width, height), facecolor=t["surface"],
                             gridspec_kw={"width_ratios": ratios})
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])
    handles = [plt.Line2D([0], [0], color=t["series"][slot], lw=2.5, marker="o", ms=5) for *_, slot in METHODS]
    fig.legend(handles, [m[0] for m in METHODS], loc="upper left", bbox_to_anchor=(0.005, 0.87), ncol=4,
               frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=1.8, columnspacing=1.8)
    return fig, axes


def line_panel(ax, data, t, stage, key, factor, ylabel, title):
    style(ax, t, xgrid=False, ygrid=True)
    sizes = sorted({c["tokens"] for c in data["cases"]})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [tick(s) for s in sizes], fontsize=8.5)
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 1.3)
    backend_index = 1 if stage == "prefill" else 2
    for method in METHODS:
        backend, slot = method[backend_index], method[3]
        xs, ys = ok(data, stage, backend, key)
        if not xs:
            continue
        ys = [y * factor for y in ys]
        color = t["series"][slot]
        dashed = backend == "triton_packed" and key == "extra_memory_bytes"  # sits exactly on Flash's line
        ax.plot(xs, ys, color=color, lw=2, ls=(0, (4, 2)) if dashed else "-", marker="o", ms=5, mec=t["surface"],
                mew=1.2, zorder=4 if dashed else 3)
        if oom(data, stage, backend):
            # Would-be value at the next size: attention work and the score table grow 4x per doubling.
            x, y = xs[-1] * 2, ys[-1] * 4
            ax.plot([xs[-1], x], [ys[-1], y], color=color, lw=1.5, ls=":", zorder=3)
            ax.plot([x], [y], marker="X", ms=10, color=color, mec=t["surface"], mew=1, zorder=4)
            ax.annotate("out of memory", (x, y), textcoords="offset points", xytext=(7, -3), ha="left",
                        va="center", fontsize=7.5, color=t["ink2"])
    ax.set_title(title, fontsize=10, color=t["ink"], loc="left")
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=8.5)
    ax.set_xlabel("context length (tokens)", color=t["ink2"], fontsize=8.5)


def method_backend(label, backend_index):
    return next(m[backend_index] for m in METHODS if m[0] == label)


def scoreboard(ax, data, t, stage, size):
    """Horizontal bars: times slower than FlashAttention at one context length."""

    style(ax, t, xgrid=True, ygrid=False)
    backend_index = 1 if stage == "prefill" else 2
    flash = next(c["ms"] for c in rows(data, stage, FLASH) if c["tokens"] == size and c["status"] == "ok")
    entries = []
    for method in METHODS:
        case = next((c for c in rows(data, stage, method[backend_index]) if c["tokens"] == size), None)
        entries.append((method[0], method[3], case))
    longest = max(c["ms"] / flash for _, _, c in entries if c and c["status"] == "ok")
    for y, (label, slot, case) in enumerate(reversed(entries)):
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                color=t["ink"])
        if not case or case["status"] != "ok":
            ax.text(longest * 0.02, y, "out of memory", va="center", fontsize=8.5, color=t["ink2"], style="italic")
            continue
        ratio = case["ms"] / flash
        ax.barh(y, ratio, height=0.62, color=t["series"][slot], zorder=3)
        if method_backend(label, backend_index) == FLASH:
            note = "fastest"
        elif ratio < 1.05:
            note = "same as Flash"
        else:
            note = f"{ratio:.1f}× slower" if ratio < 10 else f"{ratio:.0f}× slower"
        ax.text(ratio + longest * 0.02, y, f"{ms_text(case['ms'])}  ·  {note}", va="center", fontsize=8.5,
                color=t["ink"])
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_xlim(0, longest * 1.55)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"At {tick(size)} tokens: time compared with FlashAttention", fontsize=10, color=t["ink"],
                 loc="left")
    ax.set_xlabel("times slower than FlashAttention (1 = as fast)", color=t["ink2"], fontsize=8.5)


def prefill_chart(data, t):
    fig, (left, right) = frame(t, "Prefill: FlashAttention is 40–100× faster, same math",
                               "One attention layer of Qwen3-4B (32 query heads, 8 KV heads, bf16), one prompt. "
                               "X = out of memory (shown at the would-be value).")
    line_panel(left, data, t, "prefill", "ms", 1, "milliseconds (log scale)", "Time per layer vs prompt length")
    scoreboard(right, data, t, "prefill", 4096)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.62)
    return fig


def decode_chart(data, t):
    fig, (left, right) = frame(t, "Decode: every good kernel hits the memory limit",
                               "One attention layer of Qwen3-4B, 8 requests each adding one token. Flash and our kernel read the "
                               "KV cache once; the others also copy it.")
    line_panel(left, data, t, "decode", "ms", 1, "milliseconds (log scale)", "Time per layer vs context length")
    scoreboard(right, data, t, "decode", max(c["tokens"] for c in data["cases"]))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.62)
    return fig


def memory_chart(data, t):
    fig, (left, right) = frame(t, "Naive attention needs memory for a giant score table",
                               "Extra GPU memory for one prefill layer (on top of Q, K, V). FlashAttention and our "
                               "kernel only need room for the output.")
    line_panel(left, data, t, "prefill", "extra_memory_bytes", 2**-30, "GiB (log scale)",
               "Extra memory vs prompt length")
    limit = data.get("total_bytes", 23659151360) / 2**30
    left.axhline(limit, color=t["ink2"], lw=1.2, ls="--", zorder=2)
    left.text(left.get_xlim()[0] * 1.1, limit * 1.25, f"all of the L4's memory ({limit:.0f} GiB)", fontsize=8,
              color=t["ink2"])
    left.set_ylim(None, limit * 6)
    xs, ys = ok(data, "prefill", FLASH, "extra_memory_bytes")
    left.annotate("FlashAttention and our kernel:\nthe same line (just the output)", (xs[3], ys[3] * 2**-30),
                  textcoords="offset points", xytext=(0, 26), ha="center", fontsize=8, color=t["ink2"])

    # Right: bars at 4k, the largest size where every method fits.
    style(right, t, xgrid=True, ygrid=False)
    size = 4096
    entries = [(m[0], m[3], next(c for c in rows(data, "prefill", m[1]) if c["tokens"] == size)) for m in METHODS]
    top = max(c["extra_memory_bytes"] for *_, c in entries) / 2**30
    for y, (label, slot, case) in enumerate(reversed(entries)):
        gib = case["extra_memory_bytes"] / 2**30
        right.barh(y, gib, height=0.62, color=t["series"][slot], zorder=3)
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
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.62)
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
                           ("attention-decode", decode_chart)):
            fig = make(data, t)
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
