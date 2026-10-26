"""Render the Q2 roofline charts from run_roofline (light and dark SVG).

  .conda-env/bin/python minillm_l4/scripts/make_roofline_charts.py \
      minillm_l4/results/roofline_20261026/roofline.json minillm_l4/docs/assets/roofline

Charts:
* roofline          - decode and prefill on the L4 roofline (two panels):
  FP8 projections, attention, and the whole step, against the measured
  power-capped ceilings (solid) and the datasheet (dotted);
* prefill-breakdown - where prefill GPU time goes as the prompt grows.

FLOPs and bytes per category (model dimensions from the JSON):
  projections, M rows   2*M*P FLOPs; P + scales + 2*M*sum(K+N) bytes
  lm_head, R rows       2*R*V*H;     V*H*2 + 2*R*(H+V)
  prefill attention     L*4*Hq*d*T(T+1)/2 (causal); L*T*(2*Hq+2*Hkv)*d*2 (Q, K, V, O once)
  decode attention      L*4*Hq*d*b*c;              L*b*c*2*Hkv*d*2 (the KV cache)
Whole step = the sum of those (elementwise traffic of norms/RoPE/SiLU excluded).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_charts import FONT, THEMES, figure, style  # noqa: E402

# Measured on this L4 (docs/phase_notes/gemm_peak.md): best sustained cuBLAS GEMM and copy bandwidth.
CEILINGS = {"fp8": 127.5, "bf16": 63.8, "bandwidth_gbps": 231.0}
DATASHEET = {"fp8": 242.5, "bf16": 121.0, "bandwidth_gbps": 300.0}
SLOTS = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
SERIES = (("projections", "FP8 projections", 0), ("attention", "Attention (BF16)", 1), ("step", "Whole step", 2))


def costs(dims, *, rows, logits_rows, prefill_tokens=None, batch=0, context=0):
    """{category: (flops, bytes)} for one forward step."""

    L, H, V = dims["layers"], dims["hidden"], dims["vocab"]
    hq, hkv, d = dims["q_heads"], dims["kv_heads"], dims["head_dim"]
    proj = (2 * rows * dims["fp8_params"],
            dims["fp8_params"] + dims["fp8_scale_bytes"] + 2 * rows * dims["fp8_in_plus_out"])
    head = (2 * logits_rows * V * H, dims["lm_head_bytes"] + 2 * logits_rows * (H + V))
    if prefill_tokens is not None:
        t = prefill_tokens
        attn = (L * 4 * hq * d * t * (t + 1) / 2, L * t * (2 * hq + 2 * hkv) * d * 2)
    else:
        attn = (L * 4 * hq * d * batch * context, L * batch * context * 2 * hkv * d * 2)
    step = tuple(a + b + c for a, b, c in zip(proj, head, attn))
    return {"projections": proj, "lm_head": head, "attention": attn, "step": step}


def points(data):
    """Per phase: list of (label, {series: (intensity FLOP/B, TFLOPS)})."""

    dims = data["model_dims"]
    out = {"decode": [], "prefill": []}
    for case in data["decode"]:
        b, c = case["batch"], case["context"] + data["profile_steps"] / 2
        g = case["gpu_ms_by_category"]
        times = {"projections": g.get("weights_fp8_projections", 0), "attention": g.get("kv_attention", 0),
                 "step": sum(g.values())}
        out["decode"].append((f"b{b}", costs(dims, rows=b, logits_rows=b, batch=b, context=c), times))
    for case in data["prefill"]:
        t = case["tokens"]
        g = case["gpu_ms_by_category"]
        times = {"projections": g.get("weights_fp8_projections", 0), "attention": g.get("prefill_attention", 0),
                 "step": sum(g.values())}
        out["prefill"].append((f"{t:,}", costs(dims, rows=t, logits_rows=1, prefill_tokens=t), times))
    return {phase: [(label, {k: (c[k][0] / c[k][1], c[k][0] / (times[k] * 1e9)) for k, _, _ in SERIES if times[k] > 0})
                    for label, c, times in rows] for phase, rows in out.items()}


def ceilings(ax, t, xs):
    lo, hi = xs
    bw = CEILINGS["bandwidth_gbps"] * 1e9 / 1e12
    for peak, label in ((CEILINGS["fp8"], "FP8 roof (measured, 72 W)"), (CEILINGS["bf16"], "BF16 roof (measured)")):
        ridge = peak / bw
        ax.plot([lo, ridge, hi], [lo * bw, peak, peak], color=t["ink2"], lw=1.4, zorder=2)
        ax.text(hi / 1.05, peak * 1.06, label, ha="right", va="bottom", fontsize=7.5, color=t["ink2"])
    dbw = DATASHEET["bandwidth_gbps"] * 1e9 / 1e12
    ax.plot([lo, DATASHEET["fp8"] / dbw, hi], [lo * dbw, DATASHEET["fp8"], DATASHEET["fp8"]],
            color=t["muted"], lw=1, ls=":", zorder=1)
    ax.text(DATASHEET["fp8"] / dbw * 1.08, DATASHEET["fp8"] * 1.07, "FP8 datasheet", ha="left", va="bottom",
            fontsize=7.5, color=t["muted"])
    ax.text(lo * 1.15, lo * 1.15 * bw * 1.35, f"{CEILINGS['bandwidth_gbps']:.0f} GB/s", rotation=0, fontsize=7.5,
            color=t["ink2"], va="bottom")


def roofline(data, t, theme):
    pts = points(data)
    fig, axes = figure(t, 11, 5.2, ncols=2)
    colors = SLOTS[theme]
    for ax, phase, title, xs, ys, labelled in (
            (axes[0], "decode", f"Decode (batch 1–8, context {data['decode_context']})", (0.5, 3e3), (0.2, 500),
             {"b1", "b2", "b4", "b8"}),
            (axes[1], "prefill", "Prefill (one prompt, 128–8,192 tokens)", (20, 1e4), (5, 500),
             {"128", "512", "2,048", "8,192"})):
        style(ax, t, xgrid=True, ygrid=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ceilings(ax, t, xs)
        for key, label, slot in SERIES:
            series = [(lab, p[key]) for lab, p in pts[phase] if key in p]
            if not series:
                continue
            marker = "D" if key == "step" else "o"
            # Decode attention has one intensity (q heads / kv heads FLOPs per KV byte): markers, no line.
            line = 0 if (phase == "decode" and key == "attention") else 1.6
            ax.plot([x for _, (x, _) in series], [y for _, (_, y) in series], color=colors[slot], marker=marker,
                    ms=6, lw=line, zorder=4, label=label)
            if key == "step":
                for lab, (x, y) in series:
                    if lab in labelled:
                        offset, align = ((9, -9), "left") if phase == "decode" else ((0, -14), "center")
                        ax.annotate(lab, (x, y), textcoords="offset points", xytext=offset, ha=align,
                                    fontsize=7.5, color=t["ink2"])
        ax.set_xlim(*xs)
        ax.set_ylim(*ys)
        ax.set_title(title, fontsize=10, color=t["ink"], loc="left")
        ax.set_xlabel("Arithmetic intensity (FLOPs per byte read or written)", color=t["ink2"], fontsize=8.5)
    axes[0].set_ylabel("Achieved TFLOPS", color=t["ink2"], fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=t["ink2"], loc="upper left")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.8, bottom=0.13, wspace=0.12)
    fig.text(0.01, 0.975, "Decode is memory-bound, prefill is compute-bound (Qwen3-4B FP8, one L4)",
             ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    ridge = CEILINGS["fp8"] / (CEILINGS["bandwidth_gbps"] / 1000)
    fig.text(0.01, 0.915, f"Left of the ridge (FP8 ≈ {ridge:.0f} FLOP/B) a step waits on memory; right of it, on the "
             "tensor cores. Labels: batch size (decode), prompt tokens (prefill).",
             ha="left", va="top", fontsize=9.5, color=t["ink2"])
    fig.text(0.01, 0.015, "Solid roofs: measured best sustained cuBLAS GEMM and copy bandwidth under the 72 W cap. "
             "Dotted: datasheet (FP8 242 TFLOPS, 300 GB/s). Kernel times from torch.profiler.",
             ha="left", va="bottom", fontsize=8, color=t["muted"])
    return fig


def prefill_breakdown(data, t, theme):
    fig, ax = figure(t, 9.2, 4.8)
    style(ax, t, xgrid=False, ygrid=True)
    colors = SLOTS[theme] + ["#eda100" if theme == "light" else "#c98500"]
    parts = (("FP8 projections", ("weights_fp8_projections",), 0),
             ("Attention (FlashAttention, BF16)", ("prefill_attention",), 1),
             ("Norm, RoPE, SiLU", ("norm_rope_silu",), 2),
             ("lm_head, KV write, other", ("weights_lm_head", "kv_write", "other"), 3))
    cases = sorted(data["prefill"], key=lambda c: c["tokens"])
    for x, case in enumerate(cases):
        g, total, bottom = case["gpu_ms_by_category"], sum(case["gpu_ms_by_category"].values()), 0.0
        for label, keys, slot in parts:
            share = 100 * sum(g.get(k, 0) for k in keys) / total
            ax.bar(x, share, bottom=bottom, width=0.7, color=colors[slot], edgecolor=t["surface"], lw=1,
                   label=label if x == 0 else None)
            if share >= 6:
                ax.text(x, bottom + share / 2, f"{share:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="#ffffff")
            bottom += share
        ax.text(x, 101.5, f"{total:,.0f} ms", ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
    ax.set_xticks(range(len(cases)), [f"{c['tokens']:,}" for c in cases])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of prefill GPU time (%)", color=t["ink2"], fontsize=9)
    ax.set_xlabel("Prompt tokens", color=t["ink2"], fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=t["ink2"], ncol=4, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.13)
    fig.text(0.01, 0.975, "Where prefill time goes", ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.915, "Projections dominate; attention's share grows with the prompt (its cost is quadratic). "
             "Label: total GPU time.", ha="left", va="top", fontsize=9.5, color=t["ink2"])
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
        for name, make in (("roofline", roofline), ("prefill-breakdown", prefill_breakdown)):
            fig = make(data, t, theme)
            path = out / f"{name}-{theme}.svg"
            fig.savefig(path, facecolor=t["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
