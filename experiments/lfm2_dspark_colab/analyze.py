"""Turn one results folder from run_bench.py into summary.csv and report.md.

  python analyze.py results/<run>          # writes summary.csv + report.md, prints the report
  python analyze.py                        # same, for the newest folder under results/

Metrics (all from llama-server's own per-request timings):
  decode tok/s     sum(predicted_n) / sum(predicted_ms)   generation speed after the first token
  prefill tok/s    sum(prompt_n) / sum(prompt_ms)
  acceptance       sum(draft_n_accepted) / sum(draft_n)   share of drafted tokens the target kept
  tokens/step      predicted_n / (predicted_n - accepted) tokens produced per target forward
                   (each verify step yields its accepted drafts plus one token of its own)
  speedup          decode tok/s / decode tok/s of the same target without speculation
  same text        outputs identical to the no-speculation run of the same target and prompt
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Peak DRAM bandwidth by GPU (GB/s, datasheet), for the decode speed ceiling.
PEAK_BW_GBS = {"t4": 320, "l4": 300, "a100": 1555, "a10": 600, "v100": 900, "h100": 3350}


def load(run: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    env = json.loads((run / "env.json").read_text())
    configs = json.loads((run / "configs.json").read_text()) if (run / "configs.json").exists() else []
    rows = [json.loads(x) for x in (run / "requests.jsonl").read_text().splitlines() if x.strip()] \
        if (run / "requests.jsonl").exists() else []
    bench = [json.loads(x) for x in (run / "llama_bench.jsonl").read_text().splitlines() if x.strip()] \
        if (run / "llama_bench.jsonl").exists() else []
    return env, configs, rows, bench


def _agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error") and r.get("timings")]
    t = [r["timings"] for r in ok]
    pred_n = sum(x.get("predicted_n", 0) for x in t)
    pred_ms = sum(x.get("predicted_ms", 0) for x in t)
    prompt_n = sum(x.get("prompt_n", 0) for x in t)
    prompt_ms = sum(x.get("prompt_ms", 0) for x in t)
    drafted = sum(x.get("draft_n", 0) or 0 for x in t)
    accepted = sum(x.get("draft_n_accepted", 0) or 0 for x in t)
    per_req = [x["predicted_per_second"] for x in t if x.get("predicted_per_second")]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "tokens": pred_n,
        "decode_tps": pred_n / pred_ms * 1000 if pred_ms else None,
        "decode_tps_median_req": statistics.median(per_req) if per_req else None,
        "prefill_tps": prompt_n / prompt_ms * 1000 if prompt_ms else None,
        "ms_per_token": pred_ms / pred_n if pred_n else None,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": accepted / drafted if drafted else None,
        "tokens_per_step": pred_n / (pred_n - accepted) if drafted and pred_n > accepted else None,
    }


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def summarize(run: Path) -> tuple[list[dict], dict, list[dict]]:
    env, configs, rows, _ = load(run)
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    order = [c["name"] for c in configs] or list(by_cfg)
    meta = {c["name"]: c for c in configs}

    base_text = {(r["target"], r["prompt_id"], r["rep"]): r.get("text") or ""
                 for r in rows if not r["spec"]}
    summary, diffs = [], []
    for name in order:
        rs = by_cfg.get(name, [])
        m = meta.get(name, {})
        row = {"config": name, "target": m.get("target", rs[0]["target"] if rs else ""),
               "spec": m.get("spec", rs[0]["spec"] if rs else ""), "n_max": m.get("n_max", ""),
               "status": m.get("status", "ok" if rs else "missing"), **_agg(rs)}
        same = total = 0
        for r in rs:
            key = (r["target"], r["prompt_id"], r["rep"])
            if r["spec"] and key in base_text and not r.get("error"):
                total += 1
                if (r.get("text") or "") == base_text[key]:
                    same += 1
                elif r["rep"] == 0:
                    i = _first_diff(r.get("text") or "", base_text[key])
                    diffs.append({"config": name, "prompt_id": r["prompt_id"], "char": i,
                                  "base": base_text[key][max(0, i - 40): i + 40],
                                  "spec": (r.get("text") or "")[max(0, i - 40): i + 40]})
        row["same_text"] = f"{same}/{total}" if total else ""
        summary.append(row)

    base_tps = {r["target"]: r["decode_tps"] for r in summary if not r["spec"] and r["decode_tps"]}
    for r in summary:
        b = base_tps.get(r["target"])
        r["speedup"] = r["decode_tps"] / b if b and r["decode_tps"] else None

    # per-category view
    cat: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cat[(r["config"], r["category"])].append(r)
    cats = sorted({r["category"] for r in rows})
    per_cat = {}
    for (name, c), rs in cat.items():
        per_cat[(name, c)] = _agg(rs)
    return summary, {"per_cat": per_cat, "cats": cats, "base_tps_by_target": base_tps}, diffs


def _f(x, nd=1, pct=False):
    if x is None or x == "":
        return "-"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(run: Path) -> str:
    run = Path(run)
    env, configs, rows, bench = load(run)
    summary, extra, diffs = summarize(run)

    cols = ["config", "target", "spec", "n_max", "status", "requests", "errors", "tokens", "decode_tps",
            "decode_tps_median_req", "prefill_tps", "ms_per_token", "drafted", "accepted", "acceptance",
            "tokens_per_step", "speedup", "same_text"]
    with (run / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in cols})

    gpu = env.get("gpu", {})
    gname = gpu.get("name", "?")
    bw = next((v for k, v in PEAK_BW_GBS.items() if k in gname.lower().split() or f" {k}" in gname.lower()), None)
    L = [f"# LFM2.5-2.6B + DSpark on {gname}", ""]
    L.append(f"- llama.cpp `{env.get('llama_cpp', {}).get('commit', '?')[:10]}` "
             f"({env.get('llama_cpp', {}).get('subject', '')})")
    L.append(f"- driver {gpu.get('driver_version', '?')}, CUDA {gpu.get('cuda_version', '?')}, "
             f"started {env.get('started_utc', '?')} UTC")
    a = env.get("args", {})
    L.append(f"- {env.get('n_prompts', '?')} prompts x {a.get('reps', '?')} reps, max {a.get('max_tokens', '?')} "
             f"new tokens, greedy, flash attention {a.get('flash_attn', '?')}, draft "
             f"{env.get('models', {}).get('draft', {}).get('file', '?')}")
    L.append("")
    L.append("## Per config")
    L.append("")
    L.append("| config | decode tok/s | speedup | ms/token | acceptance | tokens/step | prefill tok/s | same text | errors |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        if r["status"] != "ok":
            L.append(f"| {r['config']} | {r['status']} | | | | | | | |")
            continue
        L.append(f"| {r['config']} | {_f(r['decode_tps'])} | {_f(r['speedup'], 2)}x | {_f(r['ms_per_token'], 2)} | "
                 f"{_f(r['acceptance'], pct=True)} | {_f(r['tokens_per_step'], 2)} | {_f(r['prefill_tps'], 0)} | "
                 f"{r['same_text'] or '-'} | {r['errors']} |")

    if bw:
        L += ["", f"## Decode ceiling from memory bandwidth ({gname}: {bw} GB/s peak)", "",
              "Each decode step reads every weight once, so tok/s <= bandwidth / model bytes.", "",
              "| target | file GB | ceiling tok/s | measured base tok/s | share of ceiling |", "|---|---:|---:|---:|---:|"]
        for q, m in env.get("models", {}).get("targets", {}).items():
            gb = m["bytes"] / 1e9
            ceil = bw / gb
            meas = extra["base_tps_by_target"].get(q)
            L.append(f"| {q} | {gb:.2f} | {ceil:.0f} | {_f(meas)} | {_f(meas / ceil if meas else None, pct=True)} |")

    if extra["cats"]:
        L += ["", "## Speedup and acceptance by prompt category", ""]
        L.append("| config | " + " | ".join(extra["cats"]) + " |")
        L.append("|---|" + "---:|" * len(extra["cats"]))
        for r in summary:
            if not r["spec"] or r["status"] != "ok":
                continue
            cells = []
            base_cfg = next((s["config"] for s in summary if s["target"] == r["target"] and not s["spec"]), None)
            for c in extra["cats"]:
                s = extra["per_cat"].get((r["config"], c), {})
                b = extra["per_cat"].get((base_cfg, c), {})
                sp = s.get("decode_tps") / b["decode_tps"] if s.get("decode_tps") and b.get("decode_tps") else None
                cells.append(f"{_f(sp, 2)}x / {_f(s.get('acceptance'), pct=True)}")
            L.append(f"| {r['config']} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Cells are speedup / acceptance.")

    if bench:
        L += ["", "## llama-bench (no speculation)", "", "| target | test | tok/s | stdev |", "|---|---|---:|---:|"]
        for b in bench:
            test = f"pp{b.get('n_prompt')}" if b.get("n_prompt") else f"tg{b.get('n_gen')}"
            L.append(f"| {b.get('target')} | {test} | {_f(b.get('avg_ts'))} | {_f(b.get('stddev_ts'))} |")

    if diffs:
        L += ["", "## Outputs that differ from the no-speculation run (rep 0, first 10)", ""]
        for d in diffs[:10]:
            L.append(f"- `{d['config']}` / `{d['prompt_id']}` first differs at char {d['char']}")
            L.append(f"  - base: `{d['base']!r}`")
            L.append(f"  - spec: `{d['spec']!r}`")

    failed = [c for c in configs if c.get("status") != "ok"]
    if failed:
        L += ["", "## Configs that failed to start", ""]
        for c in failed:
            L.append(f"- `{c['name']}`: see `logs/server_{c['name']}.log`")

    text = "\n".join(L) + "\n"
    (run / "report.md").write_text(text)
    return text


def main() -> None:
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = sorted(p for p in (HERE / "results").iterdir() if p.is_dir())
        if not runs:
            raise SystemExit("no results yet")
        run = runs[-1]
    print(write_report(run))
    print(f"(wrote {run / 'summary.csv'} and {run / 'report.md'})")


if __name__ == "__main__":
    main()
