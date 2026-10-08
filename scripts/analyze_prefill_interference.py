"""How much does a new request's prefill slow down requests that are decoding?

Reads a sweep's ``*_events.jsonl`` files (measured runs only).  For every
decoding request, each interval between two of its consecutive tokens is
*clean* if no other request's prefill work (``prefill_chunk_start`` ->
``prefill_chunk_end``, or ``prefill_start`` -> ``prefill_end``) overlaps it,
and *interrupted* otherwise.  The median clean interval is the normal decode
step.  An interrupted interval's excess over that step is charged to the
overlapping prefill(s), split evenly when several overlap.

Per case it reports: the normal decode step; the median and worst interrupted
interval; the stall one arriving prompt adds to each request that is decoding
(total, and per interrupted token); and how much the arrivals raise the
decoding requests' TPOT.

  .conda-env/bin/python minillm_l4/scripts/analyze_prefill_interference.py \
      minillm_l4/results/mixed_batch_sweep_20261007
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

POLICY_ORDER = ("static", "continuous", "chunked_512", "mixed_512", "mixed_2048")


def load_runs(path: Path):
    """Yield one {request: {"tokens": [...], "windows": [...], "prompt": n}} per measured run."""

    runs: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"tokens": [], "windows": [], "open": None, "whole": None}))
    for line in path.open():
        event = json.loads(line)
        if event["warmup"]:
            continue
        request = runs[event["run_id"]][event["request_id"]]
        name, t = event["event"], event["timestamp_ms"]
        if name == "token_ready":
            request["tokens"].append(t)
        elif name == "prefill_chunk_start":
            request["open"] = t
        elif name == "prefill_chunk_end" and request["open"] is not None:
            request["windows"].append((request["open"], t))
            request["open"] = None
        elif name == "prefill_start":
            request["whole"] = t
        elif name == "prefill_end" and request["whole"] is not None and not request["windows"]:
            request["windows"].append((request["whole"], t))
    return runs.values()


def analyze_case(path: Path, prompt_lengths: dict[str, int]):
    clean, interrupted = [], []
    stall_per_arrival = collections.defaultdict(float)  # (run, arrival, victim) -> excess ms
    interrupted_by_arrival = collections.defaultdict(int)
    tpot_with, decoding_victims = [], set()
    runs = list(load_runs(path))
    intervals_all = []
    for run_index, requests in enumerate(runs):
        for victim, data in requests.items():
            tokens = data["tokens"]
            for a, b in zip(tokens, tokens[1:]):
                overlapping = [
                    other for other, od in requests.items() if other != victim
                    and any(start < b and end > a for start, end in od["windows"])
                ]
                intervals_all.append((run_index, victim, a, b, overlapping))
    for _, _, a, b, overlapping in intervals_all:
        (interrupted if overlapping else clean).append(b - a)
    if not clean:
        return None
    step = statistics.median(clean)
    for run_index, victim, a, b, overlapping in intervals_all:
        if overlapping:
            excess = max(0.0, (b - a) - step)
            for other in overlapping:
                stall_per_arrival[(run_index, other, victim)] += excess / len(overlapping)
                interrupted_by_arrival[(run_index, other, victim)] += 1
                decoding_victims.add((run_index, victim))
    for run_index, requests in enumerate(runs):
        for victim, data in requests.items():
            if (run_index, victim) in decoding_victims and len(data["tokens"]) > 1:
                tokens = data["tokens"]
                tpot_with.append((tokens[-1] - tokens[0]) / (len(tokens) - 1))
    by_prompt = collections.defaultdict(list)
    per_token = collections.defaultdict(list)
    for key, excess in stall_per_arrival.items():
        _, arrival, _ = key
        length = prompt_lengths.get(arrival.rsplit("-", 1)[-1])
        by_prompt[length].append(excess)
        per_token[length].append(excess / interrupted_by_arrival[key])
    return {
        "step": step,
        "interrupted_median": statistics.median(interrupted) if interrupted else None,
        "interrupted_max": max(interrupted) if interrupted else None,
        "stall_by_prompt": {k: statistics.median(v) for k, v in by_prompt.items()},
        "stall_per_token_by_prompt": {k: statistics.median(v) for k, v in per_token.items()},
        "tpot_interrupted": statistics.median(tpot_with) if tpot_with else None,
        "interrupted_share": len(interrupted) / (len(interrupted) + len(clean)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for events in sorted(args.sweep_dir.glob("p*_g*_b*_*_events.jsonl")):
        match = re.match(r"p(\d+)_g(\d+)_b(\d+)_(.+)_events\.jsonl", events.name)
        prompt, _, batch, policy = int(match[1]), int(match[2]), int(match[3]), match[4]
        if policy in {"isolated", "prefill"}:
            continue
        workload = json.loads((args.sweep_dir / events.name.replace("_events.jsonl", "_workload.json")).read_text())
        lengths = {r["request_id"].rsplit("-", 1)[-1]: len(r["prompt_token_ids"]) for r in workload["requests"]}
        result = analyze_case(events, lengths)
        if result:
            rows.append((prompt, batch, policy, result))
    rows.sort(key=lambda r: (r[0], r[1], POLICY_ORDER.index(r[2]) if r[2] in POLICY_ORDER else 99))

    print("Decode interference from arriving prefills (medians over measured runs; ms)")
    print("step = normal decode step; gap = interval with another request's prefill in it;")
    print("stall/arrival = extra delay one arriving prompt adds to EACH request that is decoding")
    print("(split into short 128-token and long N-token arrivals); TPOT = decoding requests' TPOT.\n")
    header = (f"{'prompt':>6} {'arr':>3} {'policy':<12} {'step':>6} {'gap med':>8} {'gap max':>8} "
              f"{'stall/arrival short':>19} {'stall/arrival long':>18} {'per-token long':>14} "
              f"{'TPOT':>6} {'TPOT +%':>7}")
    print(header)
    print("-" * len(header))
    for prompt, batch, policy, r in rows:
        short = r["stall_by_prompt"].get(128)
        long = r["stall_by_prompt"].get(prompt) if prompt != 128 else None
        per_token = r["stall_per_token_by_prompt"].get(prompt)
        fmt = lambda v, w, d=0: f"{v:>{w},.{d}f}" if v is not None else f"{'-':>{w}}"
        increase = (r["tpot_interrupted"] / r["step"] - 1) * 100 if r["tpot_interrupted"] else None
        print(f"{prompt:>6} {batch:>3} {policy:<12} {r['step']:>6.1f} {fmt(r['interrupted_median'], 8)} "
              f"{fmt(r['interrupted_max'], 8)} {fmt(short, 19)} {fmt(long, 18)} {fmt(per_token, 14)} "
              f"{fmt(r['tpot_interrupted'], 6, 1)} {fmt(increase, 7)}")


if __name__ == "__main__":
    main()
