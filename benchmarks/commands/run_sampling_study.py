"""How sampling settings change answers across domains, and what sampling costs.

Sections (pinned Qwen3-4B FP8, fused kernels, merged projections, CUDA Graph
decode, no HF reference check):

* study     - 30 prompts, or one per domain with --one-prompt (medical, math,
              coding, general knowledge, creative) x 21 settings x 5 seeds
              (greedy: 1 seed), up to 300 new tokens,
              batch 32, requests shuffled so domains and settings share batches.
              Each output is scored:
              - math: the number after "Answer:" equals the expected value;
              - general knowledge, medical: the expected fact appears in the
                first 200 characters (a keyword check, so a rough one);
              - coding: the function passes unit tests, run in a subprocess
                with resource limits after a static allow-list check;
              - creative: not scored (the control domain).
              Also records seed-to-seed variety, repetition, language drift and
              the model's confidence on every token.
* speed     - decode TPOT for greedy vs sampling at batch 1/8/32.
* reproduce - the same seeds under other batch sizes and policies: do tokens repeat?

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_sampling_study \\
      --output-dir minillm_l4/results/sampling_study_<date>
"""

from __future__ import annotations

import argparse
import ast
import difflib
import gc
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

EOS = (151645, 151643)
SEEDS = 5
MAX_NEW_TOKENS = 300

MATH = "\n\nWork it out step by step. End with a final line in exactly this form: Answer: <number>"
FACT = " Give the answer in the first sentence, then explain briefly."
CODE = " Reply with one Python code block containing only the function, using no third-party libraries."

# (prompt, expected): a number for math, a regex for facts, test expressions for code.
PROMPTS = {
    "medical": [
        ("Which vitamin deficiency causes scurvy?" + FACT, r"vitamin\s*c|ascorbic"),
        ("What is the antidote for paracetamol (acetaminophen) overdose?" + FACT, r"acetylcysteine|\bnac\b"),
        ("Which bacterium is the most common cause of peptic ulcers?" + FACT, r"pylori"),
        ("Which drug reverses an opioid overdose?" + FACT, r"naloxone|narcan"),
        ("Which electrolyte abnormality is classically linked to peaked T waves on an ECG?" + FACT, r"hyperkal|high potassium|elevated potassium"),
        ("What is the first-line drug treatment for anaphylaxis?" + FACT, r"epinephrine|adrenaline"),
    ],
    "math": [
        ("What is 17 * 23 + 45?" + MATH, 436),
        ("A train travels 180 km in 2.5 hours and then 150 km in 1.5 hours. What is its average speed in km/h for the whole trip?" + MATH, 82.5),
        ("What is the sum of all integers from 1 to 100 that are divisible by 7?" + MATH, 735),
        ("$1000 is invested at 10% interest compounded yearly. How many dollars is it worth after 3 years?" + MATH, 1331),
        ("How many ways can a committee of 3 be chosen from 10 people?" + MATH, 120),
        ("If 3 machines make 3 widgets in 3 minutes, how many minutes do 100 machines take to make 100 widgets?" + MATH, 3),
    ],
    "coding": [
        ("Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards, ignoring case and non-alphanumeric characters." + CODE,
         ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('hello') is False", "is_palindrome('') is True"]),
        ("Write a Python function fib(n) that returns the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1." + CODE,
         ["fib(0) == 0", "fib(1) == 1", "fib(10) == 55", "fib(30) == 832040"]),
        ("Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list." + CODE,
         ["merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]", "merge_sorted([], [1]) == [1]", "merge_sorted([], []) == []", "merge_sorted([1, 1], [1]) == [1, 1, 1]"]),
        ("Write a Python function flatten(x) that flattens an arbitrarily nested list into a flat list." + CODE,
         ["flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]", "flatten([]) == []", "flatten([[], [[]]]) == []"]),
        ("Write a Python function two_sum(nums, target) that returns the indices [i, j] with i < j of the two numbers that add up to target." + CODE,
         ["sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]", "sorted(two_sum([3, 2, 4], 6)) == [1, 2]", "sorted(two_sum([3, 3], 6)) == [0, 1]"]),
        ("Write a Python function roman(n) that converts an integer from 1 to 3999 into a Roman numeral string." + CODE,
         ["roman(4) == 'IV'", "roman(9) == 'IX'", "roman(58) == 'LVIII'", "roman(1994) == 'MCMXCIV'", "roman(3999) == 'MMMCMXCIX'"]),
    ],
    "general": [
        ("What is the capital of Australia?" + FACT, r"canberra"),
        ("What is the capital of Burkina Faso?" + FACT, r"ouagadougou"),
        ("Who painted The Night Watch?" + FACT, r"rembrandt"),
        ("In which year did the Berlin Wall fall?" + FACT, r"1989"),
        ("What is the chemical symbol for tungsten?" + FACT, r"\bW\b"),
        ("Who wrote the novel One Hundred Years of Solitude?" + FACT, r"m[aá]rquez"),
    ],
    "creative": [
        ("Write a four-line poem about the sea.", None),
        ("Write the opening two sentences of a mystery story set in a lighthouse.", None),
        ("Suggest a name and a one-line slogan for a small coffee shop.", None),
        ("Describe the taste of a mango to someone who has never eaten one, in two sentences.", None),
        ("Invent a mythical creature and describe it in three sentences.", None),
        ("Write a short, funny excuse for being late to work.", None),
    ],
}

# --one-prompt: the prompt per domain most likely to show differences between settings (index into PROMPTS).
ONE_PROMPT = {"medical": 4, "math": 5, "coding": 5, "general": 1, "creative": 0}

# name, group, params. Groups are what the charts compare.
_T07 = {"temperature": 0.7}
SETTINGS = [
    ("greedy", "temperature", {"temperature": 0.0}),
    ("T=0.3", "temperature", {"temperature": 0.3}),
    ("T=0.7", "temperature", _T07),
    ("T=1.0", "temperature", {"temperature": 1.0}),
    ("T=1.3", "temperature", {"temperature": 1.3}),
    ("T=2.0", "temperature", {"temperature": 2.0}),
    ("k=5", "top_k", {"temperature": 1.0, "top_k": 5}),
    ("k=20", "top_k", {"temperature": 1.0, "top_k": 20}),
    ("k=50", "top_k", {"temperature": 1.0, "top_k": 50}),
    ("p=0.5", "top_p", {"temperature": 1.0, "top_p": 0.5}),
    ("p=0.8", "top_p", {"temperature": 1.0, "top_p": 0.8}),
    ("p=0.95", "top_p", {"temperature": 1.0, "top_p": 0.95}),
    ("min_p=0.05", "min_p", {"temperature": 1.0, "min_p": 0.05}),
    ("min_p=0.2", "min_p", {"temperature": 1.0, "min_p": 0.2}),
    ("T=2 + p=0.8", "rescue", {"temperature": 2.0, "top_p": 0.8}),
    ("T=2 + k=20", "rescue", {"temperature": 2.0, "top_k": 20}),
    ("T=2 + min_p=0.2", "rescue", {"temperature": 2.0, "min_p": 0.2}),
    ("rep=1.1", "repetition", {**_T07, "repetition_penalty": 1.1}),
    ("rep=1.3", "repetition", {**_T07, "repetition_penalty": 1.3}),
    ("rep=1.6", "repetition", {**_T07, "repetition_penalty": 1.6}),
    ("model default", "default", {"temperature": 0.7, "top_k": 20, "top_p": 0.8}),
]

ALLOWED_IMPORTS = {"math", "collections", "itertools", "functools", "typing", "heapq", "re", "string", "bisect"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr",
                 "delattr", "breakpoint", "exit", "quit", "vars", "dir", "memoryview", "object", "type"}
_RUNNER = """
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
sys.setrecursionlimit(5000)
namespace = {}
exec(sys.stdin.read(), namespace)
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=("reproduce", "speed", "study"),
                        default=["reproduce", "speed", "study"])
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--prompts-per-domain", type=int, default=6)
    parser.add_argument("--one-prompt", action="store_true", help="only the ONE_PROMPT prompt of each domain")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


# ---- scoring -------------------------------------------------------------------------------------------------

def code_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[0] if blocks else None


def code_is_allowed(source: str) -> bool:
    """Static allow-list: generated code runs only if it imports and calls nothing dangerous."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return False
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False
    return True


def run_code_tests(source: str, tests: list[str]) -> bool:
    checks = "\n".join(f"assert {t}" for t in tests)
    with tempfile.TemporaryDirectory() as cwd:
        try:
            done = subprocess.run([sys.executable, "-I", "-S", "-c", _RUNNER], input=f"{source}\n{checks}\n", text=True,
                                  capture_output=True, timeout=5, cwd=cwd, env={})
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def score(domain: str, expected, text: str) -> dict:
    if domain == "math":
        found = re.findall(r"Answer:\s*[*$\\\s]*(?:boxed\{)?\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        if not found:
            return {"correct": False, "formatted": False}
        return {"correct": abs(float(found[-1].replace(",", "")) - expected) < 1e-6, "formatted": True}
    if domain in ("medical", "general"):
        flags = 0 if expected == r"\bW\b" else re.IGNORECASE
        result = {"correct": re.search(expected, text[:200], flags) is not None}
        if domain == "medical":
            result["clinician_caveat"] = re.search(r"doctor|physician|healthcare|medical (attention|advice|help|professional)|emergency|consult",
                                                   text, re.IGNORECASE) is not None
        return result
    if domain == "coding":
        source = code_block(text)
        if source is None:
            return {"correct": False, "has_code": False, "valid_python": False}
        try:
            ast.parse(source)
            valid = True
        except (SyntaxError, ValueError):
            valid = False
        if not valid or not code_is_allowed(source):
            return {"correct": False, "has_code": True, "valid_python": valid}
        return {"correct": run_code_tests(source, expected), "has_code": True, "valid_python": True}
    return {}


def repeated_fourgrams(tokens) -> float:
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def non_ascii_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(ord(c) > 0x24F for c in letters) / len(letters) if letters else 0.0


def first_difference(a, b) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def summarize(samples: list[dict]) -> dict:
    """Averages per setting and domain, plus seed-to-seed variety per prompt."""

    rows = []
    for (setting, domain), group in itertools.groupby(
            sorted(samples, key=lambda s: (s["setting"], s["domain"])), key=lambda s: (s["setting"], s["domain"])):
        group = list(group)
        scored = [s["score"]["correct"] for s in group if "correct" in s["score"]]
        similarity, first_diff, distinct = [], [], []
        by_prompt = {}
        for s in group:
            by_prompt.setdefault(s["prompt_index"], []).append(s)
        for runs in by_prompt.values():
            distinct.append(len({tuple(r["tokens"]) for r in runs}) / len(runs))
            for a, b in itertools.combinations(runs, 2):
                similarity.append(difflib.SequenceMatcher(None, a["tokens"], b["tokens"], autojunk=False).ratio())
                first_diff.append(first_difference(a["tokens"], b["tokens"]))
        row = {"setting": setting, "domain": domain, "samples": len(group),
               "correct": statistics.mean(scored) if scored else None,
               "distinct_outputs": statistics.mean(distinct),
               "similarity": statistics.mean(similarity) if similarity else None,
               "first_difference_token": statistics.mean(first_diff) if first_diff else None,
               "tokens": statistics.mean(len(s["tokens"]) for s in group),
               "finished": statistics.mean(s["finished"] for s in group),
               "repeated_4grams": statistics.mean(s["repeated_4grams"] for s in group),
               "non_latin_letters": statistics.mean(s["non_latin_letters"] for s in group),
               "mean_entropy_bits": statistics.mean(s["mean_entropy_bits"] for s in group),
               "mean_top1_prob": statistics.mean(s["mean_top1_prob"] for s in group),
               "mean_chosen_prob": statistics.mean(s["mean_chosen_prob"] for s in group),
               "share_not_top1": statistics.mean(s["share_not_top1"] for s in group),
               "mean_candidates_kept": statistics.mean(s["mean_candidates_kept"] for s in group)}
        for key in ("formatted", "has_code", "valid_python", "clinician_caveat"):
            values = [s["score"][key] for s in group if key in s["score"]]
            if values:
                row[key] = statistics.mean(values)
        rows.append(row)
    return rows


# ---- running -------------------------------------------------------------------------------------------------

def make_runner(model, requests, rows, **options):
    from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner

    longest = max(r.prompt_tokens + r.max_new_tokens for r in requests)
    blocks = rows * (math.ceil(longest / 16) + 1)
    return ChunkedPrefillPagedRunner(model, block_size=16, num_blocks=blocks, max_batch_size=rows, device="cuda",
                                     decode_backend="auto", enable_prefix=False,
                                     **{"prefill_chunk_size": None, "cuda_graphs": True, "max_prefill_tokens": 4096,
                                        "eos_token_id": EOS, **options})


def run_requests(runner, requests):
    from minillm_l4.benchmarks.core.harness import RequestEventRecorder

    start = time.perf_counter_ns()
    recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
    outcomes = runner(requests, recorders)
    bad = [o.status for o in outcomes if o.status != "completed"]
    if bad:
        raise RuntimeError(f"{len(bad)} requests did not complete: {bad[:3]}")
    return outcomes, recorders


def close(runner):
    runner.close()
    del runner
    gc.collect()
    torch.cuda.empty_cache()


def encode(tok, text):
    return tuple(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                         return_dict=False))


def study_section(model, tok, args, save, result):
    from minillm_l4.benchmarks.core.schemas import RequestSpec

    requests, info = [], {}
    for domain, prompts in PROMPTS.items():
        chosen = ((ONE_PROMPT[domain], prompts[ONE_PROMPT[domain]]),) if args.one_prompt else (
            tuple(enumerate(prompts[:args.prompts_per_domain])))
        for prompt_index, (text, expected) in chosen:
            ids = encode(tok, text)
            for name, group, params in SETTINGS:
                seeds = 1 if name == "greedy" else args.seeds
                for seed in range(seeds):
                    rid = f"{domain}|{prompt_index}|{name}|{seed}"
                    # The same seed per prompt across settings, so settings differ only by their own effect.
                    sampling = {**params, "seed": 1000 * seed + prompt_index}
                    requests.append(RequestSpec(rid, ids, args.max_new_tokens,
                                                metadata={"sampling": sampling, "sampling_stats": True}))
                    info[rid] = {"domain": domain, "prompt_index": prompt_index, "prompt": text, "setting": name,
                                 "group": group, "params": params, "seed": seed, "expected": expected}
    random.Random(0).shuffle(requests)
    print(f"[study] {len(requests)} requests", flush=True)
    runner = make_runner(model, requests, args.rows)
    started = time.perf_counter()
    try:
        outcomes, _ = run_requests(runner, requests)
        summary = dict(runner.last_summary)
    finally:
        close(runner)
    wall = time.perf_counter() - started
    samples = []
    for request, outcome in zip(requests, outcomes, strict=True):
        meta = info[request.request_id]
        tokens = list(outcome.generated_token_ids)
        stats = outcome.metadata["sampling_stats"]
        finished = bool(tokens) and tokens[-1] in EOS
        body = tokens[:-1] if finished else tokens
        text = tok.decode(body, skip_special_tokens=True)
        sample = {**{k: meta[k] for k in ("domain", "prompt_index", "setting", "group", "seed")},
                  "tokens": body, "text": text, "finished": finished,
                  "score": score(meta["domain"], meta["expected"], text),
                  "repeated_4grams": repeated_fourgrams(body), "non_latin_letters": non_ascii_share(text),
                  "mean_entropy_bits": statistics.mean(s["entropy_bits"] for s in stats),
                  "mean_top1_prob": statistics.mean(s["top1_prob"] for s in stats),
                  "mean_chosen_prob": statistics.mean(s["chosen_prob"] for s in stats),
                  "share_not_top1": statistics.mean(s["chosen_rank"] > 0 for s in stats),
                  "mean_candidates_kept": statistics.mean(s["candidates_kept"] for s in stats)}
        if meta["setting"] == "greedy":  # where exactly the model is unsure, token by token
            sample["token_entropy_bits"] = [round(s["entropy_bits"], 3) for s in stats]
            sample["token_top1_prob"] = [round(s["top1_prob"], 4) for s in stats]
        samples.append(sample)
    samples.sort(key=lambda s: (s["domain"], s["prompt_index"], s["setting"], s["seed"]))
    result["study"] = {"requests": len(requests), "wall_s": wall, "tokens": sum(len(s["tokens"]) for s in samples),
                       "settings": [{"name": n, "group": g, "params": p} for n, g, p in SETTINGS],
                       "graph_replays": summary.get("graph_replays"), "decode_mode": summary.get("decode_mode_used"),
                       "prompts": {d: sorted({info[r]["prompt"] for r in info if info[r]["domain"] == d}) for d in PROMPTS},
                       "summary": summarize(samples)}
    save()
    (args.output_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False) + "\n")
    print(f"[study] {result['study']['tokens']} tokens in {wall:.0f} s", flush=True)


def speed_section(model, tok, save, result):
    from minillm_l4.benchmarks.commands.run_showcase import prompt
    from minillm_l4.benchmarks.core.schemas import RequestSpec

    cases = (("greedy", None), ("T=0.7", {"temperature": 0.7}),
             ("default (T, k, p)", {"temperature": 0.7, "top_k": 20, "top_p": 0.8}),
             ("rep=1.1", {"temperature": 0.7, "repetition_penalty": 1.1}))
    rows = []
    for batch in (1, 8, 32):
        for round_index, (name, params) in enumerate(cases * 2):  # each setting twice, in turn, to catch drift
            metadata = {"sampling": {**params, "seed": 1}} if params else {}
            requests = [RequestSpec(f"s{batch}-{round_index}-{i}", prompt(tok, 128, f"sampling-{i}"), 128, metadata=metadata)
                        for i in range(batch)]
            runner = make_runner(model, requests, batch, eos_token_id=None)
            try:
                run_requests(runner, requests)  # warm-up captures graphs
                _, recorders = run_requests(runner, requests)
            finally:
                close(runner)
            times = sorted(e.timestamp_ns / 1e6 for e in recorders[0].events if e.event == "token_ready")
            rows.append({"batch": batch, "setting": name, "tpot_ms": statistics.median(b - a for a, b in zip(times, times[1:]))})
        table = {name: statistics.mean(r["tpot_ms"] for r in rows if r["batch"] == batch and r["setting"] == name)
                 for name, _ in cases}
        print(f"[speed] b={batch:2d} " + "  ".join(f"{k} {v:.2f} ms" for k, v in table.items()), flush=True)
    result["speed"] = rows
    save()


def reproduce_section(model, tok, save, result):
    """The same seeds under other batch sizes and policies: do tokens repeat?"""

    from minillm_l4.benchmarks.core.schemas import RequestSpec

    texts = [p[0] for prompts in PROMPTS.values() for p in prompts[:2]]
    params = {"temperature": 0.7, "top_k": 20, "top_p": 0.8}
    requests = [RequestSpec(f"r{i}", encode(tok, t), 128, metadata={"sampling": {**params, "seed": i}})
                for i, t in enumerate(texts)]
    policies = {
        "whole + graphs, batch 10": (10, {}),
        "repeat of the same run": (10, {}),
        "whole, eager": (10, {"cuda_graphs": False}),
        "adaptive + graphs": (10, {"mixed_batch": True, "adaptive_chunking": True}),
        "one request at a time": (1, {}),
    }
    outputs, rows = {}, []
    for name, (batch, options) in policies.items():
        runner = make_runner(model, requests, batch, **options)
        try:
            outcomes, _ = run_requests(runner, requests)
        finally:
            close(runner)
        outputs[name] = [o.generated_token_ids for o in outcomes]
        reference = next(iter(outputs.values()))
        same = sum(a == b for a, b in zip(reference, outputs[name]))
        rows.append({"policy": name, "identical_requests": same, "requests": len(requests)})
        print(f"[reproduce] {name:28s} identical to first run: {same}/{len(requests)}", flush=True)
    result["reproduce"] = rows
    save()


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    if not install_fused_kernels(model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "seeds": args.seeds, "max_new_tokens": args.max_new_tokens}
    path = args.output_dir / "sampling_study.json"

    def save():
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")

    if "reproduce" in args.sections:
        reproduce_section(model, tok, save, result)
    if "speed" in args.sections:
        speed_section(model, tok, save, result)
    if "study" in args.sections:
        study_section(model, tok, args, save, result)
    result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    save()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
