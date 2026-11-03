"""Focused live EOS and sampled-distribution checks for one llama.cpp model pair.

The sampled check is a statistical smoke test, not proof of exact equality of
the two generation distributions. It must not compare sampled paths seed by
seed: the two modes can consume random numbers differently.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .benchmark import _port_is_free, _stop_server, _stream_chat, _wait_until_ready
from .server import build_server_command, project_root, resolve_path
from .tokenizer_check import compare_tokenizers


EOS_PROMPT = "Reply with exactly the word OK and then stop. /no_think"
ANSWER_CASES = {
    "ok": (EOS_PROMPT, "OK"),
    "arithmetic": ("What is 7 + 5? Reply with only the number. /no_think", "12"),
    "json": (
        "Return a JSON object with exactly one key named ok whose value is true. No markdown. /no_think",
        {"ok": True},
    ),
}
SAMPLING_PROMPT = "Write a single surprising sentence about a dragon. /no_think"
SAMPLING_TOKEN_POSITION = 7  # First variable position observed for this pinned pair/prompt.


def _total_variation(left: Counter[int], right: Counter[int], sample_count: int) -> float:
    return sum(abs(left[token] - right[token]) for token in left.keys() | right.keys()) / (2 * sample_count)


def _permutation_p_value(left: list[int], right: list[int], permutations: int = 2000) -> tuple[float, float]:
    """Compare two empirical categorical distributions without assuming paired RNG streams."""
    if len(left) != len(right) or not left:
        raise ValueError("non-empty, equally sized samples are required")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    size = len(left)
    observed = _total_variation(Counter(left), Counter(right), size)
    pooled = list(left) + list(right)
    rng = random.Random(20260925)
    at_least_as_different = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        simulated = _total_variation(Counter(pooled[:size]), Counter(pooled[size:]), size)
        at_least_as_different += simulated >= observed - 1e-12
    return observed, (at_least_as_different + 1) / (permutations + 1)


def _answer_text(text: str) -> str:
    return text.partition("</think>")[2].strip() if "</think>" in text else text.strip()


def _answer_is_correct(text: str, expected: str | dict[str, Any]) -> bool:
    answer = _answer_text(text)
    if isinstance(expected, str):
        return answer == expected
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, dict)
        and parsed.keys() == expected.keys()
        and all(type(parsed[key]) is type(value) and parsed[key] == value for key, value in expected.items())
    )


def _run_mode(
    config: dict[str, Any], mode: str, output_dir: Path, port: int, samples: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    host = str(config["runtime"]["host"])
    if not _port_is_free(host, port):
        raise RuntimeError(f"port {port} on {host} is already in use")
    command, env = build_server_command(config, mode, port)
    base_url = f"http://{host}:{port}"
    log_path = output_dir / f"{mode}_server.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=project_root(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_until_ready(
                base_url, process, float(config["runtime"]["startup_timeout_seconds"]), log_path
            )
            _stream_chat(base_url, EOS_PROMPT, "eos_warmup", 128, 0.0, 1234, ["temperature"])
            answers = {
                case_id: _stream_chat(base_url, prompt, case_id, 128, 0.0, 1234, ["temperature"])
                for case_id, (prompt, _) in ANSWER_CASES.items()
            }
            sampled = []
            seed_offset = 0 if mode == "baseline" else 10_000
            for sample_index in range(samples):
                seed = seed_offset + sample_index
                row = _stream_chat(
                    base_url, SAMPLING_PROMPT, "sampling", 16, 1.0, seed, ["temperature"]
                )
                row["seed"] = seed
                sampled.append(row)
                if (sample_index + 1) % 16 == 0 or sample_index + 1 == samples:
                    print(f"{mode}: sampled {sample_index + 1}/{samples}", flush=True)
            return answers, sampled
        finally:
            _stop_server(process)


def run(config_path: Path, output_dir: Path | None = None, samples: int = 64) -> dict[str, Any]:
    if samples < 8:
        raise ValueError("at least eight samples per mode are required")
    root = project_root()
    resolved_config = config_path if config_path.is_absolute() else (root / config_path).resolve()
    config_bytes = resolved_config.read_bytes()
    config = yaml.safe_load(config_bytes)
    teacher = resolve_path(root, config["models"]["teacher"]["file"])
    draft = resolve_path(root, config["models"]["draft"]["file"])
    mismatches = compare_tokenizers(teacher, draft)
    if mismatches:
        raise RuntimeError(f"draft/teacher tokenizer mismatch: {', '.join(mismatches)}")

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = root / "results" / f"validation_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.yaml").write_bytes(config_bytes)

    all_results = {}
    for mode in ("baseline", "speculative"):
        answers, sampled = _run_mode(
            config, mode, output_dir, int(config["runtime"]["base_port"]) + 20, samples
        )
        all_results[mode] = {"answers": answers, "sampled": sampled}
        (output_dir / f"{mode}_sampling.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sampled), encoding="utf-8"
        )

    baseline = all_results["baseline"]
    speculative = all_results["speculative"]
    answer_checks = {}
    for case_id, (_, expected) in ANSWER_CASES.items():
        base_answer = baseline["answers"][case_id]
        spec_answer = speculative["answers"][case_id]
        answer_checks[case_id] = {
            "passed": (
                base_answer["stop_reason"] == spec_answer["stop_reason"] == "eos"
                and base_answer["token_ids"] == spec_answer["token_ids"]
                and base_answer["text"] == spec_answer["text"]
                and _answer_is_correct(base_answer["text"], expected)
            ),
            "expected": expected,
            "baseline": base_answer,
            "speculative": spec_answer,
        }
    answers_passed = all(result["passed"] for result in answer_checks.values())
    sampled_tokens = {}
    for mode, result in all_results.items():
        rows = result["sampled"]
        if any(len(row["token_ids"]) <= SAMPLING_TOKEN_POSITION for row in rows):
            raise RuntimeError(f"{mode} produced fewer than {SAMPLING_TOKEN_POSITION + 1} tokens")
        sampled_tokens[mode] = [row["token_ids"][SAMPLING_TOKEN_POSITION] for row in rows]
    observed_tv, p_value = _permutation_p_value(
        sampled_tokens["baseline"], sampled_tokens["speculative"]
    )
    spec_draft_count = sum(
        int(row["server_timings"].get("draft_n", 0)) for row in speculative["sampled"]
    )
    sampling_passed = (
        spec_draft_count > 0
        and len(set(sampled_tokens["baseline"])) > 1
        and len(set(sampled_tokens["speculative"])) > 1
        and p_value >= 0.01
    )
    summary = {
        "model_pair": {"teacher": config["models"]["teacher"], "draft": config["models"]["draft"]},
        "answers": {
            "passed": answers_passed,
            "cases": answer_checks,
        },
        "sampling": {
            "passed_statistical_smoke_test": sampling_passed,
            "prompt": SAMPLING_PROMPT,
            "temperature": 1.0,
            "max_new_tokens": 16,
            "samples_per_mode": samples,
            "first_seed_by_mode": {"baseline": 0, "speculative": 10_000},
            "token_position_zero_based": SAMPLING_TOKEN_POSITION,
            "baseline_token_counts": dict(Counter(sampled_tokens["baseline"])),
            "speculative_token_counts": dict(Counter(sampled_tokens["speculative"])),
            "observed_total_variation": observed_tv,
            "permutation_p_value": p_value,
            "speculative_draft_tokens": spec_draft_count,
            "note": "Failure to reject a difference is not proof that complete sampled distributions match.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"EOS and completed-answer exact-match: {'pass' if answers_passed else 'fail'}", flush=True)
    print(
        f"Sampling statistical smoke: {'pass' if sampling_passed else 'fail'} "
        f"(TV={observed_tv:.3f}, p={p_value:.3f})",
        flush=True,
    )
    print(f"Results: {output_dir}", flush=True)
    if not answers_passed or not sampling_passed:
        raise SystemExit("live validation failed; inspect summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.samples)


if __name__ == "__main__":
    main()
