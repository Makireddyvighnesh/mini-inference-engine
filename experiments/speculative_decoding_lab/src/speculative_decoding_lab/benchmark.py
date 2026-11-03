"""Run matched baseline/speculative tests against the configured llama.cpp server."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .metrics import summarize
from .server import build_server_command, project_root, resolve_path
from .tokenizer_check import compare_tokenizers, tokenizer_fingerprint


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def _record_text_piece(text_parts: list[str], piece: Any, terminal: bool) -> None:
    """Keep streamed text, replacing it only when the final event has text."""
    if not isinstance(piece, str) or not piece:
        return
    if terminal:
        text_parts[:] = [piece]
    else:
        text_parts.append(piece)


def _completion_text(base_url: str, token_ids: list[int], text_parts: list[str]) -> str:
    """Use streamed text when available, otherwise decode the returned token IDs."""
    if text_parts:
        return "".join(text_parts)
    response = _http_json(
        f"{base_url}/detokenize", {"tokens": token_ids}, timeout=60
    )
    if not isinstance(response, dict) or not isinstance(response.get("content"), str):
        raise RuntimeError(f"llama-server /detokenize returned an invalid response: {response!r}")
    return response["content"]


def _iter_sse_data(response: Any):
    """Yield complete SSE data payloads and reject a truncated final event."""
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        raise RuntimeError("SSE connection ended in the middle of an event")


def _stream_chat(
    base_url: str,
    prompt: str,
    prompt_id: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    samplers: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    template_result = _http_json(
        f"{base_url}/apply-template",
        {"messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    if not isinstance(template_result, dict) or "error" in template_result:
        raise RuntimeError(f"llama-server template request failed: {template_result!r}")
    rendered_prompt = template_result.get("prompt")
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise RuntimeError("llama-server /apply-template returned no prompt")
    template_finished = time.perf_counter()

    payload = {
        "prompt": rendered_prompt,
        "n_predict": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "samplers": samplers,
        "stream": True,
        "return_tokens": True,
        "timings_per_token": True,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    generation_started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    token_ids: list[int] = []
    timings: dict[str, Any] = {}
    stop_reason: str | None = None
    saw_stop = False
    saw_partial_token = False

    with urllib.request.urlopen(request, timeout=600) as response:
        for data in _iter_sse_data(response):
            if data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON in llama-server SSE event: {data!r}") from error
            if not isinstance(event, dict):
                raise RuntimeError(f"unexpected llama-server SSE event: {event!r}")
            if "error" in event:
                raise RuntimeError(f"llama-server inference failed: {event['error']!r}")
            if isinstance(event.get("timings"), dict):
                timings.update(event["timings"])
            piece = event.get("content")
            event_tokens = event.get("tokens")
            if event_tokens is not None:
                if not isinstance(event_tokens, list) or any(
                    isinstance(token, bool) or not isinstance(token, int)
                    for token in event_tokens
                ):
                    raise RuntimeError(f"llama-server returned invalid token IDs: {event_tokens!r}")
                if event_tokens:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    if event.get("stop") is True:
                        # The terminal native response carries the complete result.
                        token_ids = list(event_tokens)
                    else:
                        token_ids.extend(event_tokens)
                        saw_partial_token = True
            elif piece and event.get("stop") is not True:
                raise RuntimeError("llama-server emitted text without raw token IDs")
            if event.get("stop") is True:
                saw_stop = True
                # The final native response repeats the full content after the
                # incremental stream, so replace it only when non-empty. Some
                # server builds return raw token IDs with an empty final content.
                _record_text_piece(text_parts, piece, terminal=True)
                reason = event.get("stop_type")
                if isinstance(reason, str) and reason:
                    stop_reason = reason
            else:
                _record_text_piece(text_parts, piece, terminal=False)

    stream_finished = time.perf_counter()
    if not saw_stop:
        raise RuntimeError("llama-server stream ended before its terminal stop event")
    if first_token_at is None or not token_ids:
        raise RuntimeError("llama-server completed without returning generated token IDs")
    if len(token_ids) > 1 and not saw_partial_token:
        raise RuntimeError("llama-server returned no token events before its terminal response; TTFT is unavailable")
    if stop_reason is None:
        raise RuntimeError("llama-server terminal event did not include stop_type")
    completion_text = _completion_text(base_url, token_ids, text_parts)
    finished = time.perf_counter()
    required_timings = {
        "predicted_n",
        "predicted_ms",
        "predicted_per_second",
        "prompt_per_second",
    }
    missing_timings = required_timings - timings.keys()
    if missing_timings:
        raise RuntimeError(
            "llama-server completion omitted timing fields: "
            + ", ".join(sorted(missing_timings))
        )
    return {
        "prompt_id": prompt_id,
        "text": completion_text,
        "token_ids": token_ids,
        "stop_reason": stop_reason,
        "ttft_ms": (first_token_at - started) * 1000.0,
        "generation_ttft_ms": (first_token_at - generation_started) * 1000.0,
        "template_ms": (template_finished - started) * 1000.0,
        "stream_e2e_ms": (stream_finished - started) * 1000.0,
        "postprocessing_ms": (finished - stream_finished) * 1000.0,
        "client_e2e_ms": (finished - started) * 1000.0,
        "server_timings": timings,
    }


def _wait_until_ready(base_url: str, process: subprocess.Popen[Any], timeout_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            _http_json(f"{base_url}/health", timeout=2.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1.0)
    detail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
    raise RuntimeError(f"llama-server failed to become ready; see {log_path}\n{detail}")


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _run_mode(
    config: dict[str, Any],
    mode: str,
    port: int,
    prompts: list[dict[str, str]],
    run_dir: Path,
    repetition: int,
) -> list[dict[str, Any]]:
    runtime = config["runtime"]
    benchmark = config["benchmark"]
    host = str(runtime["host"])
    if not _port_is_free(host, port):
        raise RuntimeError(f"port {port} on {host} is already in use; no server was started")

    command, env = build_server_command(config, mode, port)
    log_path = run_dir / f"{mode}_rep_{repetition + 1}_server.log"
    base_url = f"http://{host}:{port}"
    rows: list[dict[str, Any]] = []
    measurement_started: float | None = None
    measurement_ended: float | None = None
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
                base_url,
                process,
                float(runtime["startup_timeout_seconds"]),
                log_path,
            )
            for _ in range(int(benchmark["warmup_repetitions"])):
                for prompt in prompts:
                    _stream_chat(
                        base_url,
                        prompt["text"],
                        prompt["id"],
                        int(benchmark["max_new_tokens"]),
                        float(benchmark["temperature"]),
                        int(benchmark["seed"]),
                        list(benchmark["samplers"]),
                    )
            measurement_started = time.perf_counter()
            for prompt in prompts:
                result = _stream_chat(
                    base_url,
                    prompt["text"],
                    prompt["id"],
                    int(benchmark["max_new_tokens"]),
                    float(benchmark["temperature"]),
                    int(benchmark["seed"]),
                    list(benchmark["samplers"]),
                )
                result.update(
                    {
                        "mode": mode,
                        "repetition": repetition,
                        "category": prompt.get("category"),
                    }
                )
                rows.append(result)
                print(
                    f"{mode:11s} {prompt['id']:14s} rep={repetition + 1} "
                    f"TTFT={result['ttft_ms']:.1f} ms "
                    f"E2E={result['client_e2e_ms']:.1f} ms "
                    f"decode={result['server_timings']['predicted_per_second']:.2f} tok/s"
                )
            measurement_ended = time.perf_counter()
        finally:
            _stop_server(process)

    measured_ms = (
        (measurement_ended - measurement_started) * 1000.0
        if measurement_started is not None and measurement_ended is not None
        else 0.0
    )
    for row in rows:
        row["mode_measurement_duration_ms"] = measured_ms
    return rows


def _summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    generation_ttft = [
        row["generation_ttft_ms"]
        for row in rows
        if row.get("generation_ttft_ms") is not None
    ]
    template_times = [row["template_ms"] for row in rows if row.get("template_ms") is not None]
    e2e = [row["client_e2e_ms"] for row in rows]
    stream_e2e = [row["stream_e2e_ms"] for row in rows if row.get("stream_e2e_ms") is not None]
    postprocessing = [
        row["postprocessing_ms"] for row in rows if row.get("postprocessing_ms") is not None
    ]
    decode_tps = [
        float(row["server_timings"]["predicted_per_second"])
        for row in rows
        if row["server_timings"].get("predicted_per_second") is not None
    ]
    prompt_tps = [
        float(row["server_timings"]["prompt_per_second"])
        for row in rows
        if row["server_timings"].get("prompt_per_second") is not None
    ]
    predicted_n = sum(int(row["server_timings"].get("predicted_n", 0)) for row in rows)
    draft_n = sum(int(row["server_timings"].get("draft_n", 0)) for row in rows)
    accepted_n = sum(int(row["server_timings"].get("draft_n_accepted", 0)) for row in rows)
    duration_by_repetition: dict[int, float] = {}
    for row in rows:
        repetition = int(row["repetition"])
        duration_by_repetition[repetition] = float(
            row.get("mode_measurement_duration_ms", 0.0)
        )
    measured_duration_ms = sum(duration_by_repetition.values())
    tpot_values = []
    decode_ms = 0.0
    decode_steps = 0
    for row in rows:
        generated_tokens = int(row["server_timings"].get("predicted_n", 0))
        request_decode_ms = row["server_timings"].get("predicted_ms")
        if generated_tokens > 1 and request_decode_ms is not None:
            request_decode_ms = float(request_decode_ms)
            steps = generated_tokens - 1
            tpot_values.append(request_decode_ms / steps)
            decode_ms += request_decode_ms
            decode_steps += steps
    return {
        "requests": len(rows),
        "ttft_ms": summarize(ttft),
        "generation_ttft_ms": summarize(generation_ttft),
        "template_ms": summarize(template_times),
        "client_e2e_ms": summarize(e2e),
        "stream_e2e_ms": summarize(stream_e2e),
        "postprocessing_ms": summarize(postprocessing),
        "decode_tpot_ms": summarize(tpot_values),
        "decode_tokens_per_second": summarize(decode_tps),
        "prefill_tokens_per_second": summarize(prompt_tps),
        "measured_aggregate_output_tokens_per_second": (
            predicted_n / (measured_duration_ms / 1000.0)
            if measured_duration_ms > 0
            else None
        ),
        "measured_duration_ms": measured_duration_ms,
        "server_decode_tokens": predicted_n,
        "draft_tokens": draft_n,
        "accepted_draft_tokens": accepted_n,
        "draft_acceptance_rate": accepted_n / draft_n if draft_n else None,
        "mean_decode_tpot_ms": (
            decode_ms / decode_steps if decode_steps else None
        ),
    }


def _correctness(baseline: list[dict[str, Any]], speculative: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_key = {
        (row["prompt_id"], row["repetition"]): row for row in baseline
    }
    spec_by_key = {
        (row["prompt_id"], row["repetition"]): row for row in speculative
    }
    shared = sorted(set(base_by_key) & set(spec_by_key))
    mismatches = [
        key
        for key in shared
        if base_by_key[key].get("token_ids") != spec_by_key[key].get("token_ids")
        or base_by_key[key].get("stop_reason") != spec_by_key[key].get("stop_reason")
        or base_by_key[key].get("text") != spec_by_key[key].get("text")
    ]
    invalid = [
        key
        for key in shared
        if not base_by_key[key].get("token_ids")
        or not spec_by_key[key].get("token_ids")
        or not base_by_key[key].get("stop_reason")
        or not spec_by_key[key].get("stop_reason")
        or not isinstance(base_by_key[key].get("text"), str)
        or not base_by_key[key]["text"].strip()
        or not isinstance(spec_by_key[key].get("text"), str)
        or not spec_by_key[key]["text"].strip()
    ]
    duplicate_keys = (
        len(base_by_key) != len(baseline) or len(spec_by_key) != len(speculative)
    )
    return {
        "matched_requests": len(shared),
        "exact_token_matches": sum(
            bool(base_by_key[key].get("token_ids"))
            and base_by_key[key].get("token_ids") == spec_by_key[key].get("token_ids")
            for key in shared
        ),
        "exact_text_matches": sum(
            isinstance(base_by_key[key].get("text"), str)
            and bool(base_by_key[key]["text"].strip())
            and base_by_key[key].get("text") == spec_by_key[key].get("text")
            for key in shared
        ),
        "invalid_requests": [
            {"prompt_id": key[0], "repetition": key[1]} for key in invalid
        ],
        "duplicate_request_keys": duplicate_keys,
        "mismatches": [
            {"prompt_id": key[0], "repetition": key[1]} for key in mismatches
        ],
        "passed": (
            bool(shared)
            and not mismatches
            and not invalid
            and not duplicate_keys
            and len(shared) == len(baseline) == len(speculative)
        ),
    }


def run(
    config_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    mode_order: str = "random",
) -> Path | None:
    root = project_root()
    run_started = datetime.now(timezone.utc)
    resolved_config = config_path if config_path.is_absolute() else (root / config_path).resolve()
    config_bytes = resolved_config.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    prompt_path = resolve_path(root, config["benchmark"]["prompts_file"])
    prompts_bytes = prompt_path.read_bytes()
    prompts_config = yaml.safe_load(prompts_bytes.decode("utf-8"))
    prompts = prompts_config["prompts"]

    base_port = int(config["runtime"]["base_port"])
    modes = config["benchmark"]["modes"]
    if dry_run:
        for index, mode in enumerate(modes):
            command, _ = build_server_command(
                config, mode, base_port + index, validate_models=False
            )
            print(f"{mode}: {' '.join(command)}")
        return None

    teacher_path = resolve_path(root, config["models"]["teacher"]["file"])
    draft_path = resolve_path(root, config["models"]["draft"]["file"])
    mismatches = compare_tokenizers(teacher_path, draft_path)
    if mismatches:
        fields = ", ".join(mismatches)
        raise RuntimeError(
            "Draft/teacher tokenizer incompatibility; refusing speculative run. "
            f"Mismatched GGUF fields: {fields}"
        )
    print("Tokenizer gate: exact metadata and token-ID ordering match")

    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = resolve_path(root, config["benchmark"]["results_dir"]) / timestamp
    elif not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "experiment.yaml"
    prompts_snapshot = output_dir / "prompts.yaml"
    config_snapshot.write_bytes(config_bytes)
    prompts_snapshot.write_bytes(prompts_bytes)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    run_modes = list(modes)
    if set(run_modes) == {"baseline", "speculative"}:
        if mode_order == "random":
            if secrets.randbelow(2):
                run_modes.reverse()
        elif mode_order == "baseline-first":
            run_modes = ["baseline", "speculative"]
        elif mode_order == "speculative-first":
            run_modes = ["speculative", "baseline"]
        else:
            raise ValueError(f"unsupported mode order: {mode_order}")
    elif mode_order not in {"random", "baseline-first", "speculative-first"}:
        raise ValueError(f"unsupported mode order: {mode_order}")

    mode_order_by_repetition: dict[str, list[str]] = {}
    for repetition in range(int(config["benchmark"]["repetitions"])):
        repetition_modes = run_modes if repetition % 2 == 0 else list(reversed(run_modes))
        mode_order_by_repetition[str(repetition)] = list(repetition_modes)
        for index, mode in enumerate(repetition_modes):
            rows = _run_mode(
                config,
                mode,
                base_port + index,
                prompts,
                output_dir,
                repetition,
            )
            all_rows.setdefault(mode, []).extend(rows)

    for mode, rows in all_rows.items():
        raw_path = output_dir / f"{mode}.jsonl"
        raw_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = {mode: _summarize_mode(rows) for mode, rows in all_rows.items()}
    gate_failures: list[str] = []
    if "baseline" in all_rows and "speculative" in all_rows:
        summary["correctness"] = _correctness(all_rows["baseline"], all_rows["speculative"])
        if not summary["correctness"]["passed"]:
            gate_failures.append("baseline and speculative token outputs differ or are incomplete")
        drafted = summary["speculative"]["draft_tokens"]
        summary["speculative_gate"] = {
            "draft_tokens_observed": drafted,
            "passed": drafted > 0,
        }
        if drafted <= 0:
            gate_failures.append("speculative mode completed without drafting any tokens")
        base_tps = summary["baseline"]["measured_aggregate_output_tokens_per_second"]
        spec_tps = summary["speculative"]["measured_aggregate_output_tokens_per_second"]
        summary["aggregate_output_tps_speedup"] = (
            spec_tps / base_tps
            if summary["correctness"]["passed"] and drafted > 0 and base_tps and spec_tps
            else None
        )
    summary["gates"] = {"passed": not gate_failures, "failures": gate_failures}

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def file_record(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    input_files = {
        "experiment_config": file_record(config_snapshot),
        "prompts": file_record(prompts_snapshot),
    }

    runtime_command, runtime_env = build_server_command(config, "baseline", base_port)
    version_result = subprocess.run(
        [runtime_command[0], "--version"],
        cwd=root,
        env=runtime_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    _, teacher_tokenizer_hash = tokenizer_fingerprint(teacher_path)
    _, draft_tokenizer_hash = tokenizer_fingerprint(draft_path)
    manifest = {
        "project": config["project"],
        "started_utc": run_started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "config_path": str(resolved_config),
        "mode_order_by_repetition": mode_order_by_repetition,
        "input_files": input_files,
        "teacher": config["models"]["teacher"],
        "draft": config["models"]["draft"],
        "teacher_file": file_record(teacher_path),
        "draft_file": file_record(draft_path),
        "tokenizer_sha256": {
            "teacher": teacher_tokenizer_hash,
            "draft": draft_tokenizer_hash,
        },
        "runtime_binary": runtime_command[0],
        "runtime_version": (version_result.stdout + version_result.stderr).strip(),
        "gpu_info": (
            gpu_result.stdout.strip()
            if gpu_result.returncode == 0
            else f"nvidia-smi unavailable: {gpu_result.stderr.strip()}"
        ),
        "results_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Results: {output_dir}")
    if gate_failures:
        raise SystemExit("Benchmark gates failed: " + "; ".join(gate_failures))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode-order",
        choices=("random", "baseline-first", "speculative-first"),
        default="random",
        help="Choose server startup order; the selected order is recorded in the manifest",
    )
    args = parser.parse_args()
    run(args.config, args.output_dir, args.dry_run, args.mode_order)


if __name__ == "__main__":
    main()
