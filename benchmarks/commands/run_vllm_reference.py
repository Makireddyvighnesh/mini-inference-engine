"""Run a matched vLLM reference benchmark for the MiniLLM-L4 workloads.

This command is intentionally executed with the separate vLLM environment. It
uses vLLM's async streaming engine so first-token and per-token timestamps are
captured instead of inferring latency from final completion time.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD_DIR = PROJECT_ROOT / "results/paged_graph_matrix_20260919"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/vllm_reference"
DEFAULT_MODEL = (
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--"
    "Qwen3-4B-Instruct-2507-FP8/snapshots/"
    "8591804019c8b22094c3b5b4454e0edc05dffc98"
)
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MODEL_REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM against the MiniLLM-L4 paged-graph workloads."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workload-dir", type=Path, default=DEFAULT_WORKLOAD_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("short", "medium", "long", "xlong", "xxlong"),
        default=("short", "medium", "long"),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graphs; default keeps the normal vLLM path.",
    )
    return parser.parse_args()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        return {"available": False, "count": 0}
    ordered = sorted(normalized)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "standard_deviation": (
            statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
        ),
    }


def _prompt_sha256(prompt_token_ids: Sequence[int]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little") for token in prompt_token_ids)
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_reference(reference_dir: Path, bucket: str) -> dict[str, list[int]]:
    payload = _load_json(reference_dir / f"{bucket}.json")
    references: dict[str, list[int]] = {}
    request_payloads = payload.get("requests", [])
    if isinstance(request_payloads, Mapping):
        request_payloads = request_payloads.values()
    for request in request_payloads:
        prompt_sha = str(request.get("prompt_sha256", ""))
        if prompt_sha:
            references[prompt_sha] = [int(token) for token in request["generated_token_ids"]]
    return references


def _metric_record(
    request: Mapping[str, Any],
    generated_token_ids: Sequence[int],
    token_times_ns: Sequence[int],
    arrival_ns: int,
) -> dict[str, Any]:
    if not token_times_ns:
        raise RuntimeError(f"vLLM produced no streamed tokens for {request['request_id']}")
    times = [int(value) for value in token_times_ns]
    first_ns = times[0]
    last_ns = times[-1]
    itl_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(times, times[1:])
    ]
    generated = len(generated_token_ids)
    decode_ms = (last_ns - first_ns) / 1_000_000.0
    e2e_ms = (last_ns - arrival_ns) / 1_000_000.0
    tpot_ms = decode_ms / (generated - 1) if generated > 1 else None
    return {
        "request_id": str(request["request_id"]),
        "prompt_tokens": len(request["prompt_token_ids"]),
        "requested_output_tokens": int(request["max_new_tokens"]),
        "generated_output_tokens": generated,
        "generated_token_ids": [int(token) for token in generated_token_ids],
        "prompt_sha256": str(
            request.get("prompt_sha256")
            or _prompt_sha256(request["prompt_token_ids"])
        ),
        "ttft_ms": (first_ns - arrival_ns) / 1_000_000.0,
        "itl_ms": itl_ms,
        "tpot_ms": tpot_ms,
        "decode_ms": decode_ms,
        "e2e_latency_ms": e2e_ms,
        "tokens_per_second": (
            generated / (decode_ms / 1000.0) if decode_ms > 0 else None
        ),
        "e2e_tokens_per_second": (
            generated / (e2e_ms / 1000.0) if e2e_ms > 0 else None
        ),
    }


def _summarize_request_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "ttft_ms",
        "tpot_ms",
        "decode_ms",
        "e2e_latency_ms",
        "tokens_per_second",
        "e2e_tokens_per_second",
    )
    metrics = {
        field: _distribution(
            float(record[field])
            for record in records
            if record.get(field) is not None
        )
        for field in fields
    }
    metrics["itl_ms"] = _distribution(
        float(value) for record in records for value in record["itl_ms"]
    )
    return {
        "request_count": len(records),
        "metrics": metrics,
    }


def _correctness(
    records: Sequence[Mapping[str, Any]],
    references: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for record in records:
        prompt_sha = str(record["prompt_sha256"])
        expected = references.get(prompt_sha)
        actual = tuple(int(token) for token in record["generated_token_ids"])
        if expected is None:
            errors.append(f"{record['request_id']}: missing reference")
            continue
        requested = int(record["requested_output_tokens"])
        if actual != tuple(expected[:requested]):
            errors.append(f"{record['request_id']}: token mismatch")
        checked += 1
    return {
        "status": "pass" if not errors else "fail",
        "checked": checked,
        "errors": errors,
        "comparison": "exact_prefix_of_phase1_reference",
    }


async def _collect_request(
    engine: Any,
    request: Mapping[str, Any],
    *,
    request_id: str,
    arrival_ns: int,
    sampling_params: Any,
) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt

    generated: list[int] = []
    token_times_ns: list[int] = []
    prompt = TokensPrompt(prompt_token_ids=[int(token) for token in request["prompt_token_ids"]])
    async for output in engine.generate(
        prompt,
        sampling_params,
        request_id=request_id,
    ):
        completion = output.outputs[0]
        delta = [int(token) for token in completion.token_ids]
        if not delta:
            continue
        timestamp_ns = perf_counter_ns()
        generated.extend(delta)
        token_times_ns.extend([timestamp_ns] * len(delta))
    return _metric_record(request, generated, token_times_ns, arrival_ns)


async def _collect_batch(
    engine: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_name: str,
    repetition_index: int,
) -> dict[str, Any]:
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    arrival_ns = perf_counter_ns()
    tasks = []
    for index, request in enumerate(requests):
        params = SamplingParams(
            max_tokens=int(request["max_new_tokens"]),
            min_tokens=int(request["max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            output_kind=RequestOutputKind.DELTA,
        )
        tasks.append(
            asyncio.create_task(
                _collect_request(
                    engine,
                    request,
                    request_id=f"{case_name}-r{repetition_index}-{index}",
                    arrival_ns=arrival_ns,
                    sampling_params=params,
                )
            )
        )
    records = await asyncio.gather(*tasks)
    completed_ns = perf_counter_ns()
    duration_ms = (completed_ns - arrival_ns) / 1_000_000.0
    total_tokens = sum(int(record["generated_output_tokens"]) for record in records)
    return {
        "repetition_index": repetition_index,
        "duration_ms": duration_ms,
        "aggregate_tokens_per_second": total_tokens / (duration_ms / 1000.0),
        "aggregate_requests_per_second": len(records) / (duration_ms / 1000.0),
        "requests": records,
    }


def _case_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [record for run in runs for record in run["requests"]]
    return {
        **_summarize_request_metrics(records),
        "aggregate": {
            "tokens_per_second": _distribution(
                float(run["aggregate_tokens_per_second"]) for run in runs
            ),
            "requests_per_second": _distribution(
                float(run["aggregate_requests_per_second"]) for run in runs
            ),
            "run_duration_ms": _distribution(float(run["duration_ms"]) for run in runs),
        },
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        revision=MODEL_REVISION,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        max_num_seqs=int(args.max_num_seqs),
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=bool(args.enforce_eager),
        seed=0,
        disable_log_stats=False,
        use_tqdm_on_load=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "minillm_l4_vllm_reference",
        "status": "running",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "vllm_configuration": {
            "model": str(args.model),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "enforce_eager": args.enforce_eager,
        },
        "measurement": {
            "warmup_repetitions": args.warmup_repetitions,
            "repetitions": args.repetitions,
            "sampling": "greedy",
            "timestamp_source": "direct_async_stream_receive",
        },
        "cases": [],
    }
    manifest_path = output_dir / "vllm_manifest.json"

    try:
        for bucket in args.workloads:
            workload = _load_json(args.workload_dir / f"workload_{bucket}.json")
            requests = workload["requests"]
            references = _load_reference(args.reference_dir, bucket)
            for batch_size in args.batch_sizes:
                if batch_size > len(requests):
                    raise ValueError(
                        f"batch size {batch_size} exceeds {bucket} workload size {len(requests)}"
                    )
                selected = requests[:batch_size]
                case_name = f"{bucket}_b{batch_size}"
                for warmup_index in range(args.warmup_repetitions):
                    await _collect_batch(
                        engine,
                        selected,
                        case_name=f"warmup-{case_name}",
                        repetition_index=warmup_index,
                    )
                runs = [
                    await _collect_batch(
                        engine,
                        selected,
                        case_name=case_name,
                        repetition_index=repetition_index,
                    )
                    for repetition_index in range(args.repetitions)
                ]
                records = [record for run in runs for record in run["requests"]]
                correctness = _correctness(records, references)
                result = {
                    "schema_version": 1,
                    "benchmark": "minillm_l4_vllm_reference",
                    "case": {
                        "name": case_name,
                        "bucket": bucket,
                        "prompt_tokens": len(selected[0]["prompt_token_ids"]),
                        "batch_size": batch_size,
                        "request_count_measured": len(selected),
                    },
                    "workload": workload,
                    "runs": runs,
                    "summary": _case_summary(runs),
                    "correctness": correctness,
                    "configuration": manifest["vllm_configuration"],
                }
                result_path = output_dir / f"{case_name}.json"
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                row = {
                    "case_name": case_name,
                    "bucket": bucket,
                    "prompt_tokens": result["case"]["prompt_tokens"],
                    "batch_size": batch_size,
                    "result": str(result_path),
                    "correctness": correctness,
                    "summary": result["summary"],
                }
                manifest["cases"].append(row)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                metrics = result["summary"]["metrics"]
                aggregate = result["summary"]["aggregate"]
                print(
                    f"vllm {bucket:6s} batch={batch_size:<2d} "
                    f"TTFT_P50={metrics['ttft_ms']['median']:.3f} ms "
                    f"ITL_P50={metrics['itl_ms']['median']:.3f} ms "
                    f"TPOT_P50={metrics['tpot_ms']['median']:.3f} ms "
                    f"TPS_P50={aggregate['tokens_per_second']['median']:.3f} "
                    f"E2E_TPS_P50={metrics['e2e_tokens_per_second']['median']:.3f} "
                    f"correctness={correctness['status']}"
                )
    finally:
        engine.shutdown()

    manifest["status"] = (
        "completed"
        if manifest["cases"]
        and all(case["correctness"]["status"] == "pass" for case in manifest["cases"])
        else "failed"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
