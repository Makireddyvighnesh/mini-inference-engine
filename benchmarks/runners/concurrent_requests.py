"""Trace runner for staggered arrivals and FIFO padded static batches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from minillm_l4.engine.generation.manual import manual_greedy_generate, output_token_digest
from minillm_l4.engine.kv_cache import ContiguousKvCache
from minillm_l4.engine.request import RequestLifecycle, RequestState
from minillm_l4.engine.scheduler import ScheduledRequest, StaticBatchScheduler


def _first_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "device must be supplied when the model has no discoverable parameters"
        ) from error


def _wait_until_ms(recorder: RequestEventRecorder, target_ms: float) -> None:
    while recorder.now_ns() / 1_000_000.0 < target_ms:
        remaining_ms = target_ms - recorder.now_ns() / 1_000_000.0
        time.sleep(min(remaining_ms / 1000.0, 0.01))


def _padded_inputs(
    requests: Sequence[RequestSpec],
    *,
    device: torch.device,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], int]:
    max_prompt_tokens = max(request.prompt_tokens for request in requests)
    batch_size = len(requests)
    input_ids = torch.full(
        (batch_size, max_prompt_tokens),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, request in enumerate(requests):
        length = request.prompt_tokens
        input_ids[row, -length:] = torch.tensor(
            request.prompt_token_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, -length:] = 1
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, max_prompt_tokens


class StaticRequestTraceRunner:
    """Run an arrival trace with FIFO admission and padded static batching.

    New arrivals are considered only between model calls. A selected batch
    runs until every row reaches its output limit, making head-of-line blocking
    and padding waste visible before continuous batching is introduced.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
        cancel_request_ids: Sequence[str] = (),
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.cancel_request_ids = frozenset(str(value) for value in cancel_request_ids)
        self.last_summary: dict[str, Any] | None = None
        self.last_lifecycles: dict[str, RequestLifecycle] = {}

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("A request trace must not be empty")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        request_ids = {request.request_id for request in requests}
        unknown_cancellations = self.cancel_request_ids - request_ids
        if unknown_cancellations:
            raise ValueError(
                f"cancel_request_ids are not in the trace: {sorted(unknown_cancellations)}"
            )

        by_id = {request.request_id: request for request in requests}
        recorder_by_id = {
            recorder.request_id: recorder for recorder in recorders
        }
        lifecycles = {
            request.request_id: RequestLifecycle(request.request_id)
            for request in requests
        }
        self.last_lifecycles = lifecycles
        scheduler = StaticBatchScheduler(
            (
                ScheduledRequest(
                    request_id=request.request_id,
                    scheduled_arrival_ms=request.scheduled_arrival_ms,
                    ordinal=index,
                )
                for index, request in enumerate(requests)
            ),
            max_batch_size=self.max_batch_size,
        )
        outcomes: dict[str, RequestOutcome] = {}
        batch_records: list[dict[str, Any]] = []
        first_recorder = recorders[0]

        while not scheduler.empty:
            now_ms = first_recorder.now_ns() / 1_000_000.0
            scheduler.admit(now_ms)
            if scheduler.ready_count == 0:
                next_arrival = scheduler.next_arrival_ms
                if next_arrival is None:
                    break
                _wait_until_ms(first_recorder, next_arrival)
                continue

            cancelled = scheduler.cancel_ready(set(self.cancel_request_ids))
            for item in cancelled:
                lifecycle = lifecycles[item.request_id]
                lifecycle.transition(RequestState.CANCELLED)
                lifecycle.release_resources()
                recorder = recorder_by_id[item.request_id]
                recorder.record(
                    "cancelled",
                    metadata={"reason": "cancelled_before_execution"},
                )
                outcomes[item.request_id] = RequestOutcome(
                    status="cancelled",
                    metadata={
                        "runner": "fifo_padded_static_batch",
                        "lifecycle": list(lifecycle.history),
                        "resources_released": lifecycle.resources_released,
                    },
                )
            if scheduler.ready_count == 0:
                continue

            scheduled_batch = scheduler.next_batch()
            batch = tuple(by_id[item.request_id] for item in scheduled_batch)
            batch_recorders = tuple(
                recorder_by_id[item.request_id] for item in scheduled_batch
            )
            for request, recorder in zip(batch, batch_recorders, strict=True):
                lifecycles[request.request_id].transition(RequestState.PREFILL)
                recorder.record("admission")
                recorder.record("execution_start")

            batch_outcomes, batch_record = self._execute_batch(
                batch,
                batch_recorders,
                lifecycles,
                batch_index=len(batch_records),
            )
            outcomes.update(
                (request.request_id, outcome)
                for request, outcome in zip(batch, batch_outcomes, strict=True)
            )
            batch_records.append(batch_record)

        if len(outcomes) != len(requests):
            missing = sorted(request_ids - set(outcomes))
            raise RuntimeError(f"scheduler did not terminate requests: {missing}")

        prompt_slots = sum(record["prompt_slots"] for record in batch_records)
        output_slots = sum(record["output_slots"] for record in batch_records)
        prompt_padding = sum(record["prompt_padding_slots"] for record in batch_records)
        output_padding = sum(record["output_padding_slots"] for record in batch_records)
        total_slots = prompt_slots + output_slots
        total_padding = prompt_padding + output_padding
        self.last_summary = {
            "policy": "fifo_padded_static_batch",
            "max_batch_size": self.max_batch_size,
            "batch_count": len(batch_records),
            "batch_sizes": [record["batch_size"] for record in batch_records],
            "maximum_queue_depth": scheduler.maximum_queue_depth,
            "prompt_padding_slots": prompt_padding,
            "output_padding_slots": output_padding,
            "padding_slots": total_padding,
            "padding_waste_ratio": total_padding / total_slots if total_slots else 0.0,
            "cancelled_requests": len(self.cancel_request_ids),
            "batches": batch_records,
        }
        return tuple(outcomes[request.request_id] for request in requests)

    def _execute_batch(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
        lifecycles: Mapping[str, RequestLifecycle],
        *,
        batch_index: int,
    ) -> tuple[tuple[RequestOutcome, ...], dict[str, Any]]:
        inputs, max_prompt_tokens = _padded_inputs(
            requests,
            device=self.device,
            pad_token_id=self.pad_token_id,
        )
        output_limits = tuple(request.max_new_tokens for request in requests)
        max_output_tokens = max(output_limits)
        capacity_tokens = max_prompt_tokens + max_output_tokens - 1
        owner_ids = tuple(request.request_id for request in requests)
        cache = ContiguousKvCache(
            self.model.config,
            owner_ids=owner_ids,
            capacity_tokens=capacity_tokens,
        )
        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": "fifo_padded_static_batch",
                    "batch_index": batch_index,
                    "batch_size": len(requests),
                    "max_prompt_tokens": max_prompt_tokens,
                    "max_output_tokens": max_output_tokens,
                },
            )

        def on_prefill_end(_: torch.Tensor) -> None:
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for request, recorder in zip(requests, recorders, strict=True):
                lifecycles[request.request_id].transition(RequestState.DECODING)
                recorder.record("prefill_end", timestamp_ns=timestamp_ns)

        def on_token(token_index: int, token_ids: torch.Tensor) -> None:
            values = token_ids.detach().to(device="cpu").reshape(len(requests), -1)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for row, (request, recorder) in enumerate(
                zip(requests, recorders, strict=True)
            ):
                if token_index >= request.max_new_tokens:
                    continue
                token_id = int(values[row, 0])
                recorder.mark_token_ready(
                    token_index,
                    token_id=token_id,
                    timestamp_ns=timestamp_ns,
                )
                recorder.mark_token_sent(
                    token_index,
                    token_id=token_id,
                    timestamp_ns=timestamp_ns,
                )
                if token_index + 1 == request.max_new_tokens:
                    lifecycles[request.request_id].transition(RequestState.FINISHED)
                    recorder.record("completion", timestamp_ns=timestamp_ns)

        snapshot: dict[str, Any] | None = None
        try:
            generation = manual_greedy_generate(
                self.model,
                inputs,
                output_tokens=max_output_tokens,
                logits_mode=self.logits_mode,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                on_prefill_end=on_prefill_end,
                on_token=on_token,
                kv_cache=cache,
                sequence_output_limits=output_limits,
            )
            snapshot = cache.snapshot()
            rows = tuple(
                tuple(
                    int(token)
                    for token in generation.row(row_index)
                    .detach()
                    .to(device="cpu")
                    .tolist()
                )
                for row_index in range(len(requests))
            )
        finally:
            cache.release()

        for request in requests:
            lifecycle = lifecycles[request.request_id]
            if not lifecycle.terminal:
                lifecycle.transition(RequestState.FINISHED)
            lifecycle.release_resources()

        prompt_padding = (
            len(requests) * max_prompt_tokens
            - sum(request.prompt_tokens for request in requests)
        )
        output_padding = (
            len(requests) * max_output_tokens
            - sum(request.max_new_tokens for request in requests)
        )
        compact_cache = {
            key: value
            for key, value in (snapshot or {}).items()
            if key not in {"layers", "owner_ids"}
        }
        outcomes = tuple(
            RequestOutcome(
                status="completed",
                generated_token_ids=row,
                metadata={
                    "runner": "fifo_padded_static_batch",
                    "batch_index": batch_index,
                    "batch_size": len(requests),
                    "lifecycle": list(lifecycles[request.request_id].history),
                    "resources_released": lifecycles[
                        request.request_id
                    ].resources_released,
                    "prompt_padding_slots": max_prompt_tokens - request.prompt_tokens,
                    "output_padding_slots": max_output_tokens - request.max_new_tokens,
                    "cache": compact_cache,
                    "output_token_sha256": output_token_digest(
                        torch.tensor([row], dtype=torch.long)
                    ),
                },
            )
            for request, row in zip(requests, rows, strict=True)
        )
        return outcomes, {
            "batch_index": batch_index,
            "request_ids": list(owner_ids),
            "batch_size": len(requests),
            "max_prompt_tokens": max_prompt_tokens,
            "max_output_tokens": max_output_tokens,
            "prompt_slots": len(requests) * max_prompt_tokens,
            "output_slots": len(requests) * max_output_tokens,
            "prompt_padding_slots": prompt_padding,
            "output_padding_slots": output_padding,
            "cache_released": cache.lifecycle == "released",
        }


def write_concurrent_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
    scheduler_summary: Mapping[str, Any] | None,
) -> None:
    payload = result.to_dict()
    payload["concurrent_requests"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "scheduler": None if scheduler_summary is None else dict(scheduler_summary),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_concurrent_references(
    result: BenchmarkResult,
    reference_dir: Path,
    *,
    cancelled_request_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Check every completed row against the Phase 1 per-bucket corpora."""

    expected: dict[str, Any] = {}
    reference_paths = sorted(reference_dir.glob("*.json"))
    errors: list[str] = []
    for path in reference_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected.update(payload.get("requests", {}))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")

    request_specs = {
        request.request_id: request for request in result.workload.requests
    }
    cancelled = set(cancelled_request_ids)
    first_completed: dict[str, tuple[int, ...]] = {}
    for run in result.runs:
        for record in run["requests"]:
            request_id = str(record["request_id"])
            outcome = record["outcome"]
            if request_id in cancelled:
                if outcome["status"] != "cancelled":
                    errors.append(f"{request_id}: expected cancelled status")
                continue
            if outcome["status"] != "completed":
                errors.append(f"{request_id}: expected completed status")
                continue
            reference = expected.get(request_id)
            if reference is None:
                errors.append(f"{request_id}: missing Phase 1 reference")
                continue
            request = request_specs[request_id]
            tokens = tuple(int(value) for value in outcome["generated_token_ids"])
            if reference.get("prompt_sha256") != request.prompt_sha256:
                errors.append(f"{request_id}: prompt digest mismatch")
            if int(reference.get("max_new_tokens", -1)) != request.max_new_tokens:
                errors.append(f"{request_id}: output limit mismatch")
            if tuple(reference.get("generated_token_ids", ())) != tokens:
                errors.append(f"{request_id}: generated token mismatch")
            prior = first_completed.setdefault(request_id, tokens)
            if prior != tokens:
                errors.append(f"{request_id}: output changed across repetitions")

    return {
        "status": "pass" if not errors else "fail",
        "reference_directory": str(reference_dir),
        "reference_files": [str(path) for path in reference_paths],
        "completed_requests_checked": len(first_completed),
        "cancelled_requests_checked": len(cancelled),
        "errors": errors,
    }
