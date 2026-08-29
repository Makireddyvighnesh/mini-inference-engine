"""Benchmark cached decoding against full-prefix recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from minillm_l4.engine.generation.manual import (
    manual_greedy_generate,
    output_token_digest,
)
from minillm_l4.engine.generation.recompute import recompute_greedy_generate
from minillm_l4.engine.kv_cache import ContiguousKvCache


CACHE_MODES = ("contiguous", "recompute")


def _first_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "device must be supplied when the model has no discoverable parameters"
        ) from error


def _token_values(token_ids: torch.Tensor, *, batch_size: int) -> list[int]:
    values = token_ids.detach().to(device="cpu")
    if values.ndim == 1:
        values = values.reshape(batch_size, -1)
    if values.ndim != 2 or values.shape != (batch_size, 1):
        raise ValueError("token callbacks must provide one token per batch row")
    return [int(value) for value in values[:, 0].tolist()]


def _device_memory(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes), int(total_bytes)


def _allocated_memory(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated(device))


class KvCacheBatchRunner:
    """Run one static batch with contiguous caching or no-cache recomputation."""

    def __init__(
        self,
        model: Any,
        *,
        mode: str,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
        capacity_tokens: int | None = None,
    ) -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"mode must be one of {CACHE_MODES}")
        if capacity_tokens is not None and capacity_tokens < 1:
            raise ValueError("capacity_tokens must be positive when supplied")
        self.model = model
        self.mode = mode
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.capacity_tokens = capacity_tokens
        self.last_cache_snapshot: dict[str, Any] | None = None

    @property
    def runner_name(self) -> str:
        return (
            "contiguous_dynamic_kv"
            if self.mode == "contiguous"
            else "full_prefix_recompute"
        )

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("A generation batch must contain at least one request")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1:
            raise ValueError(
                "KV benchmark static batching requires equal prompt lengths; "
                f"received {sorted(prompt_lengths)}"
            )
        if len(output_lengths) != 1:
            raise ValueError(
                "KV benchmark static batching requires equal output lengths; "
                f"received {sorted(output_lengths)}"
            )

        batch_size = len(requests)
        prompt_tokens = next(iter(prompt_lengths))
        output_tokens = next(iter(output_lengths))
        required_cache_tokens = prompt_tokens + output_tokens - 1
        selected_capacity = self.capacity_tokens or required_cache_tokens
        owner_ids = tuple(request.request_id for request in requests)
        free_bytes, total_bytes = _device_memory(self.device)
        allocated_bytes_before = _allocated_memory(self.device)

        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": self.runner_name,
                    "batch_size": batch_size,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "cache_mode": self.mode,
                },
            )

        input_ids = torch.tensor(
            [request.prompt_token_ids for request in requests],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)

        def record_prefill_end(token_ids: torch.Tensor) -> None:
            _token_values(token_ids, batch_size=batch_size)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for recorder in recorders:
                recorder.record("prefill_end", timestamp_ns=timestamp_ns)

        def record_token(token_index: int, token_ids: torch.Tensor) -> None:
            token_values = _token_values(token_ids, batch_size=batch_size)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for recorder, token_id in zip(recorders, token_values, strict=True):
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

        cache_state: ContiguousKvCache | None = None
        self.last_cache_snapshot = None
        try:
            if self.mode == "contiguous":
                model_config = getattr(self.model, "config", None)
                if model_config is None:
                    raise ValueError("contiguous caching requires model.config")
                cache_state = ContiguousKvCache(
                    model_config,
                    owner_ids=owner_ids,
                    capacity_tokens=selected_capacity,
                )
                cache_state.assert_owners(owner_ids)
                generation = manual_greedy_generate(
                    self.model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    output_tokens=output_tokens,
                    logits_mode=self.logits_mode,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    on_prefill_end=record_prefill_end,
                    on_token=record_token,
                    kv_cache=cache_state,
                )
            else:
                generation = recompute_greedy_generate(
                    self.model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    output_tokens=output_tokens,
                    logits_mode=self.logits_mode,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    on_prefill_end=record_prefill_end,
                    on_token=record_token,
                )

            token_rows = [
                [
                    int(token)
                    for token in generation.row(row_index)
                    .detach()
                    .to(device="cpu")
                    .tolist()
                ]
                for row_index in range(batch_size)
            ]
            for recorder in recorders:
                recorder.record("completion")

            if cache_state is not None:
                self.last_cache_snapshot = cache_state.snapshot(
                    free_device_bytes_before=free_bytes,
                    total_device_bytes=total_bytes,
                    observed_allocated_bytes_before=allocated_bytes_before,
                    observed_allocated_bytes_after=_allocated_memory(self.device),
                )

            model_config = getattr(self.model, "config", None)
            outcomes: list[RequestOutcome] = []
            compact_cache = None
            if self.last_cache_snapshot is not None:
                compact_cache = {
                    key: value
                    for key, value in self.last_cache_snapshot.items()
                    if key not in {"layers", "owner_ids"}
                }
            for request, row in zip(requests, token_rows, strict=True):
                generated_count = len(row)
                if self.mode == "contiguous":
                    forward_calls = generated_count
                    input_tokens_processed = prompt_tokens + max(
                        0, generated_count - 1
                    )
                else:
                    forward_calls = generated_count
                    input_tokens_processed = (
                        generated_count * prompt_tokens
                        + generated_count * (generated_count - 1) // 2
                    )
                row_tensor = torch.tensor([row], dtype=torch.long)
                outcomes.append(
                    RequestOutcome(
                        status="completed",
                        generated_token_ids=tuple(row),
                        metadata={
                            "runner": self.runner_name,
                            "model_id": getattr(model_config, "_name_or_path", None),
                            "batch_size": batch_size,
                            "cache_mode": self.mode,
                            "forward_calls": forward_calls,
                            "model_input_tokens_processed_per_sequence": input_tokens_processed,
                            "cache": compact_cache,
                            "output_token_sha256": output_token_digest(row_tensor),
                        },
                    )
                )
            return tuple(outcomes)
        finally:
            if cache_state is not None:
                cache_state.release()


def write_kv_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
    mode: str,
    cache_snapshot: Mapping[str, Any] | None,
) -> None:
    """Write common harness data plus cache/recomputation evidence."""

    payload = result.to_dict()
    payload["kv_cache"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "mode": mode,
        "cache_snapshot": (
            None if cache_snapshot is None else dict(cache_snapshot)
        ),
        "generation": (
            "one prefill plus one-token forwards into an owned contiguous cache"
            if mode == "contiguous"
            else "full prompt and generated prefix recomputed for every token"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
