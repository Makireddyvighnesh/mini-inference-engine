"""Iteration-level continuous batching with a rebatching KV-cache path."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from .concurrent_requests import _padded_inputs
from minillm_l4.engine.generation.manual import output_token_digest
from minillm_l4.engine.request import RequestLifecycle, RequestState
from minillm_l4.engine.scheduler import ContinuousBatchScheduler, ScheduledRequest


def _select_next_token(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("model output must contain logits with shape [batch, sequence, vocab]")
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _wait_until_ms(recorder: RequestEventRecorder, target_ms: float) -> None:
    while recorder.now_ns() / 1_000_000.0 < target_ms:
        remaining_ms = target_ms - recorder.now_ns() / 1_000_000.0
        time.sleep(min(remaining_ms / 1000.0, 0.01))


def _cache_layers(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    layers = getattr(cache, "layers", None)
    if not layers:
        raise TypeError("continuous batching requires a layer-based model cache")
    values: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        layer_values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(layer_values, torch.Tensor):
            raise TypeError(f"model cache layer {index} is not initialized with tensors")
        values.append((keys, layer_values))
    return tuple(values)


def _left_pad_cache_tensor(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = int(tensor.shape[-2])
    if current_length > target_length:
        raise ValueError(
            f"cache length {current_length} exceeds rebatching target {target_length}"
        )
    padding = target_length - current_length
    if padding == 0:
        return tensor
    shape = list(tensor.shape)
    shape[-2] = padding
    prefix = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((prefix, tensor), dim=-2)


def _select_cache_rows(cache: Any, indices: Sequence[int]) -> Any | None:
    if not indices:
        return None
    layers = _cache_layers(cache)
    device = layers[0][0].device
    cache.batch_select_indices(
        torch.tensor(tuple(indices), dtype=torch.long, device=device)
    )
    return cache


def _merge_caches(
    existing: Any | None,
    incoming: Any,
    *,
    model_config: Any,
) -> Any:
    """Rebase two variable-length batches into one left-padded cache."""

    from transformers import DynamicCache

    incoming_layers = _cache_layers(incoming)
    if existing is None:
        return incoming
    existing_layers = _cache_layers(existing)
    if len(existing_layers) != len(incoming_layers):
        raise ValueError("existing and incoming caches have different layer counts")

    existing_length = int(existing.get_seq_length())
    incoming_length = int(incoming.get_seq_length())
    target_length = max(existing_length, incoming_length)
    merged_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for (existing_keys, existing_values), (incoming_keys, incoming_values) in zip(
        existing_layers,
        incoming_layers,
        strict=True,
    ):
        merged_layers.append(
            (
                torch.cat(
                    (
                        _left_pad_cache_tensor(existing_keys, target_length),
                        _left_pad_cache_tensor(incoming_keys, target_length),
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        _left_pad_cache_tensor(existing_values, target_length),
                        _left_pad_cache_tensor(incoming_values, target_length),
                    ),
                    dim=0,
                ),
            )
        )
    return DynamicCache(ddp_cache_data=tuple(merged_layers), config=model_config)


@dataclass
class _Sequence:
    request: RequestSpec
    recorder: RequestEventRecorder
    lifecycle: RequestLifecycle
    generated_token_ids: list[int] = field(default_factory=list)
    next_token: torch.Tensor | None = None
    decode_forward_calls: int = 0

    @property
    def generated_count(self) -> int:
        return len(self.generated_token_ids)

    @property
    def cached_token_count(self) -> int:
        """Prompt plus generated tokens already written into the KV cache."""

        return self.request.prompt_tokens + self.generated_count - 1


class ContinuousRequestTraceRunner:
    """Run a request trace with iteration-level in-flight batching.

    New prompts are prefetched between decode iterations. The active decode
    batch is rebuilt after every iteration, and completed rows are removed from
    the shared cache immediately. Cache rebatching is intentionally explicit
    and copy-based in this phase; paged physical storage belongs to Phase 6.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int,
        max_prefill_tokens: int,
        max_wait_ms: float = 0.0,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
        cancel_request_ids: Sequence[str] = (),
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_wait_ms = float(max_wait_ms)
        self.device = torch.device(device) if device is not None else self._model_device()
        self.logits_mode = logits_mode
        self.eos_token_id = self._normalize_eos(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.cancel_request_ids = frozenset(str(value) for value in cancel_request_ids)
        self.last_summary: dict[str, Any] | None = None
        self.last_lifecycles: dict[str, RequestLifecycle] = {}

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise ValueError("device must be supplied when the model has no parameters") from error

    @staticmethod
    def _normalize_eos(value: int | Sequence[int] | None) -> tuple[int, ...]:
        if value is None:
            return ()
        values = (value,) if isinstance(value, int) else tuple(int(item) for item in value)
        if not values or any(item < 0 for item in values):
            raise ValueError("eos_token_id must contain non-negative IDs")
        return tuple(dict.fromkeys(values))

    def _is_eos(self, token_id: int) -> bool:
        return token_id in self.eos_token_id

    def _outcome(self, state: _Sequence) -> RequestOutcome:
        return RequestOutcome(
            status="completed",
            generated_token_ids=tuple(state.generated_token_ids),
            metadata={
                "runner": "continuous_in_flight_batching",
                "lifecycle": list(state.lifecycle.history),
                "resources_released": state.lifecycle.resources_released,
                "decode_forward_calls": state.decode_forward_calls,
                "output_token_sha256": output_token_digest(
                    torch.tensor([state.generated_token_ids], dtype=torch.long)
                ),
            },
        )

    def _finish(
        self,
        state: _Sequence,
        outcomes: dict[str, RequestOutcome],
        *,
        timestamp_ns: int,
    ) -> None:
        if state.lifecycle.state == RequestState.PREFILL:
            state.lifecycle.transition(RequestState.DECODING)
        if not state.lifecycle.terminal:
            state.lifecycle.transition(RequestState.FINISHED)
        state.lifecycle.release_resources()
        state.recorder.record("completion", timestamp_ns=timestamp_ns)
        outcomes[state.request.request_id] = self._outcome(state)

    def _cancel_ready(
        self,
        scheduler: ContinuousBatchScheduler,
        states: Mapping[str, _Sequence],
        outcomes: dict[str, RequestOutcome],
    ) -> None:
        for item in scheduler.cancel_ready(set(self.cancel_request_ids)):
            state = states[item.request_id]
            state.lifecycle.transition(RequestState.CANCELLED)
            state.lifecycle.release_resources()
            state.recorder.record(
                "cancelled",
                metadata={"reason": "cancelled_before_execution"},
            )
            outcomes[item.request_id] = RequestOutcome(
                status="cancelled",
                metadata={
                    "runner": "continuous_in_flight_batching",
                    "lifecycle": list(state.lifecycle.history),
                    "resources_released": state.lifecycle.resources_released,
                },
            )

    def _prefill(
        self,
        states: Sequence[_Sequence],
        active: list[_Sequence],
        cache: Any | None,
        outcomes: dict[str, RequestOutcome],
        batch_records: list[dict[str, Any]],
        *,
        batch_index: int,
    ) -> Any | None:
        active_batch_size_before = len(active)
        active_prompt_tokens_before = sum(
            state.request.prompt_tokens for state in active
        )
        active_cached_tokens_before = sum(
            state.cached_token_count for state in active
        )
        inputs, max_prompt_tokens = _padded_inputs(
            tuple(state.request for state in states),
            device=self.device,
            pad_token_id=self.pad_token_id,
        )
        for state in states:
            state.recorder.record(
                "prefill_start",
                metadata={
                    "runner": "continuous_in_flight_batching",
                    "batch_index": batch_index,
                    "batch_size": len(states),
                    "max_prompt_tokens": max_prompt_tokens,
                },
            )
        with torch.inference_mode():
            output = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1 if self.logits_mode == "last" else 0,
            )
        incoming_cache = getattr(output, "past_key_values", None)
        if incoming_cache is None:
            raise RuntimeError("continuous prefill did not return past_key_values")
        next_tokens = _select_next_token(output)
        timestamp_ns = max(state.recorder.now_ns() for state in states)
        active_new: list[_Sequence] = []
        active_indices: list[int] = []
        prompt_padding = 0
        for row, state in enumerate(states):
            token_id = int(next_tokens[row, 0].detach().to(device="cpu").item())
            state.generated_token_ids.append(token_id)
            state.next_token = next_tokens[row : row + 1].detach()
            state.recorder.record("prefill_end", timestamp_ns=timestamp_ns)
            state.recorder.mark_token_ready(0, token_id=token_id, timestamp_ns=timestamp_ns)
            state.recorder.mark_token_sent(0, token_id=token_id, timestamp_ns=timestamp_ns)
            prompt_padding += max_prompt_tokens - state.request.prompt_tokens
            state.lifecycle.transition(RequestState.DECODING)
            if self._is_eos(token_id) or state.generated_count >= state.request.max_new_tokens:
                self._finish(state, outcomes, timestamp_ns=timestamp_ns)
            else:
                active_new.append(state)
                active_indices.append(row)
        incoming_cache = _select_cache_rows(incoming_cache, active_indices)
        active.extend(active_new)
        merged_cache = (
            _merge_caches(cache, incoming_cache, model_config=self.model.config)
            if incoming_cache is not None
            else cache
        )
        batch_records.append(
            {
                "kind": "prefill",
                "batch_index": batch_index,
                "request_ids": [state.request.request_id for state in states],
                "batch_size": len(states),
                "active_batch_size_before": active_batch_size_before,
                "active_prompt_tokens_before": active_prompt_tokens_before,
                "active_cached_tokens_before": active_cached_tokens_before,
                "max_prompt_tokens": max_prompt_tokens,
                "prompt_tokens": sum(state.request.prompt_tokens for state in states),
                "prompt_padding_slots": prompt_padding,
                "prefill_while_decoding": active_batch_size_before > 0,
                "cache_rebased": cache is not None and incoming_cache is not None,
                "active_batch_size_after": len(active),
                "active_prompt_tokens_after": sum(
                    state.request.prompt_tokens for state in active
                ),
                "active_cached_tokens_after": sum(
                    state.cached_token_count for state in active
                ),
            }
        )
        del output
        return merged_cache

    def _decode(
        self,
        active: Sequence[_Sequence],
        cache: Any,
        outcomes: dict[str, RequestOutcome],
        batch_records: list[dict[str, Any]],
        *,
        iteration_index: int,
    ) -> Any | None:
        cache_length = int(cache.get_seq_length())
        active_prompt_tokens_before = sum(
            state.request.prompt_tokens for state in active
        )
        active_cached_tokens_before = sum(
            state.cached_token_count for state in active
        )
        input_ids = torch.cat(
            tuple(state.next_token for state in active if state.next_token is not None),
            dim=0,
        ).to(self.device)
        attention_mask = torch.zeros(
            (len(active), cache_length + 1),
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.zeros(
            (len(active), 1), dtype=torch.long, device=self.device
        )
        for row, state in enumerate(active):
            cached_tokens = state.cached_token_count
            if cached_tokens > cache_length:
                raise RuntimeError(
                    f"request {state.request.request_id} cache length {cached_tokens} "
                    f"exceeds shared length {cache_length}"
                )
            attention_mask[row, cache_length - cached_tokens :] = 1
            position_ids[row, 0] = cached_tokens
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1 if self.logits_mode == "last" else 0,
            )
        updated_cache = getattr(output, "past_key_values", None)
        if updated_cache is None:
            raise RuntimeError("continuous decode did not return past_key_values")
        next_tokens = _select_next_token(output)
        timestamp_ns = max(state.recorder.now_ns() for state in active)
        finished_indices: list[int] = []
        for row, state in enumerate(active):
            token_id = int(next_tokens[row, 0].detach().to(device="cpu").item())
            token_index = state.generated_count
            state.generated_token_ids.append(token_id)
            state.next_token = next_tokens[row : row + 1].detach()
            state.decode_forward_calls += 1
            state.recorder.mark_token_ready(
                token_index,
                token_id=token_id,
                timestamp_ns=timestamp_ns,
            )
            state.recorder.mark_token_sent(
                token_index,
                token_id=token_id,
                timestamp_ns=timestamp_ns,
            )
            if self._is_eos(token_id) or state.generated_count >= state.request.max_new_tokens:
                self._finish(state, outcomes, timestamp_ns=timestamp_ns)
                finished_indices.append(row)
        finished_set = set(finished_indices)
        keep_indices = [
            index for index in range(len(active)) if index not in finished_set
        ]
        remaining_states = [active[index] for index in keep_indices]
        del output
        if not keep_indices:
            batch_records.append(
                {
                    "kind": "decode",
                    "iteration_index": iteration_index,
                    "request_ids": [state.request.request_id for state in active],
                    "batch_size": len(active),
                    "active_prompt_tokens_before": active_prompt_tokens_before,
                    "active_cached_tokens_before": active_cached_tokens_before,
                    "cache_length_before": cache_length,
                    "cache_length_after": int(updated_cache.get_seq_length()),
                    "finished_request_ids": [
                        active[index].request.request_id
                        for index in finished_indices
                    ],
                    "active_batch_size_after": 0,
                    "active_prompt_tokens_after": 0,
                    "active_cached_tokens_after": 0,
                }
            )
            active.clear()
            return None
        batch_records.append(
            {
                "kind": "decode",
                "iteration_index": iteration_index,
                "request_ids": [state.request.request_id for state in active],
                "batch_size": len(active),
                "active_prompt_tokens_before": active_prompt_tokens_before,
                "active_cached_tokens_before": active_cached_tokens_before,
                "cache_length_before": cache_length,
                "cache_length_after": int(updated_cache.get_seq_length()),
                "finished_request_ids": [
                    active[index].request.request_id
                    for index in finished_indices
                ],
                "active_batch_size_after": len(remaining_states),
                "active_prompt_tokens_after": sum(
                    state.request.prompt_tokens for state in remaining_states
                ),
                "active_cached_tokens_after": sum(
                    state.cached_token_count for state in remaining_states
                ),
            }
        )
        active[:] = [active[index] for index in keep_indices]
        updated_cache.batch_select_indices(
            torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        )
        return updated_cache

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("a request trace must not be empty")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        request_ids = {request.request_id for request in requests}
        unknown = self.cancel_request_ids - request_ids
        if unknown:
            raise ValueError(f"cancel_request_ids are not in the trace: {sorted(unknown)}")
        states = {
            request.request_id: _Sequence(
                request=request,
                recorder=recorder,
                lifecycle=RequestLifecycle(request.request_id),
            )
            for request, recorder in zip(requests, recorders, strict=True)
        }
        self.last_lifecycles = {request_id: state.lifecycle for request_id, state in states.items()}
        scheduler = ContinuousBatchScheduler(
            (
                ScheduledRequest(
                    request_id=request.request_id,
                    scheduled_arrival_ms=request.scheduled_arrival_ms,
                    ordinal=index,
                )
                for index, request in enumerate(requests)
            ),
            max_batch_size=self.max_batch_size,
            max_prefill_tokens=self.max_prefill_tokens,
            max_wait_ms=self.max_wait_ms,
        )
        prompt_tokens_by_id = {
            request.request_id: request.prompt_tokens for request in requests
        }
        outcomes: dict[str, RequestOutcome] = {}
        active: list[_Sequence] = []
        cache: Any | None = None
        batch_records: list[dict[str, Any]] = []
        first_recorder = recorders[0]
        iteration_index = 0
        prefill_index = 0

        while not scheduler.empty or active:
            now_ms = first_recorder.now_ns() / 1_000_000.0
            scheduler.admit(now_ms)
            self._cancel_ready(scheduler, states, outcomes)
            if not active and scheduler.ready_count:
                if scheduler.should_wait_for_batch(now_ms, active_count=0):
                    deadline = float(scheduler.oldest_ready_arrival_ms) + self.max_wait_ms
                    _wait_until_ms(first_recorder, deadline)
                    continue
            if not active and scheduler.ready_count == 0:
                next_arrival = scheduler.next_arrival_ms
                if next_arrival is None:
                    break
                _wait_until_ms(first_recorder, next_arrival)
                continue

            available_slots = self.max_batch_size - len(active)
            if available_slots > 0 and scheduler.ready_count:
                selected = scheduler.next_batch(
                    prompt_tokens_by_id,
                    max_requests=available_slots,
                )
                if selected:
                    selected_states = [states[item.request_id] for item in selected]
                    for state in selected_states:
                        state.lifecycle.transition(RequestState.PREFILL)
                        state.recorder.record("admission")
                        state.recorder.record("execution_start")
                    cache = self._prefill(
                        selected_states,
                        active,
                        cache,
                        outcomes,
                        batch_records,
                        batch_index=prefill_index,
                    )
                    prefill_index += 1

            if active:
                cache = self._decode(
                    active,
                    cache,
                    outcomes,
                    batch_records,
                    iteration_index=iteration_index,
                )
                iteration_index += 1

        if len(outcomes) != len(requests):
            missing = sorted(request_ids - set(outcomes))
            raise RuntimeError(f"continuous scheduler did not terminate requests: {missing}")
        prompt_padding_slots = sum(
            int(record.get("prompt_padding_slots", 0))
            for record in batch_records
            if record["kind"] == "prefill"
        )
        prompt_slots = sum(
            int(record["batch_size"]) * int(record["max_prompt_tokens"])
            for record in batch_records
            if record["kind"] == "prefill"
        )
        prefill_batch_sizes = [
            record["batch_size"]
            for record in batch_records
            if record["kind"] == "prefill"
        ]
        decode_batch_sizes = [
            record["batch_size"]
            for record in batch_records
            if record["kind"] == "decode"
        ]
        active_batch_sizes = [
            record["active_batch_size_after"]
            for record in batch_records
            if record["kind"] == "prefill"
        ] + decode_batch_sizes
        prefill_records = [
            record for record in batch_records if record["kind"] == "prefill"
        ]
        active_prompt_token_counts = [
            int(record.get("active_prompt_tokens_after", 0))
            for record in batch_records
        ]
        active_cached_token_counts = [
            int(record.get("active_cached_tokens_after", 0))
            for record in batch_records
        ]
        prefill_while_decoding = [
            record
            for record in prefill_records
            if bool(record.get("prefill_while_decoding", False))
        ]
        self.last_summary = {
            "policy": "continuous_in_flight_batching",
            "max_batch_size": self.max_batch_size,
            "max_prefill_tokens": self.max_prefill_tokens,
            "max_wait_ms": self.max_wait_ms,
            "prefill_batch_count": sum(record["kind"] == "prefill" for record in batch_records),
            "decode_iteration_count": sum(record["kind"] == "decode" for record in batch_records),
            "iteration_batch_sizes": decode_batch_sizes,
            "prefill_batch_sizes": prefill_batch_sizes,
            "maximum_prefill_batch_size": max(prefill_batch_sizes, default=0),
            "maximum_prefill_input_tokens": max(
                (int(record["prompt_tokens"]) for record in prefill_records),
                default=0,
            ),
            "maximum_prefill_compute_slots": max(
                (
                    int(record["batch_size"]) * int(record["max_prompt_tokens"])
                    for record in prefill_records
                ),
                default=0,
            ),
            "maximum_prefill_padding_slots": max(
                (int(record["prompt_padding_slots"]) for record in prefill_records),
                default=0,
            ),
            "maximum_active_prompt_tokens": max(
                active_prompt_token_counts, default=0
            ),
            "maximum_active_cached_tokens": max(
                active_cached_token_counts, default=0
            ),
            "maximum_active_batch_size": max(active_batch_sizes, default=0),
            "maximum_concurrent_requests": max(active_batch_sizes, default=0),
            "maximum_queue_depth": scheduler.maximum_queue_depth,
            "prefill_batches_while_decoding": len(prefill_while_decoding),
            "requests_prefilled_while_decoding": sum(
                int(record["batch_size"]) for record in prefill_while_decoding
            ),
            "prompt_slots": prompt_slots,
            "prompt_padding_slots": prompt_padding_slots,
            "output_padding_slots": 0,
            "padding_waste_ratio": (
                prompt_padding_slots / prompt_slots if prompt_slots else 0.0
            ),
            "padding_definition": (
                "prompt padding only; decode has one live token per active row"
            ),
            "batch_records": batch_records,
            "cancelled_requests": sum(
                outcome.status == "cancelled" for outcome in outcomes.values()
            ),
        }
        return tuple(outcomes[request.request_id] for request in requests)


def write_continuous_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
    scheduler_summary: Mapping[str, Any] | None,
) -> None:
    payload = result.to_dict()
    payload["continuous_batching"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "scheduler": None if scheduler_summary is None else dict(scheduler_summary),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
