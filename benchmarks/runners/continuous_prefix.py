"""Continuous paged decoding with exact-token prefix reuse on admission."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from ..core.harness import RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from minillm_l4.engine.kv_cache import (
    PagedKvOutOfMemoryError,
    select_decode_block_tokens,
    select_decode_split_count,
)
from minillm_l4.engine.request import RequestLifecycle, RequestState
from minillm_l4.engine.scheduler import ContinuousBatchScheduler, ScheduledRequest

from .continuous_requests import _wait_until_ms
from .paged_kv import _cache_layers
from .prefix_cache import PrefixCachedPagedRunner


@dataclass
class _LiveRequest:
    request: RequestSpec
    recorder: RequestEventRecorder
    lifecycle: RequestLifecycle
    generated: list[int] = field(default_factory=list)
    next_token: torch.Tensor | None = None
    reused_tokens: int = 0
    decode_calls: int = 0
    decode_top2_logit_gaps: list[float] = field(default_factory=list)


class ContinuousPrefixPagedRunner(PrefixCachedPagedRunner):
    """Admit one bounded prefill per iteration and batch all live decode tokens.

    Prefill is deliberately sequential (chunked/packed warm prefill is separate
    work). Decode has no token padding: one row per live request, each with its
    own physical page table and sequence length. CUDA Graph requests use the
    documented eager fallback because page ownership and active rows change.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        max_batch_size: int,
        max_prefill_tokens: int,
        max_entries: int = 8,
        max_wait_ms: float = 0.0,
        device: str | torch.device | None = None,
        decode_backend: str = "auto",
        decode_mode: str = "eager",
        enable_prefix: bool = True,
        cancel_request_ids: Sequence[str] = (),
        eos_token_id: int | Sequence[int] | None = None,
        record_logit_gaps: bool = False,
        decode_sdpa_compat: bool = False,
    ) -> None:
        if max_batch_size < 1 or max_prefill_tokens < 1:
            raise ValueError("batch size and prefill budget must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        if decode_mode not in {"eager", "graph"}:
            raise ValueError("decode_mode must be eager or graph")
        super().__init__(
            model,
            block_size=block_size,
            num_blocks=num_blocks,
            max_entries=max_entries,
            device=device,
            decode_backend=decode_backend,
            warm_prefill_backend="sdpa",
        )
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_wait_ms = float(max_wait_ms)
        self.decode_mode = decode_mode
        self.enable_prefix = bool(enable_prefix)
        self.cancel_request_ids = frozenset(cancel_request_ids)
        self.eos_token_ids = (
            frozenset((eos_token_id,))
            if isinstance(eos_token_id, int)
            else frozenset(eos_token_id or ())
        )
        # Per-step top-2 logit gaps are a tie diagnostic; computing them costs
        # an extra top-k and host transfer every decode step, so it is opt-in.
        self.record_logit_gaps = bool(record_logit_gaps)
        self.decode_sdpa_compat = bool(decode_sdpa_compat)
        self.last_summary: dict[str, Any] | None = None
        self.run_summaries: list[dict[str, Any]] = []
        self.last_lifecycles: dict[str, RequestLifecycle] = {}

    @property
    def runner_name(self) -> str:
        return "continuous_paged_prefix" if self.enable_prefix else "continuous_paged_uncached"

    def _future_blocks(self, state: _LiveRequest) -> int:
        block_size = self.allocator.block_size
        final_tokens = state.request.prompt_tokens + state.request.max_new_tokens - 1
        current_tokens = self.allocator.get_block_table(state.request.request_id).token_count
        return math.ceil(final_tokens / block_size) - math.ceil(current_tokens / block_size)

    def _prefill(self, state: _LiveRequest) -> None:
        request = state.request
        owner = request.request_id
        reused = state.reused_tokens
        state.recorder.record(
            "prefill_start",
            metadata={"runner": self.runner_name, "reused_tokens": reused},
        )
        suffix = torch.tensor(
            [request.prompt_token_ids[reused:]], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            if reused == 0:
                output = self.model(
                    input_ids=suffix,
                    attention_mask=torch.ones_like(suffix),
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                self.cache.append(owner, _cache_layers(output.past_key_values))
            else:
                dense_cache = self.cache.as_dynamic_cache(
                    (owner,), model_config=self.model.config
                )
                output = self.model(
                    input_ids=suffix,
                    attention_mask=torch.ones(
                        (1, request.prompt_tokens), dtype=torch.long, device=self.device
                    ),
                    position_ids=torch.arange(
                        reused, request.prompt_tokens, dtype=torch.long, device=self.device
                    ).unsqueeze(0),
                    past_key_values=dense_cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                self.cache.append_from_dynamic_cache(
                    (owner,), output.past_key_values,
                    previous_token_counts=(reused,),
                    appended_token_counts=(request.prompt_tokens - reused,),
                )
            state.next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True).detach()
            token = int(state.next_token.item())  # synchronizes before timestamping
            if self.enable_prefix:
                self.prefixes.publish(owner, request.prompt_token_ids)
            del output

        state.generated.append(token)
        now = state.recorder.now_ns()
        state.recorder.record("prefill_end", timestamp_ns=now)
        state.recorder.mark_token_ready(0, token_id=token, timestamp_ns=now)
        state.recorder.mark_token_sent(0, token_id=token, timestamp_ns=now)
        state.lifecycle.transition(RequestState.DECODING)

    def _decode(self, active: list[_LiveRequest]) -> None:
        owners = tuple(state.request.request_id for state in active)
        old_lengths = tuple(
            self.allocator.get_block_table(owner).token_count for owner in owners
        )
        max_length = max(
            state.request.prompt_tokens + state.request.max_new_tokens - 1
            for state in active
        )
        with torch.inference_mode():
            self.cache.reserve_append(owners, 1)
            inputs = torch.cat([state.next_token for state in active], dim=0)
            output = self.model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                position_ids=torch.tensor(
                    old_lengths, dtype=torch.long, device=self.device
                ).unsqueeze(1),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                paged_kv_cache=self.cache,
                paged_sequence_ids=owners,
                paged_query_start_positions=old_lengths,
                paged_block_tables=self.cache.block_table_tensor(
                    owners, device=self.device
                ).to(dtype=torch.int32),
                paged_sequence_lengths=torch.tensor(
                    tuple(length + 1 for length in old_lengths),
                    dtype=torch.int32,
                    device=self.device,
                ),
                paged_attention_backend=self.decode_backend,
                paged_decode_max_sequence_length=max_length,
                paged_decode_split_count=select_decode_split_count(
                    max_length, batch_size=len(active)
                ) if not self.decode_sdpa_compat else 1,
                paged_decode_block_tokens=(
                    select_decode_block_tokens(max_length)
                    if not self.decode_sdpa_compat else 128
                ),
                paged_decode_use_gqa_reuse=False,
                paged_decode_sdpa_compat=self.decode_sdpa_compat,
            )
            next_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True).detach()
            if self.record_logit_gaps:
                top_two = output.logits[:, -1, :].float().topk(2, dim=-1).values
                gaps = (top_two[:, 0] - top_two[:, 1]).tolist()
            else:
                gaps = None
            values = next_tokens[:, 0].tolist()
            del output
        now = max(state.recorder.now_ns() for state in active)
        for row, (state, token) in enumerate(zip(active, values, strict=True)):
            state.next_token = next_tokens[row : row + 1]
            index = len(state.generated)
            state.generated.append(int(token))
            state.decode_calls += 1
            if gaps is not None:
                state.decode_top2_logit_gaps.append(float(gaps[row]))
            state.recorder.mark_token_ready(index, token_id=int(token), timestamp_ns=now)
            state.recorder.mark_token_sent(index, token_id=int(token), timestamp_ns=now)

    def _complete(self, state: _LiveRequest, outcomes: dict[str, RequestOutcome]) -> None:
        owner = state.request.request_id
        state.lifecycle.transition(RequestState.FINISHED)
        self.cache.release(owner)
        state.lifecycle.release_resources()
        state.recorder.record("completion")
        outcomes[owner] = RequestOutcome(
            status="completed",
            generated_token_ids=tuple(state.generated),
            metadata={
                "runner": self.runner_name,
                "reused_tokens": state.reused_tokens,
                "computed_prefill_tokens": state.request.prompt_tokens - state.reused_tokens,
                "decode_forward_calls": state.decode_calls,
                "decode_top2_logit_gaps": (
                    state.decode_top2_logit_gaps if self.record_logit_gaps else None
                ),
                "lifecycle": list(state.lifecycle.history),
                "resources_released": True,
            },
        )

    def _cancel(self, state: _LiveRequest, outcomes: dict[str, RequestOutcome]) -> None:
        state.lifecycle.transition(RequestState.CANCELLED)
        state.lifecycle.release_resources()
        state.recorder.record("cancelled", metadata={"reason": "cancelled_before_execution"})
        outcomes[state.request.request_id] = RequestOutcome(
            status="cancelled", metadata={"runner": self.runner_name}
        )

    def _fail(self, state: _LiveRequest, outcomes: dict[str, RequestOutcome], reason: str) -> None:
        owner = state.request.request_id
        if owner in self.allocator.sequence_ids:
            self.cache.release(owner)
        state.lifecycle.transition(RequestState.FAILED)
        state.lifecycle.release_resources()
        state.recorder.record("error", metadata={"reason": reason})
        outcomes[owner] = RequestOutcome(
            status="failed", error=reason,
            metadata={"runner": self.runner_name, "resources_released": True},
        )

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests or len(requests) != len(recorders):
            raise ValueError("requests and recorders must be non-empty and equal")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("request IDs must be unique")
        unknown = self.cancel_request_ids - {request.request_id for request in requests}
        if unknown:
            raise ValueError(f"unknown cancellation IDs: {sorted(unknown)}")

        # Each repetition starts cold, while requests within that repetition
        # share entries. This prevents warm-up and prior runs from biasing hits.
        self.prefixes.clear()
        start_stats = self.prefixes.snapshot()
        states = {
            request.request_id: _LiveRequest(
                request, recorder, RequestLifecycle(request.request_id)
            )
            for request, recorder in zip(requests, recorders, strict=True)
        }
        self.last_lifecycles = {
            owner: state.lifecycle for owner, state in states.items()
        }
        scheduler = ContinuousBatchScheduler(
            (
                ScheduledRequest(request.request_id, request.scheduled_arrival_ms, index)
                for index, request in enumerate(requests)
            ),
            max_batch_size=self.max_batch_size,
            max_prefill_tokens=self.max_prefill_tokens,
            max_wait_ms=self.max_wait_ms,
        )
        prompt_costs = {request.request_id: request.prompt_tokens for request in requests}
        outcomes: dict[str, RequestOutcome] = {}
        active: list[_LiveRequest] = []
        records: list[dict[str, Any]] = []
        max_physical_blocks = 0
        deferred = 0
        budget_deferred = 0
        first_recorder = recorders[0]
        try:
            while not scheduler.empty or active:
                now_ms = first_recorder.now_ns() / 1_000_000.0
                scheduler.admit(now_ms)
                for item in scheduler.cancel_ready(set(self.cancel_request_ids)):
                    self._cancel(states[item.request_id], outcomes)
                if not active and not scheduler.ready_count:
                    next_arrival = scheduler.next_arrival_ms
                    if next_arrival is None:
                        break
                    _wait_until_ms(first_recorder, next_arrival)
                    continue
                if not active and scheduler.should_wait_for_batch(now_ms, active_count=0):
                    _wait_until_ms(
                        first_recorder,
                        float(scheduler.oldest_ready_arrival_ms) + self.max_wait_ms,
                    )
                    continue

                if scheduler.ready_count and len(active) < self.max_batch_size:
                    selected = scheduler.next_batch(prompt_costs, max_requests=1)
                    state = states[selected[0].request_id]
                    owner = state.request.request_id
                    final_length = state.request.prompt_tokens + state.request.max_new_tokens - 1
                    reusable = (
                        self.prefixes.reusable_tokens(state.request.prompt_token_ids)
                        if self.enable_prefix else 0
                    )
                    if math.ceil(final_length / self.allocator.block_size) > self.allocator.num_blocks:
                        self._fail(state, outcomes, "request exceeds total KV capacity")
                    elif active and state.request.prompt_tokens - reusable > self.max_prefill_tokens:
                        scheduler.defer_front(selected)
                        budget_deferred += 1
                    else:
                        try:
                            if self.enable_prefix:
                                state.reused_tokens = self.prefixes.attach(
                                    owner, state.request.prompt_token_ids,
                                    output_tokens=state.request.max_new_tokens,
                                )
                            else:
                                self.allocator.allocate(owner)
                        except PagedKvOutOfMemoryError:
                            # The request may fit after active work finishes.
                            if active:
                                scheduler.defer_front(selected)
                                deferred += 1
                            else:
                                self._fail(state, outcomes, "KV capacity unavailable")
                        else:
                            # Matching is only an estimate: attach may evict
                            # its own source under pressure and become cold.
                            actual_work = state.request.prompt_tokens - state.reused_tokens
                            if active and actual_work > self.max_prefill_tokens:
                                self.cache.release(owner)
                                scheduler.defer_front(selected)
                                budget_deferred += 1
                            else:
                                future = self._future_blocks(state) + sum(
                                    self._future_blocks(item) for item in active
                                )
                                enough = self.prefixes.evict_until_free(future)
                                if not enough:
                                    self.cache.release(owner)
                                    if active:
                                        scheduler.defer_front(selected)
                                        deferred += 1
                                    else:
                                        self._fail(state, outcomes, "KV capacity unavailable")
                                else:
                                    if self.enable_prefix:
                                        self.prefixes.record_admission(state.reused_tokens)
                                    state.recorder.record("admission")
                                    state.recorder.record("execution_start")
                                    state.lifecycle.transition(RequestState.PREFILL)
                                    was_decoding = bool(active)
                                    self._prefill(state)
                                    max_physical_blocks = max(
                                        max_physical_blocks, self.allocator.allocated_block_count
                                    )
                                    records.append({
                                        "kind": "prefill", "request_id": owner,
                                        "reused_tokens": state.reused_tokens,
                                        "computed_tokens": actual_work,
                                        "while_decoding": was_decoding,
                                        "active_before": len(active),
                                    })
                                    if (state.generated[-1] in self.eos_token_ids
                                            or len(state.generated) >= state.request.max_new_tokens):
                                        self._complete(state, outcomes)
                                    else:
                                        active.append(state)

                if active:
                    owners = [state.request.request_id for state in active]
                    self._decode(active)
                    max_physical_blocks = max(
                        max_physical_blocks, self.allocator.allocated_block_count
                    )
                    finished = [
                        state for state in active
                        if state.generated[-1] in self.eos_token_ids
                        or len(state.generated) >= state.request.max_new_tokens
                    ]
                    records.append({
                        "kind": "decode", "request_ids": owners,
                        "batch_size": len(active),
                        "finished_request_ids": [item.request.request_id for item in finished],
                    })
                    for state in finished:
                        self._complete(state, outcomes)
                    finished_ids = {state.request.request_id for state in finished}
                    active = [
                        state for state in active
                        if state.request.request_id not in finished_ids
                    ]
                max_physical_blocks = max(
                    max_physical_blocks, self.allocator.allocated_block_count
                )
        finally:
            for owner in tuple(self.allocator.sequence_ids):
                if owner in states:
                    self.cache.release(owner)

        if len(outcomes) != len(requests):
            raise RuntimeError("continuous prefix scheduler left requests unfinished")
        current_stats = self.prefixes.snapshot()
        hits = current_stats["hits"] - start_stats["hits"]
        misses = current_stats["misses"] - start_stats["misses"]
        self.last_summary = {
            "policy": "continuous_paged_prefix",
            "prefix_enabled": self.enable_prefix,
            "decode_mode_requested": self.decode_mode,
            "decode_mode_used": "eager",
            "graph_fallback_reason": (
                "dynamic request membership and mutable page tables"
                if self.decode_mode == "graph" else None
            ),
            "max_batch_size": self.max_batch_size,
            "max_prefill_tokens": self.max_prefill_tokens,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
            "reused_tokens": current_stats["reused_tokens"] - start_stats["reused_tokens"],
            "computed_prefill_tokens": sum(
                record["computed_tokens"] for record in records if record["kind"] == "prefill"
            ),
            "evictions": current_stats["evictions"] - start_stats["evictions"],
            "eviction_wall_ms": (
                current_stats["eviction_wall_ms"] - start_stats["eviction_wall_ms"]
            ),
            "max_physical_blocks": max_physical_blocks,
            "peak_kv_utilization": max_physical_blocks / self.allocator.num_blocks,
            "deferred_admissions": deferred,
            "budget_deferred_admissions": budget_deferred,
            "maximum_queue_depth": scheduler.maximum_queue_depth,
            "prefills_while_decoding": sum(
                bool(record["while_decoding"]) for record in records
                if record["kind"] == "prefill"
            ),
            "maximum_decode_batch_size": max(
                (record["batch_size"] for record in records if record["kind"] == "decode"),
                default=0,
            ),
            "prefill_records": [record for record in records if record["kind"] == "prefill"],
            "decode_records": [record for record in records if record["kind"] == "decode"],
            "active_request_blocks_after_run": sum(
                owner in states for owner in self.allocator.sequence_ids
            ),
        }
        self.run_summaries.append(self.last_summary)
        return tuple(outcomes[request.request_id] for request in requests)
