"""Round-robin prompt chunks interleaved with continuous paged decode."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from minillm_l4.engine.generation.chunked_prefill import ChunkedPrefillSession
from minillm_l4.engine.kv_cache import (
    PagedKvOutOfMemoryError,
    select_decode_block_tokens,
    select_decode_split_count,
)
from minillm_l4.engine.kv_cache.packed import PackedSequenceMetadata
from minillm_l4.engine.model_runner.qwen3_mixed import MixedRow, build_mixed_metadata, qwen3_mixed_forward
from minillm_l4.engine.model_runner.qwen3_packed import qwen3_packed_prefill
from minillm_l4.engine.request import RequestLifecycle, RequestState
from minillm_l4.engine.step_planner import AdaptiveChunkPlanner, PromptCandidate
from minillm_l4.engine.scheduler import ContinuousBatchScheduler, ScheduledRequest
from ..core.harness import RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from .continuous_prefix import ContinuousPrefixPagedRunner, _LiveRequest
from .continuous_requests import _wait_until_ms


@dataclass
class _PartialRequest(_LiveRequest):
    session: ChunkedPrefillSession | None = None
    prefill_chunks: int = 0
    mixed_position: int = 0
    admitted_ms: float = 0.0


class ChunkedPrefillPagedRunner(ContinuousPrefixPagedRunner):
    """Interleave bounded prompt chunks with live decode steps.

    Every arrived request is admitted while a slot and enough KV pages for
    its full length are free; requests queue only when those resources are
    exhausted.  Each iteration runs one decode step for all live rows, then
    spends up to ``max_prefill_tokens`` of prompt work, rotating fairly over
    partial prompts in chunks of at most ``prefill_chunk_size`` tokens.
    ``prefill_chunk_size=None`` is the whole-prompt control: every admitted
    prompt is prefilled before the next decode step.  With ``batched_prefill``
    those prompts share one flattened (packed, unpadded) forward whose
    attention is computed per request from ``cu_seqlens`` metadata.

    ``mixed_batch=True`` is vLLM-style iteration batching: each iteration is
    one flat forward holding every live decode token plus prompt chunks,
    shortest remaining prompt first, up to ``max_prefill_tokens`` total tokens
    (decode rows included).  A large prompt therefore advances a bounded
    chunk per step instead of stalling decode for its whole prefill.
    ``prefill_chunk_size`` optionally caps one request's chunk.
    """

    def __init__(self, model: Any, *, prefill_chunk_size: int | None = 128,
                 batched_prefill: bool = True, mixed_batch: bool = False,
                 packed_prefill_token_limit: int | None = None,
                 adaptive_chunking: bool = False, planner: AdaptiveChunkPlanner | None = None,
                 **kwargs: Any) -> None:
        if prefill_chunk_size is not None and (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size < 1
        ):
            raise ValueError("prefill_chunk_size must be positive or None")
        kwargs.setdefault("decode_sdpa_compat", True)
        super().__init__(model, **kwargs)
        self.prefill_chunk_size = prefill_chunk_size
        self.batched_prefill = bool(batched_prefill)
        self.mixed_batch = bool(mixed_batch)
        if packed_prefill_token_limit is not None and packed_prefill_token_limit < 1:
            raise ValueError("packed_prefill_token_limit must be positive or None")
        # Whole prompts are packed FIFO into one flattened forward up to this many
        # tokens (None: every pending prompt); a larger prompt runs alone.
        self.packed_prefill_token_limit = packed_prefill_token_limit
        if self.mixed_batch and self.enable_prefix:
            raise ValueError("mixed batching does not yet support prefix reuse")
        if adaptive_chunking and not self.mixed_batch:
            raise ValueError("adaptive chunking requires mixed_batch=True")
        # Adaptive mode sizes each step's prompt chunks from a step-time limit
        # (short when busy, longer when idle) instead of a fixed token budget.
        self.planner = (planner or AdaptiveChunkPlanner()) if adaptive_chunking else None
        self._last_arrival_ms: float | None = None
        self._queued = 0

    @property
    def runner_name(self) -> str:
        if self.mixed_batch:
            return "continuous_paged_mixed_batch"
        mode = "chunked" if self.prefill_chunk_size is not None else "unchunked"
        return f"continuous_paged_{mode}_prefill"

    def _mixed_step(self, active: list[_PartialRequest], prefilling: deque[_PartialRequest]) -> dict[str, Any]:
        """One forward: every decode token plus budgeted prompt chunks.

        Returns a step record; decoding and newly completed prompts have their
        next token appended.  Prompts that are still partial stay queued.
        """

        chunks: list[tuple[_PartialRequest, int, int]] = []
        plan_info: dict[str, Any] = {}
        if self.planner is not None:
            now_ms = (active or list(prefilling))[0].recorder.now_ns() / 1e6
            by_id = {state.request.request_id: state for state in prefilling}
            candidates = [PromptCandidate(state.request.request_id, state.request.prompt_tokens,
                                          state.mixed_position, now_ms - state.admitted_ms) for state in prefilling]
            busy = self.planner.is_busy(
                decode_rows=len(active), waiting_prompts=len(prefilling), queued=self._queued,
                ms_since_arrival=None if self._last_arrival_ms is None else now_ms - self._last_arrival_ms)
            planned, plan_info = self.planner.plan(candidates, decode_rows=len(active), busy=busy,
                                                   memory_token_cap=self._memory_token_cap())
            chunks = [(by_id[c.key], c.start, count) for c, count in planned]
        else:
            budget = max(self.max_prefill_tokens - len(active), 0)
            if not budget:
                budget = self.allocator.block_size  # never starve prompts behind decode rows
            cap = self.prefill_chunk_size or math.inf
            for state in sorted(prefilling, key=lambda item: item.request.prompt_tokens - item.mixed_position):
                if budget <= 0:
                    break
                remaining = state.request.prompt_tokens - state.mixed_position
                count = int(min(remaining, budget, cap))
                chunks.append((state, state.mixed_position, count))
                budget -= count
        rows = [MixedRow(state.request.request_id, (state.generated[-1],),
                         self.allocator.get_block_table(state.request.request_id).token_count, decode=True)
                for state in active]
        rows += [MixedRow(state.request.request_id,
                          tuple(state.request.prompt_token_ids[start:start + count]), start, decode=False)
                 for state, start, count in chunks]
        completing = [state for state, start, count in chunks if start + count == state.request.prompt_tokens]

        started = (active or [chunks[0][0]])[0].recorder.now_ns()
        for state, start, count in chunks:
            if start == 0:
                state.recorder.record("prefill_start", timestamp_ns=started, metadata={
                    "runner": self.runner_name, "reused_tokens": 0})
            state.recorder.record("prefill_chunk_start", timestamp_ns=started, metadata={"start_token": start})
        with torch.inference_mode():
            if active:
                self.cache.reserve_append(tuple(state.request.request_id for state in active), 1)
            for state, start, count in chunks:
                self.cache.reserve_append((state.request.request_id,), count)
            max_length = max((s.request.prompt_tokens + s.request.max_new_tokens - 1 for s in active), default=1)
            input_ids, positions, metadata = build_mixed_metadata(
                rows, self.cache, device=self.device,
                decode_attention_backend=self.decode_backend,
                decode_split_count=1 if self.decode_sdpa_compat else select_decode_split_count(
                    max_length, batch_size=max(len(active), 1)),
                decode_max_sequence_length=max_length,
                decode_block_tokens=128 if self.decode_sdpa_compat else select_decode_block_tokens(max_length),
                decode_sdpa_compat=self.decode_sdpa_compat,
            )
            output = qwen3_mixed_forward(
                self.model, input_ids, positions, metadata, paged_kv_cache=self.cache,
                prefill_logit_ids=[state.request.request_id for state in completing],
            )
            parts = []
            if output.decode_logits is not None:
                parts.append(output.decode_logits.argmax(dim=-1, keepdim=True))
            parts += [output.prefill_logits[s.request.request_id].argmax(dim=-1, keepdim=True) for s in completing]
            next_tokens = torch.cat(parts).detach() if parts else None
            values = next_tokens[:, 0].tolist() if next_tokens is not None else []
        ended = (active or [chunks[0][0]])[0].recorder.now_ns()
        if self.planner is not None:
            self.planner.cost.observe(len(active), [(start, count) for _, start, count in chunks],
                                      (ended - started) / 1e6)

        for row, state in enumerate(active):
            token = int(values[row])
            state.next_token = next_tokens[row:row + 1]
            index = len(state.generated)
            state.generated.append(token)
            state.decode_calls += 1
            state.recorder.mark_token_ready(index, token_id=token, timestamp_ns=ended)
            state.recorder.mark_token_sent(index, token_id=token, timestamp_ns=ended)
        chunk_records = []
        for state, start, count in chunks:
            state.mixed_position = start + count
            state.prefill_chunks += 1
            complete = state.mixed_position == state.request.prompt_tokens
            state.recorder.record("prefill_chunk_end", timestamp_ns=ended, metadata={
                "start_token": start, "end_token": start + count, "complete": complete})
            chunk_records.append({
                "kind": "prefill_chunk", "request_id": state.request.request_id,
                "start_token": start, "end_token": start + count, "computed_tokens": count,
                "complete": complete, "while_decoding": bool(active), "start_ms": started / 1e6,
                "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6,
            })
        for offset, state in enumerate(completing):
            row = len(active) + offset
            token = int(values[row])
            state.next_token = next_tokens[row:row + 1]
            state.generated.append(token)
            state.recorder.record("prefill_end", timestamp_ns=ended)
            state.recorder.mark_token_ready(0, token_id=token, timestamp_ns=ended)
            state.recorder.mark_token_sent(0, token_id=token, timestamp_ns=ended)
            state.lifecycle.transition(RequestState.DECODING)
        return {
            "kind": "mixed_step", "decode_rows": len(active), **plan_info,
            "prefill_tokens": sum(count for _, _, count in chunks),
            "total_tokens": len(active) + sum(count for _, _, count in chunks),
            "start_ms": started / 1e6, "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6,
            "chunk_records": chunk_records,
            "decode_request_ids": [state.request.request_id for state in active],
        }

    def _prefill_step(self, state: _PartialRequest, token_limit: int) -> dict[str, Any]:
        owner = state.request.request_id
        if state.session is None:
            dense = self.cache.as_dynamic_cache((owner,), model_config=self.model.config) if state.reused_tokens else None
            state.session = ChunkedPrefillSession(
                state.request.prompt_token_ids, device=self.device,
                start_token=state.reused_tokens, past_key_values=dense,
            )
            state.recorder.record("prefill_start", metadata={"runner": self.runner_name, "reused_tokens": state.reused_tokens})
        session = state.session
        started = state.recorder.now_ns()
        state.recorder.record("prefill_chunk_start", timestamp_ns=started, metadata={"start_token": session.next_position})
        chunk = session.step(self.model, max_tokens=token_limit)
        with torch.inference_mode():
            self.cache.append_from_dynamic_cache(
                (owner,), session.past_key_values,
                previous_token_counts=(chunk.start_token,),
                appended_token_counts=(chunk.end_token - chunk.start_token,),
            )
            if chunk.complete:
                state.next_token = chunk.logits.argmax(dim=-1, keepdim=True).detach()
                token = int(state.next_token.item())
                if self.enable_prefix:
                    self.prefixes.publish(owner, state.request.prompt_token_ids)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        ended = state.recorder.now_ns()
        state.prefill_chunks += 1
        state.recorder.record("prefill_chunk_end", timestamp_ns=ended, metadata={"start_token": chunk.start_token, "end_token": chunk.end_token, "complete": chunk.complete})
        if chunk.complete:
            state.generated.append(token)
            state.recorder.record("prefill_end", timestamp_ns=ended)
            state.recorder.mark_token_ready(0, token_id=token, timestamp_ns=ended)
            state.recorder.mark_token_sent(0, token_id=token, timestamp_ns=ended)
            state.lifecycle.transition(RequestState.DECODING)
            state.session = None
        return {
            "kind": "prefill_chunk", "request_id": owner,
            "start_token": chunk.start_token, "end_token": chunk.end_token,
            "computed_tokens": chunk.end_token - chunk.start_token,
            "complete": chunk.complete, "start_ms": started / 1e6,
            "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6,
        }

    def _packed_prefill(self, states: list[_PartialRequest]) -> list[dict[str, Any]]:
        """Prefill several whole prompts in one flattened forward.

        Every prompt starts at position 0 with no reused KV.  K/V go straight
        into each request's pages, and attention, KV, and first-token logits
        are bitwise identical to a single-request Transformers prefill.
        """

        owners = tuple(state.request.request_id for state in states)
        lengths = tuple(state.request.prompt_tokens for state in states)
        started = states[0].recorder.now_ns()
        for state in states:
            state.recorder.record("prefill_start", timestamp_ns=started, metadata={
                "runner": self.runner_name, "reused_tokens": 0, "packed_batch_size": len(states)})
            state.recorder.record("prefill_chunk_start", timestamp_ns=started, metadata={"start_token": 0})
        with torch.inference_mode():
            self.cache.reserve_append(owners, lengths)
            output = qwen3_packed_prefill(
                self.model,
                torch.tensor([t for state in states for t in state.request.prompt_token_ids],
                             dtype=torch.long, device=self.device),
                PackedSequenceMetadata.from_lengths(owners, lengths, device=self.device),
                paged_kv_cache=self.cache,
                block_tables=self.cache.block_table_tensor(owners, device=self.device).to(dtype=torch.int32),
                attention_backend="sdpa",
                row_logits=True,
            )
            next_tokens = output.logits.argmax(dim=-1, keepdim=True).detach()
            tokens = next_tokens[:, 0].tolist()
        ended = states[0].recorder.now_ns()
        records = []
        for row, state in enumerate(states):
            owner = state.request.request_id
            if self.enable_prefix:
                self.prefixes.publish(owner, state.request.prompt_token_ids)
            state.next_token = next_tokens[row:row + 1]
            state.prefill_chunks += 1
            state.recorder.record("prefill_chunk_end", timestamp_ns=ended, metadata={
                "start_token": 0, "end_token": lengths[row], "complete": True})
            state.generated.append(tokens[row])
            state.recorder.record("prefill_end", timestamp_ns=ended)
            state.recorder.mark_token_ready(0, token_id=tokens[row], timestamp_ns=ended)
            state.recorder.mark_token_sent(0, token_id=tokens[row], timestamp_ns=ended)
            state.lifecycle.transition(RequestState.DECODING)
            records.append({
                "kind": "prefill_chunk", "request_id": owner, "start_token": 0,
                "end_token": lengths[row], "computed_tokens": lengths[row], "complete": True,
                "packed_batch_size": len(states), "start_ms": started / 1e6,
                "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6,
            })
        return records

    def _memory_token_cap(self) -> int | None:
        """Prompt tokens one step may hold given free GPU memory (~128 KiB each, 1 GiB spare)."""

        if self.device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(self.device)
        return max(int((free - 2**30) // (128 * 1024)), self.allocator.block_size)

    def _complete(self, state: _PartialRequest, outcomes: dict[str, RequestOutcome]) -> None:
        super()._complete(state, outcomes)
        prior = outcomes[state.request.request_id]
        outcomes[state.request.request_id] = RequestOutcome(
            status=prior.status, generated_token_ids=prior.generated_token_ids,
            metadata={**prior.metadata, "prefill_chunks": state.prefill_chunks, "prefill_chunk_size": self.prefill_chunk_size},
        )

    def __call__(self, requests: Sequence[RequestSpec], recorders: Sequence[RequestEventRecorder]) -> tuple[RequestOutcome, ...]:
        if not requests or len(requests) != len(recorders):
            raise ValueError("requests and recorders must be non-empty and equal")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("request IDs must be unique")
        unknown = self.cancel_request_ids - {request.request_id for request in requests}
        if unknown:
            raise ValueError(f"unknown cancellation IDs: {sorted(unknown)}")
        self.prefixes.clear()
        start_stats = self.prefixes.snapshot()
        states = {
            request.request_id: _PartialRequest(request, recorder, RequestLifecycle(request.request_id))
            for request, recorder in zip(requests, recorders, strict=True)
        }
        self.last_lifecycles = {owner: state.lifecycle for owner, state in states.items()}
        scheduler = ContinuousBatchScheduler(
            (ScheduledRequest(request.request_id, request.scheduled_arrival_ms, index) for index, request in enumerate(requests)),
            max_batch_size=self.max_batch_size, max_prefill_tokens=self.max_prefill_tokens,
            max_wait_ms=self.max_wait_ms,
        )
        costs = {request.request_id: request.prompt_tokens for request in requests}
        active: list[_PartialRequest] = []
        prefilling: deque[_PartialRequest] = deque()
        outcomes: dict[str, RequestOutcome] = {}
        records: list[dict[str, Any]] = []
        deferred = budget_deferred = peak_blocks = peak_inflight = 0
        failure: str | None = None
        clock = recorders[0]
        try:
            while not scheduler.empty or active or prefilling:
                if scheduler.admit(clock.now_ns() / 1e6):
                    self._last_arrival_ms = clock.now_ns() / 1e6
                for item in scheduler.cancel_ready(set(self.cancel_request_ids)):
                    self._cancel(states[item.request_id], outcomes)
                # Decode has priority; no prefill batching wait stalls a live row.
                if active and not self.mixed_batch:
                    owners = [state.request.request_id for state in active]
                    self._decode(active)
                    peak_blocks = max(peak_blocks, self.allocator.allocated_block_count)
                    finished = [state for state in active if len(state.generated) >= state.request.max_new_tokens or state.generated[-1] in self.eos_token_ids]
                    records.append({"kind": "decode", "request_ids": owners, "batch_size": len(active), "finished_request_ids": [s.request.request_id for s in finished]})
                    for state in finished:
                        self._complete(state, outcomes)
                    finished_ids = {state.request.request_id for state in finished}
                    active = [state for state in active if state.request.request_id not in finished_ids]
                if not active and not prefilling and not scheduler.ready_count:
                    if scheduler.next_arrival_ms is None:
                        break
                    _wait_until_ms(clock, scheduler.next_arrival_ms)
                    continue
                now = clock.now_ns() / 1e6
                if scheduler.admit(now):
                    self._last_arrival_ms = now
                if not active and not prefilling and scheduler.should_wait_for_batch(now, active_count=0):
                    _wait_until_ms(clock, float(scheduler.oldest_ready_arrival_ms) + self.max_wait_ms)
                    continue
                # Admit every ready request that fits; queue only on exhausted slots or pages.
                while scheduler.ready_count and len(active) + len(prefilling) < self.max_batch_size:
                    inflight = active + list(prefilling)
                    selected = scheduler.next_batch(costs, max_requests=1)
                    state = states[selected[0].request_id]
                    owner = state.request.request_id
                    final_length = state.request.prompt_tokens + state.request.max_new_tokens - 1
                    if math.ceil(final_length / self.allocator.block_size) > self.allocator.num_blocks:
                        self._fail(state, outcomes, "request exceeds total KV capacity")
                    else:
                        try:
                            if self.enable_prefix:
                                state.reused_tokens = self.prefixes.attach(owner, state.request.prompt_token_ids, output_tokens=state.request.max_new_tokens)
                            else:
                                self.allocator.allocate(owner)
                            future = self._future_blocks(state) + sum(self._future_blocks(item) for item in inflight)
                            if not self.prefixes.evict_until_free(future):
                                raise PagedKvOutOfMemoryError("insufficient capacity for all in-flight requests to finish")
                            else:
                                if self.enable_prefix:
                                    self.prefixes.record_admission(state.reused_tokens)
                                state.recorder.record("admission")
                                state.admitted_ms = state.recorder.now_ns() / 1e6
                                state.recorder.record("execution_start")
                                state.lifecycle.transition(RequestState.PREFILL)
                                prefilling.append(state)
                                peak_inflight = max(peak_inflight, len(inflight) + 1)
                        except PagedKvOutOfMemoryError:
                            if owner in self.allocator.sequence_ids:
                                self.cache.release(owner)
                            if inflight:
                                scheduler.defer_front(selected)
                                deferred += 1
                                break
                            self._fail(state, outcomes, "KV capacity unavailable")
                if self.mixed_batch:
                    self._queued = scheduler.ready_count
                    if active or prefilling:
                        step = self._mixed_step(active, prefilling)
                        peak_blocks = max(peak_blocks, self.allocator.allocated_block_count)
                        chunk_records = step.pop("chunk_records")
                        if step["decode_rows"]:
                            records.append({"kind": "decode", "request_ids": step["decode_request_ids"],
                                            "batch_size": step["decode_rows"]})
                        records.extend(chunk_records)
                        records.append(step)
                        newly_decoding = [state for state in prefilling if state.generated]
                        for state in newly_decoding:
                            prefilling.remove(state)
                        still_active = []
                        for state in active + newly_decoding:
                            if len(state.generated) >= state.request.max_new_tokens or state.generated[-1] in self.eos_token_ids:
                                self._complete(state, outcomes)
                            else:
                                still_active.append(state)
                        active = still_active
                    continue
                decoding = bool(active)
                while (self.prefill_chunk_size is None and self.batched_prefill and len(prefilling) > 1
                        and all(state.session is None and not state.reused_tokens for state in prefilling)):
                    # Whole prompts that arrived together share one flattened forward,
                    # packed FIFO up to the per-forward token limit.
                    limit = self.packed_prefill_token_limit or math.inf
                    batch, tokens = [], 0
                    for state in prefilling:
                        if batch and tokens + state.request.prompt_tokens > limit:
                            break
                        batch.append(state)
                        tokens += state.request.prompt_tokens
                    for _ in batch:
                        prefilling.popleft()
                    # A prompt at or above the limit runs alone; packing resumes after it.
                    step_records = self._packed_prefill(batch)
                    for record in step_records:
                        record["while_decoding"] = decoding
                        records.append(record)
                    peak_blocks = max(peak_blocks, self.allocator.allocated_block_count)
                    for state in batch:
                        if len(state.generated) >= state.request.max_new_tokens or state.generated[-1] in self.eos_token_ids:
                            self._complete(state, outcomes)
                        else:
                            active.append(state)
                # Spend the iteration's prompt budget round-robin across partial prompts.
                budget = math.inf if self.prefill_chunk_size is None else self.max_prefill_tokens
                while prefilling and budget > 0:
                    state = prefilling.popleft()
                    remaining = state.request.prompt_tokens - (
                        state.session.next_position if state.session else state.reused_tokens
                    )
                    limit = remaining if self.prefill_chunk_size is None else int(min(self.prefill_chunk_size, budget))
                    if self.prefill_chunk_size is None and state.session is None and not state.reused_tokens:
                        # A whole fresh prompt takes the packed path: K/V go straight into pages.
                        record = self._packed_prefill([state])[0]
                    else:
                        record = self._prefill_step(state, limit)
                    record["while_decoding"] = decoding
                    records.append(record)
                    budget -= record["computed_tokens"]
                    peak_blocks = max(peak_blocks, self.allocator.allocated_block_count)
                    if state.generated:
                        if len(state.generated) >= state.request.max_new_tokens or state.generated[-1] in self.eos_token_ids:
                            self._complete(state, outcomes)
                        else:
                            active.append(state)
                    else:
                        prefilling.append(state)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            self.prefixes.clear()
            raise
        finally:
            for owner, state in states.items():
                state.session = None
                if owner in self.allocator.sequence_ids:
                    self.cache.release(owner)
                if not state.lifecycle.terminal:
                    state.lifecycle.transition(RequestState.FAILED)
                    state.lifecycle.release_resources()
            self._record_summary(
                states=states, scheduler=scheduler, records=records,
                start_stats=start_stats, deferred=deferred,
                budget_deferred=budget_deferred, peak_blocks=peak_blocks,
                peak_inflight=peak_inflight, failure=failure,
            )
        if len(outcomes) != len(requests):
            raise RuntimeError("chunked scheduler left requests unfinished")
        return tuple(outcomes[request.request_id] for request in requests)

    def _record_summary(
        self, *, states: dict[str, _PartialRequest], scheduler: ContinuousBatchScheduler,
        records: list[dict[str, Any]], start_stats: dict[str, Any], deferred: int,
        budget_deferred: int, peak_blocks: int, peak_inflight: int,
        failure: str | None,
    ) -> None:
        stats = self.prefixes.snapshot()
        chunks = [record for record in records if record["kind"] == "prefill_chunk"]
        hits, misses = stats["hits"] - start_stats["hits"], stats["misses"] - start_stats["misses"]
        self.last_summary = {
            "status": "failed" if failure else "completed", "error": failure,
            "policy": "mixed_batch_shortest_first_token_budget" if self.mixed_batch else "decode_first_budgeted_round_robin_prefill",
            "mixed_batch": self.mixed_batch,
            "maximum_step_tokens": max((record["total_tokens"] for record in records if record["kind"] == "mixed_step"), default=0), "prefix_enabled": self.enable_prefix,
            "prefill_chunk_size": self.prefill_chunk_size, "max_prefill_tokens": self.max_prefill_tokens,
            "max_batch_size": self.max_batch_size, "maximum_inflight_requests": peak_inflight,
            "decode_mode_requested": self.decode_mode, "decode_mode_used": "eager",
            "graph_fallback_reason": "dynamic request membership and partial prefills" if self.decode_mode == "graph" else None,
            "decode_sdpa_compat": self.decode_sdpa_compat,
            "hits": hits, "misses": misses, "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
            "reused_tokens": stats["reused_tokens"] - start_stats["reused_tokens"],
            "computed_prefill_tokens": sum(record["computed_tokens"] for record in chunks),
            "maximum_prefill_chunk_tokens": max((record["computed_tokens"] for record in chunks), default=0),
            "prefill_chunks": len(chunks), "prefill_chunks_while_decoding": sum(record["while_decoding"] for record in chunks),
            "maximum_decode_batch_size": max((record["batch_size"] for record in records if record["kind"] == "decode"), default=0),
            "deferred_admissions": deferred, "budget_deferred_admissions": budget_deferred,
            "maximum_queue_depth": scheduler.maximum_queue_depth,
            "max_physical_blocks": peak_blocks, "peak_kv_utilization": peak_blocks / self.allocator.num_blocks,
            "evictions": stats["evictions"] - start_stats["evictions"],
            "prefill_records": chunks, "execution_records": records,
            "active_request_blocks_after_run": sum(owner in states for owner in self.allocator.sequence_ids),
        }
        self.run_summaries.append(self.last_summary)
