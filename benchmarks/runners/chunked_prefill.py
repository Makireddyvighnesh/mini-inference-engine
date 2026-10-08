"""Round-robin prompt chunks interleaved with continuous paged decode."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import torch

from minillm_l4.engine.generation.chunked_prefill import ChunkedPrefillSession
from minillm_l4.engine.kv_cache import (
    PagedKvOutOfMemoryError,
    select_decode_block_tokens,
    select_decode_split_count,
    triton_is_available,
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
from .paged_kv import _first_model_device
from .decode_graph import (
    DEFAULT_GRAPH_BATCH_SIZES, DecodeGraph, DecodeGraphBuffers,
    graph_pool_memory_bytes, select_graph_bucket, validate_graph_batch_sizes,
)


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

    ``cuda_graphs=True`` lazily captures decode-only buckets (default
    1/2/4/8/16/32). ``graph_mixed_decode=True`` separates graph decode from
    eager prompt work; ``graph_mixed_decode_order`` chooses decode-first
    (default) or prefill-first. Prefix reuse and unsupported devices/dispatch
    retain eager execution. See docs/phase_notes/cuda_graphs.md.
    """

    def __init__(self, model: Any, *, prefill_chunk_size: int | None = 128,
                 batched_prefill: bool = True, mixed_batch: bool = False,
                 packed_prefill_token_limit: int | None = None,
                 adaptive_chunking: bool = False, planner: AdaptiveChunkPlanner | None = None,
                 cuda_graphs: bool = False,
                 graph_batch_sizes: Sequence[int] = DEFAULT_GRAPH_BATCH_SIZES,
                 graph_mixed_decode: bool = False,
                 graph_mixed_decode_order: str = "decode_first",
                 max_model_len: int | None = None,
                 **kwargs: Any) -> None:
        if prefill_chunk_size is not None and (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size < 1
        ):
            raise ValueError("prefill_chunk_size must be positive or None")
        if not isinstance(cuda_graphs, bool) or not isinstance(graph_mixed_decode, bool):
            raise ValueError("cuda_graphs and graph_mixed_decode must be booleans")
        self.graph_batch_sizes = validate_graph_batch_sizes(graph_batch_sizes)
        if graph_mixed_decode and (not cuda_graphs or not mixed_batch):
            raise ValueError("graph_mixed_decode requires cuda_graphs=True and mixed_batch=True")
        if graph_mixed_decode_order not in {"decode_first", "prefill_first"}:
            raise ValueError("graph_mixed_decode_order must be decode_first or prefill_first")
        self.cuda_graphs = cuda_graphs
        self.graph_mixed_decode = graph_mixed_decode
        self.graph_mixed_decode_order = graph_mixed_decode_order
        kwargs.setdefault("decode_sdpa_compat", True)
        self._graphs: dict[int, DecodeGraph] = {}
        self._graph_max_sequence_length = 0
        self._graph_scratch_owner = "__cuda_graph_scratch__"
        self._graph_scratch_blocks: tuple[int, ...] = ()
        self._graph_disabled_reasons: list[str] = []
        self._request_num_blocks = kwargs["num_blocks"]
        if cuda_graphs:
            device = torch.device(kwargs.get("device") or _first_model_device(model))
            if device.type != "cuda" or not torch.cuda.is_available():
                self._graph_disabled_reasons.append("CUDA graphs require a CUDA device")
            if kwargs.get("enable_prefix", True):
                self._graph_disabled_reasons.append("prefix reuse is enabled")
            if kwargs.get("decode_backend", "auto") == "torch" or not triton_is_available():
                self._graph_disabled_reasons.append("CUDA graphs require Triton decode")
            if not kwargs["decode_sdpa_compat"]:
                self._graph_disabled_reasons.append("graph decode requires SDPA-compatible decode to keep the reduction policy fixed")
            if not 1 <= kwargs["block_size"] <= 128:
                self._graph_disabled_reasons.append("Triton decode requires block_size in 1..128")
            rope = getattr(model.config, "rope_parameters", None) or {}
            if rope.get("rope_type", "default") != "default":
                self._graph_disabled_reasons.append("graph decode requires default RoPE (dynamic RoPE may synchronize)")
            if not self._graph_disabled_reasons:
                # Extra pages are exclusively scratch: request KV capacity is
                # unchanged, including when the caller supplied an exact pool.
                if kwargs["num_blocks"] < 1:
                    raise ValueError("num_blocks must be positive")
                kwargs["num_blocks"] += self.graph_batch_sizes[-1]
        super().__init__(model, **kwargs)
        if cuda_graphs and not self._graph_disabled_reasons:
            scratch = self.allocator.allocate(
                self._graph_scratch_owner,
                token_count=self.graph_batch_sizes[-1] * self.allocator.block_size,
            )
            self._graph_scratch_blocks = scratch.block_ids
        # Like vLLM, reject prompt + max output above the model's context length.
        self.max_model_len = int(max_model_len or getattr(model.config, "max_position_embeddings", 0) or 0) or None
        if self.max_model_len is not None and self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
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
        self._reset_graph_counters()

    def _reset_graph_counters(self) -> None:
        self._graph_replays = self._eager_decode_steps = 0
        self._graph_fallbacks = dict.fromkeys(self._graph_disabled_reasons, 0)
        self._graph_captures_this_run: list[int] = []

    def _graph_bucket(self, rows: int) -> int | None:
        if not self.cuda_graphs or self._graph_disabled_reasons:
            return None
        return select_graph_bucket(rows, self.graph_batch_sizes)

    def _note_eager_decode(self, rows: int, *, mixed_prompt: bool = False) -> None:
        self._eager_decode_steps += 1
        if not self.cuda_graphs:
            return
        reasons = self._graph_disabled_reasons or (
            ["decode batch exceeds the largest graph bucket"] if self._graph_bucket(rows) is None
            else ["prompt chunks use one eager mixed forward"] if mixed_prompt else []
        )
        for reason in reasons:
            self._graph_fallbacks[reason] = self._graph_fallbacks.get(reason, 0) + 1

    def _graph_decode_call(self, buffers: DecodeGraphBuffers) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = self.model(
            input_ids=buffers.input_ids,
            # HF mask construction may inspect device positions. Paged attention
            # ignores this mask; supply the already prepared mapping to bypass it.
            attention_mask={"full_attention": None},
            position_ids=buffers.position_ids, use_cache=False, return_dict=True, logits_to_keep=1,
            paged_kv_cache=self.cache,
            paged_sequence_ids=tuple(f"graph-row-{i}" for i in range(buffers.input_ids.shape[0])),
            paged_query_start_positions=buffers.position_ids,
            paged_block_tables=buffers.block_tables, paged_sequence_lengths=buffers.sequence_lengths,
            paged_attention_backend="triton", paged_decode_graph=True,
            paged_decode_max_sequence_length=self._graph_max_sequence_length,
            paged_decode_split_count=1, paged_decode_block_tokens=128,
            paged_decode_use_gqa_reuse=False, paged_decode_sdpa_compat=True,
        )
        logits = output.logits[:, -1, :]
        gaps = None
        if self.record_logit_gaps:
            top_two = logits.float().topk(2, dim=-1).values
            gaps = top_two[:, 0] - top_two[:, 1]
        return logits.argmax(dim=-1, keepdim=True), gaps

    def _capture_decode_graph(self, bucket: int) -> DecodeGraph:
        """Lazy capture, including warmup and one scratch-only replay.

        Each bucket owns a private graph pool. All capture-time writes target
        scratch pages, so warming a bucket cannot modify a live request's KV.
        """
        started = perf_counter()
        buffers = DecodeGraphBuffers.allocate(
            bucket, math.ceil(self._graph_max_sequence_length / self.allocator.block_size),
            self._graph_scratch_blocks, device=self.device, block_size=self.allocator.block_size,
        )
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.inference_mode(), torch.cuda.stream(warmup):
            for _ in range(3):
                self._graph_decode_call(buffers)
        current.wait_stream(warmup)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            tokens, gaps = self._graph_decode_call(buffers)
        graph.replay()
        torch.cuda.synchronize(self.device)
        captured = DecodeGraph(graph, buffers, tokens, gaps, (perf_counter() - started) * 1000,
                               graph_pool_memory_bytes(graph))
        self._graphs[bucket] = captured
        self._graph_captures_this_run.append(bucket)
        return captured

    def _decode(self, active: list[_LiveRequest]) -> None:
        bucket = self._graph_bucket(len(active))
        if bucket is None:
            self._note_eager_decode(len(active))
            super()._decode(active)  # eager call, dispatch, and ordering unchanged
            return
        owners = tuple(state.request.request_id for state in active)
        old_lengths = tuple(self.allocator.get_block_table(owner).token_count for owner in owners)
        with torch.inference_mode():
            self.cache.reserve_append(owners, 1)
            if self.allocator.shared_block_count:
                raise RuntimeError("graph decode cannot write shared KV pages")
            captured = self._graphs.get(bucket) or self._capture_decode_graph(bucket)
            captured.buffers.pack(
                [state.generated[-1] for state in active], old_lengths,
                [self.allocator.get_block_table(owner).block_ids for owner in owners],
            )
            captured.graph.replay()
            self._graph_replays += 1
            # Clone real outputs: graph-owned output storage is overwritten on
            # replay; eager fallback must retain a stable next_token per row.
            next_tokens = captured.output_tokens[:len(active)].clone().detach()
            values = next_tokens[:, 0].tolist()  # one real-row D2H copy
            gaps = captured.logit_gaps[:len(active)].tolist() if captured.logit_gaps is not None else None
        now = max(state.recorder.now_ns() for state in active)
        for row, (state, token) in enumerate(zip(active, values, strict=True)):
            state.next_token = next_tokens[row:row + 1]
            index = len(state.generated)
            state.generated.append(int(token))
            state.decode_calls += 1
            if gaps is not None:
                state.decode_top2_logit_gaps.append(float(gaps[row]))
            state.recorder.mark_token_ready(index, token_id=int(token), timestamp_ns=now)
            state.recorder.mark_token_sent(index, token_id=int(token), timestamp_ns=now)

    def close(self) -> None:
        self._graphs.clear()
        if self._graph_scratch_owner in self.allocator.sequence_ids:
            self.cache.release(self._graph_scratch_owner)
        self._graph_scratch_blocks = ()
        super().close()

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
        if active and not chunks and self._graph_bucket(len(active)) is not None:
            started = active[0].recorder.now_ns()
            self._decode(active)
            ended = active[0].recorder.now_ns()
            return {
                "kind": "mixed_step", "decode_rows": len(active), **plan_info,
                "prefill_tokens": 0, "total_tokens": len(active),
                "start_ms": started / 1e6, "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6,
                "chunk_records": [], "decode_request_ids": [s.request.request_id for s in active],
                "decode_mode": "graph", "split_decode": False,
            }
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
        split_decode = bool(active and chunks and self.graph_mixed_decode
                            and self._graph_bucket(len(active)) is not None)
        forward_active = active
        if split_decode:
            # Both orders write disjoint pages; prefill keeps one-row logits.
            # Decode-first sends its tokens immediately after replay/readback.
            if self.graph_mixed_decode_order == "decode_first":
                self._decode(active)
            forward_active = []
            rows = rows[len(active):]
        elif active:
            self._note_eager_decode(len(active), mixed_prompt=bool(chunks))
        with torch.inference_mode():
            if forward_active:
                self.cache.reserve_append(tuple(state.request.request_id for state in forward_active), 1)
            for state, start, count in chunks:
                self.cache.reserve_append((state.request.request_id,), count)
            max_length = max((s.request.prompt_tokens + s.request.max_new_tokens - 1 for s in forward_active), default=1)
            input_ids, positions, metadata = build_mixed_metadata(
                rows, self.cache, device=self.device,
                decode_attention_backend=self.decode_backend,
                decode_split_count=1 if self.decode_sdpa_compat else select_decode_split_count(
                    max_length, batch_size=max(len(forward_active), 1)),
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
        if split_decode and self.graph_mixed_decode_order == "prefill_first":
            self._decode(active)
        ended = (active or [chunks[0][0]])[0].recorder.now_ns()
        # The model predicts one eager forward; split timings also include
        # graph decode, so retain the prior and valid eager observations.
        if self.planner is not None and not split_decode:
            self.planner.cost.observe(len(active), [(start, count) for _, start, count in chunks],
                                      (ended - started) / 1e6)

        for row, state in enumerate(forward_active):
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
            row = len(forward_active) + offset
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
            "decode_mode": "graph" if split_decode else "eager" if active else None,
            "split_decode": split_decode,
            "split_decode_order": self.graph_mixed_decode_order if split_decode else None,
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
        return max(int((free - 2**30) // (128 * 1024)), 0)

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
        self._reset_graph_counters()
        if self.cuda_graphs and not self._graph_disabled_reasons:
            if any(r.request_id == self._graph_scratch_owner for r in requests):
                raise ValueError("request ID is reserved for graph scratch pages")
            max_length = min(max(r.prompt_tokens + r.max_new_tokens - 1 for r in requests),
                             self._request_num_blocks * self.allocator.block_size)
            if max_length != self._graph_max_sequence_length:
                # Graph shapes belong to a workload capacity, not a request ID.
                # Reuse across repetitions; discard when a new workload changes it.
                self._graphs.clear()
                self._graph_max_sequence_length = max_length
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
        deferred = budget_deferred = peak_blocks = peak_inflight = oom_errors = 0
        failure: str | None = None
        clock = recorders[0]
        try:
            while not scheduler.empty or active or prefilling:
                try:
                    if scheduler.admit(clock.now_ns() / 1e6):
                        self._last_arrival_ms = clock.now_ns() / 1e6
                    for item in scheduler.cancel_ready(set(self.cancel_request_ids)):
                        self._cancel(states[item.request_id], outcomes)
                    # Decode has priority; no prefill batching wait stalls a live row.
                    if active and not self.mixed_batch:
                        owners = [state.request.request_id for state in active]
                        started = clock.now_ns()
                        self._decode(active)
                        ended = clock.now_ns()
                        peak_blocks = max(peak_blocks, self.allocator.allocated_block_count)
                        finished = [state for state in active if len(state.generated) >= state.request.max_new_tokens or state.generated[-1] in self.eos_token_ids]
                        records.append({"kind": "decode", "request_ids": owners, "batch_size": len(active),
                                        "finished_request_ids": [s.request.request_id for s in finished],
                                        "decode_mode": "graph" if self._graph_bucket(len(active)) is not None else "eager",
                                        "start_ms": started / 1e6, "end_ms": ended / 1e6, "wall_ms": (ended - started) / 1e6})
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
                    for item in scheduler.cancel_ready(set(self.cancel_request_ids)):
                        self._cancel(states[item.request_id], outcomes)
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
                        if self.max_model_len is not None and final_length + 1 > self.max_model_len:
                            self._fail(state, outcomes, f"request exceeds model context length "
                                       f"({final_length + 1} > {self.max_model_len} tokens)")
                        elif math.ceil(final_length / self.allocator.block_size) > self._request_num_blocks:
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
                                                "batch_size": step["decode_rows"], "decode_mode": step["decode_mode"],
                                                "wall_ms": step["wall_ms"] if not step["prefill_tokens"] else None})
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
                except torch.OutOfMemoryError as error:
                    # A forward ran out of GPU memory part-way: the KV of every
                    # in-flight request may be half-written, so fail those requests,
                    # free their pages, and keep serving the queue.  Completed
                    # requests keep their outputs.
                    oom_errors += 1
                    lost = [s for s in states.values()  # includes a packed batch already dequeued
                            if not s.lifecycle.terminal and s.lifecycle.state is not RequestState.WAITING]
                    for state in lost:
                        state.session = None
                        self._fail(state, outcomes, f"CUDA out of memory: {error}")
                    active, prefilling = [], deque()
                    records.append({"kind": "oom", "failed_request_ids": [s.request.request_id for s in lost],
                                    "at_ms": clock.now_ns() / 1e6})
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
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
                peak_inflight=peak_inflight, failure=failure, oom_errors=oom_errors,
            )
        if len(outcomes) != len(requests):
            raise RuntimeError("chunked scheduler left requests unfinished")
        return tuple(outcomes[request.request_id] for request in requests)

    def _record_summary(
        self, *, states: dict[str, _PartialRequest], scheduler: ContinuousBatchScheduler,
        records: list[dict[str, Any]], start_stats: dict[str, Any], deferred: int,
        budget_deferred: int, peak_blocks: int, peak_inflight: int,
        failure: str | None, oom_errors: int = 0,
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
            "decode_mode_requested": "graph" if self.cuda_graphs else self.decode_mode,
            "decode_mode_used": ("mixed" if self._graph_replays and self._eager_decode_steps else
                                 "graph" if self._graph_replays else "eager"),
            "cuda_graphs": self.cuda_graphs, "graph_mixed_decode": self.graph_mixed_decode,
            "graph_mixed_decode_order": self.graph_mixed_decode_order,
            "graph_batch_sizes": list(self.graph_batch_sizes),
            "graph_replays": self._graph_replays, "eager_decode_steps": self._eager_decode_steps,
            "graph_replay_share": self._graph_replays / (self._graph_replays + self._eager_decode_steps)
                if self._graph_replays + self._eager_decode_steps else 0.0,
            "graph_fallback_reason": ("; ".join(self._graph_fallbacks) if self._graph_fallbacks else
                                      "cuda_graphs=False (legacy decode_mode request)" if self.decode_mode == "graph" and not self.cuda_graphs else None),
            "graph_fallback_steps_by_reason": dict(self._graph_fallbacks),
            "captured_buckets": sorted(self._graphs),
            "graph_captures_this_run": list(self._graph_captures_this_run),
            "capture_ms": {str(n): graph.capture_ms for n, graph in sorted(self._graphs.items())},
            "graph_pool_memory_bytes_per_bucket": {str(n): graph.pool_memory_bytes for n, graph in sorted(self._graphs.items())},
            "graph_pool_memory_bytes": sum(graph.pool_memory_bytes for graph in self._graphs.values()),
            "graph_scratch_pages": len(self._graph_scratch_blocks),
            "graph_scratch_memory_bytes": (len(self._graph_scratch_blocks) * self.cache.num_layers
                * self.cache.num_kv_heads * self.allocator.block_size * self.cache.head_dim
                * self.cache.key_blocks.element_size() * 2),
            "graph_static_buffer_memory_bytes_per_bucket": {
                str(n): sum(t.numel() * t.element_size() for t in
                            (g.buffers.input_ids, g.buffers.position_ids, g.buffers.sequence_lengths,
                             g.buffers.block_tables, g.buffers.attention_mask))
                for n, g in sorted(self._graphs.items())},
            "request_kv_num_blocks": self._request_num_blocks,
            "graph_max_sequence_length": self._graph_max_sequence_length,
            "decode_sdpa_compat": self.decode_sdpa_compat,
            "hits": hits, "misses": misses, "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
            "reused_tokens": stats["reused_tokens"] - start_stats["reused_tokens"],
            "computed_prefill_tokens": sum(record["computed_tokens"] for record in chunks),
            "maximum_prefill_chunk_tokens": max((record["computed_tokens"] for record in chunks), default=0),
            "prefill_chunks": len(chunks), "prefill_chunks_while_decoding": sum(record["while_decoding"] for record in chunks),
            "maximum_decode_batch_size": max((record["batch_size"] for record in records if record["kind"] == "decode"), default=0),
            "deferred_admissions": deferred, "budget_deferred_admissions": budget_deferred,
            "oom_errors": oom_errors,
            "oom_failed_requests": sum(len(r["failed_request_ids"]) for r in records if r["kind"] == "oom"),
            "planned_steps_over_time_limit": sum(bool(r.get("over_limit")) for r in records if r["kind"] == "mixed_step"),
            "maximum_queue_depth": scheduler.maximum_queue_depth,
            "max_physical_blocks": peak_blocks, "peak_kv_utilization": peak_blocks / self.allocator.num_blocks,
            "evictions": stats["evictions"] - start_stats["evictions"],
            "prefill_records": chunks, "execution_records": records,
            "active_request_blocks_after_run": sum(owner in states for owner in self.allocator.sequence_ids),
        }
        self.run_summaries.append(self.last_summary)
