"""Matched paged static batching and forward phase instrumentation."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch

from minillm_l4.engine.request import RequestLifecycle, RequestState
from minillm_l4.engine.scheduler import ScheduledRequest, StaticBatchScheduler
from ..core.harness import RequestEventRecorder
from ..core.metrics import summarize
from ..core.schemas import RequestOutcome, RequestSpec
from .concurrent_requests import _padded_inputs
from .continuous_prefix import ContinuousPrefixPagedRunner, _LiveRequest
from .continuous_requests import _wait_until_ms


class ForwardPhaseMeter:
    """Measure forwards without per-forward synchronization or a profiler.

    CUDA elapsed time includes stream idle/launch gaps within a forward; it
    is not kernel-busy time. Host enqueue time does not imply completed GPU
    work. Request event timings remain the serving latency authority.
    """

    def __init__(self, model: Any, device: str | torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.calls: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self.phase_override: str | None = None

    def _before(self, module, args, kwargs):
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("phase measurement requires token input tensors")
        record = {
            "phase": self.phase_override or (
                "decode" if kwargs.get("paged_kv_cache") is not None
                or inputs.shape[1] == 1 and kwargs.get("past_key_values") is not None else "prefill"
            ),
            "batch_size": int(inputs.shape[0]), "input_tokens": int(inputs.numel()),
            "start_ns": time.perf_counter_ns(),
        }
        if self.device.type == "cuda":
            record["start_event"] = torch.cuda.Event(enable_timing=True)
            record["end_event"] = torch.cuda.Event(enable_timing=True)
            record["start_event"].record()
        self._pending.append(record)

    def _after(self, module, args, kwargs, output):
        if not self._pending:
            return
        record = self._pending.pop()
        if self.device.type == "cuda":
            record["end_event"].record()
        record["host_enqueue_ms"] = (time.perf_counter_ns() - record.pop("start_ns")) / 1e6
        self.calls.append(record)

    def start(self):
        self.calls.clear()
        self._pending.clear()
        self._pre = self.model.register_forward_pre_hook(self._before, with_kwargs=True)
        self._post = self.model.register_forward_hook(self._after, with_kwargs=True, always_call=True)

    def finish(self) -> dict[str, Any]:
        self._pre.remove()
        self._post.remove()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for record in self.calls:
            record["cuda_elapsed_ms"] = (
                record.pop("start_event").elapsed_time(record.pop("end_event"))
                if self.device.type == "cuda" else None
            )
        result = {"calls": list(self.calls)}
        for phase in ["prefill", "decode"]:
            rows = [row for row in self.calls if row["phase"] == phase]
            cuda = [row["cuda_elapsed_ms"] for row in rows if row["cuda_elapsed_ms"] is not None]
            result[phase] = {
                "forward_count": len(rows), "input_tokens": sum(row["input_tokens"] for row in rows),
                "host_enqueue_ms": summarize(row["host_enqueue_ms"] for row in rows).to_dict() if rows else None,
                "cuda_elapsed_ms": summarize(cuda).to_dict() if cuda else None,
                "cuda_elapsed_total_ms": sum(cuda) if cuda else None,
            }
        return result


class MeteredRunner:
    """Keep warm-up and measured forward records separate for each run."""

    def __init__(self, runner: Any, model: Any, device: str) -> None:
        self.runner = runner
        self.meter = ForwardPhaseMeter(model, device)
        self.run_summaries: list[dict[str, Any]] = []
        # A final one-token prompt chunk is still prefill, even with past KV.
        if hasattr(runner, "_prefill_step"):
            original = runner._prefill_step

            def prefill_step(*args, **kwargs):
                previous = self.meter.phase_override
                self.meter.phase_override = "prefill"
                try:
                    return original(*args, **kwargs)
                finally:
                    self.meter.phase_override = previous

            runner._prefill_step = prefill_step

    def __call__(self, requests, recorders):
        self.meter.start()
        try:
            return self.runner(requests, recorders)
        finally:
            self.run_summaries.append(self.meter.finish())


class StaticPagedBatchRunner(ContinuousPrefixPagedRunner):
    """Batch dense prefill, then paged decode with no new request admission.

    Mixed prompt lengths are left-padded only for prefill. Valid KV suffixes
    are copied to each request's pages. Completed rows leave decode, but new
    arrivals cannot join until the entire selected static group finishes.
    """

    def __call__(self, requests: Sequence[RequestSpec], recorders: Sequence[RequestEventRecorder]):
        if not requests or len(requests) != len(recorders):
            raise ValueError("requests and recorders must be non-empty and equal")
        if len(requests) > self.max_batch_size or len({r.request_id for r in requests}) != len(requests):
            raise ValueError("invalid static batch size or duplicate request IDs")
        needed = sum(math.ceil((r.prompt_tokens + r.max_new_tokens - 1) / self.allocator.block_size) for r in requests)
        if needed > self.allocator.num_blocks:
            raise ValueError("static group exceeds the configured page pool")
        states = [_LiveRequest(r, recorder, RequestLifecycle(r.request_id)) for r, recorder in zip(requests, recorders, strict=True)]
        self.last_lifecycles = {s.request.request_id: s.lifecycle for s in states}
        outcomes: dict[str, RequestOutcome] = {}
        owners = tuple(r.request_id for r in requests)
        try:
            for state in states:
                self.allocator.allocate(state.request.request_id)
                if not any(e.event == "admission" for e in state.recorder.events):
                    state.recorder.record("admission")
                    state.recorder.record("execution_start")
                state.lifecycle.transition(RequestState.PREFILL)
                state.recorder.record("prefill_start", metadata={"runner": "static_paged", "batch_size": len(states)})
            inputs, padded_length = _padded_inputs(requests, device=self.device, pad_token_id=0)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
                self.cache.append_from_dynamic_cache(owners, output.past_key_values,
                    previous_token_counts=(0,) * len(owners),
                    appended_token_counts=tuple(r.prompt_tokens for r in requests))
                next_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True).detach()
                values = next_tokens[:, 0].tolist()
                del output
            now = max(s.recorder.now_ns() for s in states)
            active = []
            for row, (state, token) in enumerate(zip(states, values, strict=True)):
                state.next_token = next_tokens[row:row + 1]
                state.generated.append(int(token))
                state.recorder.record("prefill_end", timestamp_ns=now)
                state.recorder.mark_token_ready(0, token_id=token, timestamp_ns=now)
                state.recorder.mark_token_sent(0, token_id=token, timestamp_ns=now)
                state.lifecycle.transition(RequestState.DECODING)
                if len(state.generated) == state.request.max_new_tokens:
                    self._complete(state, outcomes)
                else:
                    active.append(state)
            while active:
                self._decode(active)
                finished = [s for s in active if len(s.generated) == s.request.max_new_tokens]
                for state in finished:
                    self._complete(state, outcomes)
                finished_ids = {s.request.request_id for s in finished}
                active = [s for s in active if s.request.request_id not in finished_ids]
            self.last_summary = {
                "batch_size": len(states), "padded_prefill_tokens": len(states) * padded_length,
                "real_prefill_tokens": sum(r.prompt_tokens for r in requests),
                "active_request_blocks_after_run": 0,
            }
            return tuple(outcomes[r.request_id] for r in requests)
        finally:
            for state in states:
                if state.request.request_id in self.allocator.sequence_ids:
                    self.cache.release(state.request.request_id)
                if not state.lifecycle.terminal:
                    state.lifecycle.transition(RequestState.FAILED)
                    state.lifecycle.release_resources()


class StaticPagedTraceRunner:
    """FIFO static admission on the same paged backend as continuous runs."""

    def __init__(self, batch_runner: StaticPagedBatchRunner) -> None:
        self.batch_runner = batch_runner
        self.last_summary: dict[str, Any] | None = None

    def __call__(self, requests, recorders):
        scheduler = StaticBatchScheduler(
            (ScheduledRequest(r.request_id, r.scheduled_arrival_ms, i) for i, r in enumerate(requests)),
            max_batch_size=self.batch_runner.max_batch_size,
        )
        by_id = {r.request_id: (r, rec) for r, rec in zip(requests, recorders, strict=True)}
        outcomes = {}
        groups = []
        while not scheduler.empty:
            scheduler.admit(recorders[0].now_ns() / 1e6)
            if not scheduler.ready_count:
                _wait_until_ms(recorders[0], scheduler.next_arrival_ms)
                continue
            selected = scheduler.next_batch()
            batch = [by_id[item.request_id] for item in selected]
            output = self.batch_runner(tuple(r for r, _ in batch), tuple(rec for _, rec in batch))
            outcomes.update({item.request_id: value for item, value in zip(selected, output, strict=True)})
            groups.append({"request_ids": [item.request_id for item in selected], **self.batch_runner.last_summary})
        self.last_summary = {"policy": "fifo_static_paged", "batches": groups, "maximum_queue_depth": scheduler.maximum_queue_depth}
        return tuple(outcomes[r.request_id] for r in requests)
