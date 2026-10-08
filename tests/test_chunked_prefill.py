from __future__ import annotations

import time
from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.engine.generation.chunked_prefill import (
    ALIGNED_PREFILL_ATTENTION,
    ChunkedPrefillSession,
    aligned_prefill_attention,
    aligned_prefill_sdpa,
)
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.step_planner import AdaptiveChunkPlanner, PromptCandidate, StepCostModel


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(47)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, use_sliding_window=False,
        sliding_window=None,
    )).eval()


def _run(runner, requests, *, repetitions=1, warmups=0):
    return BenchmarkHarness(HarnessConfig(
        repetitions=repetitions, warmup_repetitions=warmups,
        respect_arrival_schedule=True, collect_gpu=False,
        collect_system_telemetry=False,
    )).run_trace(WorkloadSpec(
        name="chunked-test", seed=47, device="cpu",
        requests=tuple(requests), arrival_pattern="fixed_rate",
    ), runner)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 16])
def test_partial_cache_positions_and_final_logits_match_full_prefill(chunk_size):
    model = _model()
    prompt = (1, 2, 3, 4, 5, 6, 7)
    calls = []
    original = model.forward

    def inspect(**kwargs):
        calls.append((kwargs["input_ids"].tolist()[0], kwargs["position_ids"].tolist()[0], kwargs["attention_mask"].shape[1]))
        return original(**kwargs)

    model.forward = inspect
    session = ChunkedPrefillSession(prompt, device="cpu")
    while not session.complete:
        output = session.step(model, max_tokens=chunk_size)
        assert session.past_key_values.get_seq_length() == output.end_token
    model.forward = original
    with torch.inference_mode():
        full = model(input_ids=torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(output.logits, full.logits[:, -1, :], atol=1e-6, rtol=1e-5)
    assert [token for inputs, _, _ in calls for token in inputs] == list(prompt)
    assert [pos for _, positions, _ in calls for pos in positions] == list(range(len(prompt)))
    assert all(len(inputs) <= chunk_size and mask_length == positions[-1] + 1 for inputs, positions, mask_length in calls)
    with pytest.raises(RuntimeError, match="already complete"):
        session.step(model, max_tokens=chunk_size)


def test_resumed_prefix_uses_absolute_positions_without_recomputing_prefix():
    model = _model()
    with torch.inference_mode():
        prefix = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    session = ChunkedPrefillSession((1, 2, 3, 4, 5, 6, 7), device="cpu", start_token=4, past_key_values=prefix.past_key_values)
    assert session.step(model, max_tokens=2).start_token == 4
    final = session.step(model, max_tokens=2)
    assert (final.start_token, final.end_token, final.complete) == (6, 7, True)
    assert session.past_key_values.get_seq_length() == 7
    with pytest.raises(ValueError, match="KV length"):
        ChunkedPrefillSession((1, 2, 3, 4), device="cpu", start_token=2)


@pytest.mark.parametrize("offset", [0, 1, 4, 7])
def test_aligned_attention_rows_are_bitwise_whole_prompt_rows(offset):
    torch.manual_seed(5)
    module = _model().model.layers[0].self_attn
    query = torch.randn(1, 4, 8, 4)
    key, value = torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)
    whole, _ = sdpa_attention_forward(module, query, key, value, None, scaling=0.5)
    chunk, _ = aligned_prefill_sdpa(module, query[:, :, offset:], key, value, None, scaling=0.5)
    assert torch.equal(chunk, whole[:, offset:])
    with pytest.raises(ValueError, match="unpadded"):
        aligned_prefill_sdpa(module, query, key, value, torch.ones(1, 1, 8, 8, dtype=torch.bool))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
def test_chunked_prefill_cache_matches_whole_prompt(chunk_size):
    model = _model()
    prompt = (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5)

    def prefill(limit):
        session = ChunkedPrefillSession(prompt, device="cpu")
        while not session.complete:
            output = session.step(model, max_tokens=limit)
        return output.logits, session.past_key_values

    whole_logits, whole = prefill(len(prompt))
    logits, chunked = prefill(chunk_size)
    torch.testing.assert_close(logits, whole_logits, atol=1e-6, rtol=1e-5)
    for layer, reference in zip(chunked.layers, whole.layers, strict=True):
        torch.testing.assert_close(layer.keys, reference.keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(layer.values, reference.values, atol=1e-6, rtol=1e-5)
    assert model.config._attn_implementation == "sdpa"


def test_aligned_prefill_attention_restores_sdpa_after_error():
    model = _model()
    with pytest.raises(RuntimeError, match="boom"):
        with aligned_prefill_attention(model):
            assert model.config._attn_implementation == ALIGNED_PREFILL_ATTENTION
            raise RuntimeError("boom")
    assert model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, None])
def test_chunked_and_unchunked_tokens_match_dense_reference(chunk_size):
    reference = _model()
    requests = (
        RequestSpec("short", (1, 2, 3, 4), 6),
        RequestSpec("long", tuple(range(1, 24)), 3),
        RequestSpec("later", (5, 6, 7, 8, 9), 4, scheduled_arrival_ms=0.1),
    )
    runner = ChunkedPrefillPagedRunner(deepcopy(reference), block_size=2, num_blocks=40,
        max_batch_size=3, max_prefill_tokens=2, prefill_chunk_size=chunk_size, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    assert result.summary["completed_requests"] == len(requests)
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        expected = manual_greedy_generate(reference, {
            "input_ids": torch.tensor([request.prompt_token_ids]),
            "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
        }, output_tokens=request.max_new_tokens)
        assert row["outcome"]["generated_token_ids"] == expected.row(0).tolist()
        events = [event for event in result.runs[0]["events"] if event["request_id"] == request.request_id]
        assert sum(event["event"] == "prefill_start" for event in events) == 1
        assert sum(event["event"] == "prefill_end" for event in events) == 1
        assert [event["token_index"] for event in events if event["event"] == "token_ready"] == list(range(request.max_new_tokens))
    if chunk_size is not None:
        assert runner.last_summary["maximum_prefill_chunk_tokens"] <= min(chunk_size, 2)
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_long_prompts_progress_while_decode_is_active_and_prefills_rotate():
    requests = (
        RequestSpec("short", (1, 2), 16),
        RequestSpec("long-a", tuple(range(1, 24)), 2),
        RequestSpec("long-b", tuple(range(2, 22)), 2),
    )
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=48,
        max_batch_size=3, max_prefill_tokens=3, prefill_chunk_size=2, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    assert result.summary["completed_requests"] == 3
    summary = runner.last_summary
    assert summary["maximum_inflight_requests"] == 3
    assert summary["prefill_chunks_while_decoding"] > 1
    records = summary["execution_records"]
    chunks = [record for record in records if record["kind"] == "prefill_chunk"]
    assert any(record["request_id"] == "long-a" and not record["complete"] and record["while_decoding"] for record in chunks)
    a_end = next(i for i, record in enumerate(chunks) if record["request_id"] == "long-a" and record["complete"])
    assert any(record["request_id"] == "long-b" for record in chunks[:a_end])
    # Prompt work between consecutive decode steps never exceeds the token budget.
    spent = 0
    for record in records:
        if record["kind"] == "decode":
            spent = 0
        else:
            assert record["computed_tokens"] <= 2
            spent += record["computed_tokens"]
            assert spent <= 3
    assert summary["budget_deferred_admissions"] == 0
    assert all(state.resources_released for state in runner.last_lifecycles.values())
    runner.close()


def test_prefix_reuse_with_chunk_boundary_and_repetition_reset():
    requests = (RequestSpec("a", (1, 2, 3, 4, 5, 6, 7, 8), 1),
                RequestSpec("b", (1, 2, 3, 4, 5, 6, 9, 10), 3))
    outputs = []
    for chunk in [None, 3]:
        runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=16,
            max_batch_size=1, max_prefill_tokens=2, prefill_chunk_size=chunk, device="cpu")
        result = _run(runner, requests, repetitions=2, warmups=1)
        assert all(summary["hits"] == 1 and summary["reused_tokens"] == 6 for summary in runner.run_summaries)
        assert all(summary["active_request_blocks_after_run"] == 0 for summary in runner.run_summaries)
        outputs.append([[row["outcome"]["generated_token_ids"] for row in run["requests"]] for run in result.runs])
        second_chunks = [row for row in runner.last_summary["prefill_records"] if row["request_id"] == "b"]
        assert second_chunks[0]["start_token"] == 6
        runner.close()
        assert runner.allocator.free_block_count == runner.allocator.num_blocks
    assert outputs[0] == outputs[1]


def test_capacity_backpressure_and_cancellation_do_not_leak_partial_sessions():
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=3,
        max_batch_size=2, max_prefill_tokens=2, prefill_chunk_size=1,
        cancel_request_ids=("cancel",), device="cpu", enable_prefix=False)
    result = _run(runner, (
        RequestSpec("cancel", (1, 2), 2),
        RequestSpec("too-big", tuple(range(1, 10)), 2),
        RequestSpec("a", (1, 2, 3, 4), 3),
        RequestSpec("b", (5, 6), 2),
    ))
    assert [row["outcome"]["status"] for row in result.runs[0]["requests"]] == ["cancelled", "failed", "completed", "completed"]
    assert runner.last_summary["deferred_admissions"] > 0
    assert runner.last_summary["active_request_blocks_after_run"] == 0
    assert all(state.resources_released for state in runner.last_lifecycles.values())
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_model_failure_releases_partial_cache_and_lifecycle():
    model = _model()
    original = model.forward
    calls = 0

    def failing(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected chunk failure")
        return original(**kwargs)

    model.forward = failing
    runner = ChunkedPrefillPagedRunner(model, block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=2, prefill_chunk_size=2, device="cpu")
    result = _run(runner, (RequestSpec("a", tuple(range(1, 10)), 2), RequestSpec("b", (4, 5), 2)))
    assert result.summary["failed_requests"] == 2
    assert runner.last_summary["status"] == "failed"
    assert runner.last_summary["active_request_blocks_after_run"] == 0
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
    assert all(state.resources_released for state in runner.last_lifecycles.values())


def test_eos_stops_after_completed_prompt_not_intermediate_chunk():
    model = _model()
    prompt = (1, 2, 3, 4)
    with torch.inference_mode():
        eos = int(model(input_ids=torch.tensor([prompt]), logits_to_keep=1).logits[:, -1, :].argmax())
    runner = ChunkedPrefillPagedRunner(model, block_size=2, num_blocks=12,
        max_batch_size=1, max_prefill_tokens=1, prefill_chunk_size=1, device="cpu", eos_token_id=eos)
    result = _run(runner, (RequestSpec("a", prompt, 8),))
    assert result.runs[0]["requests"][0]["outcome"]["generated_token_ids"] == [eos]
    assert runner.last_summary["prefill_chunks"] == len(prompt)
    assert runner.last_summary["maximum_decode_batch_size"] == 0
    runner.close()


def test_sdpa_compatible_decode_uses_its_required_tile_and_split_policy(monkeypatch):
    from minillm_l4.benchmarks.runners import continuous_prefix

    monkeypatch.setattr(continuous_prefix, "select_decode_split_count", lambda *args, **kwargs: 8)
    monkeypatch.setattr(continuous_prefix, "select_decode_block_tokens", lambda *args: 64)
    model = _model()
    original = model.forward
    policies = []

    def inspect(**kwargs):
        if "paged_kv_cache" in kwargs:
            policies.append((kwargs["paged_decode_sdpa_compat"], kwargs["paged_decode_split_count"], kwargs["paged_decode_block_tokens"]))
        return original(**kwargs)

    model.forward = inspect
    runner = ChunkedPrefillPagedRunner(model, block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=2, prefill_chunk_size=2, device="cpu")
    result = _run(runner, (RequestSpec("a", (1, 2, 3, 4), 3), RequestSpec("b", (5, 6, 7), 2)))
    assert result.summary["completed_requests"] == 2
    assert policies and all(policy == (True, 1, 128) for policy in policies)
    runner.close()


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_chunk_size_is_rejected(size):
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=8,
            max_batch_size=1, max_prefill_tokens=2, prefill_chunk_size=size, device="cpu")


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_admits_every_ready_request_while_pages_are_free(chunk_size):
    requests = tuple(RequestSpec(f"r{i}", tuple(range(1, 10 + i)), 4) for i in range(4))
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=64,
        max_batch_size=16, max_prefill_tokens=4, prefill_chunk_size=chunk_size, device="cpu", enable_prefix=False)
    assert _run(runner, requests).summary["completed_requests"] == 4
    summary = runner.last_summary
    assert summary["maximum_inflight_requests"] == 4
    assert summary["deferred_admissions"] == summary["budget_deferred_admissions"] == 0
    runner.close()


def test_queues_only_when_kv_pages_are_exhausted():
    # Each request needs 7 pages to finish; 16 pages hold two at a time.
    requests = tuple(RequestSpec(f"r{i}", tuple(range(1, 11)), 4) for i in range(3))
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=16,
        max_batch_size=16, max_prefill_tokens=4, prefill_chunk_size=2, device="cpu", enable_prefix=False)
    assert _run(runner, requests).summary["completed_requests"] == 3
    summary = runner.last_summary
    assert summary["maximum_inflight_requests"] == 2
    assert summary["deferred_admissions"] > 0
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
    runner.close()


@pytest.mark.parametrize("batched_prefill", [True, False])
def test_simultaneous_whole_prompts_share_one_flattened_prefill(batched_prefill):
    reference = _model()
    requests = (
        RequestSpec("a", (1, 2, 3, 4), 5),
        RequestSpec("b", tuple(range(1, 18)), 3),
        RequestSpec("c", (5, 6, 7, 8, 9, 10, 11), 4),
    )
    runner = ChunkedPrefillPagedRunner(deepcopy(reference), block_size=2, num_blocks=40, max_batch_size=8,
        max_prefill_tokens=4, prefill_chunk_size=None, batched_prefill=batched_prefill, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        expected = manual_greedy_generate(reference, {
            "input_ids": torch.tensor([request.prompt_token_ids]),
            "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
        }, output_tokens=request.max_new_tokens)
        assert row["outcome"]["generated_token_ids"] == expected.row(0).tolist()
    chunks = runner.last_summary["prefill_records"]
    assert len(chunks) == 3
    # Single fresh prompts also take the packed path, as a batch of one.
    assert [record.get("packed_batch_size") for record in chunks] == ([3, 3, 3] if batched_prefill else [1, 1, 1])
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
    runner.close()


def _reference_tokens(reference, request):
    return manual_greedy_generate(reference, {
        "input_ids": torch.tensor([request.prompt_token_ids]),
        "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
    }, output_tokens=request.max_new_tokens).row(0).tolist()


@pytest.mark.parametrize("budget,chunk_cap", [(6, None), (9, 4), (64, None)])
def test_mixed_batch_matches_reference_and_respects_token_budget(budget, chunk_cap):
    reference = _model()
    requests = (
        RequestSpec("anchor", (1, 2, 3), 12),
        RequestSpec("long", tuple(range(1, 30)), 4, scheduled_arrival_ms=0.05),
        RequestSpec("short", (7, 8, 9, 10), 5, scheduled_arrival_ms=0.1),
        RequestSpec("mid", tuple(range(3, 15)), 3, scheduled_arrival_ms=0.15),
    )
    runner = ChunkedPrefillPagedRunner(deepcopy(reference), block_size=2, num_blocks=64, max_batch_size=8,
        max_prefill_tokens=budget, prefill_chunk_size=chunk_cap, mixed_batch=True, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    assert result.summary["completed_requests"] == len(requests)
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        assert row["outcome"]["generated_token_ids"] == _reference_tokens(reference, request)
    summary = runner.last_summary
    steps = [record for record in summary["execution_records"] if record["kind"] == "mixed_step"]
    assert steps and summary["mixed_batch"]
    for step in steps:
        # Decode rows are never skipped, and prompt work stays inside the budget.
        assert step["prefill_tokens"] <= max(budget - step["decode_rows"], 2)
        if chunk_cap:
            assert all(r["computed_tokens"] <= chunk_cap for r in summary["prefill_records"])
    # A request decodes in every step from its first token to its last.
    for request in requests:
        decode_steps = sum(request.request_id in step["decode_request_ids"] for step in steps)
        assert decode_steps == request.max_new_tokens - 1
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
    runner.close()


def test_mixed_batch_runs_shortest_prompt_first():
    requests = (
        RequestSpec("long", tuple(range(1, 40)), 2),
        RequestSpec("short", (5, 6, 7), 2),
    )
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=64, max_batch_size=4,
        max_prefill_tokens=8, prefill_chunk_size=None, mixed_batch=True, device="cpu", enable_prefix=False)
    _run(runner, requests)
    first = [r for r in runner.last_summary["execution_records"] if r["kind"] == "mixed_step"][0]
    chunks = [r for r in runner.last_summary["prefill_records"] if r["start_ms"] == first["start_ms"]]
    assert [(c["request_id"], c["computed_tokens"]) for c in chunks] == [("short", 3), ("long", 5)]
    runner.close()


def test_flattened_prefill_packs_fifo_up_to_the_token_limit():
    reference = _model()
    requests = (
        RequestSpec("a", tuple(range(1, 6)), 2),
        RequestSpec("b", tuple(range(2, 7)), 2),
        RequestSpec("c", tuple(range(1, 10)), 2),
        RequestSpec("d", (4, 5, 6), 2),
    )
    runner = ChunkedPrefillPagedRunner(deepcopy(reference), block_size=2, num_blocks=40, max_batch_size=8,
        max_prefill_tokens=4, prefill_chunk_size=None, packed_prefill_token_limit=10, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        assert row["outcome"]["generated_token_ids"] == _reference_tokens(reference, request)
    packed = {r["request_id"]: r.get("packed_batch_size") for r in runner.last_summary["prefill_records"]}
    assert packed == {"a": 2, "b": 2, "c": 1, "d": 1}
    runner.close()


def test_adaptive_chunking_matches_reference_and_records_plans():
    reference = _model()
    requests = (
        RequestSpec("anchor", (1, 2, 3), 10),
        RequestSpec("long", tuple(i % 31 + 1 for i in range(39)), 3, scheduled_arrival_ms=0.05),
        RequestSpec("short", (7, 8, 9, 10), 4, scheduled_arrival_ms=0.1),
    )
    planner = AdaptiveChunkPlanner(busy_step_ms=1e9, idle_step_ms=1e9, min_chunk_tokens=4,
                                   max_step_tokens=12, granularity=2)
    runner = ChunkedPrefillPagedRunner(deepcopy(reference), block_size=2, num_blocks=64, max_batch_size=8,
        max_prefill_tokens=12, prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True,
        planner=planner, device="cpu", enable_prefix=False)
    result = _run(runner, requests)
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        assert row["outcome"]["generated_token_ids"] == _reference_tokens(reference, request)
    steps = [r for r in runner.last_summary["execution_records"] if r["kind"] == "mixed_step"]
    assert all("busy" in step and "limit_ms" in step for step in steps)
    assert all(step["prefill_tokens"] + step["decode_rows"] <= 12 for step in steps)
    runner.close()


def test_planner_time_limit_aging_and_tail_rule():
    planner = AdaptiveChunkPlanner()
    long = PromptCandidate("long", 8192, 0, 0.0)
    busy_plan, busy = planner.plan([long], decode_rows=4, busy=True)
    idle_plan, idle = planner.plan([long], decode_rows=0, busy=False)
    assert busy_plan[0][1] < idle_plan[0][1]
    assert busy["predicted_ms"] <= planner.busy_step_ms and idle["predicted_ms"] <= planner.idle_step_ms
    # Shortest first, but a prompt that has waited long enough moves ahead.
    fresh_short = PromptCandidate("short", 512, 0, 0.0)
    old_long = PromptCandidate("old", 4096, 0, 5000.0)
    order, _ = planner.plan([fresh_short, PromptCandidate("long2", 4096, 0, 0.0)], decode_rows=0, busy=True)
    assert order[0][0].key == "short"
    order, _ = planner.plan([fresh_short, old_long], decode_rows=0, busy=True)
    assert order[0][0].key == "old"
    # A chunk never leaves a 1..128-token remainder on the padded path.
    for start in range(0, 8192, 997):
        plan, _ = planner.plan([PromptCandidate("p", 8192, start, 0.0)], decode_rows=0, busy=True)
        tail = 8192 - start - plan[0][1]
        assert tail == 0 or tail > 128


def test_cost_model_recalibrates_toward_measurements():
    model = StepCostModel()
    for tokens in (256, 512, 1024, 2048) * 4:
        model.observe(0, [(0, tokens)], 30.0 + 0.1 * tokens)  # a faster machine than the prior
    assert abs(model.predict(0, [(0, 1024)]) - (30.0 + 0.1 * 1024)) < 25.0


@pytest.mark.parametrize("mixed", [False, True])
def test_request_cancelled_while_others_decode_is_never_run(mixed):
    model = _model()
    # Every forward path ends in the final norm; slow it so decode is busy when "late" arrives.
    model.model.norm.register_forward_pre_hook(lambda *_: time.sleep(0.005))
    runner = ChunkedPrefillPagedRunner(model, block_size=2, num_blocks=32, max_batch_size=4,
        max_prefill_tokens=8, prefill_chunk_size=None, mixed_batch=mixed,
        cancel_request_ids=("late",), device="cpu", enable_prefix=False)
    result = _run(runner, (RequestSpec("a", (1, 2, 3), 12),
                           RequestSpec("late", (4, 5), 4, scheduled_arrival_ms=20.0)))
    rows = {row["request_id"]: row["outcome"] for row in result.runs[0]["requests"]}
    assert rows["a"]["status"] == "completed"
    assert rows["late"]["status"] == "cancelled" and rows["late"]["generated_token_ids"] == []
    assert runner.last_summary["active_request_blocks_after_run"] == 0
    runner.close()


@pytest.mark.parametrize("mixed", [False, True])
def test_requests_above_model_context_length_fail_at_admission(mixed):
    runner = ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=128, max_batch_size=4,
        max_prefill_tokens=128, prefill_chunk_size=None, mixed_batch=mixed, device="cpu", enable_prefix=False)
    prompt = tuple(i % 31 + 1 for i in range(60))
    result = _run(runner, (RequestSpec("fits", prompt, 4), RequestSpec("too-long", prompt, 5),
                           RequestSpec("prompt-too-long", prompt + prompt[:10], 1)))
    rows = {row["request_id"]: row["outcome"] for row in result.runs[0]["requests"]}
    assert rows["fits"]["status"] == "completed" and len(rows["fits"]["generated_token_ids"]) == 4
    for name in ("too-long", "prompt-too-long"):
        assert rows[name]["status"] == "failed" and "context length" in rows[name]["error"]
    assert runner.max_model_len == 64
    runner.close()
    with pytest.raises(ValueError, match="max_model_len"):
        ChunkedPrefillPagedRunner(_model(), block_size=2, num_blocks=8, max_batch_size=1,
                                  max_prefill_tokens=8, max_model_len=1, device="cpu")


def test_planner_memory_cap_is_hard_and_time_overrun_is_reported():
    planner = AdaptiveChunkPlanner()
    prompt = PromptCandidate("p", 8192, 0, 0.0)
    plan, info = planner.plan([prompt], decode_rows=0, busy=True, memory_token_cap=100)
    assert sum(count for _, count in plan) <= 100  # below the 144-token progress floor
    plan, info = planner.plan([prompt], decode_rows=0, busy=True, memory_token_cap=0)
    assert sum(count for _, count in plan) == planner.granularity  # alone, a prompt still advances
    plan, info = planner.plan([prompt, PromptCandidate("q", 300, 0, 0.0)], decode_rows=40, busy=True,
                              memory_token_cap=48)
    assert sum(count for _, count in plan) + 40 <= 48
    plan, info = planner.plan([prompt], decode_rows=40, busy=True, memory_token_cap=30)
    assert plan == [] and not info["over_limit"]  # decode rows use all memory: prompts wait
    slow = AdaptiveChunkPlanner(busy_step_ms=1.0)  # nothing fits: the progress floor overruns
    plan, info = slow.plan([prompt], decode_rows=0, busy=True)
    assert plan[0][1] == slow.min_chunk_tokens and info["over_limit"]


def test_cuda_oom_fails_only_inflight_requests_and_serving_continues():
    model = _model()
    original = model.forward
    calls = 0

    def oom_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("CUDA out of memory (injected)")
        return original(**kwargs)

    model.forward = oom_once
    runner = ChunkedPrefillPagedRunner(model, block_size=2, num_blocks=32, max_batch_size=2,
        max_prefill_tokens=2, prefill_chunk_size=2, device="cpu", enable_prefix=False)
    result = _run(runner, (RequestSpec("a", tuple(range(1, 10)), 2),
                           RequestSpec("b", (4, 5), 3, scheduled_arrival_ms=30.0)))
    rows = {row["request_id"]: row["outcome"] for row in result.runs[0]["requests"]}
    assert rows["a"]["status"] == "failed" and "out of memory" in rows["a"]["error"]
    assert rows["b"]["status"] == "completed" and len(rows["b"]["generated_token_ids"]) == 3
    summary = runner.last_summary
    assert summary["status"] == "completed" and summary["oom_errors"] == 1 and summary["oom_failed_requests"] == 1
    assert summary["active_request_blocks_after_run"] == 0
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
