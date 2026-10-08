from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.benchmarks.runners.continuous_prefix import ContinuousPrefixPagedRunner
from minillm_l4.benchmarks.runners.decode_graph import (
    DecodeGraph, DecodeGraphBuffers, graph_pool_memory_bytes, select_graph_bucket,
)
from minillm_l4.engine.step_planner import AdaptiveChunkPlanner


def model(*, gpu=False):
    torch.manual_seed(47)
    head_dim = 64 if gpu else 4
    result = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=head_dim * 4, intermediate_size=head_dim * 8,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, max_position_embeddings=128,
        use_sliding_window=False, sliding_window=None,
    )).eval()
    return result.to(device="cuda", dtype=torch.bfloat16) if gpu else result


def make_runner(m=None, **options):
    return ChunkedPrefillPagedRunner(m or model(), **{
        "block_size": 2, "num_blocks": 64, "max_batch_size": 4,
        "max_prefill_tokens": 6, "prefill_chunk_size": None,
        "device": "cpu", "enable_prefix": False, **options,
    })


def run(runner, requests, *, repetitions=1, warmups=0):
    result = BenchmarkHarness(HarnessConfig(
        repetitions=repetitions, warmup_repetitions=warmups, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False,
    )).run_trace(WorkloadSpec(name="serving-graph-test", seed=47,
                             device=str(runner.device), requests=tuple(requests)), runner)
    assert result.summary["failed_requests"] == 0
    return [[row["outcome"]["generated_token_ids"] for row in measured["requests"]] for measured in result.runs]


@pytest.mark.parametrize("rows,expected", [(0, None), (1, 1), (2, 2), (3, 4), (4, 4),
                                         (5, 8), (9, 16), (17, 32), (32, 32), (33, None)])
def test_smallest_graph_bucket(rows, expected):
    assert select_graph_bucket(rows, (1, 2, 4, 8, 16, 32)) == expected
    assert select_graph_bucket(3, (2, 5, 7)) == 5


@pytest.mark.parametrize("sizes", [None, 4, (), (0,), (-1, 2), (True, 2), (1, 1), (4, 2), (1, 2.5)])
def test_invalid_buckets(sizes):
    with pytest.raises(ValueError, match="graph_batch_sizes"):
        make_runner(graph_batch_sizes=sizes)


@pytest.mark.parametrize("options", [
    {"cuda_graphs": 1}, {"graph_mixed_decode": "yes"},
    {"graph_mixed_decode": True, "mixed_batch": True},
    {"cuda_graphs": True, "graph_mixed_decode": True},
    {"graph_mixed_decode_order": "parallel"},
])
def test_invalid_graph_options(options):
    with pytest.raises(ValueError, match="graph"):
        make_runner(**options)


def test_packing_keeps_addresses_and_resets_padding_after_membership_changes():
    buffers = DecodeGraphBuffers.allocate(4, 3, (40, 41, 42, 43), device="cpu", block_size=2)
    addresses = [t.data_ptr() for t in (buffers.input_ids, buffers.position_ids,
                                      buffers.sequence_lengths, buffers.block_tables)]
    buffers.pack((7, 8, 9), (0, 2, 5), ((1,), (2, 3), (4, 5, 6)))
    assert buffers.input_ids.tolist() == [[7], [8], [9], [0]]
    assert buffers.position_ids.tolist() == [[0], [2], [5], [0]]
    assert buffers.sequence_lengths.tolist() == [1, 3, 6, 1]
    assert buffers.block_tables.tolist() == [[1, -1, -1], [2, 3, -1], [4, 5, 6], [43, -1, -1]]
    # Reordered/reused physical pages and a smaller batch must leave no stale
    # token, position, length, or real page in any padding row.
    buffers.pack((11,), (2,), ((6, 4),))
    assert buffers.sequence_lengths.tolist() == [3, 1, 1, 1]
    assert buffers.input_ids.tolist() == [[11], [0], [0], [0]]
    assert buffers.position_ids.tolist() == [[2], [0], [0], [0]]
    assert buffers.block_tables.tolist() == [[6, 4, -1], [41, -1, -1], [42, -1, -1], [43, -1, -1]]
    assert addresses == [t.data_ptr() for t in (buffers.input_ids, buffers.position_ids,
                                               buffers.sequence_lengths, buffers.block_tables)]
    buffers.pack((), (), ())
    assert buffers.block_tables[:, 0].tolist() == [40, 41, 42, 43]


@pytest.mark.parametrize("tokens,positions,blocks", [
    ((1, 2, 3), (0, 0, 0), ((1,), (2,), (3,))),
    ((1,), (), ((1,),)), ((1,), (-1,), ((1,),)),
    ((1,), (2,), ((1,),)), ((1,), (4,), ((1, 2, 3),)),
    ((1,), (0,), ((-1,),)), ((1,), (0,), ((40,),)),
])
def test_packing_rejects_invalid_or_scratch_real_rows(tokens, positions, blocks):
    buffers = DecodeGraphBuffers.allocate(2, 2, (40, 41), device="cpu", block_size=2)
    with pytest.raises(ValueError):
        buffers.pack(tokens, positions, blocks)


@pytest.mark.parametrize("scratch", [(1,), (1, 1), (-1, 2)])
def test_padding_rows_need_distinct_valid_scratch_pages(scratch):
    with pytest.raises(ValueError, match="scratch"):
        DecodeGraphBuffers.allocate(2, 2, scratch, device="cpu")


@pytest.mark.parametrize("mixed,split", [(False, False), (True, False), (True, True)])
def test_cpu_fallback_preserves_tokens_ordering_and_summary(mixed, split):
    requests = (RequestSpec("a", (1, 2, 3), 8), RequestSpec("b", tuple(range(1, 24)), 4))
    reference = model()
    eager = make_runner(deepcopy(reference), mixed_batch=mixed)
    expected = run(eager, requests)
    graph = make_runner(deepcopy(reference), cuda_graphs=True, mixed_batch=mixed, graph_mixed_decode=split)
    actual = run(graph, requests, repetitions=2, warmups=1)
    assert actual == expected * 2
    for summary in graph.run_summaries:
        assert summary["decode_mode_requested"] == "graph"
        assert summary["decode_mode_used"] == "eager"
        assert "CUDA" in summary["graph_fallback_reason"]
        assert summary["graph_replays"] == 0 and summary["eager_decode_steps"] > 0
        assert summary["eager_decode_steps"] == sum(r["kind"] == "decode" for r in summary["execution_records"])
        assert summary["captured_buckets"] == summary["graph_captures_this_run"] == []
        assert summary["capture_ms"] == summary["graph_pool_memory_bytes_per_bucket"] == {}
        assert summary["graph_pool_memory_bytes"] == summary["graph_scratch_pages"] == 0
        assert summary["graph_replay_share"] == 0
        assert not any(r.get("split_decode") for r in summary["execution_records"])
    graph.close()
    eager.close()
    assert graph.allocator.free_block_count == graph.allocator.num_blocks


def test_prefix_fallback_and_prefill_only_runs():
    runner = make_runner(cuda_graphs=True, enable_prefix=True, max_batch_size=1)
    run(runner, (RequestSpec("a", (1, 2, 3, 4), 3), RequestSpec("b", (1, 2, 3, 5), 3)))
    assert "prefix reuse" in runner.last_summary["graph_fallback_reason"]
    runner.close()
    runner = make_runner(cuda_graphs=True)
    run(runner, (RequestSpec("a", (1, 2, 3, 4), 1),))
    assert runner.last_summary["graph_replays"] == runner.last_summary["eager_decode_steps"] == 0
    assert runner.last_summary["captured_buckets"] == []
    runner.close()


def test_overflow_uses_original_eager_decode(monkeypatch):
    runner = make_runner(cuda_graphs=True, graph_batch_sizes=(1,))
    # Exercise the device-independent overflow routing on CPU; never fake CUDA.
    runner._graph_disabled_reasons = []
    monkeypatch.setattr(runner, "_capture_decode_graph", lambda n: pytest.fail("overflow must not capture"))
    calls = []
    original = ContinuousPrefixPagedRunner._decode
    def inspect(self, active):
        calls.append(len(active))
        return original(self, active)
    monkeypatch.setattr(ContinuousPrefixPagedRunner, "_decode", inspect)
    requests = tuple(RequestSpec(str(i), (1, 2, 3), 3) for i in range(2))
    run(runner, requests)
    assert calls == [2, 2]
    assert runner.last_summary["eager_decode_steps"] == 2
    assert "largest graph bucket" in runner.last_summary["graph_fallback_reason"]
    runner.close()


@pytest.mark.parametrize("order", ["decode_first", "prefill_first"])
def test_split_plumbing_and_both_orders_match_eager_on_cpu(monkeypatch, order):
    # Use the trusted eager decoder in place of replay to test the split
    # scheduler plumbing independently of CUDA and model matmul numerics.
    requests = (RequestSpec("a", (1, 2), 10), RequestSpec("b", tuple(range(1, 24)), 4))
    reference = model()
    eager = make_runner(deepcopy(reference))
    expected = run(eager, requests)
    split = make_runner(deepcopy(reference), cuda_graphs=True, mixed_batch=True,
                        graph_mixed_decode=True, graph_mixed_decode_order=order)
    monkeypatch.setattr(split, "_graph_bucket", lambda n: 4 if n else None)
    def decode(active):
        ContinuousPrefixPagedRunner._decode(split, active)
        split._graph_replays += 1
    monkeypatch.setattr(split, "_decode", decode)
    assert run(split, requests) == expected
    steps = [r for r in split.last_summary["execution_records"] if r.get("split_decode")]
    assert steps and all(r["split_decode_order"] == order for r in steps)
    assert split.last_summary["eager_decode_steps"] == 0
    split.close()
    eager.close()


@pytest.mark.parametrize("cuda_graphs,split,graph_available,order", [
    (False, False, False, "decode_first"),
    (True, False, True, "decode_first"),
    (True, True, False, "decode_first"),  # CPU fallback still observes eager work.
    (True, True, True, "decode_first"),
    (True, True, True, "prefill_first"),
])
def test_adaptive_cost_observes_only_eager_steps(monkeypatch, cuda_graphs, split, graph_available, order):
    planner = AdaptiveChunkPlanner(busy_step_ms=1e9, idle_step_ms=1e9,
                                   min_chunk_tokens=2, max_step_tokens=6, granularity=2)
    cost = planner.cost
    # Start with a fitted model so an invalid sample can change coefficients
    # immediately, rather than only after the initial four observations.
    for count in (2, 4, 6, 8):
        cost.observe(0, [(0, count)], cost.predict(0, [(0, count)]) + 10.0)
    runner = make_runner(mixed_batch=True, adaptive_chunking=True, planner=planner,
                         cuda_graphs=cuda_graphs, graph_mixed_decode=split,
                         graph_mixed_decode_order=order)
    if graph_available:
        monkeypatch.setattr(runner, "_graph_bucket", lambda n: 4 if n else None)
        # Exercise graph routing on CPU using the trusted eager decoder.
        def decode(active):
            ContinuousPrefixPagedRunner._decode(runner, active)
            runner._graph_replays += 1
        monkeypatch.setattr(runner, "_decode", decode)

    observations = []
    original_observe = cost.observe
    def observe(decode_rows, segments, measured_ms):
        observations.append((decode_rows, list(segments), measured_ms))
        original_observe(decode_rows, segments, measured_ms)
    monkeypatch.setattr(cost, "observe", observe)

    checks = []
    original_step = runner._mixed_step
    def inspect(active, prefilling):
        before = (cost.coefficients, len(cost._rows), len(observations))
        step = original_step(active, prefilling)
        checks.append((dict(step), before, cost.coefficients, len(cost._rows),
                       observations[before[2]:]))
        return step
    monkeypatch.setattr(runner, "_mixed_step", inspect)
    try:
        run(runner, (RequestSpec("a", (1, 2), 10), RequestSpec("b", tuple(range(1, 24)), 4)))
        for step, before, coefficients, observation_count, samples in checks:
            if step["decode_mode"] == "graph":
                assert samples == []
                assert observation_count == before[1]
                assert coefficients == before[0]
            else:
                segments = [(chunk["start_token"], chunk["computed_tokens"])
                            for chunk in step["chunk_records"]]
                assert samples == [(step["decode_rows"], segments, step["wall_ms"])]
                assert observation_count == before[1] + 1
        steps = [check[0] for check in checks]
        assert any(step["decode_rows"] and not step["prefill_tokens"] for step in steps)
        assert any(step["decode_rows"] and step["prefill_tokens"] for step in steps)
        assert any(step["split_decode"] for step in steps) == (split and graph_available)
        assert any(step["decode_mode"] == "graph" for step in steps) == graph_available
        assert any(not step["decode_rows"] and step["prefill_tokens"] for step in steps)
    finally:
        runner.close()


def test_pool_memory_counts_only_this_private_pool(monkeypatch):
    graph = SimpleNamespace(pool=lambda: (0, 3))
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [
        {"segment_pool_id": (0, 3), "total_size": 100},
        {"segment_pool_id": (0, 3), "total_size": 200},
        {"segment_pool_id": (0, 4), "total_size": 900},
        {"segment_pool_id": (0, 0), "total_size": 500},
    ])
    assert graph_pool_memory_bytes(graph) == 300


def test_mixed_summary_counts_default_eager_prompt_steps_and_graph_decode_steps(monkeypatch):
    runner = make_runner(cuda_graphs=True, mixed_batch=True)
    monkeypatch.setattr(runner, "_graph_bucket", lambda n: 4 if n else None)
    def decode(active):
        ContinuousPrefixPagedRunner._decode(runner, active)
        runner._graph_replays += 1
    monkeypatch.setattr(runner, "_decode", decode)
    run(runner, (RequestSpec("a", (1, 2), 10), RequestSpec("b", tuple(range(1, 24)), 4)))
    summary = runner.last_summary
    assert summary["decode_mode_used"] == "mixed"
    assert summary["graph_replays"] > 0 and summary["eager_decode_steps"] > 0
    assert 0 < summary["graph_replay_share"] < 1
    assert sum(r["kind"] == "decode" for r in summary["execution_records"]) == summary["graph_replays"] + summary["eager_decode_steps"]
    runner.close()


def test_capture_failure_releases_reserved_request_slots(monkeypatch):
    runner = make_runner(cuda_graphs=True)
    runner._graph_disabled_reasons = []
    def fail(bucket):
        raise RuntimeError("injected capture failure")
    monkeypatch.setattr(runner, "_capture_decode_graph", fail)
    requests = (RequestSpec("a", (1, 2, 3), 3), RequestSpec("b", (4, 5, 6), 3))
    result = BenchmarkHarness(HarnessConfig(
        repetitions=1, warmup_repetitions=0, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False,
    )).run_trace(WorkloadSpec(name="capture-failure", seed=47, requests=requests), runner)
    assert result.summary["failed_requests"] == 2
    assert runner.last_summary["status"] == "failed"
    assert "capture failure" in runner.last_summary["error"]
    assert runner.last_summary["active_request_blocks_after_run"] == 0
    assert runner.allocator.free_block_count == 64
    assert all(lifecycle.resources_released for lifecycle in runner.last_lifecycles.values())
    runner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mixed,split,order", [(False, False, "decode_first"), (True, False, "decode_first"),
                                            (True, True, "decode_first"), (True, True, "prefill_first")])
def test_cuda_graph_tokens_variable_membership_page_crossings_and_reuse(mixed, split, order):
    reference = model(gpu=True)
    requests = (RequestSpec("a", (1, 2, 3), 10), RequestSpec("b", tuple(range(1, 26)), 6),
                RequestSpec("c", (7, 8, 9, 10, 11), 4))
    options = dict(device="cuda", block_size=4, mixed_batch=mixed, max_prefill_tokens=8)
    eager = make_runner(reference, **options)
    expected = run(eager, requests)
    eager.close()
    graph = make_runner(reference, cuda_graphs=True, graph_mixed_decode=split,
                        graph_mixed_decode_order=order, **options)
    try:
        assert run(graph, requests, repetitions=2, warmups=1) == expected * 2
        assert graph.last_summary["graph_replays"] > 0
        assert graph.last_summary["captured_buckets"]
        assert graph.last_summary["graph_pool_memory_bytes"] > 0
        assert graph.last_summary["active_request_blocks_after_run"] == 0
        assert graph.allocator.free_block_count == 64  # scratch pages are additional
        renamed = tuple(RequestSpec(f"new-{r.request_id}", r.prompt_token_ids, r.max_new_tokens,
                                    scheduled_arrival_ms=r.scheduled_arrival_ms) for r in requests)
        assert run(graph, renamed) == expected
        assert graph.last_summary["graph_captures_this_run"] == []
        # New IDs and a changed max capacity invalidate shapes, not request data.
        changed = (RequestSpec("new-a", (1, 2, 3, 4, 5, 6), 6),)
        control = make_runner(reference, **options)
        assert run(graph, changed) == run(control, changed)
        control.close()
        assert graph.last_summary["graph_captures_this_run"] == [1]
    finally:
        graph.close()
    assert graph.allocator.free_block_count == graph.allocator.num_blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_oversized_batch_falls_back_then_reuses_graphs():
    reference = model(gpu=True)
    requests = (RequestSpec("a", (1, 2, 3), 8), RequestSpec("b", (4, 5, 6), 5),
                RequestSpec("c", (7, 8, 9), 3))
    eager = make_runner(reference, device="cuda", block_size=4)
    expected = run(eager, requests)
    eager.close()
    graph = make_runner(reference, device="cuda", block_size=4, cuda_graphs=True, graph_batch_sizes=(1, 2))
    try:
        assert run(graph, requests) == expected
        assert graph.last_summary["decode_mode_used"] == "mixed"
        assert graph.last_summary["graph_replays"] > 0 and graph.last_summary["eager_decode_steps"] > 0
        assert "largest graph bucket" in graph.last_summary["graph_fallback_reason"]
    finally:
        graph.close()


def test_graph_readback_and_reservation_without_cuda(monkeypatch):
    runner = make_runner(cuda_graphs=True, graph_batch_sizes=(4,))
    runner._graph_disabled_reasons = []
    # A CPU graph stand-in checks packing/reservation/output ownership without
    # claiming to test CUDA kernel math.
    runner._graph_max_sequence_length = 40
    buffers = DecodeGraphBuffers.allocate(4, 20, (100, 101, 102, 103), device="cpu", block_size=2)
    tokens = torch.zeros((4, 1), dtype=torch.long)
    def replay():
        tokens.copy_((buffers.input_ids + buffers.position_ids + 1) % 32)
    captured = DecodeGraph(SimpleNamespace(replay=replay), buffers, tokens, None, 1.0, 100)
    def capture(bucket):
        assert bucket == 4
        runner._graphs[bucket] = captured
        runner._graph_captures_this_run.append(bucket)
        return captured
    monkeypatch.setattr(runner, "_capture_decode_graph", capture)
    requests = (RequestSpec("a", (1, 2, 3), 5), RequestSpec("b", (4, 5), 3))
    run(runner, requests)
    assert runner.last_summary["graph_replays"] == 4
    assert runner.last_summary["decode_mode_used"] == "graph"
    assert runner.last_summary["capture_ms"] == {"4": 1.0}
    assert buffers.sequence_lengths[1:].tolist() == [1, 1, 1]
    assert runner.allocator.free_block_count == 64
    runner.close()
