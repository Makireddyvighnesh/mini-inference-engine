from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm_l4.benchmarks.core.harness import (
    BenchmarkHarness,
    RequestEventRecorder,
    write_events_jsonl,
    write_result,
)
from minillm_l4.benchmarks.core.hardware import capture_gpu_snapshot
from minillm_l4.benchmarks.core.metrics import (
    calculate_request_metrics,
    percentile,
    summarize,
)
from minillm_l4.benchmarks.runners.simulated import make_simulated_runner
from minillm_l4.benchmarks.core.schemas import (
    HarnessConfig,
    RequestOutcome,
    RequestSpec,
)
from minillm_l4.benchmarks.core.timing import measure_timer_overhead
from minillm_l4.benchmarks.core.tracing import ExecutionTrace
from minillm_l4.benchmarks.core.workloads import (
    PromptBucket,
    build_fixed_workload,
    build_mixed_workload,
    load_workload,
    save_workload,
)


def test_fixed_workload_is_deterministic_and_exact_length() -> None:
    bucket = PromptBucket("tiny", prompt_tokens=6, output_tokens=3)
    first = build_fixed_workload(bucket, count=3, seed=41)
    second = build_fixed_workload(bucket, count=3, seed=41)

    assert first.to_dict() == second.to_dict()
    assert [request.prompt_tokens for request in first.requests] == [6, 6, 6]
    assert [request.max_new_tokens for request in first.requests] == [3, 3, 3]
    assert len({request.prompt_sha256 for request in first.requests}) == 3


def test_mixed_workload_contains_all_default_buckets() -> None:
    workload = build_mixed_workload(count=9, seed=17)

    assert {request.category for request in workload.requests} == {
        "short",
        "medium",
        "long",
    }
    assert [
        request.category for request in workload.requests[:3]
    ] == ["short", "medium", "long"]


def test_workload_json_round_trip(tmp_path: Path) -> None:
    workload = build_fixed_workload("short", count=2, seed=9)
    path = tmp_path / "workload.json"

    save_workload(workload, path)

    assert load_workload(path).to_dict() == workload.to_dict()


def test_percentiles_use_documented_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    distribution = summarize([1, 2, 3, 4, 5])
    assert distribution.p50 == pytest.approx(3.0)
    assert distribution.p90 == pytest.approx(4.6)
    assert distribution.p95 == pytest.approx(4.8)
    assert distribution.p99 == pytest.approx(4.96)


def test_request_metrics_derive_ttft_itl_and_tpot() -> None:
    request = RequestSpec(
        request_id="request-0",
        prompt_token_ids=(10, 11),
        max_new_tokens=3,
    )
    recorder = RequestEventRecorder(request.request_id, run_started_ns=0)
    recorder.record("arrival", timestamp_ns=0)
    recorder.record("admission", timestamp_ns=1_000_000)
    recorder.record("prefill_start", timestamp_ns=1_000_000)
    recorder.record("prefill_end", timestamp_ns=3_000_000)
    recorder.mark_token_ready(0, token_id=20, timestamp_ns=5_000_000)
    recorder.mark_token_ready(1, token_id=21, timestamp_ns=7_000_000)
    recorder.mark_token_ready(2, token_id=22, timestamp_ns=9_000_000)
    recorder.record("completion", timestamp_ns=10_000_000)

    metrics = calculate_request_metrics(
        request,
        recorder.events,
        RequestOutcome(generated_token_ids=(20, 21, 22)),
    )

    assert metrics.queue_delay_ms == pytest.approx(1.0)
    assert metrics.prefill_ms == pytest.approx(2.0)
    assert metrics.ttft_ms == pytest.approx(5.0)
    assert metrics.itl_ms == pytest.approx((2.0, 2.0))
    assert metrics.tpot_ms == pytest.approx(2.0)
    assert metrics.decode_ms == pytest.approx(5.0)
    assert metrics.e2e_latency_ms == pytest.approx(10.0)
    assert metrics.tokens_per_second == pytest.approx(600.0)


def test_harness_repeats_records_raw_events_and_writes_artifacts(tmp_path: Path) -> None:
    workload = build_fixed_workload(
        PromptBucket("tiny", prompt_tokens=4, output_tokens=3),
        count=2,
        seed=3,
    )
    configuration = HarnessConfig(
        warmup_repetitions=1,
        repetitions=3,
        collect_gpu=False,
        collect_system_telemetry=False,
        timer_overhead_iterations=10,
        runner_name="simulated_cpu",
        timing_mode="wall",
    )
    result = BenchmarkHarness(configuration).run(
        workload,
        make_simulated_runner(prefill_ms=0.01, token_ms=0.01),
    )

    assert len(result.warmup_durations_ms) == 1
    assert len(result.runs) == 3
    assert result.summary["repetitions"] == 3
    assert result.summary["completed_requests"] == 6
    assert result.summary["metrics"]["ttft_ms"]["count"] == 6
    assert result.summary["metrics"]["itl_ms"]["count"] == 12
    assert result.diagnostics["timer_overhead"]["iterations"] == 10
    assert all(run["gpu"]["samples"] == [] for run in result.runs)
    assert all(
        {event["event"] for event in run["events"]}
        >= {"arrival", "prefill_start", "prefill_end", "token_ready", "completion"}
        for run in result.runs
    )

    result_path = tmp_path / "result.json"
    events_path = tmp_path / "events.jsonl"
    write_result(result, result_path)
    write_events_jsonl(result, events_path)
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    event_lines = events_path.read_text(encoding="utf-8").splitlines()
    assert loaded["benchmark"] == "minillm_l4_harness"
    assert len(event_lines) == sum(len(run["events"]) for run in result.runs)


def test_harness_records_runner_failure_without_losing_the_run() -> None:
    workload = build_fixed_workload(
        PromptBucket("tiny", prompt_tokens=2, output_tokens=2),
        count=1,
        seed=5,
    )

    def failing_runner(request: RequestSpec, recorder: RequestEventRecorder):
        del request, recorder
        raise RuntimeError("intentional failure")

    result = BenchmarkHarness(
        HarnessConfig(
            repetitions=1,
            warmup_repetitions=0,
            collect_gpu=False,
            collect_system_telemetry=False,
            timer_overhead_iterations=5,
        )
    ).run(workload, failing_runner)

    assert result.summary["failed_requests"] == 1
    assert result.runs[0]["requests"][0]["outcome"]["status"] == "failed"
    assert result.runs[0]["requests"][0]["outcome"]["error"].startswith(
        "RuntimeError:"
    )


def test_gpu_snapshot_is_best_effort_without_system_queries() -> None:
    snapshot = capture_gpu_snapshot("test", include_system_telemetry=False)

    assert snapshot["label"] == "test"
    assert "cuda_available" in snapshot
    assert "torch_memory_allocated_bytes" in snapshot


def test_timer_overhead_has_requested_sample_count() -> None:
    overhead = measure_timer_overhead(iterations=20)

    assert overhead["iterations"] == 20
    assert overhead["minimum_ns"] >= 0
    assert overhead["maximum_ns"] >= overhead["minimum_ns"]


def test_execution_trace_keeps_component_spans_and_aggregates() -> None:
    trace = ExecutionTrace(started_ns=0)
    with trace.span("input_setup", category="request_preparation"):
        pass
    with trace.span("decode_step", category="model_execution"):
        pass
    with trace.span("decode_step", category="model_execution"):
        pass

    payload = trace.to_dict()

    assert payload["schema_version"] == 1
    assert len(payload["spans"]) == 3
    assert payload["components"]["decode_step"]["count"] == 2
    assert payload["components"]["decode_step"]["wall_ms_total"] >= 0.0


def test_disabled_execution_trace_does_not_record_or_synchronize() -> None:
    trace = ExecutionTrace(enabled=False)
    with trace.span("decode_step", category="model_execution", gpu=True):
        pass
    trace.counter("tokens", 1)

    payload = trace.to_dict()

    assert payload["enabled"] is False
    assert payload["spans"] == []
    assert payload["components"] == {}
    assert payload["counters"] == {}
def test_environment_records_project_git_revision_when_available() -> None:
    from pathlib import Path

    from minillm_l4.benchmarks.core.hardware import collect_environment_metadata

    repo = Path(__file__).resolve().parents[1]
    metadata = collect_environment_metadata()
    if (repo / ".git").exists():
        assert isinstance(metadata["git_commit_sha"], str)
        assert len(metadata["git_commit_sha"]) == 40
        assert isinstance(metadata["git_worktree_dirty"], bool)


def test_blocking_gpu_snapshots_are_outside_the_timed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from minillm_l4.benchmarks.core import harness as harness_module

    snapshot_ms = 60.0

    class SlowSnapshotSampler(harness_module.GpuSampler):
        def snapshot(self, label, *, include_system_telemetry=None):
            time.sleep(snapshot_ms / 1000.0)
            return {"label": label}

        def start(self, *, started_ns=None) -> None:
            self._started_ns = started_ns

        def stop(self):
            return list(self.samples)

    monkeypatch.setattr(harness_module, "GpuSampler", SlowSnapshotSampler)
    workload = build_fixed_workload(
        PromptBucket("tiny", prompt_tokens=2, output_tokens=2),
        count=1,
        seed=7,
    )
    result = BenchmarkHarness(
        HarnessConfig(
            repetitions=1,
            warmup_repetitions=0,
            respect_arrival_schedule=True,
            collect_gpu=True,
            collect_system_telemetry=False,
            timer_overhead_iterations=5,
        )
    ).run(workload, make_simulated_runner(prefill_ms=0.01, token_ms=0.01))

    run = result.runs[0]
    assert run["gpu"]["before"] == {"label": "before"}
    assert run["gpu"]["after"] == {"label": "after"}
    assert run["duration_ms"] < snapshot_ms
    assert result.summary["metrics"]["ttft_ms"]["maximum"] < snapshot_ms
