from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .hardware import GpuSampler, collect_environment_metadata
from .metrics import (
    RequestMetrics,
    calculate_request_metrics,
    summarize,
    summarize_request_metrics,
)
from .schemas import (
    EventRecord,
    HarnessConfig,
    RequestOutcome,
    RequestSpec,
    WorkloadSpec,
)
from .timing import measure_timer_overhead


Runner = Callable[[RequestSpec, "RequestEventRecorder"], RequestOutcome | Sequence[int] | None]
BatchRunner = Callable[
    [Sequence[RequestSpec], Sequence["RequestEventRecorder"]],
    Sequence[RequestOutcome | Sequence[int] | None],
]
TraceRunner = BatchRunner


class RequestEventRecorder:
    """Record canonical request events on a run-relative monotonic clock."""

    def __init__(self, request_id: str, *, run_started_ns: int) -> None:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        self.request_id = request_id
        self.run_started_ns = run_started_ns
        self._events: list[EventRecord] = []

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(self._events)

    @property
    def last_timestamp_ns(self) -> int:
        return self._events[-1].timestamp_ns if self._events else 0

    def now_ns(self) -> int:
        return max(0, time.perf_counter_ns() - self.run_started_ns)

    def record(
        self,
        event: str,
        *,
        timestamp_ns: int | None = None,
        token_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        relative_ns = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        if relative_ns < 0:
            raise ValueError("event timestamp cannot be before the run start")
        if self._events and relative_ns < self.last_timestamp_ns:
            raise ValueError("request events must be recorded in timestamp order")
        event_record = EventRecord(
            request_id=self.request_id,
            event=event,
            timestamp_ns=relative_ns,
            sequence=len(self._events),
            token_index=token_index,
            metadata=metadata or {},
        )
        self._events.append(event_record)
        return event_record

    def mark_token_ready(
        self,
        token_index: int,
        *,
        token_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        metadata = {} if token_id is None else {"token_id": int(token_id)}
        event_timestamp = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        self.record(
            "token_ready",
            timestamp_ns=event_timestamp,
            token_index=token_index,
            metadata=metadata,
        )
        if token_index == 0:
            self.record(
                "first_token_ready",
                timestamp_ns=event_timestamp,
                token_index=token_index,
                metadata=metadata,
            )

    def mark_token_sent(
        self,
        token_index: int,
        *,
        token_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        metadata = {} if token_id is None else {"token_id": int(token_id)}
        event_timestamp = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        self.record(
            "token_sent",
            timestamp_ns=event_timestamp,
            token_index=token_index,
            metadata=metadata,
        )
        if token_index == 0:
            self.record(
                "first_token_sent",
                timestamp_ns=event_timestamp,
                token_index=token_index,
                metadata=metadata,
            )


@dataclass(frozen=True)
class BenchmarkResult:
    workload: WorkloadSpec
    configuration: HarnessConfig
    captured_at_utc: str
    environment: dict[str, Any]
    warmup_durations_ms: tuple[float, ...]
    runs: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    benchmark_name: str = "minillm_l4_harness"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark_name,
            "captured_at_utc": self.captured_at_utc,
            "environment": self.environment,
            "workload": self.workload.to_dict(),
            "configuration": self.configuration.to_dict(),
            "warmup_durations_ms": list(self.warmup_durations_ms),
            "runs": list(self.runs),
            "summary": self.summary,
            "diagnostics": self.diagnostics,
        }


class BenchmarkHarness:
    """Run request traces with reusable instrumentation.

    ``run`` preserves the original one-request-at-a-time Phase 0 contract.
    ``run_batched`` adds a deterministic static-batch path for Phase 1 without
    changing request-level events or metric definitions.
    """

    def __init__(
        self,
        configuration: HarnessConfig | None = None,
        *,
        benchmark_name: str = "minillm_l4_harness",
    ) -> None:
        self.configuration = configuration or HarnessConfig()
        if not benchmark_name.strip():
            raise ValueError("benchmark_name must not be empty")
        self.benchmark_name = benchmark_name

    def run(self, workload: WorkloadSpec, runner: Runner) -> BenchmarkResult:
        if not callable(runner):
            raise TypeError("runner must be callable")
        def single_request_batch_runner(
            requests: Sequence[RequestSpec],
            recorders: Sequence[RequestEventRecorder],
        ) -> tuple[RequestOutcome | Sequence[int] | None, ...]:
            return tuple(
                runner(request, recorder)
                for request, recorder in zip(requests, recorders, strict=True)
            )

        return self._run_with_batch_runner(
            workload,
            single_request_batch_runner,
            batch_size=1,
            runner_for_signature=runner,
        )

    def run_batched(
        self,
        workload: WorkloadSpec,
        batch_size: int,
        runner: BatchRunner,
    ) -> BenchmarkResult:
        """Run ordered request groups through one static batch runner call."""

        if not callable(runner):
            raise TypeError("runner must be callable")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self._run_with_batch_runner(
            workload,
            runner,
            batch_size=batch_size,
            runner_for_signature=runner,
        )

    def run_trace(
        self,
        workload: WorkloadSpec,
        runner: TraceRunner,
    ) -> BenchmarkResult:
        """Give an entire arrival trace to a lifecycle-aware scheduler runner."""

        if not callable(runner):
            raise TypeError("runner must be callable")
        if not self.configuration.respect_arrival_schedule:
            raise ValueError("trace execution requires respect_arrival_schedule=True")
        return self._run_with_batch_runner(
            workload,
            runner,
            batch_size=len(workload.requests),
            runner_for_signature=runner,
            runner_manages_lifecycle=True,
        )

    def _run_with_batch_runner(
        self,
        workload: WorkloadSpec,
        batch_runner: BatchRunner,
        *,
        batch_size: int,
        runner_for_signature: Any,
        runner_manages_lifecycle: bool = False,
    ) -> BenchmarkResult:
        warmup_durations: list[float] = []
        for warmup_index in range(self.configuration.warmup_repetitions):
            warmup = self._run_once(
                workload,
                batch_runner,
                batch_size=batch_size,
                repetition_index=warmup_index,
                warmup=True,
                runner_manages_lifecycle=runner_manages_lifecycle,
            )
            warmup_durations.append(float(warmup["duration_ms"]))

        runs = tuple(
            self._run_once(
                workload,
                batch_runner,
                batch_size=batch_size,
                repetition_index=repetition_index,
                warmup=False,
                runner_manages_lifecycle=runner_manages_lifecycle,
            )
            for repetition_index in range(self.configuration.repetitions)
        )
        return BenchmarkResult(
            workload=workload,
            configuration=self.configuration,
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            environment=collect_environment_metadata(),
            warmup_durations_ms=tuple(warmup_durations),
            runs=runs,
            summary=_summarize_runs(runs),
            diagnostics={
                "timer_overhead": measure_timer_overhead(
                    iterations=self.configuration.timer_overhead_iterations
                ),
                "runner_signature": _runner_signature(runner_for_signature),
                "batch_size": batch_size,
                "runner_manages_lifecycle": runner_manages_lifecycle,
                "warmup_excluded_from_summary": True,
            },
            benchmark_name=self.benchmark_name,
        )

    def _run_once(
        self,
        workload: WorkloadSpec,
        batch_runner: BatchRunner,
        *,
        batch_size: int,
        repetition_index: int,
        warmup: bool,
        runner_manages_lifecycle: bool = False,
    ) -> dict[str, Any]:
        sampler = GpuSampler(
            device=workload.device,
            interval_seconds=self.configuration.sample_interval_seconds,
            enabled=self.configuration.collect_gpu,
            include_system_telemetry=self.configuration.collect_system_telemetry,
        )
        sampler.reset_peak_memory()
        # The before/after snapshots block on nvidia-smi (~25 ms on the L4), so
        # they stay outside the timed window that arrivals and throughput use.
        gpu_before = (
            sampler.snapshot("before", include_system_telemetry=True)
            if self.configuration.collect_gpu
            else None
        )
        run_started_ns = time.perf_counter_ns()
        run_started_utc = datetime.now(timezone.utc).isoformat()
        sampler.start(started_ns=run_started_ns)
        request_records: list[dict[str, Any]] = []
        all_events: list[EventRecord] = []
        runner_diagnostics: list[dict[str, Any]] = []

        for batch_start in range(0, len(workload.requests), batch_size):
            batch = workload.requests[batch_start : batch_start + batch_size]
            recorders: list[RequestEventRecorder] = []
            for request in batch:
                arrival_ns = self._arrival_timestamp_ns(
                    request,
                    run_started_ns=run_started_ns,
                )
                if (
                    self.configuration.respect_arrival_schedule
                    and not runner_manages_lifecycle
                ):
                    _wait_until_ns(run_started_ns + arrival_ns)
                recorder = RequestEventRecorder(
                    request.request_id,
                    run_started_ns=run_started_ns,
                )
                recorder.record(
                    "arrival",
                    timestamp_ns=arrival_ns,
                    metadata={
                        "scheduled_arrival_ms": request.scheduled_arrival_ms,
                        "arrival_schedule_respected": self.configuration.respect_arrival_schedule,
                    },
                )
                if not runner_manages_lifecycle:
                    recorder.record("admission")
                    recorder.record("execution_start")
                recorders.append(recorder)

            batch_call_started_ns = time.perf_counter_ns()
            try:
                raw_outcomes = tuple(batch_runner(batch, tuple(recorders)))
                if len(raw_outcomes) != len(batch):
                    raise ValueError(
                        "batch runner returned an unexpected number of outcomes: "
                        f"expected {len(batch)}, received {len(raw_outcomes)}"
                    )
            except Exception as error:
                raw_outcomes = tuple(
                    RequestOutcome(
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                    for _ in batch
                )
                batch_error = error
            else:
                batch_error = None
            batch_call_ended_ns = time.perf_counter_ns()
            execution_trace = getattr(batch_runner, "last_execution_trace", None)
            if callable(getattr(execution_trace, "to_dict", None)):
                execution_trace = execution_trace.to_dict()
            runner_diagnostics.append(
                {
                    "batch_index": batch_start // batch_size,
                    "request_ids": [request.request_id for request in batch],
                    "batch_size": len(batch),
                    "start_ms": (batch_call_started_ns - run_started_ns) / 1_000_000.0,
                    "end_ms": (batch_call_ended_ns - run_started_ns) / 1_000_000.0,
                    "batch_runner_wall_ms": (
                        batch_call_ended_ns - batch_call_started_ns
                    )
                    / 1_000_000.0,
                    "execution_trace": execution_trace,
                    "error": (
                        None
                        if batch_error is None
                        else f"{type(batch_error).__name__}: {batch_error}"
                    ),
                }
            )

            for request, recorder, raw_outcome in zip(
                batch,
                recorders,
                raw_outcomes,
                strict=True,
            ):
                outcome_error: Exception | None = batch_error
                outcome: RequestOutcome
                if outcome_error is None:
                    try:
                        outcome = _normalize_outcome(raw_outcome)
                        if len(outcome.generated_token_ids) > request.max_new_tokens:
                            raise ValueError(
                                f"runner generated {len(outcome.generated_token_ids)} tokens for "
                                f"request limit {request.max_new_tokens}"
                            )
                        _ensure_outcome_events(recorder, outcome)
                    except Exception as error:
                        outcome_error = error
                        outcome = RequestOutcome(
                            status="failed",
                            error=f"{type(error).__name__}: {error}",
                        )
                else:
                    outcome = raw_outcome  # type: ignore[assignment]

                if outcome_error is not None:
                    recorder.record(
                        "error",
                        metadata={
                            "exception_type": type(outcome_error).__name__,
                            "message": str(outcome_error),
                        },
                    )

                if outcome.status == "completed":
                    if not any(event.event == "completion" for event in recorder.events):
                        recorder.record("completion")
                elif outcome.status == "cancelled":
                    if not any(event.event == "cancelled" for event in recorder.events):
                        recorder.record("cancelled")
                elif outcome.status == "failed":
                    if not any(event.event == "error" for event in recorder.events):
                        recorder.record("error", metadata={"message": outcome.error})

                metrics = calculate_request_metrics(
                    request,
                    recorder.events,
                    outcome,
                )
                all_events.extend(recorder.events)
                request_records.append(
                    {
                        "request_id": request.request_id,
                        "outcome": outcome.to_dict(),
                        "metrics": metrics.to_dict(),
                    }
                )

        duration_ms = (time.perf_counter_ns() - run_started_ns) / 1_000_000.0
        gpu_after = (
            sampler.snapshot("after", include_system_telemetry=True)
            if self.configuration.collect_gpu
            else None
        )
        gpu_samples = sampler.stop()
        request_metrics = [
            _request_metrics_from_record(record["metrics"])
            for record in request_records
        ]
        run = {
            "run_id": f"{workload.name}-{'warmup' if warmup else 'measured'}-{repetition_index:03d}",
            "repetition_index": repetition_index,
            "warmup": warmup,
            "started_at_utc": run_started_utc,
            "duration_ms": duration_ms,
            "request_count": len(request_records),
            "requests": request_records,
            "events": [event.to_dict() for event in all_events],
            "runner_diagnostics": runner_diagnostics,
            "gpu": {
                "before": gpu_before,
                "after": gpu_after,
                "samples": gpu_samples if self.configuration.collect_gpu else [],
            },
            "summary": _summarize_run(request_metrics, duration_ms, gpu_samples),
        }
        return run

    def _arrival_timestamp_ns(
        self,
        request: RequestSpec,
        *,
        run_started_ns: int,
    ) -> int:
        if self.configuration.respect_arrival_schedule:
            return int(round(request.scheduled_arrival_ms * 1_000_000.0))
        return max(0, time.perf_counter_ns() - run_started_ns)

def _runner_signature(runner: Runner) -> str:
    try:
        return str(inspect.signature(runner))
    except (TypeError, ValueError):
        return type(runner).__name__


def _wait_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        time.sleep(min(remaining_ns / 1_000_000_000.0, 0.01))


def _normalize_outcome(value: RequestOutcome | Sequence[int] | None) -> RequestOutcome:
    if isinstance(value, RequestOutcome):
        return value
    if value is None:
        return RequestOutcome()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("runner output must be token IDs, not text")
    return RequestOutcome(generated_token_ids=tuple(int(token) for token in value))


def _ensure_outcome_events(
    recorder: RequestEventRecorder,
    outcome: RequestOutcome,
) -> None:
    if outcome.status != "completed":
        return
    if not any(event.event == "token_ready" for event in recorder.events):
        for token_index, token_id in enumerate(outcome.generated_token_ids):
            recorder.mark_token_ready(token_index, token_id=token_id)
            recorder.mark_token_sent(token_index, token_id=token_id)


def _request_metrics_from_record(payload: dict[str, Any]) -> RequestMetrics:
    return RequestMetrics(
        request_id=str(payload["request_id"]),
        category=str(payload["category"]),
        status=str(payload["status"]),
        prompt_tokens=int(payload["prompt_tokens"]),
        requested_output_tokens=int(payload["requested_output_tokens"]),
        generated_output_tokens=int(payload["generated_output_tokens"]),
        queue_delay_ms=_optional_float(payload.get("queue_delay_ms")),
        prefill_ms=_optional_float(payload.get("prefill_ms")),
        ttft_ms=_optional_float(payload.get("ttft_ms")),
        itl_ms=tuple(float(value) for value in payload.get("itl_ms", [])),
        tpot_ms=_optional_float(payload.get("tpot_ms")),
        decode_ms=_optional_float(payload.get("decode_ms")),
        e2e_latency_ms=_optional_float(payload.get("e2e_latency_ms")),
        tokens_per_second=_optional_float(payload.get("tokens_per_second")),
        e2e_tokens_per_second=_optional_float(payload.get("e2e_tokens_per_second")),
        error=payload.get("error"),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _summarize_run(
    request_metrics: Sequence[RequestMetrics],
    duration_ms: float,
    gpu_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    request_summary = summarize_request_metrics(request_metrics)
    completed = [metric for metric in request_metrics if metric.status == "completed"]
    completed_tokens = sum(metric.generated_output_tokens for metric in completed)
    duration_seconds = duration_ms / 1000.0
    run_summary: dict[str, Any] = {
        **request_summary,
        "duration_ms": duration_ms,
        "completed_output_tokens": completed_tokens,
        "requests_per_second": (
            len(completed) / duration_seconds if duration_seconds > 0 else None
        ),
        "tokens_per_second": (
            completed_tokens / duration_seconds if duration_seconds > 0 else None
        ),
        "memory": _memory_summary(gpu_samples),
        "gpu_utilization_percent": _gpu_utilization_summary(gpu_samples),
    }
    return run_summary


def _summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one measured run is required")
    all_metrics = [
        _request_metrics_from_record(request["metrics"])
        for run in runs
        for request in run["requests"]
    ]
    run_durations = [float(run["duration_ms"]) for run in runs]
    requests_per_second = [
        float(run["summary"]["requests_per_second"])
        for run in runs
        if run["summary"]["requests_per_second"] is not None
    ]
    tokens_per_second = [
        float(run["summary"]["tokens_per_second"])
        for run in runs
        if run["summary"]["tokens_per_second"] is not None
    ]
    all_gpu_samples = [
        sample
        for run in runs
        for sample in run["gpu"]["samples"]
    ]
    summary = summarize_request_metrics(all_metrics)
    summary.update(
        {
            "repetitions": len(runs),
            "run_duration_ms": summarize(run_durations).to_dict(),
            "requests_per_second": (
                summarize(requests_per_second).to_dict()
                if requests_per_second
                else {"available": False, "count": 0}
            ),
            "tokens_per_second": (
                summarize(tokens_per_second).to_dict()
                if tokens_per_second
                else {"available": False, "count": 0}
            ),
            "memory": _memory_summary(all_gpu_samples),
            "gpu_utilization_percent": _gpu_utilization_summary(all_gpu_samples),
            "run_summaries": [run["summary"] for run in runs],
        }
    )
    return summary


def _memory_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    allocated = [
        float(sample["torch_memory_allocated_bytes"])
        for sample in samples
        if sample.get("torch_memory_allocated_bytes") is not None
    ]
    reserved = [
        float(sample["torch_memory_reserved_bytes"])
        for sample in samples
        if sample.get("torch_memory_reserved_bytes") is not None
    ]
    peak_allocated = [
        float(sample["torch_peak_memory_allocated_bytes"])
        for sample in samples
        if sample.get("torch_peak_memory_allocated_bytes") is not None
    ]
    peak_reserved = [
        float(sample["torch_peak_memory_reserved_bytes"])
        for sample in samples
        if sample.get("torch_peak_memory_reserved_bytes") is not None
    ]
    values: dict[str, Any] = {
        "available": bool(allocated or reserved or peak_allocated or peak_reserved),
        "sample_count": len(samples),
    }
    for name, series in (
        ("allocated_bytes", allocated),
        ("reserved_bytes", reserved),
        ("peak_allocated_bytes", peak_allocated),
        ("peak_reserved_bytes", peak_reserved),
    ):
        values[name] = max(series) if series else None
    return values


def _gpu_utilization_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # Prefer NVML readings taken periodically during the run; the before/after
    # nvidia-smi snapshots alone are two endpoint readings, not a run profile.
    periodic = [
        float(sample["gpu_utilization_percent"])
        for sample in samples
        if sample.get("label") == "periodic" and sample.get("gpu_utilization_percent") is not None
    ]
    if periodic:
        return {**summarize(periodic).to_dict(), "source": "nvml_periodic"}
    values = [
        float(sample["nvidia_smi"]["gpu_utilization_percent"])
        for sample in samples
        if isinstance(sample.get("nvidia_smi"), dict)
        and sample["nvidia_smi"].get("gpu_utilization_percent") is not None
    ]
    return (
        {**summarize(values).to_dict(), "source": "nvidia_smi_endpoints"}
        if values
        else {"available": False, "count": 0}
    )


def write_result(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_events_jsonl(result: BenchmarkResult, path: Path) -> None:
    """Write raw request events as one independently streamable JSON line each."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for run in result.runs:
        for event in run["events"]:
            lines.append(
                json.dumps(
                    {
                        "run_id": run["run_id"],
                        "repetition_index": run["repetition_index"],
                        "warmup": run["warmup"],
                        **event,
                    },
                    sort_keys=True,
                )
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
