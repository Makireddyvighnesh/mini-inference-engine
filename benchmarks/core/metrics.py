from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schemas import EventRecord, RequestOutcome, RequestSpec


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile.

    The definition is intentionally shared by every Phase 0 report so small
    samples do not silently switch between library-specific percentile rules.
    """

    if not values:
        raise ValueError("At least one value is required")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("values must be finite")
    ordered = sorted(normalized)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    standard_deviation: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "standard_deviation": self.standard_deviation,
        }


def summarize(values: Iterable[float]) -> DistributionSummary:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("Cannot summarize an empty measurement list")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("Measurements must be finite")
    ordered = sorted(normalized)
    return DistributionSummary(
        count=len(normalized),
        mean=statistics.fmean(normalized),
        median=statistics.median(normalized),
        p50=percentile(ordered, 50),
        p90=percentile(ordered, 90),
        p95=percentile(ordered, 95),
        p99=percentile(ordered, 99),
        minimum=ordered[0],
        maximum=ordered[-1],
        standard_deviation=(
            statistics.pstdev(normalized) if len(normalized) > 1 else 0.0
        ),
    )


def _event_times(events: Sequence[EventRecord], name: str) -> list[int]:
    return [event.timestamp_ns for event in events if event.event == name]


def _first_time(events: Sequence[EventRecord], name: str) -> int | None:
    times = _event_times(events, name)
    return min(times) if times else None


def _duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    if end_ns < start_ns:
        raise ValueError(f"Event order is invalid: {start_ns} -> {end_ns}")
    return (end_ns - start_ns) / 1_000_000.0


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    category: str
    status: str
    prompt_tokens: int
    requested_output_tokens: int
    generated_output_tokens: int
    queue_delay_ms: float | None
    prefill_ms: float | None
    ttft_ms: float | None
    itl_ms: tuple[float, ...]
    tpot_ms: float | None
    decode_ms: float | None
    e2e_latency_ms: float | None
    tokens_per_second: float | None
    e2e_tokens_per_second: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "category": self.category,
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "generated_output_tokens": self.generated_output_tokens,
            "queue_delay_ms": self.queue_delay_ms,
            "prefill_ms": self.prefill_ms,
            "ttft_ms": self.ttft_ms,
            "itl_ms": list(self.itl_ms),
            "tpot_ms": self.tpot_ms,
            "decode_ms": self.decode_ms,
            "e2e_latency_ms": self.e2e_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "e2e_tokens_per_second": self.e2e_tokens_per_second,
            "error": self.error,
        }


def calculate_request_metrics(
    request: RequestSpec,
    events: Sequence[EventRecord],
    outcome: RequestOutcome,
) -> RequestMetrics:
    """Derive serving metrics from canonical request events."""

    if any(event.request_id != request.request_id for event in events):
        raise ValueError("All events must belong to the supplied request")
    ordered_events = sorted(events, key=lambda event: event.sequence)
    arrival_ns = _first_time(ordered_events, "arrival")
    admission_ns = _first_time(ordered_events, "admission")
    prefill_start_ns = _first_time(ordered_events, "prefill_start")
    prefill_end_ns = _first_time(ordered_events, "prefill_end")
    completion_ns = _first_time(ordered_events, "completion")
    token_events = sorted(
        (event for event in ordered_events if event.event == "token_ready"),
        key=lambda event: (event.token_index if event.token_index is not None else -1, event.sequence),
    )
    token_times = [event.timestamp_ns for event in token_events]
    if any(right < left for left, right in zip(token_times, token_times[1:])):
        raise ValueError("token_ready events must be monotonic")
    itl_ms = tuple(
        (right - left) / 1_000_000.0
        for left, right in zip(token_times, token_times[1:])
    )
    first_token_ns = token_times[0] if token_times else _first_time(
        ordered_events, "first_token_ready"
    )
    generated_count = len(outcome.generated_token_ids) or len(token_events)
    tpot_ms = (
        (token_times[-1] - token_times[0]) / 1_000_000.0 / (len(token_times) - 1)
        if len(token_times) > 1
        else None
    )
    decode_ms = _duration_ms(first_token_ns, completion_ns)
    e2e_latency_ms = _duration_ms(arrival_ns, completion_ns)

    tokens_per_second = (
        generated_count / (decode_ms / 1000.0)
        if decode_ms is not None and decode_ms > 0 and generated_count > 0
        else None
    )
    e2e_tokens_per_second = (
        generated_count / (e2e_latency_ms / 1000.0)
        if e2e_latency_ms is not None
        and e2e_latency_ms > 0
        and generated_count > 0
        else None
    )
    return RequestMetrics(
        request_id=request.request_id,
        category=request.category,
        status=outcome.status,
        prompt_tokens=request.prompt_tokens,
        requested_output_tokens=request.max_new_tokens,
        generated_output_tokens=generated_count,
        queue_delay_ms=_duration_ms(arrival_ns, admission_ns),
        prefill_ms=_duration_ms(prefill_start_ns, prefill_end_ns),
        ttft_ms=_duration_ms(arrival_ns, first_token_ns),
        itl_ms=itl_ms,
        tpot_ms=tpot_ms,
        decode_ms=decode_ms,
        e2e_latency_ms=e2e_latency_ms,
        tokens_per_second=tokens_per_second,
        e2e_tokens_per_second=e2e_tokens_per_second,
        error=outcome.error,
    )


def _present_values(metrics: Sequence[RequestMetrics], field_name: str) -> list[float]:
    values: list[float] = []
    for metric in metrics:
        value = getattr(metric, field_name)
        if value is not None:
            values.append(float(value))
    return values


def summarize_request_metrics(
    metrics: Sequence[RequestMetrics],
    *,
    include_categories: bool = True,
) -> dict[str, Any]:
    """Aggregate request metrics while retaining missing-value counts."""

    if not metrics:
        raise ValueError("At least one request metric is required")
    metric_fields = (
        "queue_delay_ms",
        "prefill_ms",
        "ttft_ms",
        "tpot_ms",
        "decode_ms",
        "e2e_latency_ms",
        "tokens_per_second",
        "e2e_tokens_per_second",
    )
    distributions: dict[str, Any] = {}
    for field_name in metric_fields:
        values = _present_values(metrics, field_name)
        distributions[field_name] = (
            summarize(values).to_dict()
            if values
            else {"available": False, "count": 0}
        )
    all_itl = [itl for metric in metrics for itl in metric.itl_ms]
    distributions["itl_ms"] = (
        summarize(all_itl).to_dict()
        if all_itl
        else {"available": False, "count": 0}
    )

    completed = sum(metric.status == "completed" for metric in metrics)
    failed = sum(metric.status == "failed" for metric in metrics)
    cancelled = sum(metric.status == "cancelled" for metric in metrics)
    return {
        "request_count": len(metrics),
        "completed_requests": completed,
        "failed_requests": failed,
        "cancelled_requests": cancelled,
        "metrics": distributions,
        "by_category": (
            summarize_by_category(metrics) if include_categories else {}
        ),
    }


def summarize_by_category(metrics: Sequence[RequestMetrics]) -> dict[str, Any]:
    grouped: dict[str, list[RequestMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(metric.category, []).append(metric)
    return {
        category: summarize_request_metrics(category_metrics, include_categories=False)
        for category, category_metrics in sorted(grouped.items())
    }
