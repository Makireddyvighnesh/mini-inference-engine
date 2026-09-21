"""Low-overhead component timing for model-runner diagnostics.

The benchmark's request events answer *when* a request became ready.  This
module answers *what the runner was doing* between those events.  GPU spans
use synchronized wall time plus CUDA events; CPU spans use the monotonic wall
clock.  Transformer layers are intentionally represented by one opaque
``model_forward`` span so the trace focuses on serving infrastructure.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass
class SpanMeasurement:
    """Mutable handle populated when an :class:`ExecutionTrace` span closes."""

    name: str
    category: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    wall_ms: float = 0.0
    device_ms: float | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


class ExecutionTrace:
    """Record sequential runner stages and aggregate them by component."""

    def __init__(
        self,
        *,
        device: Any = None,
        started_ns: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.device = device
        self.enabled = bool(enabled)
        self.started_ns = time.perf_counter_ns() if started_ns is None else int(started_ns)
        self._spans: list[SpanMeasurement] = []
        self._counters: dict[str, Any] = {}

    @property
    def spans(self) -> tuple[SpanMeasurement, ...]:
        return tuple(self._spans)

    def counter(self, name: str, value: Any) -> None:
        """Record a JSON-compatible scalar or object for the trace summary."""

        if not self.enabled:
            return
        self._counters[str(name)] = value

    def _relative_ms(self, timestamp_ns: int) -> float:
        return max(0.0, (timestamp_ns - self.started_ns) / 1_000_000.0)

    def _cuda_context(self, enabled: bool) -> tuple[Any, Any] | None:
        if not enabled:
            return None
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        device = torch.device(self.device) if self.device is not None else torch.device(
            "cuda", torch.cuda.current_device()
        )
        if device.type != "cuda":
            return None
        return torch, device

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str,
        gpu: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanMeasurement]:
        """Measure one stage.

        GPU spans synchronize before and after the stage.  That makes their
        wall time user-visible and prevents earlier asynchronous work from
        being charged to the wrong component.  CPU spans do not synchronize.
        """

        measurement = SpanMeasurement(
            name=str(name),
            category=str(category),
            metadata=dict(metadata or {}),
        )
        if not self.enabled:
            yield measurement
            return

        cuda_context = self._cuda_context(gpu)
        started_ns = time.perf_counter_ns()
        if cuda_context is not None:
            torch, device = cuda_context
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(device))
        else:
            torch = device = start_event = end_event = None

        measurement = SpanMeasurement(
            name=str(name),
            category=str(category),
            start_ms=self._relative_ms(started_ns),
            metadata=dict(metadata or {}),
        )
        error: BaseException | None = None
        try:
            yield measurement
        except BaseException as caught:
            error = caught
            raise
        finally:
            if cuda_context is not None:
                assert torch is not None
                assert device is not None
                assert end_event is not None
                end_event.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                assert start_event is not None
                measurement.device_ms = float(start_event.elapsed_time(end_event))
            ended_ns = time.perf_counter_ns()
            measurement.end_ms = self._relative_ms(ended_ns)
            measurement.wall_ms = max(
                0.0,
                (ended_ns - started_ns) / 1_000_000.0,
            )
            if error is not None:
                measurement.error = f"{type(error).__name__}: {error}"
            self._spans.append(measurement)

    def to_dict(self) -> dict[str, Any]:
        """Return an inspectable trace with per-span and aggregate views."""

        grouped: dict[str, list[SpanMeasurement]] = defaultdict(list)
        for span in self._spans:
            grouped[span.name].append(span)

        span_wall_total = float(sum(span.wall_ms for span in self._spans))
        trace_start_ms = min((span.start_ms for span in self._spans), default=0.0)
        trace_end_ms = max((span.end_ms for span in self._spans), default=trace_start_ms)
        trace_window_ms = max(0.0, trace_end_ms - trace_start_ms)
        components: dict[str, Any] = {}
        for name, spans in sorted(grouped.items()):
            wall_values = [span.wall_ms for span in spans]
            device_values = [
                span.device_ms for span in spans if span.device_ms is not None
            ]
            components[name] = {
                "category": spans[0].category,
                "count": len(spans),
                "wall_ms_total": float(sum(wall_values)),
                "wall_ms_mean": float(statistics.fmean(wall_values)),
                "wall_ms_min": float(min(wall_values)),
                "wall_ms_max": float(max(wall_values)),
                "device_ms_total": (
                    float(sum(device_values)) if device_values else None
                ),
                "device_ms_mean": (
                    float(statistics.fmean(device_values)) if device_values else None
                ),
                "host_overhead_ms_total": (
                    float(sum(wall_values) - sum(device_values))
                    if device_values
                    else float(sum(wall_values))
                ),
                "share_of_recorded_span_wall_percent": (
                    float(sum(wall_values) / span_wall_total * 100.0)
                    if span_wall_total > 0
                    else 0.0
                ),
            }

        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "clock": "perf_counter_ns_relative_to_benchmark_run",
            "gpu_span_clock": "synchronized_wall_and_cuda_event"
            if any(span.device_ms is not None for span in self._spans)
            else "not_used",
            "counters": dict(self._counters),
            "timing_totals": {
                "recorded_span_wall_ms": span_wall_total,
                "trace_window_ms": trace_window_ms,
                "unattributed_wall_ms": max(0.0, trace_window_ms - span_wall_total),
            },
            "components": components,
            "spans": [
                {
                    "name": span.name,
                    "category": span.category,
                    "start_ms": span.start_ms,
                    "end_ms": span.end_ms,
                    "wall_ms": span.wall_ms,
                    "device_ms": span.device_ms,
                    "metadata": dict(span.metadata or {}),
                    "error": span.error,
                }
                for span in self._spans
            ],
        }


__all__ = ["ExecutionTrace", "SpanMeasurement"]
