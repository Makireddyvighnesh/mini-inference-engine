from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TimingMeasurement:
    wall_ms: float
    device_ms: float | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "device_ms": self.device_ms,
            "source": self.source,
        }


def _cuda_context(device: Any) -> tuple[Any, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if device is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())
    else:
        cuda_device = torch.device(device)
        if cuda_device.type != "cuda":
            return None
    return torch, cuda_device


def measure_call(
    function: Callable[[], Any],
    *,
    device: Any = None,
    timing_mode: str = "auto",
) -> tuple[Any, TimingMeasurement]:
    """Measure a call with synchronized wall time and optional CUDA events.

    ``device_ms`` measures work recorded on the current CUDA stream. The wall
    measurement includes the synchronization required to make the result
    observable, which is the appropriate boundary for a user-visible timing.
    """

    if timing_mode not in {"auto", "wall", "cuda"}:
        raise ValueError("timing_mode must be auto, wall, or cuda")
    cuda_context = _cuda_context(device)
    if timing_mode == "cuda" and cuda_context is None:
        raise RuntimeError("CUDA timing was requested but CUDA is unavailable")

    if cuda_context is None:
        started_ns = time.perf_counter_ns()
        result = function()
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return result, TimingMeasurement(
            wall_ms=elapsed_ms,
            device_ms=None,
            source="wall_clock",
        )

    torch, cuda_device = cuda_context
    if timing_mode == "wall":
        torch.cuda.synchronize(cuda_device)
        started_ns = time.perf_counter_ns()
        result = function()
        torch.cuda.synchronize(cuda_device)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return result, TimingMeasurement(
            wall_ms=elapsed_ms,
            device_ms=None,
            source="synchronized_wall_clock",
        )

    torch.cuda.synchronize(cuda_device)
    started_ns = time.perf_counter_ns()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = function()
    end_event.record()
    end_event.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return result, TimingMeasurement(
        wall_ms=elapsed_ms,
        device_ms=float(start_event.elapsed_time(end_event)),
        source="cuda_events",
    )


def measure_timer_overhead(*, iterations: int = 1_000) -> dict[str, Any]:
    """Measure the cost of one monotonic-clock timing pair in this process."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    samples_ns: list[int] = []
    for _ in range(iterations):
        started_ns = time.perf_counter_ns()
        ended_ns = time.perf_counter_ns()
        samples_ns.append(ended_ns - started_ns)
    ordered = sorted(samples_ns)

    def p(value: float) -> float:
        position = (len(ordered) - 1) * value / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)

    return {
        "iterations": iterations,
        "mean_ns": float(statistics.fmean(samples_ns)),
        "median_ns": float(statistics.median(samples_ns)),
        "p95_ns": p(95),
        "minimum_ns": float(ordered[0]),
        "maximum_ns": float(ordered[-1]),
    }
