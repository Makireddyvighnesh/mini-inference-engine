"""Benchmark workload and harness settings. Request and event types live in `engine/events.py`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from minillm_l4.engine.events import (  # noqa: F401 - re-exported for benchmark code
    CANONICAL_EVENTS,
    EVENT_SCHEMA_VERSION,
    REQUEST_STATUSES,
    EventRecord,
    RequestOutcome,
    RequestSpec,
    _validate_json_object,
    _validate_nonempty_string,
)

SCHEMA_VERSION = 1
WORKLOAD_SCHEMA_VERSION = 1
ARRIVAL_PATTERNS = frozenset({"closed_loop", "fixed_rate", "poisson"})


@dataclass(frozen=True)
class WorkloadSpec:
    """A reproducible ordered request trace and its benchmark identity."""

    name: str
    seed: int
    requests: tuple[RequestSpec, ...]
    model_id: str = "synthetic-fixture"
    model_revision: str = "fixture-v1"
    dtype: str = "fp32"
    device: str = "cpu"
    arrival_pattern: str = "closed_loop"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.name, name="name")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not self.requests:
            raise ValueError("requests must not be empty")
        normalized_requests = tuple(self.requests)
        if any(not isinstance(request, RequestSpec) for request in normalized_requests):
            raise TypeError("requests must contain RequestSpec values")
        object.__setattr__(self, "requests", normalized_requests)
        identifiers = [request.request_id for request in normalized_requests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("request IDs must be unique within a workload")
        arrivals = [request.scheduled_arrival_ms for request in normalized_requests]
        if arrivals != sorted(arrivals):
            raise ValueError("requests must be ordered by scheduled_arrival_ms")
        for field_name in ("model_id", "model_revision", "dtype", "device"):
            _validate_nonempty_string(getattr(self, field_name), name=field_name)
        if self.arrival_pattern not in ARRIVAL_PATTERNS:
            allowed = ", ".join(sorted(ARRIVAL_PATTERNS))
            raise ValueError(
                f"arrival_pattern must be one of {allowed}; got {self.arrival_pattern!r}"
            )
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "request_count": len(self.requests),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dtype": self.dtype,
            "device": self.device,
            "arrival_pattern": self.arrival_pattern,
            "metadata": dict(self.metadata),
            "requests": [request.to_dict() for request in self.requests],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkloadSpec":
        version = int(payload.get("schema_version", WORKLOAD_SCHEMA_VERSION))
        if version != WORKLOAD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported workload schema version: {version}")
        requests = tuple(
            RequestSpec.from_dict(request_payload)
            for request_payload in payload["requests"]
        )
        return cls(
            name=str(payload["name"]),
            seed=int(payload["seed"]),
            requests=requests,
            model_id=str(payload.get("model_id", "synthetic-fixture")),
            model_revision=str(payload.get("model_revision", "fixture-v1")),
            dtype=str(payload.get("dtype", "fp32")),
            device=str(payload.get("device", "cpu")),
            arrival_pattern=str(payload.get("arrival_pattern", "closed_loop")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class HarnessConfig:
    """Run controls recorded with every benchmark result."""

    warmup_repetitions: int = 1
    repetitions: int = 3
    respect_arrival_schedule: bool = False
    sample_interval_seconds: float = 0.25
    collect_gpu: bool = True
    collect_system_telemetry: bool = True
    runner_name: str = "custom"
    timing_mode: str = "auto"
    seed: int = 17
    timer_overhead_iterations: int = 1_000

    def __post_init__(self) -> None:
        if self.warmup_repetitions < 0:
            raise ValueError("warmup_repetitions must be non-negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.sample_interval_seconds <= 0 or not math.isfinite(
            self.sample_interval_seconds
        ):
            raise ValueError("sample_interval_seconds must be finite and positive")
        if self.timer_overhead_iterations <= 0:
            raise ValueError("timer_overhead_iterations must be positive")
        _validate_nonempty_string(self.runner_name, name="runner_name")
        if self.timing_mode not in {"auto", "wall", "cuda"}:
            raise ValueError("timing_mode must be auto, wall, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_repetitions": self.warmup_repetitions,
            "repetitions": self.repetitions,
            "respect_arrival_schedule": self.respect_arrival_schedule,
            "sample_interval_seconds": self.sample_interval_seconds,
            "collect_gpu": self.collect_gpu,
            "collect_system_telemetry": self.collect_system_telemetry,
            "runner_name": self.runner_name,
            "timing_mode": self.timing_mode,
            "seed": self.seed,
            "timer_overhead_iterations": self.timer_overhead_iterations,
        }
