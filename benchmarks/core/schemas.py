from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = 1
WORKLOAD_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1

ARRIVAL_PATTERNS = frozenset({"closed_loop", "fixed_rate", "poisson"})
REQUEST_STATUSES = frozenset({"completed", "failed", "cancelled"})
CANONICAL_EVENTS = frozenset(
    {
        "arrival",
        "admission",
        "execution_start",
        "prefill_start",
        "prefill_end",
        "first_token_ready",
        "token_ready",
        "first_token_sent",
        "token_sent",
        "completion",
        "error",
        "cancelled",
    }
)


def _validate_json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = dict(value)
    try:
        json.dumps(normalized)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain JSON-compatible values") from error
    return normalized


def _validate_nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RequestSpec:
    """A deterministic request input independent of any model runtime."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    scheduled_arrival_ms: float = 0.0
    category: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.request_id, name="request_id")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        normalized_ids = tuple(int(token_id) for token_id in self.prompt_token_ids)
        if any(token_id < 0 or token_id > 2**32 - 1 for token_id in normalized_ids):
            raise ValueError("prompt_token_ids must contain unsigned 32-bit IDs")
        object.__setattr__(self, "prompt_token_ids", normalized_ids)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(float(self.scheduled_arrival_ms)):
            raise ValueError("scheduled_arrival_ms must be finite")
        if self.scheduled_arrival_ms < 0:
            raise ValueError("scheduled_arrival_ms must be non-negative")
        _validate_nonempty_string(self.category, name="category")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prompt_sha256(self) -> str:
        payload = b"".join(
            struct.pack("<I", token_id) for token_id in self.prompt_token_ids
        )
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_tokens": self.prompt_tokens,
            "prompt_sha256": self.prompt_sha256,
            "max_new_tokens": self.max_new_tokens,
            "scheduled_arrival_ms": float(self.scheduled_arrival_ms),
            "category": self.category,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequestSpec":
        return cls(
            request_id=str(payload["request_id"]),
            prompt_token_ids=tuple(int(value) for value in payload["prompt_token_ids"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            scheduled_arrival_ms=float(payload.get("scheduled_arrival_ms", 0.0)),
            category=str(payload.get("category", "default")),
            metadata=payload.get("metadata", {}),
        )


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
class EventRecord:
    """One request event with a timestamp relative to a benchmark run."""

    request_id: str
    event: str
    timestamp_ns: int
    sequence: int
    token_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.request_id, name="request_id")
        _validate_nonempty_string(self.event, name="event")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.token_index is not None and self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "request_id": self.request_id,
            "event": self.event,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_ms": self.timestamp_ns / 1_000_000.0,
            "sequence": self.sequence,
            "token_index": self.token_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RequestOutcome:
    """Normalized result returned by a model runner."""

    status: str = "completed"
    generated_token_ids: tuple[int, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in REQUEST_STATUSES:
            allowed = ", ".join(sorted(REQUEST_STATUSES))
            raise ValueError(f"status must be one of {allowed}")
        normalized_ids = tuple(int(token_id) for token_id in self.generated_token_ids)
        if any(token_id < 0 or token_id > 2**32 - 1 for token_id in normalized_ids):
            raise ValueError("generated_token_ids must contain unsigned 32-bit IDs")
        object.__setattr__(self, "generated_token_ids", normalized_ids)
        if self.status == "failed" and not self.error:
            raise ValueError("failed outcomes must include an error")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_tokens": len(self.generated_token_ids),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


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
