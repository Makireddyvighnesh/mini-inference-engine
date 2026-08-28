from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import RequestSpec, WorkloadSpec


@dataclass(frozen=True)
class PromptBucket:
    """A named prompt/output shape used in a reproducible benchmark."""

    name: str
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("bucket name must not be empty")
        if self.prompt_tokens <= 0:
            raise ValueError("bucket prompt_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError("bucket output_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


DEFAULT_BUCKETS: tuple[PromptBucket, ...] = (
    PromptBucket("short", prompt_tokens=128, output_tokens=32),
    PromptBucket("medium", prompt_tokens=512, output_tokens=64),
    PromptBucket("long", prompt_tokens=2048, output_tokens=128),
)


def _bucket_by_name(bucket: str | PromptBucket) -> PromptBucket:
    if isinstance(bucket, PromptBucket):
        return bucket
    for candidate in DEFAULT_BUCKETS:
        if candidate.name == bucket:
            return candidate
    allowed = ", ".join(candidate.name for candidate in DEFAULT_BUCKETS)
    raise ValueError(f"Unknown prompt bucket {bucket!r}; expected one of {allowed}")


def _token_ids(*, length: int, seed: int, vocab_size: int) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    generator = random.Random(seed)
    return tuple(generator.randrange(1, vocab_size) for _ in range(length))


def _arrival_offsets(
    *,
    count: int,
    pattern: str,
    arrival_rate_per_second: float | None,
    generator: random.Random,
) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if pattern == "closed_loop":
        return [0.0] * count
    if arrival_rate_per_second is None or arrival_rate_per_second <= 0:
        raise ValueError(
            "arrival_rate_per_second must be positive for timed arrival patterns"
        )
    interval_ms = 1000.0 / arrival_rate_per_second
    if pattern == "fixed_rate":
        return [index * interval_ms for index in range(count)]
    if pattern == "poisson":
        offsets: list[float] = []
        elapsed_ms = 0.0
        for _ in range(count):
            # random.expovariate accepts a rate in events per unit; use ms.
            elapsed_ms += generator.expovariate(arrival_rate_per_second / 1000.0)
            offsets.append(elapsed_ms)
        return offsets
    raise ValueError(f"Unsupported arrival pattern: {pattern}")


def _workload_metadata(
    *,
    generator_seed: int,
    buckets: Iterable[PromptBucket],
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "generator": "minillm_l4.benchmarks.core.workloads",
        "generator_version": 1,
        "generator_seed": generator_seed,
        "vocab_size": vocab_size,
        "buckets": [bucket.to_dict() for bucket in buckets],
    }


def build_fixed_workload(
    bucket: str | PromptBucket,
    *,
    count: int = 8,
    seed: int = 17,
    arrival_pattern: str = "closed_loop",
    arrival_rate_per_second: float | None = None,
    model_id: str = "synthetic-fixture",
    model_revision: str = "fixture-v1",
    dtype: str = "fp32",
    device: str = "cpu",
    vocab_size: int = 32_000,
    request_prefix: str | None = None,
) -> WorkloadSpec:
    """Build a fixed-shape workload with deterministic token IDs."""

    selected_bucket = _bucket_by_name(bucket)
    if count <= 0:
        raise ValueError("count must be positive")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    generator = random.Random(seed)
    offsets = _arrival_offsets(
        count=count,
        pattern=arrival_pattern,
        arrival_rate_per_second=arrival_rate_per_second,
        generator=generator,
    )
    prefix = request_prefix or selected_bucket.name
    requests = tuple(
        RequestSpec(
            request_id=f"{prefix}-{index:04d}",
            prompt_token_ids=_token_ids(
                length=selected_bucket.prompt_tokens,
                seed=generator.randrange(2**63),
                vocab_size=vocab_size,
            ),
            max_new_tokens=selected_bucket.output_tokens,
            scheduled_arrival_ms=offsets[index],
            category=selected_bucket.name,
            metadata={
                "bucket": selected_bucket.name,
                "prompt_tokens": selected_bucket.prompt_tokens,
                "output_tokens": selected_bucket.output_tokens,
            },
        )
        for index in range(count)
    )
    return WorkloadSpec(
        name=f"fixed_{selected_bucket.name}",
        seed=seed,
        requests=requests,
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        device=device,
        arrival_pattern=arrival_pattern,
        metadata=_workload_metadata(
            generator_seed=seed,
            buckets=(selected_bucket,),
            vocab_size=vocab_size,
        ),
    )


def build_mixed_workload(
    *,
    count: int = 12,
    seed: int = 17,
    buckets: tuple[PromptBucket, ...] = DEFAULT_BUCKETS,
    weights: tuple[float, ...] | None = None,
    arrival_pattern: str = "closed_loop",
    arrival_rate_per_second: float | None = None,
    model_id: str = "synthetic-fixture",
    model_revision: str = "fixture-v1",
    dtype: str = "fp32",
    device: str = "cpu",
    vocab_size: int = 32_000,
    request_prefix: str = "mixed",
) -> WorkloadSpec:
    """Build a deterministic mixed-length workload.

    When there are at least as many requests as buckets, one request from each
    bucket is placed first. This makes the Phase 0 fixture reliably exercise
    short, medium, and long shapes while the remaining requests are sampled
    using the supplied weights.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if not buckets:
        raise ValueError("buckets must not be empty")
    if len({bucket.name for bucket in buckets}) != len(buckets):
        raise ValueError("bucket names must be unique")
    if weights is None:
        normalized_weights = tuple(1.0 for _ in buckets)
    else:
        if len(weights) != len(buckets):
            raise ValueError("weights must have the same length as buckets")
        normalized_weights = tuple(float(weight) for weight in weights)
    if any(weight < 0 or not math.isfinite(weight) for weight in normalized_weights):
        raise ValueError("weights must be finite and non-negative")
    if sum(normalized_weights) <= 0:
        raise ValueError("at least one weight must be positive")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    generator = random.Random(seed)
    selected: list[PromptBucket] = list(buckets[:count])
    while len(selected) < count:
        selected.append(
            generator.choices(buckets, weights=normalized_weights, k=1)[0]
        )

    offsets = _arrival_offsets(
        count=count,
        pattern=arrival_pattern,
        arrival_rate_per_second=arrival_rate_per_second,
        generator=generator,
    )
    requests = tuple(
        RequestSpec(
            request_id=f"{request_prefix}-{index:04d}",
            prompt_token_ids=_token_ids(
                length=selected[index].prompt_tokens,
                seed=generator.randrange(2**63),
                vocab_size=vocab_size,
            ),
            max_new_tokens=selected[index].output_tokens,
            scheduled_arrival_ms=offsets[index],
            category=selected[index].name,
            metadata={
                "bucket": selected[index].name,
                "prompt_tokens": selected[index].prompt_tokens,
                "output_tokens": selected[index].output_tokens,
            },
        )
        for index in range(count)
    )
    return WorkloadSpec(
        name="mixed_lengths",
        seed=seed,
        requests=requests,
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        device=device,
        arrival_pattern=arrival_pattern,
        metadata={
            **_workload_metadata(
                generator_seed=seed,
                buckets=buckets,
                vocab_size=vocab_size,
            ),
            "weights": list(normalized_weights),
        },
    )


def build_fixture_workloads(
    *,
    seed: int = 17,
    fixed_count: int = 4,
    mixed_count: int = 12,
    arrival_pattern: str = "closed_loop",
    arrival_rate_per_second: float | None = None,
    model_id: str = "synthetic-fixture",
    model_revision: str = "fixture-v1",
    dtype: str = "fp32",
    device: str = "cpu",
    vocab_size: int = 32_000,
) -> dict[str, WorkloadSpec]:
    """Return the canonical short/medium/long/mixed fixture workloads."""

    workloads = {
        bucket.name: build_fixed_workload(
            bucket,
            count=fixed_count,
            seed=seed,
            arrival_pattern=arrival_pattern,
            arrival_rate_per_second=arrival_rate_per_second,
            model_id=model_id,
            model_revision=model_revision,
            dtype=dtype,
            device=device,
            vocab_size=vocab_size,
        )
        for bucket in DEFAULT_BUCKETS
    }
    workloads["mixed"] = build_mixed_workload(
        count=mixed_count,
        seed=seed,
        arrival_pattern=arrival_pattern,
        arrival_rate_per_second=arrival_rate_per_second,
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        device=device,
        vocab_size=vocab_size,
    )
    return workloads


def save_workload(workload: WorkloadSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_workload(path: Path) -> WorkloadSpec:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Workload file must contain a JSON object: {path}")
    return WorkloadSpec.from_dict(payload)
