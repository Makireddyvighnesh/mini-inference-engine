"""Metric helpers shared by the benchmark runner and unit tests."""

from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Return the nearest-rank percentile (for example, P95)."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    rank = max(1, math.ceil(percentile_value / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize measurements with median and tail percentiles."""
    observed = sorted(float(value) for value in values)
    if not observed:
        return {"count": 0}
    return {
        "count": len(observed),
        "mean": sum(observed) / len(observed),
        "p50": percentile(observed, 50),
        "p90": percentile(observed, 90),
        "p95": percentile(observed, 95),
        "p99": percentile(observed, 99),
        "min": observed[0],
        "max": observed[-1],
    }
