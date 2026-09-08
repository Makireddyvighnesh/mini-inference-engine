import json
from pathlib import Path

import pytest

from minillm_l4.benchmarks.plots.concurrency_stress import load_stress_metrics


def _entry(batch_size: int, *, correctness: str = "pass") -> dict:
    def metric(p50: float, p95: float) -> dict:
        return {"p50": p50, "p95": p95, "p99": p95}

    return {
        "name": "mixed",
        "max_batch_size": batch_size,
        "correctness": {"status": correctness},
        "scheduler": {
            "padding_waste_ratio": 0.25,
            "batch_count": 3,
            "maximum_queue_depth": 4,
        },
        "summary": {
            "repetitions": 3,
            "metrics": {
                "ttft_ms": metric(100, 500),
                "e2e_latency_ms": metric(1000, 3000),
                "tpot_ms": metric(70, 80),
            },
            "tokens_per_second": metric(20, 20),
            "requests_per_second": metric(1, 1),
            "memory": {"peak_reserved_bytes": 2**30},
            "gpu_utilization_percent": metric(70, 90),
        },
    }


def test_load_stress_metrics_sorts_and_extracts_resource_metrics(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"workloads": [_entry(8), _entry(2), _entry(4)]}),
        encoding="utf-8",
    )

    rows = load_stress_metrics(path, workload_name="mixed")

    assert [row["max_batch_size"] for row in rows] == [2, 4, 8]
    assert rows[0]["peak_reserved_vram_gib"] == 1
    assert rows[0]["padding_waste_percent"] == 25
    assert rows[0]["gpu_utilization_p95"] == 90


def test_load_stress_metrics_rejects_failed_correctness(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"workloads": [_entry(2), _entry(4, correctness="fail")]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass correctness"):
        load_stress_metrics(path, workload_name="mixed")
