import json
from pathlib import Path

import pytest

from minillm_l4.benchmarks.plots.prefill_decode import (
    load_context_metrics,
    validate_context_comparison,
)


def _summary(prefill: float, tpot: float, decode: float) -> dict:
    def distribution(value: float) -> dict:
        return {"p50": value, "p95": value + 1, "p99": value + 2}

    return {
        "repetitions": 3,
        "metrics": {
            "prefill_ms": distribution(prefill),
            "tpot_ms": distribution(tpot),
            "decode_ms": distribution(decode),
            "ttft_ms": distribution(prefill),
        },
        "tokens_per_second": distribution(10.0),
    }


def test_load_context_metrics_filters_batch_and_sorts_context(tmp_path: Path) -> None:
    manifest = {
        "workloads": [
            {
                "name": "long",
                "prompt_tokens": 2048,
                "output_tokens": 32,
                "batch_size": 1,
                "summary": _summary(400, 72, 9000),
                "correctness": {"status": "pass"},
            },
            {
                "name": "short",
                "prompt_tokens": 128,
                "output_tokens": 32,
                "batch_size": 1,
                "summary": _summary(74, 70, 2200),
                "correctness": {"status": "pass"},
            },
            {
                "name": "ignored",
                "prompt_tokens": 128,
                "output_tokens": 32,
                "batch_size": 2,
                "summary": _summary(80, 71, 2300),
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = load_context_metrics(path, batch_size=1)

    assert [row["prompt_tokens"] for row in rows] == [128, 2048]
    assert rows[0]["prefill_p50_ms"] == 74
    assert rows[1]["decode_p95_ms"] == 9001
    assert validate_context_comparison(rows) == 32


def test_load_context_metrics_requires_multiple_contexts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "workloads": [
                    {
                        "name": "short",
                        "prompt_tokens": 128,
                        "output_tokens": 32,
                        "batch_size": 1,
                        "summary": _summary(74, 70, 2200),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two context sizes"):
        load_context_metrics(path, batch_size=1)


def test_context_comparison_rejects_variable_output_lengths() -> None:
    with pytest.raises(ValueError, match="one fixed output length"):
        validate_context_comparison(
            (
                {"output_tokens": 32, "correctness": "pass"},
                {"output_tokens": 64, "correctness": "pass"},
            )
        )
