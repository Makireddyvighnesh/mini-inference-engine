import io
import json

import speculative_decoding_lab.benchmark as benchmark
from speculative_decoding_lab.benchmark import (
    _completion_text,
    _correctness,
    _record_text_piece,
    _stream_chat,
    _summarize_mode,
)


def test_empty_terminal_content_preserves_streamed_text():
    parts = ["Hello", " world"]

    _record_text_piece(parts, "", terminal=True)

    assert "".join(parts) == "Hello world"


def test_final_nonempty_content_replaces_streamed_chunks():
    parts = ["Hello", " ", "world"]

    _record_text_piece(parts, "Hello world!", terminal=True)

    assert parts == ["Hello world!"]


def test_missing_stream_text_is_detokenized(monkeypatch):
    calls = []

    def fake_http_json(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"content": "decoded answer"}

    monkeypatch.setattr(benchmark, "_http_json", fake_http_json)

    assert _completion_text("http://127.0.0.1:18180", [17, 23], []) == "decoded answer"
    assert calls == [
        ("http://127.0.0.1:18180/detokenize", {"tokens": [17, 23]}, 60)
    ]


def test_e2e_includes_detokenization_after_stream(monkeypatch):
    calls = []

    def fake_http_json(url, payload, timeout):
        calls.append(url)
        if url.endswith("/apply-template"):
            return {"prompt": "rendered prompt"}
        if url.endswith("/detokenize"):
            return {"content": "decoded answer"}
        raise AssertionError(f"unexpected request: {url}")

    events = [
        {"tokens": [17], "content": "", "stop": False},
        {
            "tokens": [17],
            "content": "",
            "stop": True,
            "stop_type": "limit",
            "timings": {
                "predicted_n": 1,
                "predicted_ms": 0.0,
                "predicted_per_second": 0.0,
                "prompt_per_second": 500.0,
            },
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    ticks = iter([10.0, 11.0, 12.0, 13.0, 14.0, 16.0])
    monkeypatch.setattr(benchmark, "_http_json", fake_http_json)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(stream))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    row = _stream_chat("http://127.0.0.1:18180", "question", "p", 1, 0.0, 1, ["temperature"])

    assert row["text"] == "decoded answer"
    assert row["ttft_ms"] == 3000.0
    assert row["stream_e2e_ms"] == 4000.0
    assert row["postprocessing_ms"] == 2000.0
    assert row["client_e2e_ms"] == 6000.0
    assert calls == ["http://127.0.0.1:18180/apply-template", "http://127.0.0.1:18180/detokenize"]


def test_summary_reports_decode_metrics_and_acceptance():
    row = {
        "prompt_id": "sample",
        "repetition": 0,
        "text": "same generated text",
        "ttft_ms": 80.0,
        "client_e2e_ms": 1000.0,
        "stream_e2e_ms": 950.0,
        "postprocessing_ms": 50.0,
        "mode_measurement_duration_ms": 5000.0,
        "server_timings": {
            "predicted_n": 10,
            "predicted_ms": 900.0,
            "predicted_per_second": 10.0 / 0.9,
            "prompt_per_second": 500.0,
            "draft_n": 20,
            "draft_n_accepted": 10,
        },
    }

    summary = _summarize_mode([row])

    assert summary["ttft_ms"]["p50"] == 80.0
    assert summary["stream_e2e_ms"]["p50"] == 950.0
    assert summary["postprocessing_ms"]["p50"] == 50.0
    assert summary["decode_tpot_ms"]["p50"] == 100.0
    assert summary["draft_acceptance_rate"] == 0.5
    assert summary["measured_aggregate_output_tokens_per_second"] == 2.0


def test_correctness_requires_identical_tokens_text_and_stop_reason():
    baseline = [
        {
            "prompt_id": "p",
            "repetition": 0,
            "text": "answer",
            "token_ids": [42],
            "stop_reason": "limit",
        }
    ]
    speculative = [dict(baseline[0])]
    assert _correctness(baseline, speculative)["passed"]

    speculative[0]["text"] = "different"
    assert not _correctness(baseline, speculative)["passed"]

    speculative[0]["text"] = "answer"
    speculative[0]["token_ids"] = [43]
    assert not _correctness(baseline, speculative)["passed"]

    speculative[0]["token_ids"] = [42]
    speculative[0]["stop_reason"] = "eos"
    assert not _correctness(baseline, speculative)["passed"]

    speculative[0]["stop_reason"] = "limit"
    speculative[0].pop("token_ids")
    assert not _correctness(baseline, speculative)["passed"]


def test_correctness_rejects_two_empty_text_outputs():
    row = {
        "prompt_id": "p",
        "repetition": 0,
        "text": "",
        "token_ids": [42],
        "stop_reason": "limit",
    }
    result = _correctness([row], [dict(row)])
    assert result["exact_token_matches"] == 1
    assert result["exact_text_matches"] == 0
    assert not result["passed"]
    assert result["invalid_requests"] == [{"prompt_id": "p", "repetition": 0}]
