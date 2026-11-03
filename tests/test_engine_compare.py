"""CPU checks for the four-engine comparison: round order, pooling, and the report."""

import json

from minillm_l4.benchmarks.commands import run_engine_compare as compare


def _run(ttft, tpot, tokens=(5, 6, 7), duration=1000.0, status="completed"):
    return {"duration_ms": duration, "requests": [{
        "id": "r0", "ttft_ms": ttft, "tpot_ms": tpot, "max_gap_ms": tpot, "tokens": len(tokens),
        "expected_tokens": 3, "token_ids": list(tokens), "status": status, "error": None}]}


def _entry(*runs):
    return {**compare.base.summarize(list(runs), expected=len(runs)), "runs": list(runs)}


def _write_session(root, round_index, engine, cases):
    directory = root / f"round{round_index}" / engine
    directory.mkdir(parents=True)
    (directory / f"{engine}.json").write_text(json.dumps({"engine": engine, "config": {"name": engine}, "cases": cases}))


def _workloads(root):
    (root / "workloads.json").write_text(json.dumps([{
        "section": "compare", "name": "p512_b1", "prompt_tokens": 512, "batch": 1, "output_tokens": 3,
        "label": "x", "requests": []}]))


def test_rounds_alternate_engine_order():
    assert compare.session_order(1) == compare.ENGINES
    assert compare.session_order(2) == tuple(reversed(compare.ENGINES))
    assert compare.session_order(3) == compare.ENGINES


def test_combine_rounds_pools_runs_and_keeps_round_values():
    combined = compare.combine_rounds([_entry(_run(10.0, 2.0)), _entry(_run(20.0, 4.0))])
    assert combined["ttft_p50_ms"] == 15.0
    assert combined["rounds"]["ttft_p50_ms"] == [10.0, 20.0]
    assert combined["e2e_ms"] == 1000.0
    assert compare.round_spread([10.0, 20.0]) == 10.0 / 15.0
    assert compare.round_spread([10.0]) is None


def test_report_tables_and_output_agreement(tmp_path):
    _workloads(tmp_path)
    _write_session(tmp_path, 1, "hf", {"p512_b1": _entry(_run(100.0, 50.0))})
    _write_session(tmp_path, 1, "minillm", {"p512_b1": _entry(_run(60.0, 20.0))})
    _write_session(tmp_path, 1, "sglang", {"p512_b1": _entry(_run(70.0, 25.0, tokens=(5, 9, 7)))})
    text, data = compare.build_report(tmp_path)
    assert "| Prompt tokens | HF generate | SGLang | MiniLLM (ours) |" in text
    assert "| 512 | 100.0 | 70.0 | 60.0 |" in text
    assert "| MiniLLM (ours) | 1/1 | - |" in text
    assert "| SGLang | 0/1 | 1 |" in text
    assert data["invalid"] == []


def test_report_marks_a_failed_case_invalid(tmp_path):
    _workloads(tmp_path)
    _write_session(tmp_path, 1, "hf", {"p512_b1": _entry(_run(100.0, 50.0, tokens=(5,), status="failed"))})
    text, data = compare.build_report(tmp_path)
    assert "(invalid)" in text
    assert len(data["invalid"]) == 1


def test_telemetry_summary_uses_busy_samples(tmp_path):
    path = tmp_path / "gpu.csv"
    path.write_text("2026/11/03 11:00:00.000, 40, 16.0, 210, 405, 100, 0, Not Active\n"
                    "2026/11/03 11:00:01.000, 60, 72.0, 1200, 6000, 9000, 95, Active\n"
                    "2026/11/03 11:00:02.000, 62, 70.0, 1300, 6000, 9500, 90, Not Active\n")
    summary = compare.telemetry_summary(path)
    assert summary["busy_samples"] == 2
    assert summary["power_w"] == 71.0
    assert summary["temp_max_c"] == 62.0
    assert summary["memory_peak_mib"] == 9500.0
    assert summary["power_capped_share"] == 0.5


def test_tensor_core_report_counts_token_differences():
    from minillm_l4.benchmarks.commands import run_tensor_core_decode as tc

    tokens = tc.compare_tokens([(1, 2, 3), (4, 5, 6)], [(1, 2, 3), (4, 9, 6)])
    assert tokens == {"requests": 2, "identical": 1, "earliest_difference": 1}
    report = tc.format_report([{"batch": 1, "prompt_tokens": 512, "exact_tpot_ms": 20.0, "tensor_core_tpot_ms": 19.0,
                                "tokens": tokens}])
    assert "1/2" in report and "+5.0%" in report and "earliest difference at output token 1" in report
