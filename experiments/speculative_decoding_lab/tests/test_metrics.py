import pytest

from speculative_decoding_lab.metrics import percentile, summarize


def test_nearest_rank_percentile():
    assert percentile([9, 1, 4, 2, 3, 8, 5, 6, 7, 10], 50) == 5
    assert percentile([9, 1, 4, 2, 3, 8, 5, 6, 7, 10], 95) == 10


def test_percentile_requires_valid_input():
    with pytest.raises(ValueError):
        percentile([], 95)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_summary_includes_tail_percentiles():
    result = summarize([10, 20, 30, 40])
    assert result["count"] == 4
    assert result["p50"] == 20
    assert result["p95"] == 40
