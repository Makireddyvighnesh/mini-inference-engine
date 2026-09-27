import argparse

import pytest

from minillm_l4.benchmarks.commands.run_profile_decode import (
    attribute_gpu_work,
    merged_duration_ns,
    parse_shape,
)


def test_merged_duration_counts_overlapping_work_once() -> None:
    assert merged_duration_ns([]) == 0
    assert merged_duration_ns([(0, 10), (5, 15), (20, 30)]) == 25
    assert merged_duration_ns([(20, 30), (0, 10), (10, 12)]) == 22


def test_gpu_work_is_attributed_to_the_step_that_contains_it() -> None:
    steps = [(0, 100), (100, 200)]
    gpu = [(10, 20), (15, 30), (120, 140), (190, 210)]

    per_step = attribute_gpu_work(steps, gpu)

    assert per_step[0] == {"wall_ms": 1e-4, "gpu_busy_ms": 2e-5, "gpu_operations": 2.0}
    # (190, 210) spills past the window and must not be counted.
    assert per_step[1]["gpu_busy_ms"] == pytest.approx(2e-5)
    assert per_step[1]["gpu_operations"] == 1.0


def test_parse_shape() -> None:
    assert parse_shape("2048x4") == (2048, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("2048")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("0x1")
