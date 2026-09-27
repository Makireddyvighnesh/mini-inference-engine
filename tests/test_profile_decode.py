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


def test_every_shape_is_timed_before_any_profiler_session() -> None:
    from minillm_l4.benchmarks.commands.run_profile_decode import measure_shapes

    calls: list[tuple[str, int, int]] = []

    def time_fn(prompt: int, batch: int) -> list[float]:
        calls.append(("time", prompt, batch))
        return [10.0, 12.0, 11.0]

    def profile_fn(prompt: int, batch: int) -> list[dict[str, float]]:
        calls.append(("profile", prompt, batch))
        return [
            {"wall_ms": 20.0, "gpu_busy_ms": 3.0, "gpu_operations": 7.0},
            {"wall_ms": 22.0, "gpu_busy_ms": 5.0, "gpu_operations": 7.0},
        ]

    rows = measure_shapes([(128, 1), (2048, 4)], time_fn=time_fn, profile_fn=profile_fn)

    assert calls == [
        ("time", 128, 1),
        ("time", 2048, 4),
        ("profile", 128, 1),
        ("profile", 2048, 4),
    ]
    assert rows[1]["prompt_tokens"] == 2048 and rows[1]["batch_size"] == 4
    assert rows[0]["step_ms_p50"] == 11.0
    assert rows[0]["gpu_busy_ms_p50"] == 4.0
    assert rows[0]["gpu_busy_fraction"] == 4.0 / 11.0
    assert rows[0]["profiled_step_ms_p50"] == 21.0
