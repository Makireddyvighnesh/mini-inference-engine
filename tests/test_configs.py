from pathlib import Path

from minillm_l4.configs import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_workload_configs_are_yaml_and_validate() -> None:
    configurations = (
        ("minillm_l4_harness.yaml", 0),
        ("qwen3_fp8_baseline.yaml", 1),
        ("qwen3_fp8_manual.yaml", 2),
        ("qwen3_fp8_kv_cache.yaml", 3),
        ("qwen3_fp8_concurrent.yaml", 4),
        ("qwen3_fp8_continuous.yaml", 5),
        ("qwen3_fp8_paged.yaml", 6),
    )
    for filename, expected_phase in configurations:
        path = PROJECT_ROOT / "configs" / "workloads" / filename
        payload = load_yaml_config(path, expected_phase=expected_phase)
        assert payload["project"] == "MiniLLM-L4"
        assert payload["phase"] == expected_phase


def test_kv_cache_config_selects_both_reference_paths() -> None:
    payload = load_yaml_config(
        PROJECT_ROOT / "configs/workloads/qwen3_fp8_kv_cache.yaml",
        expected_phase=3,
    )

    assert payload["cache"]["modes"] == ["contiguous", "recompute"]
    assert payload["cache"]["capacity"] == "request"


def test_continuous_config_selects_iteration_scheduler() -> None:
    payload = load_yaml_config(
        PROJECT_ROOT / "configs/workloads/qwen3_fp8_continuous.yaml",
        expected_phase=5,
    )

    assert payload["scheduler"]["policy"] == "continuous_in_flight_batching"
    assert payload["scheduler"]["max_prefill_tokens"] == 4096
    assert payload["scheduler"]["max_wait_ms"] == 2.0


def test_paged_config_selects_fixed_block_storage() -> None:
    payload = load_yaml_config(
        PROJECT_ROOT / "configs/workloads/qwen3_fp8_paged.yaml",
        expected_phase=6,
    )

    assert payload["cache"]["modes"] == ["contiguous", "paged_hybrid"]
    assert payload["cache"]["block_sizes"] == [8, 16, 32, 64]
    assert payload["cache"]["capacity_token_slots"] == 32768
    assert payload["cache"]["decode_backend"] == "auto"
    assert payload["model"]["fp8_kernel_path"] == "sm89"
