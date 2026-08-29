from pathlib import Path

from minillm_l4.configs import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_workload_configs_are_yaml_and_validate() -> None:
    configurations = (
        ("minillm_l4_harness.yaml", 0),
        ("qwen3_fp8_baseline.yaml", 1),
        ("qwen3_fp8_manual.yaml", 2),
        ("qwen3_fp8_kv_cache.yaml", 3),
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
