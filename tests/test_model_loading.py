import pytest

from minillm_l4.engine.model_loading import configure_kernel_strategy


def test_sm89_kernel_strategy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", raising=False)

    strategy = configure_kernel_strategy("fp8", fp8_kernel_path="sm89")

    assert strategy["fp8_sm89_custom"] is True
    assert strategy["expected_kernel_path"] == "minillm_sm89_triton_fp8"
    assert strategy["environment_transformers_disable_deepgemm"] == "1"


def test_unknown_fp8_kernel_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, triton, sm89"):
        configure_kernel_strategy("fp8", fp8_kernel_path="unknown")
