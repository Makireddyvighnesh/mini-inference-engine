"""Minimal loading helpers for the pinned checkpoint-native FP8 model."""

from __future__ import annotations

import os
from typing import Any

import torch


def checkpoint_quantization_config(model_config: Any) -> dict[str, Any] | None:
    """Return a JSON-compatible checkpoint quantization configuration."""

    value = getattr(model_config, "quantization_config", None)
    if value is None and isinstance(model_config, dict):
        value = model_config.get("quantization_config")
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
    raise TypeError("Checkpoint quantization_config must be dictionary-like")


def _quant_method(model_config: Any) -> str | None:
    quantization = checkpoint_quantization_config(model_config)
    if quantization is None:
        return None
    method = quantization.get("quant_method")
    return None if method is None else str(method).lower()


def _validate_options(
    *,
    fp8_fallback_dtype: str,
    fp8_kernel_path: str,
) -> None:
    if fp8_fallback_dtype not in {"auto", "fp16", "bfloat16"}:
        raise ValueError(
            "fp8_fallback_dtype must be one of: auto, fp16, bfloat16"
        )
    if fp8_kernel_path not in {"auto", "triton", "sm89"}:
        raise ValueError("fp8_kernel_path must be one of: auto, triton, sm89")


def configure_kernel_strategy(
    precision: str,
    *,
    fp8_kernel_path: str = "auto",
) -> dict[str, Any]:
    """Configure the supported Transformers FP8 dispatch before import."""

    if precision != "fp8":
        raise ValueError("MiniLLM-L4 currently supports only native FP8 loading")
    _validate_options(
        fp8_fallback_dtype="auto",
        fp8_kernel_path=fp8_kernel_path,
    )
    if fp8_kernel_path in {"triton", "sm89"}:
        os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"

    capability = (
        list(torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None
    )
    deepgemm_eligible = capability is not None and capability[0] >= 9
    return {
        "fp8_kernel_path_requested": fp8_kernel_path,
        "gpu_compute_capability": capability,
        "fp8_deepgemm_eligible": deepgemm_eligible,
        "fp8_triton_forced": fp8_kernel_path == "triton",
        "fp8_sm89_custom": fp8_kernel_path == "sm89",
        "expected_kernel_path": (
            "minillm_sm89_triton_fp8"
            if fp8_kernel_path == "sm89"
            else "triton_finegrained_fp8"
            if fp8_kernel_path == "triton" or not deepgemm_eligible
            else "auto_deepgemm_or_triton"
        ),
        "environment_transformers_disable_deepgemm": os.environ.get(
            "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"
        ),
        "kernel_verification_note": (
            "Expected dispatch is a hypothesis; profiling must verify the "
            "kernel that actually executed."
        ),
    }


def model_load_kwargs(
    precision: str,
    *,
    model_config: Any,
    fp8_fallback_dtype: str = "auto",
) -> tuple[dict[str, Any], str]:
    """Return validated Transformers arguments for native FP8 loading."""

    if precision != "fp8":
        raise ValueError("MiniLLM-L4 currently supports only native FP8 loading")
    _validate_options(
        fp8_fallback_dtype=fp8_fallback_dtype,
        fp8_kernel_path="auto",
    )
    if _quant_method(model_config) != "fp8":
        raise RuntimeError(
            "FP8 requires a checkpoint whose config declares quant_method='fp8'"
        )
    fallback_dtype: str | torch.dtype = {
        "auto": "auto",
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[fp8_fallback_dtype]
    return (
        {
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "dtype": fallback_dtype,
        },
        "checkpoint-defined fine-grained FP8 with runtime-selected compute "
        f"and {fp8_fallback_dtype} fallback modules",
    )


def precision_execution_metadata(
    precision: str,
    *,
    model_config: Any,
    fp8_fallback_dtype: str = "auto",
    fp8_kernel_path: str = "auto",
) -> dict[str, Any]:
    """Describe the configured precision without claiming profiled execution."""

    if precision != "fp8":
        raise ValueError("MiniLLM-L4 currently supports only native FP8 loading")
    _validate_options(
        fp8_fallback_dtype=fp8_fallback_dtype,
        fp8_kernel_path=fp8_kernel_path,
    )
    quantization = checkpoint_quantization_config(model_config)
    return {
        "requested_precision": "fp8",
        "resolved_precision": "native_fp8",
        "checkpoint_quantization_config": quantization,
        "weight_format": "fp8_e4m3",
        "activation_format": (
            "dynamic_fp8"
            if quantization
            and quantization.get("activation_scheme") == "dynamic"
            else "checkpoint_defined_fp8"
        ),
        "compute_format": "checkpoint_and_runtime_selected",
        "quantization_backend": "transformers_fine_grained_fp8",
        "non_fp8_module_dtype": fp8_fallback_dtype,
        "native_tensor_core_execution_verified": False,
        "verification_note": (
            "Stored tensor formats do not prove which GPU instructions ran; "
            "kernel profiling is required."
        ),
    }
