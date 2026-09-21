"""Model-specific execution paths used by the inference engine."""

from .qwen3_packed import PackedPrefillOutput, qwen3_packed_prefill

__all__ = ["PackedPrefillOutput", "qwen3_packed_prefill"]
