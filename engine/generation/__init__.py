"""Generation backends used by the MiniLLM-L4 engine."""

from .huggingface import transformers_greedy_generate
from .manual import ManualGenerationResult, manual_greedy_generate

__all__ = [
    "ManualGenerationResult",
    "manual_greedy_generate",
    "transformers_greedy_generate",
]
