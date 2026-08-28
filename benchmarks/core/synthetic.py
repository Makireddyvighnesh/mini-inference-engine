"""Exact-token materialization for deterministic synthetic prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    category: str
    target_prompt_tokens: int
    target_output_tokens: int
    seed_text: str


def read_synthetic_samples(path: Path) -> list[SyntheticSample]:
    samples: list[SyntheticSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                sample = SyntheticSample(
                    sample_id=str(payload["sample_id"]),
                    category=str(payload["category"]),
                    target_prompt_tokens=int(payload["target_prompt_tokens"]),
                    target_output_tokens=int(payload["target_output_tokens"]),
                    seed_text=str(payload["seed_text"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid sample on line {line_number}: {error}"
                ) from error
            if sample.target_prompt_tokens <= 0:
                raise ValueError(
                    f"{sample.sample_id}: target_prompt_tokens must be positive"
                )
            if sample.target_output_tokens <= 0:
                raise ValueError(
                    f"{sample.sample_id}: target_output_tokens must be positive"
                )
            if not sample.seed_text.strip():
                raise ValueError(f"{sample.sample_id}: seed_text is empty")
            samples.append(sample)

    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Synthetic sample IDs must be unique")
    return samples


def samples_for_length(
    samples: Iterable[SyntheticSample],
    prompt_tokens: int,
) -> list[SyntheticSample]:
    return [
        sample
        for sample in samples
        if sample.target_prompt_tokens == prompt_tokens
    ]


def exact_token_ids(tokenizer: Any, sample: SyntheticSample) -> list[int]:
    """Expand seed text deterministically and truncate to an exact length."""

    sections: list[str] = []
    section_index = 1
    token_ids: list[int] = []
    while len(token_ids) < sample.target_prompt_tokens:
        sections.append(
            f"Section {section_index}. {sample.seed_text} "
            "Continue the analysis using concrete evidence, explicit "
            "assumptions, and a concise conclusion for case "
            f"{sample.sample_id}-{section_index}."
        )
        token_ids = tokenizer.encode(
            "\n\n".join(sections),
            add_special_tokens=False,
        )
        section_index += 1
    return token_ids[: sample.target_prompt_tokens]
