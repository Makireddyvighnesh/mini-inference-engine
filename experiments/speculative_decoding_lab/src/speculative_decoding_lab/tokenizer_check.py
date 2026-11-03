"""Strictly compare tokenizer metadata embedded in two GGUF model files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TOKENIZER_FIELDS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.remove_extra_whitespaces",
    "tokenizer.ggml.precompiled_charsmap",
)

REQUIRED_TOKENIZER_FIELDS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.eos_token_id",
)


def _normalize(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def _field_value(field: Any) -> Any:
    return _normalize(field.contents())


def tokenizer_fingerprint(model_path: Path) -> tuple[dict[str, Any], str]:
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(
            "GGUF tokenizer inspection needs the optional dependency: pip install '.[dev]'"
        ) from error

    reader = GGUFReader(str(model_path))
    values: dict[str, Any] = {}
    missing: list[str] = []
    for key in TOKENIZER_FIELDS:
        field = reader.fields.get(key)
        if field is None:
            missing.append(key)
        else:
            values[key] = _field_value(field)
    values["__missing_fields__"] = missing
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return values, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compare_tokenizers(teacher_path: Path, draft_path: Path) -> list[str]:
    teacher, _ = tokenizer_fingerprint(teacher_path)
    draft, _ = tokenizer_fingerprint(draft_path)

    def valid_required_field(values: dict[str, Any], key: str) -> bool:
        value = values.get(key)
        if key == "tokenizer.ggml.eos_token_id":
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0
        return bool(value)

    return [
        key
        for key in TOKENIZER_FIELDS
        if teacher.get(key) != draft.get(key)
        or (
            key in REQUIRED_TOKENIZER_FIELDS
            and (not valid_required_field(teacher, key) or not valid_required_field(draft, key))
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("draft", type=Path)
    args = parser.parse_args()

    for model_path in (args.teacher, args.draft):
        if not model_path.is_file():
            parser.error(f"GGUF file does not exist: {model_path}")

    _, teacher_hash = tokenizer_fingerprint(args.teacher)
    _, draft_hash = tokenizer_fingerprint(args.draft)
    mismatches = compare_tokenizers(args.teacher, args.draft)
    print(f"teacher tokenizer SHA256: {teacher_hash}")
    print(f"draft tokenizer SHA256:   {draft_hash}")
    if mismatches:
        print("INCOMPATIBLE tokenizer metadata fields:")
        for key in mismatches:
            print(f"  - {key}")
        raise SystemExit(1)
    print("PASS: tokenizer metadata and token-ID ordering match exactly")


if __name__ == "__main__":
    main()
