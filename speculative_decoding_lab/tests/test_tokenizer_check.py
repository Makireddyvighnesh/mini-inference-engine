from pathlib import Path

import speculative_decoding_lab.tokenizer_check as tokenizer_check


def test_required_tokenizer_fields_cannot_be_missing_from_both_models(monkeypatch):
    monkeypatch.setattr(tokenizer_check, "tokenizer_fingerprint", lambda path: ({}, "same"))

    mismatches = tokenizer_check.compare_tokenizers(Path("teacher.gguf"), Path("draft.gguf"))

    assert set(mismatches) == set(tokenizer_check.REQUIRED_TOKENIZER_FIELDS)


def test_optional_absent_fields_do_not_fail_identical_tokenizers(monkeypatch):
    fields = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 1,
    }
    monkeypatch.setattr(tokenizer_check, "tokenizer_fingerprint", lambda path: (fields, "same"))

    assert tokenizer_check.compare_tokenizers(Path("teacher.gguf"), Path("draft.gguf")) == []


def test_empty_required_token_list_fails(monkeypatch):
    fields = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": [],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 0,
    }
    monkeypatch.setattr(tokenizer_check, "tokenizer_fingerprint", lambda path: (fields, "same"))

    assert tokenizer_check.compare_tokenizers(Path("teacher.gguf"), Path("draft.gguf")) == [
        "tokenizer.ggml.tokens"
    ]


def test_different_token_order_fails(monkeypatch):
    fields = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 1,
    }

    def fingerprint(path):
        current = dict(fields)
        if path.name == "draft.gguf":
            current["tokenizer.ggml.tokens"] = ["b", "a"]
        return current, "unused"

    monkeypatch.setattr(tokenizer_check, "tokenizer_fingerprint", fingerprint)
    assert tokenizer_check.compare_tokenizers(Path("teacher.gguf"), Path("draft.gguf")) == [
        "tokenizer.ggml.tokens"
    ]
