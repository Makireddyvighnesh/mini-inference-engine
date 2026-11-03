"""Shared YAML configuration loading and common validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(
    path: Path,
    *,
    expected_phase: int | None = None,
    label: str = "MiniLLM-L4",
) -> dict[str, Any]:
    """Load one project configuration from a YAML mapping."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} config must contain a YAML mapping")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported configuration schema version")
    if payload.get("project") != "MiniLLM-L4":
        raise ValueError(f"{label} config belongs to a different project")
    if expected_phase is not None and int(payload.get("phase", -1)) != expected_phase:
        raise ValueError(f"{label} config must set phase={expected_phase}")
    return payload
