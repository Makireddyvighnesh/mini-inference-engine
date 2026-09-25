"""Build or launch a single-mode llama.cpp server from the YAML experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_server_command(
    config: dict[str, Any],
    mode: str,
    port: int | None = None,
    validate_models: bool = True,
) -> tuple[list[str], dict[str, str]]:
    if mode not in {"baseline", "speculative"}:
        raise ValueError("mode must be 'baseline' or 'speculative'")

    root = project_root()
    runtime = config["runtime"]
    model_config = config["models"]
    binary_override = os.environ.get("SPECULATIVE_LLAMA_SERVER_BINARY")
    binary = resolve_path(root, binary_override or runtime["binary"])
    teacher = resolve_path(root, model_config["teacher"]["file"])
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server binary does not exist: {binary}")
    if validate_models and not teacher.is_file():
        raise FileNotFoundError(f"teacher GGUF does not exist: {teacher}")

    selected_port = int(port if port is not None else runtime["base_port"])
    command = [
        str(binary),
        "--model",
        str(teacher),
        "--host",
        str(runtime["host"]),
        "--port",
        str(selected_port),
        "--ctx-size",
        str(runtime["context_tokens"]),
        "--parallel",
        str(runtime["server_slots"]),
        "-ngl",
        str(runtime["gpu_layers"]),
        "--flash-attn",
        "on" if runtime["flash_attention"] else "off",
        "--jinja",
        "--no-webui",
        "--no-cache-prompt",
    ]

    if mode == "speculative":
        draft = resolve_path(root, model_config["draft"]["file"])
        if validate_models and not draft.is_file():
            raise FileNotFoundError(
                f"draft GGUF does not exist: {draft}\n"
                "Download the configured draft GGUF; see README.md."
            )
        command.extend(
            [
                "--model-draft",
                str(draft),
                "--spec-type",
                str(config["speculative"]["type"]),
                "--spec-draft-n-max",
                str(config["speculative"]["draft_tokens"]),
                "-ngld",
                str(runtime["draft_gpu_layers"]),
            ]
        )

    env = os.environ.copy()
    library_override = os.environ.get("SPECULATIVE_LLAMA_SERVER_LIBRARY_DIR")
    library_dir = (
        resolve_path(root, library_override)
        if library_override
        else binary.parent if binary_override else resolve_path(root, runtime["library_dir"])
    )
    prior_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(library_dir) + (
        f":{prior_library_path}" if prior_library_path else ""
    )
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--mode", choices=("baseline", "speculative"), required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else project_root() / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command, env = build_server_command(config, args.mode, args.port)
    if args.print_command:
        print(" ".join(command))
        return
    raise SystemExit(subprocess.run(command, cwd=project_root(), env=env).returncode)


if __name__ == "__main__":
    main()
