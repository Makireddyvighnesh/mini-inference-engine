from pathlib import Path

import speculative_decoding_lab.server as server


def _config() -> dict:
    return {
        "runtime": {
            "binary": "bin/llama-server",
            "library_dir": "bin",
            "host": "127.0.0.1",
            "base_port": 18180,
            "context_tokens": 8192,
            "server_slots": 1,
            "gpu_layers": 99,
            "draft_gpu_layers": 99,
            "flash_attention": True,
        },
        "models": {"teacher": {"file": "models/teacher.gguf"}, "draft": {"file": "models/draft.gguf"}},
        "speculative": {"type": "draft-simple", "draft_tokens": 4},
    }


def test_default_runtime_uses_configured_binary(monkeypatch, tmp_path: Path):
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setattr(server, "project_root", lambda: tmp_path)
    monkeypatch.delenv("SPECULATIVE_LLAMA_SERVER_BINARY", raising=False)
    monkeypatch.delenv("SPECULATIVE_LLAMA_SERVER_LIBRARY_DIR", raising=False)

    command, env = server.build_server_command(_config(), "speculative", validate_models=False)

    assert command[0] == str(binary)
    assert command[command.index("--model-draft") + 1] == str(tmp_path / "models" / "draft.gguf")
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(binary.parent)


def test_runtime_binary_can_be_overridden_for_other_checkouts(monkeypatch, tmp_path: Path):
    binary = tmp_path / "external" / "llama-server"
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setattr(server, "project_root", lambda: tmp_path)
    monkeypatch.setenv("SPECULATIVE_LLAMA_SERVER_BINARY", str(binary))
    monkeypatch.delenv("SPECULATIVE_LLAMA_SERVER_LIBRARY_DIR", raising=False)

    command, env = server.build_server_command(_config(), "baseline", validate_models=False)

    assert command[0] == str(binary)
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(binary.parent)

    library_dir = tmp_path / "custom-libs"
    monkeypatch.setenv("SPECULATIVE_LLAMA_SERVER_LIBRARY_DIR", str(library_dir))
    _, env = server.build_server_command(_config(), "baseline", validate_models=False)
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(library_dir)
