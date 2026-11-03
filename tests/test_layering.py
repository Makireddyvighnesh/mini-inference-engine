"""The engine must work without the benchmark code; only benchmarks may import the engine."""

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine"


def test_engine_never_imports_benchmarks():
    offenders = []
    for path in ENGINE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            offenders += [f"{path.name}: {name}" for name in names if "benchmarks" in name]
    assert offenders == []
