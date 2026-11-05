"""Structural rules for the codebase, so the layering doesn't erode as it grows."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
    return out


def _sources(*dirs: str) -> list[Path]:
    return [p for d in dirs for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]


def test_the_library_never_imports_entry_points():
    for path in _sources("brink"):
        for module, _ in _imports(path):
            assert not module.startswith(("experiments", "scripts", "app")), f"{path.relative_to(ROOT)}: {module}"


def test_entry_points_share_code_through_the_library_not_each_other():
    """Shared plumbing (loaders, thresholds, evaluators) lives in brink/, not in another experiment script."""
    for path in _sources("experiments"):
        for module, _ in _imports(path):
            assert not module.startswith("experiments"), f"{path.relative_to(ROOT)} imports {module}"


def test_no_private_names_imported_across_modules():
    for path in _sources("brink", "experiments", "scripts", "app"):
        for module, names in _imports(path):
            if module.startswith("brink"):
                private = [n for n in names if n.startswith("_")]
                assert not private, f"{path.relative_to(ROOT)} imports private {private} from {module}"
