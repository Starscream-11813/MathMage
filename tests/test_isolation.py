"""Enforce the answer-vault boundary in every agent-facing module."""

import ast
from pathlib import Path


def test_agent_modules_cannot_import_gold_sources():
    root = Path(__file__).resolve().parents[1]
    forbidden = {"mathmage.grading.vault", "mathmage.grading.verify", "mathmage.data.loaders",
                 "mathmage.analysis", "mathmage.experiments"}
    for directory in ["team", "scaffolds", "tools", "prompts", "llm"]:
        for file in (root / "mathmage" / directory).rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    assert not any(name == f or name.startswith(f + ".") for f in forbidden), (file, name)


def test_agent_view_has_no_answers_or_solutions():
    from mathmage.data.registry import Item, ItemView
    assert not any(w in ItemView.__dataclass_fields__ for w in ("answer", "solution", "correct"))
    item = Item("fixture/1", "aime_2024", "fixture", "Compute 2+2", "aime_integer", "algebra",
                "family-1", "test", "2024-02", "olympiad", "olympiad")
    assert set(item.view().__dict__) == {"item_id", "benchmark", "problem", "format_instruction", "kind", "answer_format"}
