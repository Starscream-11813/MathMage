"""The answer vault: reference answers and solutions.

Only the grader (:mod:`mathmage.grading.verify`), analysis scripts, offline learning on
training items, and the mock model may import this module. The team pipeline, prompts,
and LLM client never do; tests/test_isolation.py enforces this statically.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from mathmage import paths

ANSWERS_FILE = paths.REGISTRY / "answers.parquet"


@dataclass(frozen=True)
class VaultRecord:
    item_id: str
    answer: str
    answer_type: str
    answer_format: str
    solution: str


class AnswerVault:
    def __init__(self, df: pd.DataFrame):
        self._rec = {r["item_id"]: VaultRecord(**r) for r in df.to_dict("records")}

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._rec

    def __getitem__(self, item_id: str) -> VaultRecord:
        return self._rec[item_id]

    def __len__(self) -> int:
        return len(self._rec)


@lru_cache(maxsize=2)
def load_vault(path: str = str(ANSWERS_FILE)) -> AnswerVault:
    return AnswerVault(pd.read_parquet(path))


def save_vault(records: list[VaultRecord], path=ANSWERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in records]).to_parquet(path, index=False)
