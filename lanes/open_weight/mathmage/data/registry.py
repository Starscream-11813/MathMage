"""The item registry: problems, provenance, families, and splits (no answers)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from mathmage import paths

ITEMS_FILE = paths.REGISTRY / "items.parquet"

# Answer-format instructions shown to agents. They describe the required form of the
# answer, never its value.
FORMAT_INSTRUCTIONS = {
    "aime_integer": "The answer is an integer from 0 to 999 inclusive.",
    "exact": "Give the exact value in simplest form.",
    "tqa_float": "The answer is a single real number (give a decimal value).",
    "tqa_integer": "The answer is a single integer.",
    "tqa_list_float": "The answer is a list of real numbers; write it as [a, b, ...].",
    "tqa_list_integer": "The answer is a list of integers; write it as [a, b, ...].",
}

BENCHMARK_LABELS = {
    "aime_2023": "AIME 2023",
    "aime_2024": "AIME 2024",
    "aime_2025": "AIME 2025",
    "aime_2026": "AIME 2026",
    "hmmt_feb_2025": "HMMT Feb 2025",
    "hmmt_feb_2026": "HMMT Feb 2026",
    "tqa_phys": "TheoremQA-physics",
    "math500": "MATH-500",
}

SHORT_LABELS = {
    "aime_2024": "AIME24",
    "aime_2025": "AIME25",
    "aime_2026": "AIME26",
    "hmmt_feb_2026": "HMMT26",
    "tqa_phys": "TQA-phys",
    "math500": "MATH-500",
    "aime_2023": "AIME23",
    "hmmt_feb_2025": "HMMT25",
}


@dataclass(frozen=True)
class Item:
    item_id: str        # e.g. "aime_2025/16" (problem II-1), "hmmt_feb_2026/06"
    benchmark: str      # key of BENCHMARK_LABELS
    source: str         # dataset of origin
    problem: str
    answer_format: str  # key of FORMAT_INSTRUCTIONS
    domain: str         # problem type when the source provides one
    family_id: str      # near-duplicate family (used for splits)
    split: str          # train | dev | test
    item_date: str      # competition month, YYYY-MM
    band: str           # olympiad | physics | foundations
    kind: str           # token-budget class: olympiad | physics | math500
    extra: str = ""     # JSON with source-specific metadata (level, subject, ...)

    def view(self) -> "ItemView":
        return ItemView(self.item_id, self.benchmark, self.problem, FORMAT_INSTRUCTIONS[self.answer_format], self.kind,
                        self.answer_format)


@dataclass(frozen=True)
class ItemView:
    """Everything an agent may see about an item."""

    item_id: str
    benchmark: str
    problem: str
    format_instruction: str
    kind: str
    answer_format: str = "exact"


class Registry:
    def __init__(self, items: Iterable[Item]):
        self._items = {it.item_id: it for it in items}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item_id: str) -> Item:
        return self._items[item_id]

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def all(self) -> list[Item]:
        return list(self._items.values())

    def select(self, benchmarks: Optional[Iterable[str]] = None, split: Optional[str] = None,
               limit_per_benchmark: Optional[int] = None) -> list[Item]:
        bset = set(benchmarks) if benchmarks is not None else None
        out = [it for it in self._items.values()
               if (bset is None or it.benchmark in bset) and (split is None or it.split == split)]
        out.sort(key=lambda it: (it.benchmark, it.item_id))
        if limit_per_benchmark is not None:
            seen: dict[str, int] = {}
            keep = []
            for it in out:
                seen[it.benchmark] = seen.get(it.benchmark, 0) + 1
                if seen[it.benchmark] <= limit_per_benchmark:
                    keep.append(it)
            out = keep
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(it) for it in self._items.values()])

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Registry":
        fields = Item.__dataclass_fields__.keys()
        return cls(Item(**{k: (row[k] if k in row else "") for k in fields}) for row in df.to_dict("records"))

    def save(self, path=ITEMS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(path, index=False)


@lru_cache(maxsize=2)
def load_registry(path: str = str(ITEMS_FILE)) -> Registry:
    return Registry.from_frame(pd.read_parquet(path))
