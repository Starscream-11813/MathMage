"""Build the item registry, answer vault, and named subsets (free; no model calls).

    python scripts/build_registry.py

Outputs (data/registry/):
  items.parquet     problems, provenance, families, splits (no answers)
  answers.parquet   reference answers and solutions (the vault; grader-only)
  subsets.json      named item subsets (pilot_20, sat_suite, ablation_45, math500_100, ...)
  data_card.json    counts, near-duplicate pairs, and exclusions
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathmage import paths  # noqa: E402
from mathmage.data.loaders import ERRATA, load_all_raw, write_raw_snapshot  # noqa: E402
from mathmage.data.registry import Registry  # noqa: E402
from mathmage.data.splits import assign_splits, make_subsets, near_duplicate_families  # noqa: E402
from mathmage.grading.vault import VaultRecord, save_vault  # noqa: E402


def main() -> None:
    paths.ensure_dirs()
    raw = load_all_raw()
    print(f"loaded {len(raw)} raw items:", dict(Counter(r.benchmark for r in raw)))
    fams, pairs = near_duplicate_families([r.item_id for r in raw], [r.problem for r in raw], threshold=0.92)
    items = assign_splits(raw, fams)
    reg = Registry(items)
    reg.save()
    save_vault([VaultRecord(item_id=r.item_id, answer=r.answer, answer_type=r.answer_type,
                            answer_format=r.answer_format, solution=r.solution) for r in raw])
    subsets = make_subsets(items)
    (paths.REGISTRY / "subsets.json").write_text(json.dumps(subsets, indent=1), encoding="utf-8")
    write_raw_snapshot(raw, paths.REGISTRY / "problems_snapshot.jsonl")
    card = {
        "n_items": len(items),
        "by_benchmark": dict(Counter(it.benchmark for it in items)),
        "by_split": dict(Counter(f"{it.benchmark}:{it.split}" for it in items)),
        "near_duplicate_pairs": [{"a": a, "b": b, "cos": round(c, 4)} for a, b, c in pairs],
        "excluded_training_items": [it.item_id for it in items if it.split == "excluded"],
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "errata": {k: {"wrong": w, "official": o, "note": n} for k, (w, o, n) in ERRATA.items()},
        "notes": [
            "TheoremQA-physics: 114 text-only items of 131; 17 picture items excluded (DeepSeek-V3 is text-only).",
            "AIME ids 01-15 are AIME I and 16-30 are AIME II.",
            "Training pool: AIME 2023 and HMMT February 2025, split by family into train and dev.",
        ],
    }
    (paths.REGISTRY / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in card.items() if k != "near_duplicate_pairs"}, indent=1))
    print(f"near-duplicate pairs: {len(pairs)}")
    for p in card["near_duplicate_pairs"][:30]:
        print("  ", p)


if __name__ == "__main__":
    main()
