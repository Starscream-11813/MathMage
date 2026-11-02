"""v6 registry: fresh post-cutoff contests transcribed from the organizers' official PDFs.

Sources (all held after every member's reported training cutoff, the latest being Claude Haiku 4.5's
July 2025 training data): HMMT November 2025 Guts, HMMT February 2026 Guts and Team, PUMaC 2025
(Division A subject tests and Team), CMIMC 2026 (individual rounds and Team) and BrUMO 2026 (Division A
Individual and Team, and Guts). MathArena had not released these sets when this registry was built.

Each problem was transcribed from page images of the official problem PDF and its answer copied from
the official solutions (one JSONL file per contest, kept with the registry as ``transcriptions/``).
Only problems with a single exact final answer are kept; estimation, proof, set-valued, chained and
figure-dependent problems are excluded with a recorded reason.

The frozen v3 registry, the v4 extension and their vaults are never modified. New items go to
``data/registry_v6/`` with their own vault and a manifest of sources, hashes and exclusions.
Near-duplicates: every new problem is screened against the base and extension registries and the
other new problems (normalized text, difflib ratio >= 0.85), exactly as in scripts/v4_data.py.

    python -I scripts/v6_data.py <folder with the transcription JSONL files>
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mathmage import paths  # noqa: E402
from mathmage.data.registry import Item, Registry, load_registry  # noqa: E402
from mathmage.experiments.artifacts import atomic_json, file_digest  # noqa: E402
from mathmage.grading.vault import VaultRecord, save_vault  # noqa: E402

V6 = paths.DATA / "registry_v6"
EXT = paths.DATA / "registry_ext"
THRESHOLD = 0.85
FILES = ("hmmt_nov_2025_guts", "hmmt_feb_2026_guts", "hmmt_feb_2026_team", "pumac_2025", "cmimc_2026",
         "brumo_2026")
REQUIRED = ("item_id", "benchmark", "round", "number", "item_date", "source_url", "problem", "answer",
            "answer_type", "domain", "included", "exclude_reason")
# Benchmarks as reported (one macro-averaging cell per contest). HMMT February 2026 Team items that survive the
# screen would join the Guts cell; all three usable ones repeat earlier test items (MathArena's HMMT Feb 2026 set).
CELL = {"hmmt_nov_2025_guts": "hmmt_nov_2025_guts", "hmmt_feb_2026_guts": "hmmt_feb_2026_guts",
        "hmmt_feb_2026_team": "hmmt_feb_2026_guts", "pumac_2025": "pumac_2025", "cmimc_2026": "cmimc_2026",
        "brumo_2026": "brumo_2026"}
CUTOFFS = {"o3-mini-low": "2023-10", "deepseek-v3": "2024-07", "haiku-4.5": "2025-07 (training data)",
           "gpt-oss-120b-low": "2024-06"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def load_rows(folder: Path, partial: bool = False) -> list[dict]:
    rows = []
    for name in FILES:
        path = folder / f"{name}.jsonl"
        if not path.exists():
            if partial:                          # testing only: a provisional registry from finished files
                continue
            raise SystemExit(f"missing transcription file {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                missing = [key for key in REQUIRED if key not in row]
                if missing:
                    raise SystemExit(f"{path.name}: {row.get('item_id')} lacks {missing}")
                rows.append(row)
    ids = [row["item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate item_id across transcription files")
    return rows


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if folder is None or not folder.is_dir():
        raise SystemExit(__doc__)
    partial = "--partial" in sys.argv
    rows = load_rows(folder, partial)
    # Curator review after transcription: problems whose official key or statement is flawed are excluded here,
    # with the reason recorded (the transcription files themselves are kept as written).
    overrides_path = folder / "curation_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    unknown = sorted(set(overrides) - {row["item_id"] for row in rows})
    if unknown and not partial:
        raise SystemExit(f"curation overrides name unknown items: {unknown}")
    for row in rows:
        if row["item_id"] in overrides and row["included"]:
            row["included"], row["exclude_reason"] = False, overrides[row["item_id"]]
    kept = [row for row in rows if row["included"]]
    for row in kept:
        if not str(row["problem"]).strip() or not str(row["answer"]).strip():
            raise SystemExit(f"{row['item_id']}: empty problem or answer")
        if row["item_date"] <= "2025-07":
            raise SystemExit(f"{row['item_id']}: not after every member's cutoff")

    base = load_registry()
    ext = Registry.from_frame(pd.read_parquet(EXT / "items.parquet"))
    reference = [(it, normalize(it.problem)) for it in base.all() + ext.all()]
    family, excluded, matches = {}, {}, []
    for k, row in enumerate(kept):
        text = normalize(row["problem"])
        family[row["item_id"]] = f"fam-{row['item_id']}"
        for it, other in reference:
            matcher = difflib.SequenceMatcher(None, text, other, autojunk=False)
            if matcher.quick_ratio() >= THRESHOLD and matcher.ratio() >= THRESHOLD:
                matches.append({"new": row["item_id"], "registry": it.item_id, "split": it.split})
                # A fresh set must not repeat a problem any earlier run has seen (training, tuning or test).
                excluded[row["item_id"]] = f"near-duplicate of earlier {it.split} item {it.item_id}"
        for earlier in kept[:k]:
            matcher = difflib.SequenceMatcher(None, text, normalize(earlier["problem"]), autojunk=False)
            if matcher.quick_ratio() >= THRESHOLD and matcher.ratio() >= THRESHOLD:
                matches.append({"new": row["item_id"], "registry": earlier["item_id"], "split": "new"})
                family[row["item_id"]] = family[earlier["item_id"]]

    items, answers = [], []
    for row in kept:
        if row["item_id"] in excluded:
            continue
        answer = str(row["answer"]).strip()
        items.append(Item(item_id=row["item_id"], benchmark=CELL[row["benchmark"]], source=row["source_url"],
                          problem=str(row["problem"]).strip(), answer_format="exact", domain=row["domain"],
                          family_id=family[row["item_id"]], split="test", item_date=row["item_date"],
                          band="olympiad", kind="olympiad",
                          extra=json.dumps({"contest": row["benchmark"], "round": row["round"],
                                            "number": int(row["number"])})))
        answers.append(VaultRecord(item_id=row["item_id"], answer=answer,
                                   answer_type="integer" if re.fullmatch(r"-?\d+", answer) else "latex",
                                   answer_format="exact", solution=str(row.get("solution_excerpt", ""))))
    V6.mkdir(parents=True, exist_ok=True)
    Registry(items).save(V6 / "items.parquet")
    save_vault(answers, V6 / "answers.parquet")
    copies = V6 / "transcriptions"
    copies.mkdir(exist_ok=True)
    present = [name for name in FILES if (folder / f"{name}.jsonl").exists()]
    for name in present:
        shutil.copy2(folder / f"{name}.jsonl", copies / f"{name}.jsonl")
    if overrides_path.exists():
        shutil.copy2(overrides_path, copies / "curation_overrides.json")
    counts = {}
    for row in rows:
        entry = counts.setdefault(row["benchmark"], {"total": 0, "kept": 0, "excluded": {}})
        entry["total"] += 1
        reason = (excluded.get(row["item_id"]) if row["included"] else row["exclude_reason"]) or ""
        if row["included"] and row["item_id"] not in excluded:
            entry["kept"] += 1
        else:
            key = reason.split(":")[0][:60] or "unspecified"
            entry["excluded"][key] = entry["excluded"].get(key, 0) + 1
    manifest = {
        "purpose": "v6 follow-up: fresh post-cutoff contests transcribed from official PDFs; frozen registries unchanged",
        "items": len(items), "cells": sorted({it.benchmark for it in items}),
        "per_contest": counts, "near_duplicate_threshold": THRESHOLD, "near_duplicates": matches,
        "excluded_as_training_or_dev_duplicates": excluded,
        "items_sha256": file_digest(V6 / "items.parquet"), "answers_sha256": file_digest(V6 / "answers.parquet"),
        "transcriptions_sha256": {name: file_digest(copies / f"{name}.jsonl") for name in present},
        "curation_overrides": overrides,
        "partial_testing_registry": partial,
        "base_registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
        "ext_registry_sha256": file_digest(EXT / "items.parquet"),
        "training_cutoffs": CUTOFFS,
        "strict_post_cutoff": "every v6 item postdates every member of both rosters (latest cutoff 2025-07)",
    }
    atomic_json(V6 / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("items", "cells", "per_contest", "near_duplicates",
                                               "excluded_as_training_or_dev_duplicates")}, indent=1))


if __name__ == "__main__":
    main()
