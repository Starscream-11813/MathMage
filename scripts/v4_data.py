"""Extension registry for the v4 follow-up: post-cutoff contests from MathArena.

The frozen v3 registry and answer vault are never modified: the frozen offline
artifacts bind their hashes. New items go to ``data/registry_ext/`` with their own
vault, family assignment and a manifest recording sources, file hashes and, per
roster, whether each contest postdates every member's training cutoff.

Near-duplicates: every new problem is compared with every registry problem and with
the other new problems (normalized text, difflib ratio). A near-duplicate of a
training or development problem is excluded; other near-duplicates share a family.

    python -I scripts/v4_data.py
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mathmage import paths  # noqa: E402
from mathmage.data.loaders import _domain  # noqa: E402
from mathmage.data.registry import Item, Registry, load_registry  # noqa: E402
from mathmage.experiments.artifacts import atomic_json, file_digest  # noqa: E402
from mathmage.grading.vault import VaultRecord, save_vault  # noqa: E402

EXT = paths.DATA / "registry_ext"
RAW = paths.RAW / "hf_v4"
THRESHOLD = 0.85
# MathArena repository -> competition month. Training cutoffs used for the contamination
# classes: o3-mini 2023-10, DeepSeek-V3 (2024-12 release) 2024-07, Claude Haiku 4.5
# training data through 2025-07, gpt-oss-120b 2024-06.
CONTESTS = {"hmmt_nov_2025": "2025-11", "brumo_2025": "2025-04", "cmimc_2025": "2025-04", "smt_2025": "2025-04"}
CUTOFFS = {"sat": "2025-07", "open": "2024-06"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def download(repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=f"MathArena/{repo}", filename="data/train-00000-of-00001.parquet",
                                repo_type="dataset", cache_dir=str(RAW)))


def main():
    base = load_registry()
    base_items = base.all()
    new, answers, sources = [], [], {}
    for repo, month in CONTESTS.items():
        path = download(repo)
        sources[repo] = {"repo": f"MathArena/{repo}", "file_sha256": file_digest(path), "competition_month": month}
        frame = pd.read_parquet(path)
        for row in sorted(frame.to_dict("records"), key=lambda r: int(r["problem_idx"])):
            item_id = f"{repo}/{int(row['problem_idx']):02d}"
            answer = str(row["answer"]).strip()
            new.append({"item_id": item_id, "benchmark": repo, "source": f"MathArena/{repo}",
                        "problem": str(row["problem"]).strip(), "domain": _domain(row.get("problem_type")),
                        "item_date": month})
            answers.append(VaultRecord(item_id=item_id, answer=answer,
                                       answer_type="integer" if answer.lstrip("-").isdigit() else "latex",
                                       answer_format="exact", solution=""))
        sources[repo]["items"] = len(frame)

    # Near-duplicate screen against every registry problem and among the new problems.
    base_norm = [(it, normalize(it.problem)) for it in base_items]
    family, excluded, matches = {}, set(), []
    for k, entry in enumerate(new):
        text = normalize(entry["problem"])
        family.setdefault(entry["item_id"], f"fam-{entry['item_id']}")
        for it, other in base_norm:
            ratio = difflib.SequenceMatcher(None, text, other, autojunk=False).quick_ratio()
            if ratio >= THRESHOLD and difflib.SequenceMatcher(None, text, other, autojunk=False).ratio() >= THRESHOLD:
                matches.append({"new": entry["item_id"], "registry": it.item_id, "split": it.split})
                if it.split in ("train", "dev"):
                    excluded.add(entry["item_id"])
                else:
                    family[entry["item_id"]] = it.family_id
        for earlier in new[:k]:
            other = normalize(earlier["problem"])
            if difflib.SequenceMatcher(None, text, other, autojunk=False).ratio() >= THRESHOLD:
                matches.append({"new": entry["item_id"], "registry": earlier["item_id"], "split": "new"})
                family[entry["item_id"]] = family[earlier["item_id"]]

    items = [Item(item_id=e["item_id"], benchmark=e["benchmark"], source=e["source"], problem=e["problem"],
                  answer_format="exact", domain=e["domain"], family_id=family[e["item_id"]], split="test",
                  item_date=e["item_date"], band="olympiad", kind="olympiad", extra="{}")
             for e in new if e["item_id"] not in excluded]
    kept = {it.item_id for it in items}
    EXT.mkdir(parents=True, exist_ok=True)
    Registry(items).save(EXT / "items.parquet")
    save_vault([a for a in answers if a.item_id in kept], EXT / "answers.parquet")
    manifest = {
        "purpose": "v4 follow-up: post-cutoff contests; the frozen v3 registry and vault are unchanged",
        "sources": sources, "near_duplicate_threshold": THRESHOLD, "near_duplicates": matches,
        "excluded_as_training_or_dev_duplicates": sorted(excluded), "items": len(items),
        "items_sha256": file_digest(EXT / "items.parquet"), "answers_sha256": file_digest(EXT / "answers.parquet"),
        "base_registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
        "base_vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
        "training_cutoffs": {"o3-mini-low": "2023-10", "deepseek-v3": "2024-07", "haiku-4.5": "2025-07 (training data)",
                             "gpt-oss-120b-low": "2024-06"},
        "strict_post_cutoff": {roster: sorted({it.benchmark for it in items if it.item_date > cutoff})
                               for roster, cutoff in CUTOFFS.items()},
    }
    atomic_json(EXT / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("items", "near_duplicates", "excluded_as_training_or_dev_duplicates",
                                               "strict_post_cutoff")}, indent=1))


if __name__ == "__main__":
    main()
