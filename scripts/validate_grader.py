"""Validate the MathMage grader against MathArena's own correctness labels (free).

For every public MathArena attempt on AIME 2025/2026 and HMMT February 2025/2026, we
re-extract the final answer from the raw response, grade it against our vault, and
compare with MathArena's `correct` label (produced by MathArena's parser, which SAT used
for HMMT 2026). Disagreements are written out for audit.

    python scripts/validate_grader.py [--max-per-competition N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from mathmage import paths  # noqa: E402
from mathmage.data.loaders import matharena_outputs  # noqa: E402
from mathmage.grading.vault import load_vault  # noqa: E402
from mathmage.data.loaders import ERRATA  # noqa: E402
from mathmage.grading.verify import grade_response  # noqa: E402

COMPETITIONS = ["aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-competition", type=int, default=None)
    args = ap.parse_args()
    vault = load_vault()
    rows = []
    for comp in COMPETITIONS:
        df = matharena_outputs(comp, columns=["problem_idx", "model_name", "idx_answer", "answer",
                                              "gold_answer", "parsed_answer", "correct"])
        if args.max_per_competition:
            df = df.sample(n=min(args.max_per_competition, len(df)), random_state=0)
        for r in df.to_dict("records"):
            item_id = f"{comp}/{int(r['problem_idx']):02d}"
            g = grade_response(item_id, r["answer"] or "", vault)
            rows.append({"competition": comp, "item_id": item_id, "model": r["model_name"], "attempt": r["idx_answer"],
                         "ours_extracted": g.extracted, "ours_method": g.method, "ours_correct": bool(g.correct),
                         "ma_parsed": str(r["parsed_answer"]),
                         "ma_correct": bool(r["correct"]), "gold": str(r["gold_answer"]),
                         "vault_gold": vault[item_id].answer})
        print(comp, "done", len(df))
    out = pd.DataFrame(rows)
    out["agree"] = out.ours_correct == out.ma_correct
    gold_mismatch = out[out.gold.str.strip() != out.vault_gold.str.strip()][["item_id", "gold", "vault_gold"]].drop_duplicates()
    summary = {
        "n_attempts": int(len(out)),
        "agreement": float(out.agree.mean()),
        "agreement_excluding_errata_items": float(out[~out.item_id.isin(list(ERRATA))].agree.mean()),
        "lastint_fallback_rate": float(out.ours_method.str.endswith("+lastint").mean()),
        "by_competition": {c: {"n": int(len(g)), "agreement": float(g.agree.mean()),
                               "ours_acc": float(g.ours_correct.mean()), "ma_acc": float(g.ma_correct.mean()),
                               "ours_only_correct": int((g.ours_correct & ~g.ma_correct).sum()),
                               "ma_only_correct": int((~g.ours_correct & g.ma_correct).sum())}
                           for c, g in out.groupby("competition")},
        "gold_string_mismatches": gold_mismatch.to_dict("records"),
    }
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / "grader_validation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    dis = out[~out.agree].sort_values(["competition", "item_id"])
    dis.to_csv(paths.TABLES / "grader_disagreements.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "gold_string_mismatches"}, indent=1))
    print("gold string mismatches (string form only; may still be equivalent):", len(gold_mismatch))


if __name__ == "__main__":
    main()
