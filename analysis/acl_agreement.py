"""Who is right when the team disagrees? Stage-0 member agreement vs member and program accuracy (read-only).

    python analysis/acl_agreement.py

For every graded unit, the three members' stage-0 answers are clustered by mathematical equivalence (an
abstention counts as its own answer) and the unit is classed as unanimous, a 2-1 split, or all different.
Reports the share of units, the members' mean accuracy, and the executed program's accuracy on the units where
it returned an answer. Covers the original suites (E1, E2) and the 2025-26 contests (v4 + v5 MathMage runs).
Writes results/tables/acl_agreement.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v5_results as V  # noqa: E402
from mathmage.grading.equiv import cluster  # noqa: E402

V4R, R = V.V4R, V.R
OUT = REPO / "results/tables/acl_agreement.json"


def pattern_stats(rows, registry, benchmarks=None):
    stats = defaultdict(lambda: {"units": 0, "member_correct": 0, "member_n": 0, "program_answered": 0,
                                 "program_correct": 0})
    for row in rows:
        if benchmarks and row["benchmark"] not in benchmarks:
            continue
        cands = [c for c in row["_candidates"] if c["id"].startswith("s0-")]
        grades = row["candidate_grades"]
        members = sorted((c for c in cands if c["phase"] not in R.TOOL_PHASES), key=lambda c: c["id"])
        program = [c for c in cands if c["phase"] in R.TOOL_PHASES]
        fmt = registry[row["item_id"]].answer_format
        answered = [c for c in members if c.get("valid", True) and c.get("answer") is not None]
        labels = dict(zip([c["id"] for c in answered], cluster([c["answer"] for c in answered], fmt))) if answered else {}
        keys = [labels.get(c["id"], f"abstain:{c['id']}") for c in members]
        distinct = len(set(keys))
        kind = {1: "unanimous", 2: "2-1 split"}.get(distinct, "all differ")
        s = stats[kind]
        s["units"] += 1
        for c in members:
            s["member_n"] += 1
            s["member_correct"] += int(bool(grades.get(c["id"], {}).get("correct")))
        for c in program:
            if c.get("valid", True) and c.get("answer") is not None:
                s["program_answered"] += 1
                s["program_correct"] += int(bool(grades.get(c["id"], {}).get("correct")))
    total = sum(s["units"] for s in stats.values())
    return {k: {**s, "share": s["units"] / total, "member_acc": s["member_correct"] / max(1, s["member_n"]),
                "program_acc": s["program_correct"] / max(1, s["program_answered"])}
            for k, s in stats.items()}


def main():
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    new1 = V4R.mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"])
    new2 = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    out = {"E1": pattern_stats(rows1, registry), "E2": pattern_stats(rows2, registry),
           "sat216": pattern_stats(rows1 + new1, registry, set(V.FRESH_E1)),
           "open1968": pattern_stats(rows2 + new2, registry, set(V.FRESH_E2))}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, block in out.items():
        print(name)
        for kind in ("unanimous", "2-1 split", "all differ"):
            s = block.get(kind)
            if s:
                print(f"  {kind:10s} share {100 * s['share']:.0f}% ({s['units']})  members {s['member_acc']:.2f}  "
                      f"program {s['program_acc']:.2f} ({s['program_answered']})")


if __name__ == "__main__":
    main()
