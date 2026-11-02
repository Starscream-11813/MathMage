"""Who is right when the team disagrees, on the fresh set (read-only; no API calls).

    python analysis/v6_agreement.py

The fresh-set counterpart of analysis/acl_agreement.py: every graded MathMage-16 unit of each team is classed by its
members' stage-0 agreement (unanimous, a 2-1 split, all different; an abstention is its own answer), and each
class reports its share of units, every member's accuracy and the executed program's accuracy where it answered.
Writes results/tables/v6_agreement.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v6_results as V6R  # noqa: E402  (installs the grading guard before any equivalence check)
from mathmage.grading.equiv import cluster  # noqa: E402

R = V6R.R
OUT = REPO / "results/tables/v6_agreement.json"


def pattern_stats(rows, registry):
    stats = defaultdict(lambda: {"units": 0, "members": defaultdict(lambda: [0, 0]), "program": [0, 0]})
    for row in rows:
        cands = [c for c in row["_candidates"] if c["id"].startswith("s0-")]
        grades = row["candidate_grades"]
        members = sorted((c for c in cands if c["phase"] not in R.TOOL_PHASES), key=lambda c: c["id"])
        program = [c for c in cands if c["phase"] in R.TOOL_PHASES]
        fmt = registry[row["item_id"]].answer_format
        answered = [c for c in members if c.get("valid", True) and c.get("answer") is not None]
        labels = dict(zip([c["id"] for c in answered], cluster([c["answer"] for c in answered], fmt))) if answered else {}
        keys = [labels.get(c["id"], f"abstain:{c['id']}") for c in members]
        kind = {1: "unanimous", 2: "2-1 split"}.get(len(set(keys)), "all differ")
        s = stats[kind]
        s["units"] += 1
        for c in members:
            tally = s["members"][c.get("model_key") or c["id"]]
            tally[0] += int(bool(grades.get(c["id"], {}).get("correct")))
            tally[1] += 1
        for c in program:
            if c.get("valid", True) and c.get("answer") is not None:
                s["program"][0] += int(bool(grades.get(c["id"], {}).get("correct")))
                s["program"][1] += 1
    total = sum(s["units"] for s in stats.values())
    out = {}
    for kind, s in stats.items():
        member_right = sum(v[0] for v in s["members"].values())
        member_n = sum(v[1] for v in s["members"].values())
        out[kind] = {"units": s["units"], "share": s["units"] / total, "member_acc": member_right / max(1, member_n),
                     "by_member": {k: {"acc": v[0] / max(1, v[1]), "n": v[1]} for k, v in sorted(s["members"].items())},
                     "program_acc": s["program"][0] / max(1, s["program"][1]), "program_answered": s["program"][1]}
    return out


def main():
    registry = V6R.Registry6()
    out = {}
    for package in ("sat26", "lane"):
        rows = V6R.mathmage_rows(package, "fresh_mm")
        if rows:
            out[package] = {"n_units": len(rows), "patterns": pattern_stats(rows, registry)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for package, block in out.items():
        print(f"{package} ({block['n_units']} units)")
        for kind in ("unanimous", "2-1 split", "all differ"):
            s = block["patterns"].get(kind)
            if s:
                members = ", ".join(f"{k} {v['acc']:.2f}" for k, v in s["by_member"].items())
                print(f"  {kind:10s} share {100 * s['share']:.0f}% ({s['units']})  members {s['member_acc']:.2f} "
                      f"[{members}]  program {s['program_acc']:.2f} ({s['program_answered']})")


if __name__ == "__main__":
    main()
