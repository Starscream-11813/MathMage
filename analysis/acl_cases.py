"""Material for the qualitative appendix: full problems, stage-0 answers, and the team's decisions (read-only).

    python analysis/acl_cases.py

For selected units it prints the problem, the gold answer, each member's and the program's stage-0 answer, the
majority vote's pick, the stage-0 posterior's pick, and the logged action trajectory with VOC values. It also
searches the original suites for units where the vote is wrong but the stage-0 posterior is right because it
weighs the program. Writes results/tables/acl_cases.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v5_results as V  # noqa: E402
from mathmage.grading.equiv import cluster  # noqa: E402

V4R, R = V.V4R, V.R
OUT = REPO / "results/tables/acl_cases.json"
WANT = [("e1", "aime_2025/15", 0), ("e1", "aime_2026/02", 0), ("e1", "aime_2025/12", 0), ("e1", "aime_2024/08", 0)]


def describe(row, registry, members):
    item = registry[row["item_id"]]
    cands = {c["id"]: c for c in row["_candidates"]}
    grades = row["candidate_grades"]
    stage0 = []
    for cid in ("s0-0", "s0-1", "s0-2", "s0-tool"):
        c = cands.get(cid)
        if c is None:
            continue
        who = "program" if cid == "s0-tool" else members.get(cid, c.get("model_key"))
        stage0.append({"who": who, "answer": c.get("answer"), "correct": bool(grades.get(cid, {}).get("correct"))})
    steps = []
    for snap in row.get("anytime", [])[1:]:
        steps.append({k: snap.get(k) for k in ("action", "voc", "ratio", "cost_usd", "decision", "decision_correct",
                                                "confidence") if k in snap})
    return {"item_id": row["item_id"], "seed": row.get("seed"), "benchmark": row["benchmark"],
            "problem": item.problem, "gold": getattr(item, "answer", None), "stage0": stage0,
            "final_correct": row.get("final_correct"), "steps": steps}


def main():
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    out = {"vignettes": [], "weighing_examples": []}
    by_key = {("e1", r["item_id"], int(r.get("seed", 0))): r for r in rows1}
    by_key.update({("e2", r["item_id"], int(r.get("seed", 0))): r for r in rows2})
    for key in WANT:
        row = by_key.get(key)
        if row:
            out["vignettes"].append(describe(row, registry, e1.members if key[0] == "e1" else e2.members))
    # stage-0 cases where the vote is wrong and the weighed pool (program included) is right
    for study, rows in (("e1", rows1),):
        for row in rows:
            cands = [c for c in row["_candidates"] if c["id"] in ("s0-0", "s0-1", "s0-2", "s0-tool")]
            fmt = registry[row["item_id"]].answer_format
            valid = [c for c in cands if c.get("valid", True) and c.get("answer") is not None]
            if len(valid) < 4:
                continue
            labels = cluster([c["answer"] for c in valid], fmt)
            lab = {c["id"]: l for c, l in zip(valid, labels)}
            members = [lab[c["id"]] for c in valid if c["id"] != "s0-tool"]
            top, count = Counter(members).most_common(1)[0]
            if count != 2:
                continue
            grades = row["candidate_grades"]
            vote_right = any(grades.get(c["id"], {}).get("correct") for c in valid if lab[c["id"]] == top)
            tool_right = grades.get("s0-tool", {}).get("correct")
            if not vote_right and tool_right and lab["s0-tool"] != top:
                d = describe(row, registry, e1.members)
                d["vote_answer"] = next(c["answer"] for c in valid if lab[c["id"]] == top)
                out["weighing_examples"].append(d)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for v in out["vignettes"] + out["weighing_examples"][:6]:
        print("\n==", v["item_id"], "seed", v["seed"], "gold", v["gold"], "final", v["final_correct"])
        print("  ", v["problem"][:160].replace("\n", " "))
        print("   stage0:", [(s["who"], s["answer"], s["correct"]) for s in v["stage0"]])
        for s in v["steps"][:5]:
            print("   step:", s)
    print("\nweighing examples:", len(out["weighing_examples"]))


if __name__ == "__main__":
    main()
