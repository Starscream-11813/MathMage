"""SAT-style team on the problems that postdate EVERY member's training cutoff (read-only; descriptive).

    python analysis/strict_cutoff.py

Training cutoffs recorded in data/registry_ext/manifest.json: o3-mini 2023-10, DeepSeek-V3 2024-07, Claude Haiku
4.5 training data through 2025-07, gpt-oss-120b 2024-06. The April 2025 contests (BRUMO, CMIMC, SMT) therefore
postdate two of the three SAT-team members but fall inside Claude Haiku 4.5's training window. Only AIME 2026,
HMMT Feb 2026 and HMMT Nov 2025 (93 problems) are strictly post-cutoff for the whole SAT team. (For the open team,
every 2025-26 contest is strictly post-cutoff.) This script reports MathMage against all five baseline families on
those 93 problems, with the same family-clustered bootstrap as the pre-registered analyses. The v4 pre-registered
H3 contrasts already cover vote, SC, USC and debate on this set; mixture of agents is added here descriptively.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v5_results as V  # noqa: E402

V4R, R = V.V4R, V.R
STRICT = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025")
SAMPLES = 2000


def main():
    registry = V4R.Combined()
    e1, _ = R.STUDIES
    a1 = json.loads(e1.artifact.read_text(encoding="utf-8"))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    m1 = V4R.mathmage_methods(rows1, e1, a1, registry)
    n1 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e1_mathmage_nov"]), e1, a1, registry)
    sets = {
        "mu*": V4R.subset(m1["mu*"] + n1["mu*"], STRICT),
        "vote": V4R.subset(m1["vote"] + n1["vote"], STRICT),
        "SC@10 (DeepSeek-V3)": V4R.subset(V4R.sc_outcomes(V4R.graded("e1_sc"), registry, [10])[10]
                                          + V4R.sc_outcomes(V4R.graded("new_e1_sc_nov"), registry, [10])[10], STRICT),
        "USC": V4R.subset(V4R.usc_outcomes(V4R.graded("e1_usc")) + V4R.usc_outcomes(V4R.graded("new_e1_usc_nov")), STRICT),
        "debate": V4R.subset(V4R.debate_outcomes(V4R.graded("e1_debate"), registry)
                             + V4R.debate_outcomes(V4R.graded("new_e1_debate_nov"), registry), STRICT),
        "MoA": V4R.subset(V.moa_outcomes(V.graded5("e1_moa")) + V.moa_outcomes(V.graded5("nov_e1_moa")), STRICT),
    }
    out = {"benchmarks": STRICT, "n": len(sets["mu*"]),
           "methods": {k: V4R.summarize(v, SAMPLES) for k, v in sets.items()},
           "contrasts": {f"mu* vs {k}": V4R.contrast(sets["mu*"], v, SAMPLES) for k, v in sets.items() if k != "mu*"}}
    (V.OUT_T / "strict_sat.json").write_text(json.dumps(V.to_jsonable(out), indent=1), encoding="utf-8")
    lines = ["# SAT-style team: problems after every member's training cutoff (descriptive)", "",
             "AIME 2026, HMMT Feb 2026 and HMMT Nov 2025: 93 problems. Claude Haiku 4.5's training data runs to July 2025, so "
             "the April 2025 contests are excluded here.", ""]
    lines += V4R.method_table(out["methods"], "Accuracy and cost")
    lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in out["contrasts"].items()]
    (V.OUT_T / "strict_sat.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
