"""Exploratory robustness of the v5 wins: leave one contest out (read-only; not pre-registered).

    python analysis/v5_robustness.py

For the H6/H7 debate and MoA contrasts, drops each contest in turn and recomputes the paired macro
difference (family-clustered bootstrap, 2,000 draws). The pre-registered tests are in v5_results.py;
this only asks whether one contest carries a result.
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


def main():
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    n1 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e1_mathmage_apr"]), e1, a1, registry)
    n2 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e2_mathmage_apr"]), e2, a2, registry)
    later = V.APRIL + ("aime_2026", "hmmt_feb_2026")
    panels = [("H6 · SAT roster · vs debate", n1["mu*"], V4R.debate_outcomes(V.graded5("apr_e1_debate"), registry), V.APRIL),
              ("H6 · SAT roster · vs MoA", n1["mu*"], V.moa_outcomes(V.graded5("apr_e1_moa")), V.APRIL),
              ("H7 · open roster · vs debate", n2["mu*"], V4R.debate_outcomes(V.graded5("apr_e2_debate"), registry), later),
              ("H7 · open roster · vs MoA", n2["mu*"], V.moa_outcomes(V.graded5("apr_e2_moa")), later)]
    lines = ["# v5 wins without each contest (exploratory; not pre-registered)", "",
             "| Contrast | " + " | ".join(f"without {V4R.LABEL[c]}" for c in later) + " |",
             "|---|" + "---|" * len(later)]
    for name, mm, base, contests in panels:
        cells = []
        for drop in later:
            if drop not in contests:
                cells.append("--")
                continue
            keep = tuple(c for c in contests if c != drop)
            c = V4R.contrast(V4R.subset(mm, keep), V4R.subset(base, keep), 2000)
            cells.append(f"{100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], p = {c['p']:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    out = V.OUT_T / "v5_robustness.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
