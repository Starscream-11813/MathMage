"""Hint-trial diagnostics by difficulty band (read-only; no API calls).

    python analysis/v7_hint_strata.py

Answers a reviewer's question about the fourth wave's crossed hint trial (analysis/v6_results.py, S8): did the deployed
screen block a right tutor's hints disproportionately on the hardest states, where a hint might help most? The unit
is an initially wrong member of a recorded stage-0 state that has a right tutor, exactly as in S8; the band is how
many of the state's three members were right at stage 0 (one of three: harder; two of three: easier). For each band
it reports the right hint's delivery rate, the rescue rates of one fresh attempt and of the right tutor's hint
(intention to treat), and their paired difference, with problem-clustered bootstrap intervals.
Writes results/tables/v7_hint_strata.json and .md.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("v6_results", REPO / "analysis" / "v6_results.py")
V6R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V6R)
OUT = REPO / "results/tables"
DRAWS = 2000


def units(package):
    """S8's units (initially wrong members), each with its state's band and problem family."""
    out = []
    for row in V6R.mech_records(package):
        correct0 = row["stage0_member_correct"]
        band = int(sum(bool(c) for c in correct0))
        by_arm = [(g, e.get("hint")) for g, e in zip(row["arms"], row["_events"])]
        for s, ok in enumerate(correct0):
            if ok:
                continue
            u = {"band": band, "family": row["family_id"]}
            for g, hint in by_arm:
                arm, new = g["arm"], g["new_correct"]
                if arm["kind"] == "sample" and arm["member"] == s and arm["k"] == 1:
                    u["fresh"] = bool(new and new[0])
                elif arm["kind"] == "hint" and arm["solver"] == s and arm["arm"] == "right":
                    u["right"] = bool(new and new[0])
                    u["delivered"] = bool(hint and hint.get("delivered"))
            if "right" in u and "fresh" in u:                 # the right-tutor units, as in the paper's figure
                out.append(u)
    return out


def summarize(rows):
    def mean(values, clusters):
        return V6R._cluster_mean(values, clusters, DRAWS)
    fam = [u["family"] for u in rows]
    delivered = [u for u in rows if u["delivered"]]
    return {"n": len(rows),
            "delivery": mean([float(u["delivered"]) for u in rows], fam),
            "right_delivered": mean([float(u["right"]) for u in delivered], [u["family"] for u in delivered]),
            "fresh": mean([float(u["fresh"]) for u in rows], fam),
            "right_itt": mean([float(u["right"]) for u in rows], fam),
            "right_minus_fresh": mean([float(u["right"]) - float(u["fresh"]) for u in rows], fam)}


def main():
    result = {}
    for package, team in (("lane", "open-weight"), ("sat26", "2026 team")):
        rows = units(package)
        bands = defaultdict(list)
        for u in rows:
            bands[u["band"]].append(u)
        result[team] = {"all": summarize(rows),
                        "by_band": {f"{b} of 3 right": summarize(v) for b, v in sorted(bands.items())}}
    OUT.joinpath("v7_hint_strata.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    fmt = lambda r: "--" if not r else f"{100 * r['mean']:.1f} [{100 * r['low']:.1f}, {100 * r['high']:.1f}]"  # noqa: E731
    lines = ["# Hint trial by difficulty band (right-tutor units)", ""]
    for team, d in result.items():
        lines += [f"## {team}", "", "| Band | n | Right hint delivered | Fresh attempt | Right hint (ITT) | "
                  "Right hint, if delivered | Hint - fresh |", "|---|---|---|---|---|---|---|"]
        for name, s in [("all", d["all"])] + list(d["by_band"].items()):
            lines.append(f"| {name} | {s['n']} | {fmt(s['delivery'])} | {fmt(s['fresh'])} | {fmt(s['right_itt'])} | "
                         f"{fmt(s['right_delivered'])} | {fmt(s['right_minus_fresh'])} |")
        lines.append("")
    OUT.joinpath("v7_hint_strata.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
