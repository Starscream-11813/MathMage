"""Numbers for the ACL paper: every method on every evaluation set, with bootstrap SDs (read-only; no API calls).

    python analysis/acl_numbers.py              # 2,000 draws, as in the pre-registered analyses

Re-assembles exactly the unit rows used by v4_results.py, v5_results.py and strict_cutoff.py, and re-runs their
family-clustered, crossed-seed bootstrap with the same seed. Besides the stored macro accuracy and 95% interval it
records the standard deviation of the bootstrap draws (the bootstrap standard error, "sd") and, where every
benchmark shares a multi-seed schedule, the standard deviation of macro accuracy across seeds ("seed_sd").
Every recomputed macro accuracy and interval is checked against the stored summaries; any mismatch is reported.
Output: results/tables/acl_numbers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v5_results as V  # noqa: E402

V4R, R = V.V4R, V.R
TABLES = REPO / "results/tables"
APRIL, FRESH_E1, FRESH_E2 = V.APRIL, V.FRESH_E1, V.FRESH_E2
STRICT = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025")
NEW = ("hmmt_nov_2025",) + APRIL                      # the 153 new contest problems (v4)
LATER2 = APRIL + ("aime_2026", "hmmt_feb_2026")        # open team, v5 primary set (H7)
STAGE0 = ["o3-mini (low)", "Claude Haiku 4.5", "DeepSeek-V3", "single sample (mean of 3)", "program", "vote",
          "vote + program", "weighted vote + program", "posterior", "mu*", "maximal", "oracle (stage 0)",
          "oracle (final pool)"]


def seed_sd(rows):
    """SD across seeds of macro accuracy, when every benchmark shares one multi-seed schedule."""
    schedules = defaultdict(set)
    for r in rows:
        schedules[r["benchmark"]].add(int(r["seed"]))
    seeds = {tuple(sorted(s)) for s in schedules.values()}
    if len(seeds) != 1 or len(next(iter(seeds))) < 2:
        return None
    per_seed = []
    for s in next(iter(seeds)):
        by = defaultdict(list)
        for r in rows:
            if int(r["seed"]) == s:
                by[r["benchmark"]].append(r["final_correct"])
        per_seed.append(float(np.mean([np.mean(v) for v in by.values()])))
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1))}


def bench_seed_sd(rows):
    """Per benchmark: accuracy and SD across seeds (None for single-seed benchmarks)."""
    out = {}
    for b in sorted({r["benchmark"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["benchmark"] == b:
                by[int(r["seed"])].append(r["final_correct"])
        acc = [float(np.mean(v)) for _, v in sorted(by.items())]
        out[b] = {"acc": float(np.mean([r["final_correct"] for r in rows if r["benchmark"] == b])),
                  "seed_sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else None, "seeds": len(acc),
                  "n": sum(r["benchmark"] == b for r in rows)}
    return out


def summarize(rows, samples):
    estimate, draws = V4R.bootstrap(rows, "final_correct", samples)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"macro": estimate, "sd": float(np.std(draws, ddof=1)), "low": float(lo), "high": float(hi),
            "cost": float(np.mean([r["cost_usd"] for r in rows])), "n": len(rows),
            "by_benchmark": bench_seed_sd(rows), "seed_sd": seed_sd(rows)}


def contrast(left, right, samples):
    index = {(r["item_id"], r["seed"]): r for r in right}
    rows = [{**r, "delta": r["final_correct"] - index[(r["item_id"], r["seed"])]["final_correct"]}
            for r in left if (r["item_id"], r["seed"]) in index]
    if not rows:
        return None
    estimate, draws = V4R.bootstrap(rows, "delta", samples)
    lo, hi = np.quantile(draws, [.025, .975])
    p = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    return {"estimate": estimate, "sd": float(np.std(draws, ddof=1)), "low": float(lo), "high": float(hi), "p": p,
            "n": len(rows), "wins": int(sum(r["delta"] > 0 for r in rows)),
            "losses": int(sum(r["delta"] < 0 for r in rows))}


def assemble():
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    m1 = V4R.mathmage_methods(rows1, e1, a1, registry)
    m2 = V4R.mathmage_methods(rows2, e2, a2, registry)
    rows2_new = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    n1 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"]), e1, a1, registry)
    n2 = V4R.mathmage_methods(rows2_new, e2, a2, registry)

    g, g5 = V4R.graded, V.graded5
    deb = lambda jobs: sum((V4R.debate_outcomes(g(j), registry) for j in jobs), [])          # noqa: E731
    usc = lambda jobs: sum((V4R.usc_outcomes(g(j)) for j in jobs), [])                       # noqa: E731
    deb5 = lambda jobs: sum((V4R.debate_outcomes(g5(j), registry) for j in jobs), [])        # noqa: E731
    usc5 = lambda jobs: sum((V4R.usc_outcomes(g5(j)) for j in jobs), [])                     # noqa: E731
    moa = lambda jobs: sum((V.moa_outcomes(g5(j)) for j in jobs), [])                        # noqa: E731
    sc10 = lambda rows: V4R.sc_outcomes(rows, registry, [10])[10] if rows else []            # noqa: E731

    sat_deb = deb(["e1_debate", "new_e1_debate_nov"]) + deb5(["apr_e1_debate"])
    sat_usc = usc(["e1_usc", "new_e1_usc_nov"]) + usc5(["apr_e1_usc"])
    sat_moa = moa(["e1_moa", "nov_e1_moa", "apr_e1_moa"])
    sat_sc = sc10(g("e1_sc")) + sc10(g("new_e1_sc_nov")) + sc10(g5("apr_e1_sc"))
    open_deb = deb(["e2_debate", "new_e2_debate_nov"]) + deb5(["apr_e2_debate"])
    open_usc = usc(["e2_usc", "new_e2_usc_nov"]) + usc5(["apr_e2_usc"])
    open_moa = moa(["e2_moa", "nov_e2_moa", "apr_e2_moa"])
    sc_e2 = V4R.pooled_sc(rows2, registry, [1, 3, 5, 8, 12, 24])[5]  # exactly as in v4 H2 (AIME, 480)
    sc_open = V4R.pooled_sc(rows2 + rows2_new, registry, [5])[5]   # as in v5 S1

    sat_all = {k: m1.get(k, []) + n1.get(k, []) for k in set(m1) | set(n1)}
    open_all = {k: m2.get(k, []) + n2.get(k, []) for k in set(m2) | set(n2)}
    sub = V4R.subset

    def team(mm, bench, baselines):
        out = {k: sub(mm[k], bench) for k in STAGE0 if k in mm and sub(mm[k], bench)}
        out.update({k: sub(v, bench) for k, v in baselines.items() if sub(v, bench)})
        return out

    sat_base = {"SC": sat_sc, "USC": sat_usc, "debate": sat_deb, "MoA": sat_moa}
    sets = {
        "E1": ("SAT", team(m1, ("aime_2024", "aime_2025", "aime_2026", "hmmt_feb_2026", "tqa_phys"),
                           {"SC": sc10(g("e1_sc")), "USC": usc(["e1_usc"]), "debate": deb(["e1_debate"]),
                            "MoA": moa(["e1_moa"])})),
        "strict93": ("SAT", team(sat_all, STRICT, sat_base)),
        "april123": ("SAT", team(n1, APRIL, sat_base)),
        "sat216": ("SAT", team(sat_all, FRESH_E1, sat_base)),
        "new153_sat": ("SAT", team(n1, NEW, sat_base)),
        "E2": ("open", team(m2, ("aime_2024", "aime_2025", "math500"),
                            {"USC": usc(["e2_usc"]), "debate": deb(["e2_debate"]), "MoA": moa(["e2_moa"])})),
        "E2_aime_sc": ("open", {"mu*": sub(m2["mu*"], ("aime_2024", "aime_2025")),
                                "maximal": sub(m2["maximal"], ("aime_2024", "aime_2025")), "SC": sc_e2}),
        "open1488": ("open", team(n2, LATER2, {"USC": open_usc, "debate": open_deb, "MoA": open_moa})),
        "open1968": ("open", team(open_all, FRESH_E2, {"SC": sc_open, "USC": open_usc, "debate": open_deb,
                                                       "MoA": open_moa})),
        "new153_open": ("open", team(n2, NEW, {"SC": sc_open, "USC": open_usc, "debate": open_deb, "MoA": open_moa})),
        # the six 2025-26 contests the two teams share (216 problems; x 8 seeds on the open team = v4 H4)
        "open1728": ("open", team(open_all, FRESH_E1, {"SC": sc_open, "USC": open_usc, "debate": open_deb,
                                                       "MoA": open_moa})),
    }
    return sets


def stored():
    """The stored summaries, keyed like this script's sets, for cross-checking."""
    read = lambda name: json.loads((TABLES / name).read_text(encoding="utf-8"))  # noqa: E731
    v4, v5, strict = read("v4_summary.json"), read("v5_summary.json"), read("strict_sat.json")
    s1s, s1o = v5["S1"]["SAT roster"]["methods"], v5["S1"]["open roster"]["methods"]
    rename = {"SC@10 (DeepSeek-V3)": "SC", "SC@5 (pooled seeds)": "SC"}
    norm = lambda d: {rename.get(k, k): v for k, v in d.items()}  # noqa: E731
    return {
        "E1": {**norm({k: v for k, v in v4["H1"]["methods"].items() if not k.startswith("SC@") or "10" in k}),
               "MoA": v5["H8"]["E1"]["methods"]["MoA"]},
        "strict93": norm(strict["methods"]),
        "april123": norm(v5["H6"]["methods"]),
        "sat216": norm(s1s),
        "E2": {**{k: v for k, v in v4["H2"]["methods"].items() if not k.startswith("SC@")},
               "MoA": v5["H8"]["E2"]["methods"]["MoA"]},
        "E2_aime_sc": {"SC": v4["H2"]["methods"]["SC@5 (pooled seeds)"]},
        "open1488": norm(v5["H7"]["methods"]),
        "open1728": {k: v for k, v in v4["H4"]["e2"]["methods"].items()},
        "new153_sat": {k: v for k, v in v4["H4"]["e1"]["methods"].items()},
        "open1968": {**norm(s1o), **{k: v for k, v in v4["H3"]["e2"]["methods"].items()
                                     if k in ("posterior", "maximal", "weighted vote + program")}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--only", nargs="*", help="recompute these sets and merge into the existing JSON")
    args = ap.parse_args()
    sets = assemble()
    path = TABLES / "acl_numbers.json"
    out = json.loads(path.read_text(encoding="utf-8")) if args.only and path.exists() else {}
    if args.only:
        sets = {k: v for k, v in sets.items() if k in args.only}
    mismatches = []
    reference = stored()
    for name, (roster, methods) in sets.items():
        block = {"roster": roster, "methods": {k: summarize(v, args.samples) for k, v in methods.items()},
                 "contrasts": {f"mu* vs {k}": contrast(methods["mu*"], v, args.samples)
                               for k, v in methods.items() if k != "mu*" and "oracle" not in k}}
        for k, ref in reference.get(name, {}).items():
            got = block["methods"].get(k)
            if got is None:
                mismatches.append(f"{name}/{k}: missing")
                continue
            for field in ("macro", "low", "high"):
                if abs(got[field] - ref[field]) > 5e-4:
                    mismatches.append(f"{name}/{k}/{field}: stored {ref[field]:.4f} recomputed {got[field]:.4f}")
            if got["n"] != ref["n"]:
                mismatches.append(f"{name}/{k}/n: stored {ref['n']} recomputed {got['n']}")
        out[name] = block
        mu = block["methods"]["mu*"]
        print(f"{name:12s} {roster:4s} n={mu['n']:5d}  mu* {100 * mu['macro']:.1f} ± {100 * mu['sd']:.1f}")
    path.write_text(json.dumps(V.to_jsonable(out), indent=1), encoding="utf-8")
    print("\ncross-check against stored summaries:", "all match" if not mismatches else f"{len(mismatches)} mismatches")
    for line in mismatches:
        print("  ", line)


if __name__ == "__main__":
    main()
