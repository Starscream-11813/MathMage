"""v5 follow-up analysis: the completed same-roster baseline matrix (read-only; no API calls).

    python analysis/v5_results.py            # tables + figure (LuaLaTeX)
    python analysis/v5_results.py --draft    # fast figure preview

Implements the analysis plan frozen in runs/live_20261018/program_v5/plan.json (written before any v5
result existed; see PREREGISTRATION.md there):

  Primary (v5 data only; Holm across the nine contrasts)
    H6  SAT roster, 123 April-2025 problems: mu* vs debate, USC, MoA, DeepSeek-V3 SC@10
    H7  open roster, 1,488 evaluations: mu* vs debate, USC, MoA
    H8  MoA on the original suites: mu* vs MoA on E1 (237) and E2 (580)
  Secondary
    S1  complete contamination-free sets (v4 + v5): mu* vs debate (two-look Pocock, nominal alpha 0.0294),
        USC, MoA and SC (descriptive)
    S2  the maximal operating point vs each baseline on the H6-H8 sets
    S3  dollars per correct answer
    S4  same-roster MoA on E1 vs SAT's reported MoA (57.3), descriptive
    S5  non-inferiority to debate: pooled over every roster-benchmark cell where debate ran (v4 + v5),
        families resampled jointly across rosters; non-inferior if the 95% lower bound exceeds -3 pp
A job that is incomplete is reported as such, never silently pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v4_results as V4R  # noqa: E402
from mathmage.analysis.stats import holm  # noqa: E402

R = V4R.R
V5 = R.RUNS / "program_v5"
OUT_T, OUT_F = REPO / "results/tables", REPO / "results/figures"
APRIL = ("brumo_2025", "cmimc_2025", "smt_2025")
FRESH_E1 = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025") + APRIL
FRESH_E2 = ("aime_2025", "aime_2026", "hmmt_feb_2026", "hmmt_nov_2025") + APRIL
POCOCK = 0.0294
MARGIN = 0.03
SAT_REPORTED_MOA = 0.573


def graded5(job):
    directory = V5 / job / "graded"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))] if directory.exists() else []


def moa_outcomes(rows):
    return [V4R.unit(r, r["moa_correct"], r["resource_cost_usd"]) for r in rows]


def tag(units, roster):
    return [{**u, "benchmark": f"{roster}:{u['benchmark']}", "item_id": f"{roster}:{u['item_id']}"} for u in units]


def analyze(samples):
    plan = json.loads((V5 / "plan.json").read_text(encoding="utf-8"))
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    m1 = V4R.mathmage_methods(rows1, e1, a1, registry)
    m2 = V4R.mathmage_methods(rows2, e2, a2, registry)
    n1 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"]), e1, a1, registry)
    n2 = V4R.mathmage_methods(V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"]), e2, a2, registry)
    planned = {j["id"]: len(j["units"]) for package in plan["jobs"].values() for j in package}
    status = {job: {"graded": len(graded5(job)), "planned": n} for job, n in planned.items()}
    out = {"plan_sha256": plan["sha256"], "status": status,
           "complete": all(s["graded"] == s["planned"] for s in status.values())}

    # baselines from v5
    deb_apr1 = V4R.debate_outcomes(graded5("apr_e1_debate"), registry)
    usc_apr1 = V4R.usc_outcomes(graded5("apr_e1_usc"))
    moa_apr1 = moa_outcomes(graded5("apr_e1_moa"))
    sc_apr1 = V4R.sc_outcomes(graded5("apr_e1_sc"), registry, [10])[10] if graded5("apr_e1_sc") else []
    deb_apr2 = V4R.debate_outcomes(graded5("apr_e2_debate"), registry)
    usc_apr2 = V4R.usc_outcomes(graded5("apr_e2_usc"))
    moa_apr2 = moa_outcomes(graded5("apr_e2_moa"))
    moa_e1, moa_e2 = moa_outcomes(graded5("e1_moa")), moa_outcomes(graded5("e2_moa"))
    moa_nov1, moa_nov2 = moa_outcomes(graded5("nov_e1_moa")), moa_outcomes(graded5("nov_e2_moa"))

    # MathMage on the same units
    mm_apr1 = V4R.subset(n1["mu*"], APRIL)
    mx_apr1 = V4R.subset(n1["maximal"], APRIL)
    later2 = APRIL + ("aime_2026", "hmmt_feb_2026")
    mm_apr2, mx_apr2 = V4R.subset(n2["mu*"], later2), V4R.subset(n2["maximal"], later2)

    def block(mm, mx, baselines):
        methods = {"mu*": mm, "maximal": mx, **{k: v for k, v in baselines.items() if v}}
        return {"methods": {k: V4R.summarize(v, samples) for k, v in methods.items() if v},
                "contrasts": {f"mu* vs {k}": V4R.contrast(mm, v, samples) for k, v in baselines.items() if v},
                "secondary": {f"maximal vs {k}": V4R.contrast(mx, v, samples) for k, v in baselines.items() if v}}

    out["H6"] = block(mm_apr1, mx_apr1, {"debate": deb_apr1, "USC": usc_apr1, "MoA": moa_apr1,
                                         "SC@10 (DeepSeek-V3)": sc_apr1})
    out["H7"] = block(mm_apr2, mx_apr2, {"debate": deb_apr2, "USC": usc_apr2, "MoA": moa_apr2})
    out["H8"] = {"E1": block(m1["mu*"], m1["maximal"], {"MoA": moa_e1}),
                 "E2": block(m2["mu*"], m2["maximal"], {"MoA": moa_e2})}
    primary = ([("H6", None, k) for k in out["H6"]["contrasts"]] + [("H7", None, k) for k in out["H7"]["contrasts"]] +
               [("H8", s, k) for s in ("E1", "E2") for k in out["H8"][s]["contrasts"]])

    def get(h, s, k):
        return out[h]["contrasts"][k] if s is None else out[h][s]["contrasts"][k]
    pvalues = [(get(h, s, k) or {}).get("p", 1.0) for h, s, k in primary]
    for (h, s, k), adjusted in zip(primary, holm(pvalues) if pvalues else []):
        if get(h, s, k):
            get(h, s, k)["holm_p"] = adjusted
    out["primary_family"] = [f"{h}{'/' + s if s else ''}: {k}" for h, s, k in primary]

    # S1: complete contamination-free sets (v4 + v5)
    e1_debate = V4R.debate_outcomes(V4R.graded("e1_debate"), registry)
    e1_usc = V4R.usc_outcomes(V4R.graded("e1_usc"))
    e1_sc = V4R.sc_outcomes(V4R.graded("e1_sc"), registry, [10])[10]
    # "vote" (majority vote of the members) was added on 2026-10-21 after the pre-registered analysis ran, as a
    # descriptive row for the cross-study synthesis; it is not part of the pre-registered family.
    sat_fresh = {"mu*": V4R.subset(m1["mu*"] + n1["mu*"], FRESH_E1),
                 "vote": V4R.subset(m1["vote"] + n1["vote"], FRESH_E1),
                 "debate": V4R.subset(e1_debate + V4R.debate_outcomes(V4R.graded("new_e1_debate_nov"), registry) + deb_apr1, FRESH_E1),
                 "USC": V4R.subset(e1_usc + V4R.usc_outcomes(V4R.graded("new_e1_usc_nov")) + usc_apr1, FRESH_E1),
                 "MoA": V4R.subset(moa_e1 + moa_nov1 + moa_apr1, FRESH_E1),
                 "SC@10 (DeepSeek-V3)": V4R.subset(e1_sc + V4R.sc_outcomes(V4R.graded("new_e1_sc_nov"), registry, [10])[10]
                                                   + sc_apr1, FRESH_E1)}
    e2_debate = V4R.debate_outcomes(V4R.graded("e2_debate"), registry)
    e2_usc = V4R.usc_outcomes(V4R.graded("e2_usc"))
    rows2_new = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    pooled = V4R.pooled_sc(rows2 + rows2_new, registry, [5])
    open_fresh = {"mu*": V4R.subset(m2["mu*"] + n2["mu*"], FRESH_E2),
                  "vote": V4R.subset(m2["vote"] + n2["vote"], FRESH_E2),
                  "debate": V4R.subset(e2_debate + V4R.debate_outcomes(V4R.graded("new_e2_debate_nov"), registry) + deb_apr2, FRESH_E2),
                  "USC": V4R.subset(e2_usc + V4R.usc_outcomes(V4R.graded("new_e2_usc_nov")) + usc_apr2, FRESH_E2),
                  "MoA": V4R.subset(moa_e2 + moa_nov2 + moa_apr2, FRESH_E2),
                  "SC@5 (pooled seeds)": V4R.subset(pooled[5], FRESH_E2)}
    out["S1"] = {}
    for roster, sets in (("SAT roster", sat_fresh), ("open roster", open_fresh)):
        out["S1"][roster] = {"methods": {k: V4R.summarize(v, samples) for k, v in sets.items() if v},
                             "contrasts": {f"mu* vs {k}": V4R.contrast(sets["mu*"], v, samples)
                                           for k, v in sets.items() if k != "mu*" and v}}
        debate = out["S1"][roster]["contrasts"].get("mu* vs debate")
        if debate:
            debate["pocock_alpha"] = POCOCK
            debate["significant_at_pocock"] = debate["p"] < POCOCK

    # S3: dollars per correct answer
    out["S3"] = {}
    for name, sets in (("SAT roster · contamination-free", sat_fresh), ("open roster · contamination-free", open_fresh)):
        out["S3"][name] = {k: {"usd_per_correct": float(np.mean([u["cost_usd"] for u in v]) /
                                                        max(1e-9, V4R.summarize(v, 200)["macro"]))}
                           for k, v in sets.items() if v}

    # S4: MoA on E1 vs SAT's reported MoA
    if moa_e1:
        out["S4"] = {"ours_MoA_E1": V4R.summarize(moa_e1, samples)["macro"], "sat_reported_MoA": SAT_REPORTED_MOA}

    # S5: non-inferiority to debate over every cell where debate ran (v4 + v5)
    sat_mm = m1["mu*"] + n1["mu*"]
    sat_deb = (e1_debate + V4R.debate_outcomes(V4R.graded("new_e1_debate_nov"), registry) + deb_apr1)
    open_mm = m2["mu*"] + n2["mu*"]
    open_deb = (e2_debate + V4R.debate_outcomes(V4R.graded("new_e2_debate_nov"), registry) + deb_apr2)
    keep = lambda mm, deb: [u for u in mm if (u["item_id"], u["seed"]) in {(d["item_id"], d["seed"]) for d in deb}]  # noqa: E731
    s5 = V4R.contrast(tag(keep(sat_mm, sat_deb), "sat") + tag(keep(open_mm, open_deb), "open"),
                      tag(sat_deb, "sat") + tag(open_deb, "open"), samples)
    if s5:
        s5["margin"] = -MARGIN
        s5["non_inferior"] = s5["low"] > -MARGIN
        cost_mm = np.mean([u["cost_usd"] for u in keep(sat_mm, sat_deb)]) / np.mean([u["cost_usd"] for u in sat_deb])
        cost_open = np.mean([u["cost_usd"] for u in keep(open_mm, open_deb)]) / np.mean([u["cost_usd"] for u in open_deb])
        s5["cost_ratio_mathmage_over_debate"] = {"SAT roster": float(cost_mm), "open roster": float(cost_open)}
    out["S5"] = s5
    return out


# ----------------------------------------------------------------------------- output
def write(out, path):
    f = V4R.fmt_contrast
    lines = ["# v5 follow-up results (pre-registered; see runs/live_20261018/program_v5/PREREGISTRATION.md)", "",
             f"Plan {out['plan_sha256'][:12]}; complete: {out['complete']}. Units graded / planned: " +
             ", ".join(f"{k} {v['graded']}/{v['planned']}" for k, v in out["status"].items()), "",
             "Primary family (Holm across these): " + "; ".join(out["primary_family"]), ""]
    for key, title in (("H6", "H6 · SAT roster, 123 April-2025 problems (BRUMO, CMIMC, SMT)"),
                       ("H7", "H7 · open roster, 1,488 evaluations (April 2025 contests, AIME 2026, HMMT Feb 2026)")):
        lines += [f"## {title}", ""] + V4R.method_table(out[key]["methods"], "Accuracy and cost")
        lines += ["Primary contrasts:"] + [f"- {k}: {f(c)}" for k, c in out[key]["contrasts"].items()]
        lines += ["", "Secondary (S2):"] + [f"- {k}: {f(c)}" for k, c in out[key]["secondary"].items()] + [""]
    for s in ("E1", "E2"):
        lines += [f"## H8 · MoA on the original suite, {s}", ""] + V4R.method_table(out["H8"][s]["methods"], "Accuracy and cost")
        lines += ["Primary contrast:"] + [f"- {k}: {f(c)}" for k, c in out["H8"][s]["contrasts"].items()]
        lines += ["", "Secondary (S2):"] + [f"- {k}: {f(c)}" for k, c in out["H8"][s]["secondary"].items()] + [""]
    for roster, block in out["S1"].items():
        lines += [f"## S1 · complete contamination-free set, {roster} (v4 + v5)", ""]
        lines += V4R.method_table(block["methods"], "Accuracy and cost")
        for k, c in block["contrasts"].items():
            extra = (f"; Pocock two-look boundary {c['pocock_alpha']}: "
                     f"{'significant' if c['significant_at_pocock'] else 'not significant'}") if "pocock_alpha" in c else ""
            lines.append(f"- {k}: {f(c)}{extra}")
        lines.append("")
    lines += ["## S3 · dollars per correct answer", ""]
    for name, methods in out["S3"].items():
        lines.append(f"**{name}**: " + "; ".join(f"{k} ${v['usd_per_correct']:.4f}" for k, v in
                                                 sorted(methods.items(), key=lambda kv: kv[1]["usd_per_correct"])))
        lines.append("")
    if "S4" in out:
        lines += ["## S4 · same-roster MoA vs SAT's reported MoA (descriptive)", "",
                  f"Ours (E1, Claude Haiku 4.5 aggregating the SAT roster): {V4R.pct(out['S4']['ours_MoA_E1'])}; "
                  f"SAT reports {V4R.pct(out['S4']['sat_reported_MoA'])} with its original roster.", ""]
    s5 = out.get("S5")
    if s5:
        lines += ["## S5 · non-inferiority to debate (all cells where debate ran, v4 + v5)", "",
                  f"- mu* − debate: {f(s5)}",
                  f"- margin {100 * s5['margin']:+.0f} pp: {'NON-INFERIOR' if s5['non_inferior'] else 'not shown'} "
                  f"(95% lower bound {100 * s5['low']:+.1f} pp)",
                  f"- MathMage cost relative to debate: SAT roster {s5['cost_ratio_mathmage_over_debate']['SAT roster']:.2f}, "
                  f"open roster {s5['cost_ratio_mathmage_over_debate']['open roster']:.2f}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def figure(out, draft):
    from mathmage.analysis import style
    style.use(publication=not draft, base_size=8.0)
    import matplotlib.pyplot as plt
    P = style.P
    rows = []
    for key, label in (("H6", "SAT · April"), ("H7", "open · April + 2026")):
        for k, c in out[key]["contrasts"].items():
            if c:
                rows.append((f"{label} · {k.replace('mu* vs ', '')}", c))
    for s in ("E1", "E2"):
        for k, c in out["H8"][s]["contrasts"].items():
            if c:
                rows.append((f"{s} · {k.replace('mu* vs ', '')}", c))
    if out.get("S5"):
        rows.append(("all debate cells (S5)", out["S5"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(rows) + 0.8))
    y = np.arange(len(rows))[::-1]
    for yy, (label, c) in zip(y, rows):
        significant = c.get("holm_p", c["p"]) < 0.05 if "holm_p" in c else c.get("non_inferior", False)
        color = P["purple"] if significant else P["lav1"]
        ax.errorbar(100 * c["estimate"], yy, xerr=[[100 * (c["estimate"] - c["low"])], [100 * (c["high"] - c["estimate"])]],
                    fmt="D", ms=3.6, color=color, elinewidth=1, capsize=1.5)
    ax.axvline(0, color=P["faint"], lw=.7)
    ax.axvline(-100 * MARGIN, color=P["pink_deep"], lw=.7, ls=":")
    ax.set_yticks(y, [label.replace("·", r"$\cdot$") if not draft else label for label, _ in rows], fontsize=6.6)
    ax.set_xlabel("MathMage at mu* minus baseline (points)" if draft else r"MathMage at $\mu^\star$ minus baseline (points)")
    ax.grid(axis="x", alpha=.5)
    fig.tight_layout()
    style.save(fig, "v5_contrasts", OUT_F)
    plt.close(fig)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    out = analyze(args.samples)
    (OUT_T / "v5_summary.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    write(out, OUT_T / "v5_results.md")
    figure(out, args.draft)
    print((OUT_T / "v5_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
