"""v4 follow-up analysis: same-roster baselines and post-cutoff contests (read-only).

    python analysis/v4_results.py            # publication figures (LuaLaTeX, Latin Modern TrueType)
    python analysis/v4_results.py --draft    # fast previews

Implements the pre-registered analysis plan in runs/live_20261018/program_v4/plan.json:
  H1  SAT roster (237 problems): MathMage at dev mu* vs DeepSeek-V3 SC@10, debate, USC
  H2  open roster (580 evaluations): mu* and maximal vs debate and USC; maximal vs SC at matched cost
  H3  contamination-free sets on both rosters: mu* vs majority vote and the available baselines
  H4  the 153 new contest problems on both rosters
  H5  (if run) our o3-mini SC@10 vs SAT's reported self-consistency
Primary contrasts are Holm-corrected together. Outcomes that are expectations (SC@k for k < 10,
ties broken uniformly at random) enter the bootstrap as fractions. Intervals and p-values use the
same family-clustered, crossed-seed bootstrap as the v3 analysis.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import results as R  # noqa: E402
from mathmage.analysis import style  # noqa: E402
from mathmage.analysis.frontier import frontier, pareto, stop_index  # noqa: E402
from mathmage.analysis.metrics import macro_accuracy  # noqa: E402
from mathmage.analysis.stats import holm  # noqa: E402
from mathmage.data.registry import Registry, load_registry  # noqa: E402
from mathmage.grading.equiv import cluster  # noqa: E402

V4 = R.RUNS / "program_v4"
EXT = REPO / "data" / "registry_ext"
LABEL = {**R.BENCH, "hmmt_nov_2025": "HMMT Nov 2025", "brumo_2025": "BRUMO 2025", "cmimc_2025": "CMIMC 2025",
         "smt_2025": "SMT 2025"}
SAT_REPORTED_SC = {"aime_2026": 0.534, "hmmt_feb_2026": 0.319}
FRESH_E1 = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025")
FRESH_E2 = ("aime_2025", "aime_2026", "hmmt_feb_2026", "hmmt_nov_2025", "brumo_2025", "cmimc_2025", "smt_2025")


class Combined:
    """Item lookup across the frozen registry and the v4 extension registry."""

    def __init__(self):
        self.base = load_registry()
        self.ext = Registry.from_frame(pd.read_parquet(EXT / "items.parquet")) if (EXT / "items.parquet").exists() else None

    def __getitem__(self, item_id):
        return self.base[item_id] if item_id in self.base else self.ext[item_id]


# ----------------------------------------------------------------------------- loading
def graded(job):
    directory = V4 / job / "graded"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))] if directory.exists() else []


def mathmage_rows(jobs):
    rows = []
    for job in jobs:
        for path in sorted((V4 / job / "graded").glob("*.json")) if (V4 / job / "graded").exists() else []:
            row = json.loads(path.read_text(encoding="utf-8"))
            record = json.loads((V4 / job / "agent" / path.name).read_text(encoding="utf-8"))
            row["_calls"] = [{k: c.get(k) for k in ("phase", "role", "model_key", "resource_cost_usd")}
                             for c in record.get("meta", {}).get("calls", [])]
            row["_candidates"] = [{k: c.get(k) for k in ("id", "phase", "model_key", "answer", "valid", "evidence",
                                                         "parent_id")} for c in record.get("candidates", [])]
            row["_steps"] = [{k: s.get(k) for k in ("best_candidate", "best_answer", "confidence", "chosen")}
                             for s in record.get("meta", {}).get("anytime", [])]
            row["_trace"] = [t for t in record.get("trace", []) if t.get("event") in ("policy_action", "judge", "hint", "review")]
            rows.append(row)
    return rows


def unit(row, correct, cost):
    return {"benchmark": row["benchmark"], "item_id": row["item_id"], "family_id": row.get("family_id", row["item_id"]),
            "seed": int(row.get("seed", 0)), "final_correct": float(correct), "cost_usd": float(cost)}


# ----------------------------------------------------------------------------- aggregation
def plurality(answers, correct, fmt, subset=None):
    """Expected plurality correctness with ties broken uniformly at random."""
    index = [i for i in (subset if subset is not None else range(len(answers))) if answers[i]]
    if not index:
        return 0.0
    labels = cluster([answers[i] for i in index], fmt)
    right = {lab for i, lab in zip(index, labels) if correct[i]}
    counts = Counter(labels)
    best = max(counts.values())
    tops = [lab for lab, n in counts.items() if n == best]
    return sum(lab in right for lab in tops) / len(tops)


def sc_outcomes(rows, registry, ks):
    """SC@k per unit: exact expectation over all k-subsets of the K saved samples."""
    out = {k: [] for k in ks}
    for row in rows:
        answers = [s["answer"] for s in row["samples"]]
        correct = [s["correct"] for s in row["samples"]]
        fmt = registry[row["item_id"]].answer_format
        index = [i for i, a in enumerate(answers) if a]
        labels = dict(zip(index, cluster([answers[i] for i in index], fmt))) if index else {}
        right = {labels[i] for i in index if correct[i]}
        per_sample = float(np.mean([s["resource_cost_usd"] for s in row["samples"]]))
        for k in ks:
            values = []
            for subset in itertools.combinations(range(len(answers)), k):
                labs = [labels[i] for i in subset if i in labels]
                if not labs:
                    values.append(0.0)
                    continue
                counts = Counter(labs)
                best = max(counts.values())
                tops = [lab for lab, n in counts.items() if n == best]
                values.append(sum(lab in right for lab in tops) / len(tops))
            out[k].append(unit(row, np.mean(values), k * per_sample))
    return out


def debate_outcomes(rows, registry):
    return [unit(r, plurality([s["answer"] for s in r["samples"]], [s["correct"] for s in r["samples"]],
                              registry[r["item_id"]].answer_format), r["resource_cost_usd"]) for r in rows]


def usc_outcomes(rows):
    return [unit(r, r["usc_correct"], r["resource_cost_usd"]) for r in rows]


def pooled_sc(rows, registry, ks, draws=300, seed=0):
    """Open roster: SC@k per evaluation from the item's pooled stage-0 samples over all seeds.

    Answers are clustered once per item; subsets only count cluster labels.
    """
    rng = np.random.default_rng(seed)
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    out = {k: [] for k in ks}
    for item_id, item_rows in by_item.items():
        fmt = registry[item_id].answer_format
        pool = [(c.get("answer"), bool(r["candidate_grades"].get(c["id"], {}).get("correct")))
                for r in item_rows for c in r["_candidates"] if c["id"] in ("s0-0", "s0-1", "s0-2")]
        cost = float(np.mean([c["resource_cost_usd"] for r in item_rows for c in r["_calls"] if c["phase"] == "stage0"]))
        index = [i for i, (a, _) in enumerate(pool) if a]
        labels = dict(zip(index, cluster([pool[i][0] for i in index], fmt))) if index else {}
        right = {labels[i] for i in index if pool[i][1]}

        def vote(subset):
            labs = [labels[i] for i in subset if i in labels]
            if not labs:
                return 0.0
            counts = Counter(labs)
            best = max(counts.values())
            tops = [lab for lab, n in counts.items() if n == best]
            return sum(lab in right for lab in tops) / len(tops)
        for k in ks:
            if k > len(pool):            # e.g. single-seed MATH-500 items: SC@k is undefined, not SC@3 at k's cost
                continue
            value = (float(np.mean([vote(rng.choice(len(pool), k, replace=False)) for _ in range(draws)]))
                     if k < len(pool) else vote(range(len(pool))))
            out[k] += [unit(r, value, k * cost) for r in item_rows]
    return out


def mathmage_methods(rows, study, artifact, registry):
    stage0, _, _ = R.stage0_analysis(rows, artifact["posterior_params"], registry)
    methods = R.method_rows(study, rows, float(artifact["mu_star"]), stage0)
    return {k: [{**m, "final_correct": float(m["final_correct"])} for m in v] for k, v in methods.items()}


# ----------------------------------------------------------------------------- statistics
def _macro(values, benches, n_benchmarks):
    sums = np.bincount(benches, weights=values, minlength=n_benchmarks)
    counts = np.bincount(benches, minlength=n_benchmarks)
    present = counts > 0
    return float(np.mean(sums[present] / counts[present]))


def bootstrap(rows, field, samples, seed=0):
    """Macro mean of ``field`` with a family-clustered, crossed-seed bootstrap (vectorized).

    Same scheme as the v3 analysis: families are resampled within benchmark-membership
    strata, and seeds are resampled once per draw for each benchmark's seed schedule.
    """
    names = sorted({r["benchmark"] for r in rows})
    code = {b: i for i, b in enumerate(names)}
    values = np.array([float(r[field]) for r in rows])
    benches = np.array([code[r["benchmark"]] for r in rows])
    cells, families, schedules = defaultdict(list), defaultdict(set), defaultdict(set)
    for i, r in enumerate(rows):
        cells[(r["family_id"], r["benchmark"], int(r["seed"]))].append(i)
        families[r["family_id"]].add(r["benchmark"])
        schedules[r["benchmark"]].add(int(r["seed"]))
    cells = {k: np.array(v) for k, v in cells.items()}
    schedules = {b: tuple(sorted(s)) for b, s in schedules.items()}
    strata = defaultdict(list)
    for family, benchmarks in families.items():
        strata[tuple(sorted(benchmarks))].append(family)
    strata = sorted(strata.items())
    panels = sorted(set(schedules.values()))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for d in range(samples):
        seed_draws = {panel: [int(s) for s in rng.choice(panel, len(panel), replace=True)] for panel in panels}
        parts = []
        for stratum, members in strata:
            for j in rng.integers(0, len(members), len(members)):
                family = members[j]
                for benchmark in stratum:
                    for s in seed_draws[schedules[benchmark]]:
                        cell = cells.get((family, benchmark, s))
                        if cell is not None:
                            parts.append(cell)
        index = np.concatenate(parts)
        draws[d] = _macro(values[index], benches[index], len(names))
    return _macro(values, benches, len(names)), draws


def summarize(rows, samples):
    estimate, draws = bootstrap(rows, "final_correct", samples)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"macro": estimate, "low": float(lo), "high": float(hi), "cost": float(np.mean([r["cost_usd"] for r in rows])),
            "n": len(rows), "by_benchmark": {b: float(np.mean([r["final_correct"] for r in rows if r["benchmark"] == b]))
                                             for b in sorted({r["benchmark"] for r in rows})}}


def contrast(left, right, samples):
    """Paired macro difference on the shared units, with a two-sided bootstrap p-value."""
    index = {(r["item_id"], r["seed"]): r for r in right}
    rows = [{**r, "delta": r["final_correct"] - index[(r["item_id"], r["seed"])]["final_correct"]}
            for r in left if (r["item_id"], r["seed"]) in index]
    if not rows:
        return None
    estimate, draws = bootstrap(rows, "delta", samples)
    lo, hi = np.quantile(draws, [.025, .975])
    p = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    return {"estimate": estimate, "low": float(lo), "high": float(hi), "p": p, "n": len(rows), "samples": samples,
            "wins": sum(r["delta"] > 0 for r in rows), "losses": sum(r["delta"] < 0 for r in rows)}


def subset(rows, benchmarks):
    return [r for r in rows if r["benchmark"] in benchmarks]


# ----------------------------------------------------------------------------- analysis
def analyze(samples, parts=("H1", "H2", "H3", "H4", "H5")):
    registry = Combined()
    e1, e2 = R.STUDIES
    a1 = json.loads(e1.artifact.read_text(encoding="utf-8"))
    a2 = json.loads(e2.artifact.read_text(encoding="utf-8"))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    m1 = mathmage_methods(rows1, e1, a1, registry)
    m2 = mathmage_methods(rows2, e2, a2, registry)
    needs_new = bool({"H3", "H4"} & set(parts))
    new1 = mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"]) if needs_new else []
    new2 = mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"]) if needs_new else []
    n1 = mathmage_methods(new1, e1, a1, registry) if new1 else {}
    n2 = mathmage_methods(new2, e2, a2, registry) if new2 else {}
    ks = [1, 2, 3, 5, 7, 10]
    out = {"status": {job: len(graded(job)) for job in
                      ["e1_sc", "e1_debate", "e1_usc", "e2_debate", "e2_usc", "new_e1_mathmage_nov", "new_e1_sc_nov",
                       "new_e1_debate_nov", "new_e1_usc_nov", "new_e1_mathmage_apr", "new_e2_mathmage_nov",
                       "new_e2_debate_nov", "new_e2_usc_nov", "new_e2_mathmage_apr", "e1_sat_sc"]}}

    # ---- H1: SAT roster, original 237 problems
    sc1 = sc_outcomes(graded("e1_sc"), registry, ks)
    deb1, usc1 = debate_outcomes(graded("e1_debate"), registry), usc_outcomes(graded("e1_usc"))
    h1 = {"vote": m1["vote"], "weighted vote + program": m1["weighted vote + program"], "posterior": m1["posterior"],
          "mu*": m1["mu*"], "maximal": m1["maximal"], "debate": deb1, "USC": usc1,
          **{f"SC@{k} (DeepSeek-V3)": sc1[k] for k in ks}}
    out["H1"] = {"methods": {k: summarize(v, samples) for k, v in h1.items() if v},
                 "contrasts": {f"mu* vs {b}": contrast(m1["mu*"], h1[b], samples)
                               for b in ["SC@10 (DeepSeek-V3)", "debate", "USC"] if h1[b]},
                 "secondary": {f"{a} vs {b}": contrast(m1[a], h1[b], samples) for a in ["posterior", "maximal"]
                               for b in ["SC@10 (DeepSeek-V3)", "debate", "USC"] if h1[b]}}

    # ---- H2: open roster, original 580 evaluations
    deb2, usc2 = debate_outcomes(graded("e2_debate"), registry), usc_outcomes(graded("e2_usc"))
    sc2 = pooled_sc(rows2, registry, [1, 3, 5, 8, 12, 24])
    h2 = {"vote": m2["vote"], "weighted vote + program": m2["weighted vote + program"], "posterior": m2["posterior"],
          "mu*": m2["mu*"], "maximal": m2["maximal"], "debate": deb2, "USC": usc2,
          **{f"SC@{k} (pooled seeds)": v for k, v in sc2.items()}}
    matched = min(sc2, key=lambda k: abs(np.mean([r["cost_usd"] for r in sc2[k]]) -
                                         np.mean([r["cost_usd"] for r in m2["maximal"]])))
    out["H2"] = {"methods": {k: summarize(v, samples) for k, v in h2.items() if v},
                 "matched_sc_k": matched,
                 "contrasts": {**{f"{a} vs {b}": contrast(m2[a], h2[b], samples) for a in ["mu*", "maximal"]
                                  for b in ["debate", "USC"] if h2[b]},
                               f"maximal vs SC@{matched} (matched cost)": contrast(m2["maximal"], sc2[matched], samples)}}

    # ---- H3: contamination-free sets
    if "H3" in parts:
        fresh1 = {name: subset(rows, FRESH_E1) for name, rows in {
            "vote": m1["vote"] + n1.get("vote", []), "posterior": m1["posterior"] + n1.get("posterior", []),
            "mu*": m1["mu*"] + n1.get("mu*", []), "maximal": m1["maximal"] + n1.get("maximal", []),
            "weighted vote + program": m1["weighted vote + program"] + n1.get("weighted vote + program", []),
            "SC@10 (DeepSeek-V3)": sc1[10] + sc_outcomes(graded("new_e1_sc_nov"), registry, [10])[10],
            "debate": deb1 + debate_outcomes(graded("new_e1_debate_nov"), registry),
            "USC": usc1 + usc_outcomes(graded("new_e1_usc_nov"))}.items()}
        rows2_all = rows2 + new2
        sc2_all = pooled_sc(rows2_all, registry, [3, 5, 8]) if new2 else {}
        fresh2 = {name: subset(rows, FRESH_E2) for name, rows in {
            "vote": m2["vote"] + n2.get("vote", []), "posterior": m2["posterior"] + n2.get("posterior", []),
            "mu*": m2["mu*"] + n2.get("mu*", []), "maximal": m2["maximal"] + n2.get("maximal", []),
            "weighted vote + program": m2["weighted vote + program"] + n2.get("weighted vote + program", []),
            **{f"SC@{k} (pooled seeds)": v for k, v in sc2_all.items()},
            "debate": deb2 + debate_outcomes(graded("new_e2_debate_nov"), registry),
            "USC": usc2 + usc_outcomes(graded("new_e2_usc_nov"))}.items()}
        out["H3"] = {"e1": {"benchmarks": sorted({r["benchmark"] for r in fresh1["mu*"]}),
                            "methods": {k: summarize(v, samples) for k, v in fresh1.items() if v},
                            "contrasts": {f"mu* vs {b}": contrast(fresh1["mu*"], fresh1[b], samples)
                                          for b in ["vote", "SC@10 (DeepSeek-V3)", "debate", "USC"] if fresh1[b]}},
                     "e2": {"benchmarks": sorted({r["benchmark"] for r in fresh2["mu*"]}),
                            "methods": {k: summarize(v, samples) for k, v in fresh2.items() if v},
                            "contrasts": {f"mu* vs {b}": contrast(fresh2["mu*"], fresh2[b], samples)
                                          for b in ["vote", "SC@5 (pooled seeds)", "debate", "USC"] if fresh2.get(b)}}}

    # ---- H4: the 153 new problems
    out["H4"] = {}
    for key, methods, study in [("e1", n1, e1), ("e2", n2, e2)] if "H4" in parts else []:
        if not methods:
            continue
        names = (list(study.members.values()) if not study.homogeneous else ["single sample (mean of 3)"]) + \
            ["program", "vote", "vote + program", "weighted vote + program", "posterior", "mu*", "maximal",
             "oracle (stage 0)", "oracle (final pool)"]
        out["H4"][key] = {"methods": {n: summarize(methods[n], samples) for n in names if methods.get(n)},
                          "contrasts": {f"{a} vs vote": contrast(methods[a], methods["vote"], samples)
                                        for a in ["posterior", "mu*", "maximal"]}}

    # ---- H5: SAT's self-consistency protocol, re-run
    sat = graded("e1_sat_sc") if "H5" in parts else []
    if sat:
        sc_sat = sc_outcomes(sat, registry, [1, 10])
        out["H5"] = {b: {"ours_SC@10": float(np.mean([r["final_correct"] for r in sc_sat[10] if r["benchmark"] == b])),
                         "ours_SC@1": float(np.mean([r["final_correct"] for r in sc_sat[1] if r["benchmark"] == b])),
                         "sat_reported": SAT_REPORTED_SC[b]} for b in SAT_REPORTED_SC}

    # ---- Holm across the pre-registered primary contrasts (H1-H3)
    # The pre-registered family is H1-H3 together; an interim H1/H2-only run reports unadjusted p.
    primary = ([(h, k) for h in ("H1", "H2") for k in out[h]["contrasts"]] +
               [("H3", (r, k)) for r in ("e1", "e2") for k in out["H3"][r]["contrasts"]]) if "H3" in out else []
    pvalues = []
    for h, k in primary:
        c = out[h]["contrasts"][k] if h != "H3" else out["H3"][k[0]]["contrasts"][k[1]]
        pvalues.append(c["p"] if c else 1.0)
    for (h, k), adjusted in zip(primary, holm(pvalues)):
        c = out[h]["contrasts"][k] if h != "H3" else out["H3"][k[0]]["contrasts"][k[1]]
        if c:
            c["holm_p"] = adjusted
    out["_frontier"] = {"e1": frontier(rows1), "e2": frontier(rows2)}
    out["_curves"] = {"e1_sc": {k: (float(np.mean([r["cost_usd"] for r in sc1[k]])) if sc1[k] else None,
                                    macro_accuracy(sc1[k]) if sc1[k] else None) for k in ks},
                      "e2_sc": {k: (float(np.mean([r["cost_usd"] for r in v])), macro_accuracy(v)) for k, v in sc2.items()}}
    # Figure panel for the open roster on one population: AIME 2024/25 (8 seeds), where SC@k exists for every k.
    aime = lambda rows: [r for r in rows if r["benchmark"] != "math500"]  # noqa: E731
    out["_e2_aime"] = {"frontier": frontier(aime(rows2)),
                       "methods": {name: summarize(aime(h2[name]), min(samples, 500))
                                   for name in ["vote", "weighted vote + program", "debate", "USC", "mu*"] if h2.get(name)},
                       "sc": {k: (float(np.mean([r["cost_usd"] for r in aime(v)])), macro_accuracy(aime(v)))
                              for k, v in sc2.items() if aime(v)}}
    return out


# ----------------------------------------------------------------------------- output
def pct(x):
    return f"{100 * x:.1f}"


def fmt_p(p, samples):
    return f"< {1 / samples:.1g}" if p == 0 else f"= {p:.3g}"


def fmt_contrast(c):
    if not c:
        return "—"
    samples = c.get("samples", 2000)
    return (f"{100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], {c['wins']}/{c['losses']} "
            f"(n = {c['n']}), p {fmt_p(c['p'], samples)}" + (f", Holm p {fmt_p(c['holm_p'], samples)}" if "holm_p" in c else ""))


def method_table(methods, title):
    benches = sorted({b for t in methods.values() for b in t["by_benchmark"]}, key=lambda b: list(LABEL).index(b)
                     if b in LABEL else 99)
    lines = [f"### {title}", "", "| Method | " + " | ".join(LABEL.get(b, b) for b in benches) +
             " | Macro [95% CI] | $/problem | n |", "|---" * (len(benches) + 4) + "|"]
    for name, t in methods.items():
        lines.append(f"| {name} | " + " | ".join(pct(t["by_benchmark"][b]) if b in t["by_benchmark"] else "--" for b in benches)
                     + f" | {pct(t['macro'])} [{pct(t['low'])}, {pct(t['high'])}] | {t['cost']:.4f} | {t['n']} |")
    return lines + [""]


def write(out, path):
    lines = ["# v4 follow-up results (pre-registered; see runs/live_20261018/program_v4/PREREGISTRATION.md)", "",
             "Units graded per job: " + ", ".join(f"{k} {v}" for k, v in out["status"].items()), ""]
    lines += ["## H1 · SAT roster, 237 problems: same-roster baselines", ""]
    lines += method_table(out["H1"]["methods"], "Accuracy and cost")
    lines += ["Primary contrasts (paired macro difference, pp; 95% clustered CI; wins/losses):"] + \
             [f"- {k}: {fmt_contrast(c)}" for k, c in out["H1"]["contrasts"].items()] + \
             ["", "Secondary:"] + [f"- {k}: {fmt_contrast(c)}" for k, c in out["H1"]["secondary"].items()] + [""]
    lines += ["## H2 · open roster, 580 evaluations: same-roster baselines", ""]
    lines += method_table(out["H2"]["methods"], "Accuracy and cost (SC uses each problem's 24 pooled seed samples)")
    lines += ["Primary contrasts:"] + [f"- {k}: {fmt_contrast(c)}" for k, c in out["H2"]["contrasts"].items()] + [""]
    for roster, title in ([("e1", "SAT roster"), ("e2", "open roster")] if "H3" in out else []):
        h = out["H3"][roster]
        lines += [f"## H3 · contamination-free set, {title} ({', '.join(LABEL.get(b, b) for b in h['benchmarks'])})", ""]
        lines += method_table(h["methods"], "Accuracy and cost")
        lines += ["Primary contrasts:"] + [f"- {k}: {fmt_contrast(c)}" for k, c in h["contrasts"].items()] + [""]
    for roster, title in [("e1", "SAT roster"), ("e2", "open roster, 8 seeds")]:
        if roster in out["H4"]:
            lines += [f"## H4 · the 153 new contest problems, {title}", ""]
            lines += method_table(out["H4"][roster]["methods"], "Accuracy and cost")
            lines += [f"- {k}: {fmt_contrast(c)}" for k, c in out["H4"][roster]["contrasts"].items()] + [""]
    if "H5" in out:
        lines += ["## H5 · SAT's self-consistency protocol (o3-mini, K = 10) re-run in our harness", "",
                  "| Benchmark | ours SC@1 | ours SC@10 | SAT reported SC@10 |", "|---|---|---|---|"]
        lines += [f"| {LABEL[b]} | {pct(v['ours_SC@1'])} | {pct(v['ours_SC@10'])} | {pct(v['sat_reported'])} |"
                  for b, v in out["H5"].items()] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_baselines(out, figures):
    import matplotlib.pyplot as plt
    P, M = style.P, style.METHOD
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    panels = [(axes[0], out["H1"], out["_frontier"]["e1"], out["_curves"]["e1_sc"], "E1 · SAT roster (237 problems)"),
              (axes[1], out["_e2_aime"], out["_e2_aime"]["frontier"], out["_e2_aime"]["sc"],
               "E2 · open roster, AIME 2024/25 (60 problems × 8 seeds)")]
    for ax, h, points, sc_curve, title in panels:
        front = pareto(points)
        ax.plot([p["cost_usd"] for p in front], [p["accuracy"] for p in front], color=M["mathmage"], lw=1.5,
                label="MathMage anytime frontier", zorder=4)
        curve = [(c, a) for c, a in sc_curve.values() if c is not None]
        if curve:
            ax.plot([c for c, _ in curve], [a for _, a in curve], color=P["lav2"], marker="o", ms=2.6, lw=1.1,
                    label="self-consistency (SC@k)")
        for name, marker, color, label in [("vote", "P", M["member_vote"], "majority vote"),
                                           ("weighted vote + program", "X", P["cyan_deep"], "weighted vote + program"),
                                           ("debate", "s", P["pink"], "debate (1 round)"),
                                           ("USC", "^", P["peri_outer"], "universal self-consistency"),
                                           ("mu*", "*", P["pink_deep"], r"MathMage at dev $\mu^\star$")]:
            t = h["methods"].get(name)
            if t:
                ax.errorbar(t["cost"], t["macro"], yerr=[[t["macro"] - t["low"]], [t["high"] - t["macro"]]], fmt=marker,
                            ms=6.5 if marker == "*" else 4.5, color=color, mec=P["ink"], mew=.4, elinewidth=.7, capsize=1.5,
                            zorder=5, label=label)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("mean cost per problem (USD, log scale)")
        ax.grid(True, color=P["grid"], lw=.5, which="both")
    axes[0].set_ylabel("macro accuracy")
    handles, labels = [], []
    for ax in axes:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.2, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.subplots_adjust(bottom=.3, top=.9, left=.08, right=.985, wspace=.22)
    style.save(fig, "v4_baselines", figures)
    plt.close(fig)


def fig_fresh(out, figures):
    import matplotlib.pyplot as plt
    P, M = style.P, style.METHOD
    order = [("vote", M["member_vote"]), ("SC@10 (DeepSeek-V3)", P["lav2"]), ("SC@5 (pooled seeds)", P["lav2"]),
             ("debate", P["pink"]), ("USC", P["peri_outer"]), ("weighted vote + program", P["cyan_deep"]),
             ("posterior", M["mathmage_light"]), ("mu*", P["pink_deep"]), ("maximal", M["mathmage"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    notes = []
    for ax, roster, title in [(axes[0], "e1", "SAT roster"), (axes[1], "e2", "open roster")]:
        methods = out["H3"][roster]["methods"]
        names = [(n, c) for n, c in order if n in methods]
        full = max(methods[n]["n"] for n, _ in names)
        labels = []
        for i, (name, color) in enumerate(names):
            t = methods[name]
            partial = t["n"] < full                    # computed on a subset of the post-cutoff problems
            ax.bar(i, t["macro"], .7, color=color, edgecolor=P["ink"], lw=.3, hatch="////" if partial else None)
            ax.errorbar(i, t["macro"], yerr=[[t["macro"] - t["low"]], [t["high"] - t["macro"]]], color=P["ink"],
                        elinewidth=.6, capsize=1.5)
            label = (name.replace(" (DeepSeek-V3)", "").replace(" (pooled seeds)", "")
                     .replace("weighted vote + program", "weighted vote").replace("mu*", r"$\mu^\star$"))
            labels.append(label + (r"$^\dagger$" if partial else ""))
            if partial:
                subset = ", ".join(LABEL.get(b, b) for b in t["by_benchmark"])
                note = rf"$^\dagger${title}: {subset} only ({t['n']} of {full} evaluations)"
                if note not in notes:
                    notes.append(note)
        ax.set_xticks(range(len(names)), labels, rotation=35, ha="right", fontsize=6)
        ax.set_title(f"{title}: post-cutoff contests", fontsize=8, loc="left")
        ax.grid(True, axis="y", color=P["grid"], lw=.5)
    axes[0].set_ylabel("macro accuracy")
    fig.tight_layout()
    if notes:
        fig.subplots_adjust(bottom=.36)
        fig.text(.01, .01, "   ".join(notes), fontsize=5.6, color=P["muted"], ha="left", va="bottom")
    style.save(fig, "v4_postcutoff", figures)
    plt.close(fig)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--parts", default="H1,H2,H3,H4,H5", help="hypotheses to compute, e.g. H1,H2")
    args = parser.parse_args()
    tables, figures = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    style.use(publication=not args.draft)
    out = analyze(args.samples, tuple(args.parts.split(",")))
    write(out, tables / "v4_results.md")
    (tables / "v4_summary.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    fig_baselines(out, figures)
    if "H3" in out:
        fig_fresh(out, figures)
    print((tables / "v4_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
