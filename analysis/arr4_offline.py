"""Offline analyses for the ARR revision (read-only; no API calls, no new samples).

    python analysis/arr4_offline.py            # tables + figure (LuaLaTeX)
    python analysis/arr4_offline.py --draft    # fast figure preview

All inputs are sealed v3/v4 results. Hyperparameters are the original papers' defaults, fixed before
anything was computed (2026-10-20):

  X1  Adaptive compute from the saved sample pools (expectation over 200 random sample orders per problem)
      Adaptive-Consistency (Aggarwal et al., 2023): Beta stopping rule on the two leading answers,
        stop once P(p1 > p2) >= 0.95; at most the pool size.
      Early-Stopping Self-Consistency (Li et al., 2024): windows of 5; stop at the first unanimous window;
        otherwise majority over everything drawn.
      Pools: SAT roster, the K = 10 DeepSeek-V3 SC samples (v4); open roster, the 24 pooled stage-0
      samples per problem (3 members x 8 seeds).
  X2  Mixture-of-thought cascade (Yue et al., 2024) from saved stage-0 answers: accept the cheap model when
      its chain-of-thought answer and the program-of-thought answer agree, otherwise escalate.
  X3  MathMage vs debate pooled over every roster-benchmark cell where debate ran (exploratory: not part of a
      pre-registered family), with families resampled jointly across rosters; and the problems needed for 80%
      power at the observed effect.
  X4  Dollars per correct answer on every evaluation set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import betainc
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v4_results as V4R  # noqa: E402

R = V4R.R
OUT_T, OUT_F = REPO / "results/tables", REPO / "results/figures"
ORDERS = 200
AC_THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
ESC_WINDOWS = (2, 3, 4, 5)
SAMPLES = 2000


# ----------------------------------------------------------------------------- pools
def pool_from_sc(rows, registry):
    """SAT roster: each problem's 10 DeepSeek-V3 samples as (label, correct) with the mean per-sample cost."""
    pools = {}
    for r in rows:
        answers = [s["answer"] for s in r["samples"]]
        index = [i for i, a in enumerate(answers) if a]
        labels = dict(zip(index, V4R.cluster([answers[i] for i in index], registry[r["item_id"]].answer_format))) \
            if index else {}
        pools[(r["item_id"], int(r.get("seed", 0)))] = {
            "labels": [labels.get(i, -1 - i) for i in range(len(answers))],
            "correct": [bool(s["correct"]) for s in r["samples"]],
            "cost": float(np.mean([s["resource_cost_usd"] for s in r["samples"]]))}
    return pools


def pool_from_stage0(rows, registry):
    """Open roster: the 24 pooled stage-0 member samples of each problem (3 members x 8 seeds)."""
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    pools = {}
    for item_id, item_rows in by_item.items():
        pairs = [(c.get("answer"), bool(r["candidate_grades"].get(c["id"], {}).get("correct")))
                 for r in item_rows for c in r["_candidates"] if c["id"] in ("s0-0", "s0-1", "s0-2")]
        index = [i for i, (a, _) in enumerate(pairs) if a]
        labels = dict(zip(index, V4R.cluster([pairs[i][0] for i in index], registry[item_id].answer_format))) \
            if index else {}
        cost = float(np.mean([c["resource_cost_usd"] for r in item_rows for c in r["_calls"] if c["phase"] == "stage0"]))
        pool = {"labels": [labels.get(i, -1 - i) for i in range(len(pairs))], "correct": [p[1] for p in pairs],
                "cost": cost}
        for r in item_rows:
            pools[(item_id, int(r.get("seed", 0)))] = pool
    return pools


# ----------------------------------------------------------------------------- policies
def majority(labels, correct):
    """Expected correctness of the plurality answer with ties broken uniformly at random."""
    valid = [lab for lab in labels if lab >= 0]
    if not valid:
        return 0.0
    right = {lab for lab, ok in zip(labels, correct) if ok and lab >= 0}
    counts = Counter(valid)
    best = max(counts.values())
    tops = [lab for lab, n in counts.items() if n == best]
    return sum(lab in right for lab in tops) / len(tops)


def adaptive_consistency(labels, correct, threshold):
    for n in range(1, len(labels) + 1):
        counts = Counter(lab for lab in labels[:n] if lab >= 0).most_common(2)
        v1 = counts[0][1] if counts else 0
        v2 = counts[1][1] if len(counts) > 1 else 0
        if v1 and 1 - betainc(v1 + 1, v2 + 1, 0.5) >= threshold:      # P(p1 > p2) under Beta(v1 + 1, v2 + 1)
            return majority(labels[:n], correct[:n]), n
    return majority(labels, correct), len(labels)


def early_stopping(labels, correct, window):
    n = 0
    while n < len(labels):
        chunk = labels[n:n + window]
        n = min(len(labels), n + window)
        if len(chunk) == window and chunk[0] >= 0 and all(lab == chunk[0] for lab in chunk):
            return majority(labels[:n], correct[:n]), n
    return majority(labels, correct), len(labels)


def simulate(pools, policy, parameter, seed=0):
    """Expected correctness and samples per unit over random sample orders (common orders across policies)."""
    rng = np.random.default_rng(seed)
    cache, out = {}, {}
    for key, pool in pools.items():
        ident = id(pool)
        if ident not in cache:
            values, sizes = [], []
            for _ in range(ORDERS):
                order = rng.permutation(len(pool["labels"]))
                labels = [pool["labels"][i] for i in order]
                correct = [pool["correct"][i] for i in order]
                v, n = policy(labels, correct, parameter)
                values.append(v)
                sizes.append(n)
            cache[ident] = (float(np.mean(values)), float(np.mean(sizes)))
        value, size = cache[ident]
        out[key] = {"final_correct": value, "samples": size, "cost_usd": size * pool["cost"]}
    return out


def to_units(template_rows, outcomes):
    return [{**t, "final_correct": outcomes[(t["item_id"], t["seed"])]["final_correct"],
             "cost_usd": outcomes[(t["item_id"], t["seed"])]["cost_usd"],
             "samples": outcomes[(t["item_id"], t["seed"])]["samples"]} for t in template_rows]


# ----------------------------------------------------------------------------- cascade
def cascade(rows, registry, weak, strong, tool="s0-tool"):
    """Accept the weak member when its answer and the program's agree; otherwise take the strong member's answer."""
    out = []
    for r in rows:
        fmt = registry[r["item_id"]].answer_format
        cands = {c["id"]: c for c in r["_candidates"]}
        grades = r["candidate_grades"]
        a, b = cands.get(weak, {}).get("answer"), cands.get(tool, {}).get("answer")
        agree = bool(a and b and len(set(V4R.cluster([a, b], fmt))) == 1)
        cost = R.call_cost(r, "stage0", f"member={weak[-1]}") + R.call_cost(r, "verify_forward")
        if agree:
            correct = bool(grades.get(weak, {}).get("correct"))
        elif strong == "team":
            answers = [cands.get(f"s0-{i}", {}).get("answer") for i in range(3)] + [b]
            ok = [bool(grades.get(f"s0-{i}", {}).get("correct")) for i in range(3)] + [bool(grades.get(tool, {}).get("correct"))]
            correct = tiebreak_vote(answers, ok, fmt)
            cost += sum(R.call_cost(r, "stage0", f"member={i}") for i in range(3) if f"s0-{i}" != weak)
        else:
            correct = bool(grades.get(strong, {}).get("correct"))
            cost += R.call_cost(r, "stage0", f"member={strong[-1]}")
        out.append({"benchmark": r["benchmark"], "item_id": r["item_id"], "family_id": r.get("family_id", r["item_id"]),
                    "seed": int(r.get("seed", 0)), "final_correct": float(correct), "cost_usd": float(cost),
                    "escalated": float(not agree)})
    return out


def tiebreak_vote(answers, correct, fmt):
    """Members plus program; the program (last) breaks ties, as in the calibration-free rule of the v3 analysis."""
    index = [i for i, a in enumerate(answers) if a]
    if not index:
        return 0.0
    labels = dict(zip(index, V4R.cluster([answers[i] for i in index], fmt)))
    score = defaultdict(float)
    for i in index:
        score[labels[i]] += 1.5 if i == len(answers) - 1 else 1.0
    best = max(score.values())
    tops = [lab for lab, s in score.items() if s == best]
    right = {labels[i] for i in index if correct[i]}
    return sum(lab in right for lab in tops) / len(tops)


# ----------------------------------------------------------------------------- analysis
def summarize(units):
    s = V4R.summarize(units, SAMPLES)
    s["samples"] = float(np.mean([u["samples"] for u in units])) if units and "samples" in units[0] else None
    if units and "escalated" in units[0]:
        s["escalated"] = float(np.mean([u["escalated"] for u in units]))
    return s


def sets():
    registry = V4R.Combined()
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    nov1 = V4R.mathmage_rows(["new_e1_mathmage_nov"])
    open_new = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    m1 = V4R.mathmage_methods(rows1, e1, a1, registry)
    m1n = V4R.mathmage_methods(nov1, e1, a1, registry)
    m2 = V4R.mathmage_methods(rows2, e2, a2, registry)
    m2n = V4R.mathmage_methods(open_new, e2, a2, registry)
    sc1 = V4R.graded("e1_sc") + V4R.graded("new_e1_sc_nov")
    return {"registry": registry, "rows1": rows1, "rows2": rows2, "nov1": nov1, "open_new": open_new,
            "m1": m1, "m1n": m1n, "m2": m2, "m2n": m2n, "sc1": sc1}


def adaptive_section(S):
    registry = S["registry"]
    out = {}
    fresh = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025")
    open_fresh = ("aime_2025", "aime_2026", "hmmt_feb_2026", "hmmt_nov_2025", "brumo_2025", "cmimc_2025", "smt_2025")
    sat_pools = pool_from_sc(S["sc1"], registry)
    open_pools = pool_from_stage0(S["rows2"] + S["open_new"], registry)
    panels = {
        "SAT roster · E1 suite (237)": (S["m1"], sat_pools, None),
        "SAT roster · post-cutoff (93)": ({k: V4R.subset(S["m1"][k], fresh[:2]) + S["m1n"][k] for k in ("mu*", "maximal")},
                                          sat_pools, fresh),
        "Open roster · E2 AIME (480)": ({k: V4R.subset(S["m2"][k], ("aime_2024", "aime_2025")) for k in ("mu*", "maximal")},
                                        open_pools, None),
        "Open roster · post-cutoff (1,968)": ({k: V4R.subset(S["m2"][k], ("aime_2025",)) + S["m2n"][k] for k in ("mu*", "maximal")},
                                              open_pools, open_fresh),
    }
    for name, (methods, pools, keep) in panels.items():
        template = [{k: u[k] for k in ("benchmark", "item_id", "family_id", "seed")} for u in methods["mu*"]
                    if (u["item_id"], u["seed"]) in pools]
        mm = [u for u in methods["mu*"] if (u["item_id"], u["seed"]) in pools]
        mx = [u for u in methods["maximal"] if (u["item_id"], u["seed"]) in pools]
        panel = {"n": len(template), "mu*": V4R.summarize(mm, SAMPLES), "maximal": V4R.summarize(mx, SAMPLES)}
        full = to_units(template, simulate(pools, lambda l, c, _: (majority(l, c), len(l)), None))
        panel["SC (full pool)"] = summarize(full)
        frontier = {"AC": [], "ESC": []}
        for t in AC_THRESHOLDS:
            units = to_units(template, simulate(pools, adaptive_consistency, t))
            s = summarize(units)
            frontier["AC"].append({"parameter": t, **{k: s[k] for k in ("macro", "low", "high", "cost", "samples")}})
            if t == 0.95:
                panel["Adaptive-Consistency (0.95)"] = s
                panel["contrast mu* vs AC"] = V4R.contrast(mm, units, SAMPLES)
        for w in ESC_WINDOWS:
            units = to_units(template, simulate(pools, early_stopping, w))
            s = summarize(units)
            frontier["ESC"].append({"parameter": w, **{k: s[k] for k in ("macro", "low", "high", "cost", "samples")}})
            if w == 5:
                panel["ESC (window 5)"] = s
                panel["contrast mu* vs ESC"] = V4R.contrast(mm, units, SAMPLES)
        panel["frontier"] = frontier
        out[name] = panel
    return out


def cascade_section(S):
    registry = S["registry"]
    out = {}
    e1 = {"weak DeepSeek-V3 + program, escalate to o3-mini": ("s0-2", "s0-0"),
          "weak DeepSeek-V3 + program, escalate to Claude Haiku 4.5": ("s0-2", "s0-1"),
          "weak DeepSeek-V3 + program, escalate to the team (vote + program tie-break)": ("s0-2", "team")}
    for name, (weak, strong) in e1.items():
        units = cascade(S["rows1"], registry, weak, strong)
        out[f"SAT roster · E1 · {name}"] = {**summarize(units), "contrast mu* vs cascade": V4R.contrast(S["m1"]["mu*"], units, SAMPLES)}
    units = cascade(S["rows2"], registry, "s0-0", "team")
    out["Open roster · E2 · one sample + program, escalate to the team"] = {
        **summarize(units), "contrast mu* vs cascade": V4R.contrast(S["m2"]["mu*"], units, SAMPLES)}
    out["reference"] = {"SAT mu*": V4R.summarize(S["m1"]["mu*"], SAMPLES), "open mu*": V4R.summarize(S["m2"]["mu*"], SAMPLES)}
    return out


def debate_section(S):
    """MathMage at mu* minus debate over every roster-benchmark cell where debate ran (exploratory)."""
    registry = S["registry"]

    def tag(units, roster):
        return [{**u, "benchmark": f"{roster}:{u['benchmark']}", "item_id": f"{roster}:{u['item_id']}"} for u in units]

    sat_mm = S["m1"]["mu*"] + S["m1n"]["mu*"]
    sat_deb = V4R.debate_outcomes(V4R.graded("e1_debate") + V4R.graded("new_e1_debate_nov"), registry)
    open_mm = S["m2"]["mu*"] + [u for u in S["m2n"]["mu*"] if u["benchmark"] == "hmmt_nov_2025"]
    open_deb = V4R.debate_outcomes(V4R.graded("e2_debate") + V4R.graded("new_e2_debate_nov"), registry)
    out = {"SAT roster (6 cells, 267 problems)": V4R.contrast(tag(sat_mm, "sat"), tag(sat_deb, "sat"), SAMPLES),
           "open roster (4 cells, 820 evaluations)": V4R.contrast(tag(open_mm, "open"), tag(open_deb, "open"), SAMPLES),
           "both rosters (10 cells)": V4R.contrast(tag(sat_mm, "sat") + tag(open_mm, "open"),
                                                   tag(sat_deb, "sat") + tag(open_deb, "open"), SAMPLES)}
    for name, c in out.items():
        se = (c["high"] - c["low"]) / (2 * 1.96)
        c["se"] = se
        # independent problems needed for 80% power (two-sided 0.05) at this effect, scaling the observed SE
        n_units = c["n"]
        c["n_for_80_power"] = (int(math.ceil(n_units * ((norm.ppf(.975) + norm.ppf(.8)) * se / c["estimate"]) ** 2))
                               if c["estimate"] > 0 else None)
    return out


def cost_per_correct():
    synthesis = json.loads((REPO / "results/tables/v4_summary.json").read_text(encoding="utf-8"))
    out = {}
    for key, title in (("H1", "SAT roster · E1 suite"), ("H2", "Open roster · E2 suite")):
        methods = synthesis[key]["methods"]
        out[title] = {m: {"accuracy": v["macro"], "cost": v["cost"], "usd_per_correct": v["cost"] / v["macro"]}
                      for m, v in methods.items() if v.get("macro")}
    for roster in ("e1", "e2"):
        methods = synthesis["H3"][roster]["methods"]
        title = "SAT roster · post-cutoff" if roster == "e1" else "Open roster · post-cutoff"
        out[title] = {m: {"accuracy": v["macro"], "cost": v["cost"], "usd_per_correct": v["cost"] / v["macro"]}
                      for m, v in methods.items() if v.get("macro")}
    return out


# ----------------------------------------------------------------------------- output
def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def fmt_c(c):
    if not c:
        return "--"
    p = c["p"]
    pp = "< 0.0005" if p < 1 / c["samples"] else f"{p:.3f}"
    return f"{100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], {c['wins']}/{c['losses']}, p = {pp}"


def write(result):
    lines = ["# Offline analyses for the ARR revision (no API calls)", "",
             "Generated by `python analysis/arr4_offline.py` from sealed v3/v4 results. Hyperparameters are the original "
             "papers' defaults, fixed before computing (see the script docstring). Intervals: family-clustered "
             "bootstrap (crossed with seeds on the open roster), 2,000 draws. These are exploratory analyses, outside "
             "the pre-registered families.", ""]
    lines += ["## X1 · Adaptive-compute baselines on the saved sample pools", "",
              "| Set | n | MathMage mu* | MathMage maximal | SC (full pool) | Adaptive-Consistency (0.95) | ESC (window 5) | "
              "mu* − AC | mu* − ESC |", "|---|---|---|---|---|---|---|---|---|"]
    for name, p in result["adaptive"].items():
        cell = lambda s: f"{pct(s['macro'])} @ ${s['cost']:.4f}" + (f" ({s['samples']:.1f} samples)" if s.get("samples") else "")  # noqa: E731
        lines.append(f"| {name} | {p['n']} | {cell(p['mu*'])} | {cell(p['maximal'])} | {cell(p['SC (full pool)'])} | "
                     f"{cell(p['Adaptive-Consistency (0.95)'])} | {cell(p['ESC (window 5)'])} | "
                     f"{fmt_c(p['contrast mu* vs AC'])} | {fmt_c(p['contrast mu* vs ESC'])} |")
    lines += ["", "Frontiers (accuracy @ $/problem):", ""]
    for name, p in result["adaptive"].items():
        ac = "; ".join(f"{f['parameter']}: {pct(f['macro'])} @ ${f['cost']:.4f}" for f in p["frontier"]["AC"])
        esc = "; ".join(f"w={f['parameter']}: {pct(f['macro'])} @ ${f['cost']:.4f}" for f in p["frontier"]["ESC"])
        lines += [f"- {name}: AC {ac}", f"- {name}: ESC {esc}"]
    lines += ["", "## X2 · Mixture-of-thought cascade from saved stage-0 answers", "",
              "| Cascade | Macro [95% CI] | $/problem | escalated | mu* − cascade |", "|---|---|---|---|---|"]
    for name, c in result["cascade"].items():
        if name == "reference":
            continue
        lines.append(f"| {name} | {pct(c['macro'])} [{pct(c['low'])}, {pct(c['high'])}] | ${c['cost']:.4f} | "
                     f"{100 * c['escalated']:.0f}% | {fmt_c(c['contrast mu* vs cascade'])} |")
    ref = result["cascade"]["reference"]
    lines += ["", f"Reference: MathMage mu* {pct(ref['SAT mu*']['macro'])} @ ${ref['SAT mu*']['cost']:.4f} (SAT), "
              f"{pct(ref['open mu*']['macro'])} @ ${ref['open mu*']['cost']:.4f} (open).", ""]
    lines += ["## X3 · MathMage vs debate, pooled over every cell where debate ran (exploratory)", "",
              "| Pool | mu* − debate [95% CI], wins/losses, bootstrap p | n | problems for 80% power |", "|---|---|---|---|"]
    for name, c in result["debate"].items():
        lines.append(f"| {name} | {fmt_c(c)} | {c['n']} | {c['n_for_80_power'] or '--'} |")
    lines += ["", "Pooling is exploratory: the pre-registered v4 tests were per set and Holm-corrected. Families are "
              "resampled jointly across rosters, so shared problems (AIME 2024/25, HMMT Nov 2025) are not double "
              "counted as independent.", "", "## X4 · Dollars per correct answer", ""]
    for title, methods in result["cost_per_correct"].items():
        lines.append(f"**{title}**: " + "; ".join(f"{m} ${v['usd_per_correct']:.4f}" for m, v in
                                                     sorted(methods.items(), key=lambda kv: kv[1]['usd_per_correct'])))
        lines.append("")
    (OUT_T / "arr4_offline.md").write_text("\n".join(lines), encoding="utf-8")


def figure(result, draft):
    from mathmage.analysis import style
    style.use(publication=not draft, base_size=8.0)
    import matplotlib.pyplot as plt
    P = style.P
    names = list(result["adaptive"])
    fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.35))
    for ax, name in zip(axes, names):
        p = result["adaptive"][name]
        for key, color, marker, label in (("AC", P["cyan_deep"], "o", "Adaptive-Consistency"),
                                          ("ESC", P["lav2"], "s", "Early-stopping SC")):
            pts = p["frontier"][key]
            ax.plot([f["cost"] for f in pts], [100 * f["macro"] for f in pts], marker=marker, ms=2.8, lw=1, color=color,
                    label=label)
        sc = p["SC (full pool)"]
        ax.plot(sc["cost"], 100 * sc["macro"], "^", ms=4, color=P["ink"], label="SC, full pool")
        for key, marker, label in (("mu*", "*", r"MathMage at $\mu^\star$"), ("maximal", "D", "MathMage maximal")):
            s = p[key]
            ax.plot(s["cost"], 100 * s["macro"], marker, ms=6 if marker == "*" else 3.6, color=P["purple"] if key == "mu*"
                    else P["violet"], label=label, zorder=5)
        ax.set_xscale("log")
        ax.set_title(name.replace("·", r"$\cdot$") if not draft else name, fontsize=7.2, loc="left")
        ax.set_xlabel("cost per problem (USD, log)", fontsize=7)
        ax.grid(alpha=.5)
    axes[0].set_ylabel("macro accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.6, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .1, 1, 1))
    style.save(fig, "arr4_adaptive", OUT_F)
    plt.close(fig)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    S = sets()
    result = {"adaptive": adaptive_section(S), "cascade": cascade_section(S), "debate": debate_section(S),
              "cost_per_correct": cost_per_correct()}
    (OUT_T / "arr4_offline.json").write_text(json.dumps(to_jsonable(result), indent=1), encoding="utf-8")
    write(result)
    figure(result, args.draft)
    print((OUT_T / "arr4_offline.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
