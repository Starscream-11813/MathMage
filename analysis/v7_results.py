"""v7 analysis: program-augmented baselines on the fresh set and two simple stage-0 rules (read-only; no API calls).

    python analysis/v7_results.py              # 2,000 draws
    python analysis/v7_results.py --draft      # fewer draws
    python analysis/v7_results.py --rules-only # Q3 only (needs no v7 run)

Implements the analysis plan frozen in runs/live_20261018/program_v7/plan_v7.json, written before any v7 call and
before the simple rules were computed:
  P1/P2  MathMage-16 at mu* vs USC + program, MoA + program and debate + program on each team's fresh set (Holm
         across each team's three contrasts)
  Q1     maximal run vs each program-augmented baseline
  Q2     each program-augmented baseline vs its v6 original (the value of showing the program)
  Q3     two simple stage-0 rules on every main-table set, vs MathMage's stage-0 posterior and vs mu*:
         program first (the program's answer when it returned one, else the members' vote) and an accuracy-weighted
         vote (each author weighted by its Laplace-smoothed accuracy on its team's 16 training problems)
Same estimators as v6 (family-clustered, crossed-seed bootstrap; paired macro differences).
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

import v6_results as V6R  # noqa: E402  (installs the grading guard and the cached bootstrap)
from mathmage.analysis.stats import holm  # noqa: E402

V4R, V5R, R = V6R.V4R, V6R.V5R, V6R.R
V7 = R.RUNS / "program_v7"
OUT_T = REPO / "results/tables"
FRESH_E1 = ("aime_2026", "hmmt_feb_2026", "hmmt_nov_2025", "brumo_2025", "cmimc_2025", "smt_2025")
FRESH_E2 = ("aime_2025",) + FRESH_E1
E1_BENCH = ("aime_2024", "aime_2025", "aime_2026", "hmmt_feb_2026", "tqa_phys")
E2_BENCH = ("aime_2024", "aime_2025", "math500")
OFFLINE = {"sat": R.RUNS / "program_v3" / "offline_sat", "open": R.RUNS / "program_open" / "offline_open",
           "sat26": R.RUNS / "program_v6" / "offline_sat26"}
PLUS = {"USC + program": ("fresh_usc_p", "USC"), "MoA + program": ("fresh_moa_p", "MoA"),
        "debate + program": ("fresh_debate_p", "debate")}


def author(c):
    return "tool" if c["phase"] in R.TOOL_PHASES or c["id"] == "s0-tool" else c["model_key"]


# ----------------------------------------------------------------------------- Q3: simple rules
def training_weights(team):
    """Laplace-smoothed accuracy (k + 1) / (n + 2) of each author over the answers it gave on the team's 16 training
    problems (stage-0 panels of the frozen offline phase)."""
    from mathmage.experiments.artifacts import read_record
    from mathmage.grading.vault import load_vault
    from mathmage.grading.verify import grade_answer
    root = OFFLINE[team]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train = set(manifest["train_ids"])
    vault = load_vault()
    tally = defaultdict(lambda: [0, 0])
    items = set()
    for path in sorted((root / "observations").glob("*.json")):
        try:
            data = read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"])["data"]
        except Exception:     # noqa: BLE001 - observations of other kinds lack this schema
            continue
        if not (isinstance(data, dict) and data.get("item_id") in train and "stage0" in data and "members" in data):
            continue
        items.add(data["item_id"])
        for c in data["stage0"]:
            if c.get("valid", True) and c.get("answer") is not None and not c.get("error"):
                k = author(c)
                tally[k][1] += 1
                tally[k][0] += bool(grade_answer(data["item_id"], c["answer"], vault).correct)
    if len(items) != len(train):
        raise ValueError(f"{team}: found training panels for {len(items)} of {len(train)} problems")
    return {k: {"correct": c, "answered": n, "weight": (c + 1) / (n + 2)} for k, (c, n) in sorted(tally.items())}


def rule_rows(rows, registry, weights):
    """Program first and the accuracy-weighted vote, per unit, from the saved stage-0 answers."""
    out = {"program first": [], "accuracy-weighted vote": []}
    for row in rows:
        stage0 = sorted((c for c in row["_candidates"] if c["id"].startswith("s0-")),
                        key=lambda c: (author(c) == "tool", c["id"]))          # members in roster order, then program
        grades = row["candidate_grades"]
        cost = sum(R.call_cost(row, "stage0", f"member={i}") for i in range(3)) + R.call_cost(row, "verify_forward")
        pool = [c for c in stage0 if c.get("valid", True) and c.get("answer") is not None]
        tool = next((c for c in pool if author(c) == "tool"), None)
        first = bool(grades.get(tool["id"], {}).get("correct")) if tool else bool(row["initial_correct"])
        out["program first"].append(R.base(row, first, cost))
        weighted = False
        if pool:
            from mathmage.grading.equiv import cluster
            labels = cluster([c["answer"] for c in pool], registry[row["item_id"]].answer_format)
            score = defaultdict(float)
            for c, lab in zip(pool, labels):
                score[lab] += weights[author(c)]["weight"]
            top = max(score, key=lambda lab: (score[lab], -labels.index(lab)))
            weighted = any(grades.get(c["id"], {}).get("correct") for c, lab in zip(pool, labels) if lab == top)
        out["accuracy-weighted vote"].append(R.base(row, weighted, cost))
    return out


def rule_sets(registry):
    """(team, raw MathMage rows, MathMage methods) for every main-table set, built as in acl_numbers.py / v6."""
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    new1 = V4R.mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"])
    new2 = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    sub = V4R.subset
    out = {}
    for key, team, rows, study, artifact in (
            ("E1", "sat", sub(rows1, E1_BENCH), e1, a1), ("sat216", "sat", sub(rows1 + new1, FRESH_E1), e1, a1),
            ("E2", "open", sub(rows2, E2_BENCH), e2, a2), ("open1968", "open", sub(rows2 + new2, FRESH_E2), e2, a2)):
        out[key] = (team, rows, V4R.mathmage_methods(rows, study, artifact, registry))
    for key, package, team in (("fresh26", "sat26", "sat26"), ("fresh_open", "lane", "open")):
        t = V6R.team_methods(package, registry, 1)
        out[key] = (team, t["rows"], t["methods"])
    return out


def simple_rules(registry, samples):
    weights = {team: training_weights(team) for team in OFFLINE}
    result = {"weights": weights, "sets": {}}
    for key, (team, rows, methods) in rule_sets(registry).items():
        rules = rule_rows(rows, registry, weights[team])
        table = {name: V6R.summarize(methods[name], samples)
                 for name in ("vote", "vote + program", "weighted vote + program", "posterior", "mu*") if methods.get(name)}
        table.update({name: V6R.summarize(v, samples) for name, v in rules.items()})
        contrasts = {}
        for name, v in rules.items():
            contrasts[f"posterior vs {name}"] = V4R.contrast(methods["posterior"], v, samples)
            contrasts[f"mu* vs {name}"] = V4R.contrast(methods["mu*"], v, samples)
        result["sets"][key] = {"team": team, "n": len(rows), "methods": table, "contrasts": contrasts}
        print(f"{key:10s} n={len(rows):5d}  " + "  ".join(f"{k} {100 * t['macro']:.1f}" for k, t in table.items()))
    return result


# ----------------------------------------------------------------------------- P1/P2, Q1, Q2: parity runs
def graded7(package, job):
    directory = V7 / package / job / "graded"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))] if directory.exists() else []


def parity(registry, samples):
    plan = json.loads((V7 / "plan_v7.json").read_text(encoding="utf-8"))
    out = {"plan": plan["sha256"]}
    for package, key in (("sat26", "P1"), ("lane", "P2")):
        team = V6R.team_methods(package, registry, samples)
        planned = {j["id"]: j["n_units"] for j in plan["jobs"][package]}
        plus, status = {}, {}
        for name, (job, _) in PLUS.items():
            rows = graded7(package, job)
            status[job] = {"graded": len(rows), "planned": planned[job]}
            if not rows:
                continue
            if job == "fresh_usc_p":
                plus[name] = V4R.usc_outcomes(rows)
            elif job == "fresh_moa_p":
                plus[name] = V5R.moa_outcomes(rows)
            else:
                plus[name] = V4R.debate_outcomes(rows, registry)
            status[job].update(reused=sum(bool(r.get("reused_from")) for r in rows),
                               new_cost_usd=float(sum(r["new_cost_usd"] for r in rows)))
        methods = team["methods"]
        block = {"status": status,
                 "methods": {name: V6R.summarize(v, samples) for name, v in
                             {**{k: methods[k] for k in ("vote", "vote + program", "USC", "MoA", "debate", "posterior",
                                                         "mu*", "maximal")}, **plus}.items()},
                 "contrasts": {f"mu* vs {name}": V4R.contrast(methods["mu*"], v, samples) for name, v in plus.items()},
                 "maximal": {f"maximal vs {name}": V4R.contrast(methods["maximal"], v, samples) for name, v in plus.items()},
                 "gain": {f"{name} vs {PLUS[name][1]}": V4R.contrast(v, methods[PLUS[name][1]], samples)
                          for name, v in plus.items()}}
        keys = [k for k, c in block["contrasts"].items() if c]
        for k, adjusted in zip(keys, holm([block["contrasts"][k]["p"] for k in keys]) if keys else []):
            block["contrasts"][k]["holm_p"] = adjusted
        block["complete"] = all(s["graded"] == s["planned"] for s in status.values())
        out[key] = block
    return out


# ----------------------------------------------------------------------------- output
def write(out, path):
    lines = ["# v7 results (plan runs/live_20261018/program_v7/plan_v7.json)", ""]
    for key, title in (("P1", "sat26 team, fresh set (one seed)"), ("P2", "open roster, fresh set (8 seeds)")):
        if key not in out.get("parity", {}):
            continue
        b = out["parity"][key]
        lines += [f"## {key} · {title} · complete: {b['complete']}", ""]
        lines += [f"- {job}: {s}" for job, s in b["status"].items()] + [""]
        lines += V4R.method_table(b["methods"], "Accuracy and cost")
        lines += ["Primary (mu* vs program-augmented baselines; Holm across 3):"]
        lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in b["contrasts"].items()]
        lines += ["", "Q1 (maximal run):"] + [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in b["maximal"].items()]
        lines += ["", "Q2 (value of showing the program):"] + [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in b["gain"].items()]
        lines.append("")
    if out.get("rules"):
        lines += ["## Q3 · simple stage-0 rules", "", "Training weights (Laplace-smoothed accuracy):"]
        for team, w in out["rules"]["weights"].items():
            lines.append(f"- {team}: " + ", ".join(f"{k} {v['weight']:.3f} ({v['correct']}/{v['answered']})" for k, v in w.items()))
        lines.append("")
        for key, s in out["rules"]["sets"].items():
            lines += [f"### {key} ({s['team']}, n = {s['n']})", ""]
            lines += V4R.method_table(s["methods"], "Accuracy and cost")
            lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in s["contrasts"].items()] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()
    samples = args.samples or (300 if args.draft else 2000)
    registry = V6R.Registry6()
    path = OUT_T / "v7_results.json"
    out = json.loads(path.read_text(encoding="utf-8")) if args.rules_only and path.exists() else {}
    out["rules"] = simple_rules(registry, samples)
    if not args.rules_only:
        out["parity"] = parity(registry, samples)
    OUT_T.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(V6R.to_jsonable(out), indent=1), encoding="utf-8")
    write(out, OUT_T / "v7_results.md")
    print((OUT_T / "v7_results.md").read_text(encoding="utf-8")[:8000])


if __name__ == "__main__":
    main()
