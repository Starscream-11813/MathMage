"""MathMage paper results from the frozen live runs (read-only; never dispatches a request).

    python analysis/results.py            # publication figures (LuaLaTeX, Latin Modern TrueType)
    python analysis/results.py --draft    # fast previews

Outputs: results/tables/*.{md,json} and results/figures/*.{pdf,png}.
Every number is recomputed from sealed graded rows and observations. A phase that is still
running is labelled INTERIM in its table and figure.

Methods compared inside one harness, on identical saved answers where possible:
  member / single sample   one roster member's stage-0 answer (its own call cost)
  program                  the independent program-of-thought answer (its own call cost)
  vote                     plurality over the three members (ties: roster order)
  vote + program           plurality over members and program (free baseline)
  weighted vote + program  plurality weighted by training-fitted abilities (free baseline)
  posterior                MathMage's label-free posterior pick after stage 0 (no extra calls)
  mu*                      MathMage stopped at the development-frozen threshold (prefix replay)
  maximal                  MathMage after its bounded maximal action sequence (mu = 0)
  oracle (stage 0)         a member or the program proposed a correct answer
  oracle (final pool)      any candidate in the final pool is correct
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mathmage.analysis import style  # noqa: E402
from mathmage.analysis.frontier import frontier, pareto, stop_index  # noqa: E402
from mathmage.analysis.metrics import macro_accuracy  # noqa: E402
from mathmage.analysis.stats import cluster_interval, holm, paired_contrast  # noqa: E402

RUNS = REPO / "runs" / "live_20261018"
BENCH = {"aime_2024": "AIME 2024", "aime_2025": "AIME 2025", "aime_2026": "AIME 2026",
         "hmmt_feb_2026": "HMMT Feb 2026", "tqa_phys": "TheoremQA phys.", "math500": "MATH-500"}
ACTIONS = ["sample", "coach", "tool_alt", "diversify", "review", "judge"]
ACTION_LABEL = {"sample": "fresh sample", "coach": "coach (hint + re-solve)", "tool_alt": "program attempt",
                "diversify": "diversified re-solve", "review": "review", "judge": "judge"}
TOOL_PHASES = {"tool_forward", "verify_forward", "tool_alt"}
ABLATIONS = {"A1": "uniform budget (no early stop)", "A2": "SAT-style certificate judge", "A3": "unconditional revision",
             "A4": "fixed action order", "A5": "open discussion for hints", "A6": "random tutor",
             "A7": "no tools / program", "A8": "no strategy cards", "C1": "best member alone, sequential",
             "C2": "3 × best member, independent", "R1": "operational rule router"}
SAMPLES = 2000


@dataclass
class Study:
    key: str
    title: str
    root: Path
    phase: str
    artifact: Path
    members: dict
    homogeneous: bool


STUDIES = [
    Study("e1", "E1 · SAT roster", RUNS / "program_v3", "e1_primary", RUNS / "program_v3/offline_sat/artifacts.json",
          {"s0-0": "o3-mini (low)", "s0-1": "Claude Haiku 4.5", "s0-2": "DeepSeek-V3"}, False),
    Study("e2", "E2 · open roster", RUNS / "program_open", "e2", RUNS / "program_open/offline_open/artifacts.json",
          {"s0-0": "gpt-oss-120b #1", "s0-1": "gpt-oss-120b #2", "s0-2": "gpt-oss-120b #3"}, True),
]


# ----------------------------------------------------------------------------- loading
def load_phase(root: Path, phase: str, conditions=None):
    directory = root / phase
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    execution = directory / "execution.json"
    complete = execution.exists() and json.loads(execution.read_text(encoding="utf-8")).get("status") == "complete"
    order = {item: index for index, item in enumerate(manifest["item_ids"])}
    rows = []
    for path in (directory / "graded").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if conditions and row.get("condition") not in conditions:
            continue
        record = json.loads((directory / "agent" / path.name).read_text(encoding="utf-8"))
        row["_calls"] = [{k: c.get(k) for k in ("phase", "role", "model_key", "resource_cost_usd")}
                         for c in record.get("meta", {}).get("calls", [])]
        row["_candidates"] = [{k: c.get(k) for k in ("id", "phase", "model_key", "answer", "valid", "evidence", "parent_id")}
                              for c in record.get("candidates", [])]
        row["_steps"] = [{k: s.get(k) for k in ("best_candidate", "best_answer", "confidence", "chosen")}
                         for s in record.get("meta", {}).get("anytime", [])]
        row["_trace"] = [t for t in record.get("trace", []) if t.get("event") in ("policy_action", "judge", "hint", "review")]
        rows.append(row)
    rows.sort(key=lambda r: (order.get(r["item_id"], 10 ** 6), r.get("seed", 0), r.get("condition", "")))
    return rows, manifest, complete


def call_cost(row, phase, role=None):
    return sum(float(c.get("resource_cost_usd") or 0) for c in row["_calls"]
               if c.get("phase") == phase and (role is None or c.get("role") == role))


def base(row, correct, cost):
    return {"benchmark": row["benchmark"], "item_id": row["item_id"], "family_id": row.get("family_id", row["item_id"]),
            "seed": row.get("seed", 0), "final_correct": correct, "cost_usd": float(cost)}


# ----------------------------------------------------------------------------- stage-0 reconstruction
def posterior_variants(params_dict):
    from mathmage.team.posterior import AgentProfile, PosteriorParams, TeamPosterior

    full = PosteriorParams.from_dict(params_dict)
    member_keys = [k for k in full.agents if "@" not in k and k != "tool"]
    mean_theta = float(np.mean([full.agents[k].theta for k in member_keys + ["tool"] if k in full.agents]))
    equal = replace(full, agents={k: (AgentProfile(mean_theta, v.a, v.abstain) if "@" not in k else v)
                                  for k, v in full.agents.items()})
    # (parameters, include the program answer, use partial-certificate check evidence)
    variants = {"posterior (full)": (full, True, True), "posterior, no program answer": (full, False, True),
                "posterior, no check evidence": (full, True, False),
                "posterior, equal abilities": (equal, True, True),
                "posterior, difficulty fixed": (replace(full, b_sd=0.05), True, True)}
    if full.trap_weight >= 0.01:
        variants["posterior, no trap mixture"] = (replace(full, trap_weight=0.0, kappa_trap=0.0), True, True)
    return {name: (TeamPosterior(p), tool, checks) for name, (p, tool, checks) in variants.items()}, full


def stage0_analysis(rows, params_dict, registry):
    """Rebuild stage-0 decisions from saved answers: posterior variants and free baselines."""
    from mathmage.grading.equiv import cluster
    from mathmage.team.posterior import Evidence, Obs

    variants, full = posterior_variants(params_dict)
    out, confidence = defaultdict(list), defaultdict(list)
    mismatches = Counter()
    for row in rows:
        stage0 = [c for c in row["_candidates"] if c["id"].startswith("s0-")]
        fmt = registry[row["item_id"]].answer_format
        grades = row["candidate_grades"]

        def pool_of(include_tool):
            included = [c for c in stage0 if include_tool or c["phase"] not in TOOL_PHASES]
            pool = [c for c in included if c.get("valid", True) and c.get("answer") is not None]
            labels = cluster([c["answer"] for c in pool], fmt) if pool else []
            return included, pool, labels

        def decide(posterior, include_tool, checks=True):
            included, pool, labels = pool_of(include_tool)
            label_of = {c["id"]: lab for c, lab in zip(pool, labels)}
            obs = [Obs("tool" if c["phase"] in TOOL_PHASES else c["model_key"], label_of.get(c["id"], -1))
                   for c in included if not (c["phase"] in TOOL_PHASES and label_of.get(c["id"], -1) < 0)]
            evidence = [Evidence("check_failed" if (c.get("evidence") or {}).get("status") == "failed" else "check_passed",
                                 label_of[c["id"]]) for c in pool
                        if checks and (c.get("evidence") or {}).get("source") == "certificate_program"
                        and (c.get("evidence") or {}).get("status") in ("failed", "partial")]
            C = max(labels, default=-1) + 1
            if C == 0:
                return False, 0.0, None
            result = posterior.compute(obs, evidence, C)
            correct = any(grades.get(c["id"], {}).get("correct") for c in pool if label_of[c["id"]] == result.best)
            return correct, result.confidence, result.best

        program_cost = call_cost(row, "verify_forward")
        check_cost = call_cost(row, "verify_partial")
        vote_cost = sum(call_cost(row, "stage0", f"member={i}") for i in range(3))
        b = base(row, None, row["anytime"][0]["cost_usd"])
        chosen = {}
        for name, (posterior, include_tool, checks) in variants.items():
            correct, conf, chosen[name] = decide(posterior, include_tool, checks)
            cost = b["cost_usd"] - (0.0 if include_tool else program_cost) - (0.0 if checks else check_cost)
            out[name].append({**b, "final_correct": correct, "cost_usd": cost})
            if name == "posterior (full)":
                confidence["posterior"].append((conf, correct))
        included, pool, labels = pool_of(True)
        label_of = {c["id"]: lab for c, lab in zip(pool, labels)}
        logged = (row.get("_steps") or [{}])[0].get("best_candidate")
        mismatches["correctness"] += out["posterior (full)"][-1]["final_correct"] != bool(row["anytime"][0]["correct"])
        mismatches["choice"] += logged is not None and label_of.get(logged) != chosen["posterior (full)"]
        mismatches["n"] += 1

        def pick(weight):
            if not pool:
                return False, 0.0, None
            score = defaultdict(float)
            for c, lab in zip(pool, labels):
                score[lab] += weight(c)
            top = max(score, key=lambda lab: (score[lab], -labels.index(lab)))
            total = sum(weight(c) for c in included)
            return (any(grades.get(c["id"], {}).get("correct") for c, lab in zip(pool, labels) if lab == top),
                    score[top] / total if total else 0.0, top)

        def ability(c):
            key = "tool" if c["phase"] in TOOL_PHASES else c["model_key"]
            return float(1 / (1 + math.exp(-full.agent(key).theta)))
        plain, _, _ = pick(lambda c: 1.0)
        weighted, wshare, wtop = pick(ability)
        # Calibration-free rule: the program breaks ties (outvotes one member, never a pair).
        tiebreak, _, _ = pick(lambda c: 1.5 if c["phase"] in TOOL_PHASES else 1.0)
        mismatches["weighted_vote_same_choice"] += wtop == chosen["posterior (full)"]
        out["vote + program"].append(base(row, plain, vote_cost + program_cost))
        out["vote + program as tie-breaker"].append(base(row, tiebreak, vote_cost + program_cost))
        out["weighted vote + program"].append(base(row, weighted, vote_cost + program_cost))
        confidence["weighted vote share"].append((wshare, weighted))
        member_labels = [lab for c, lab in zip(pool, labels) if c["phase"] not in TOOL_PHASES]
        share = Counter(member_labels).most_common(1)[0][1] / 3 if member_labels else 0.0
        confidence["vote share"].append((share, bool(row["initial_correct"])))
    return out, confidence, mismatches


def method_rows(study, rows, mu_star, stage0):
    out = defaultdict(list)
    for r in rows:
        grades = r["candidate_grades"]
        if study.homogeneous:
            out["single sample"].append(base(r, bool(grades.get("s0-0", {}).get("correct")),
                                             call_cost(r, "stage0", "member=0")))
            out["single sample (mean of 3)"].append(base(r, float(np.mean([bool(grades.get(c, {}).get("correct"))
                                                                           for c in study.members])),
                                                         np.mean([call_cost(r, "stage0", f"member={i}") for i in range(3)])))
        else:
            for cid, name in study.members.items():
                out[name].append(base(r, bool(grades.get(cid, {}).get("correct")),
                                      call_cost(r, "stage0", f"member={cid[-1]}")))
        out["program"].append(base(r, bool(grades.get("s0-tool", {}).get("correct")), call_cost(r, "verify_forward")))
        out["vote"].append(base(r, bool(r["initial_correct"]), sum(call_cost(r, "stage0", f"member={i}") for i in range(3))))
        steps = r["anytime"]
        out["posterior"].append(base(r, bool(steps[0]["correct"]), steps[0]["cost_usd"]))
        cut = steps[stop_index(steps, mu_star)]
        out["mu*"].append(base(r, bool(cut["correct"]), cut["cost_usd"]))
        out["maximal"].append(base(r, bool(steps[-1]["correct"]), steps[-1]["cost_usd"]))
        out["oracle (stage 0)"].append(base(r, bool(r["covered_with_tool"]), steps[0]["cost_usd"]))
        out["oracle (final pool)"].append(base(r, bool(r["pool_covered"]), steps[-1]["cost_usd"]))
    out["vote + program"] = stage0["vote + program"]
    out["weighted vote + program"] = stage0["weighted vote + program"]
    return out


def summarize(methods, samples):
    table = {}
    for name, rs in methods.items():
        by = defaultdict(list)
        for r in rs:
            by[r["benchmark"]].append(r)
        ci = cluster_interval(rs, macro_accuracy, samples=samples)
        table[name] = {"macro": ci["estimate"], "low": ci["low"], "high": ci["high"],
                       "cost": float(np.mean([r["cost_usd"] for r in rs])),
                       "by_benchmark": {b: float(np.mean([float(x["final_correct"]) for x in v])) for b, v in by.items()},
                       "n": len(rs)}
    return table


def contrasts(methods, references, targets, samples):
    out = {}
    for target in targets:
        for ref in references:
            out[f"{target} vs {ref}"] = paired_contrast(methods[target], methods[ref], samples=samples)
    keys = [k for k, v in out.items() if v.get("mcnemar_exact_p") is not None]
    for key, adjusted in zip(keys, holm([out[k]["mcnemar_exact_p"] for k in keys]) if keys else []):
        out[key]["holm_p"] = adjusted
    return out


# ----------------------------------------------------------------------------- calibration & actions
def calibration(pairs, bins=10):
    confidence = np.array([c for c, _ in pairs], float)
    correct = np.array([float(y) for _, y in pairs])
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) | (hi == 1.0))
        if mask.any():
            rows.append({"n": int(mask.sum()), "confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {"ece": float(ece), "brier": float(np.mean((confidence - correct) ** 2)), "bins": rows, "n": len(pairs)}


def risk_coverage(pairs):
    confidence = np.array([c for c, _ in pairs], float)
    correct = np.array([float(y) for _, y in pairs])
    order = np.argsort(-confidence, kind="stable")
    c, y = confidence[order], correct[order]
    smoothed, start = y.copy(), 0
    while start < len(c):
        end = start
        while end + 1 < len(c) and c[end + 1] == c[start]:
            end += 1
        smoothed[start:end + 1] = y[start:end + 1].mean()
        start = end + 1
    selective = np.cumsum(smoothed) / np.arange(1, len(smoothed) + 1)
    coverage = np.arange(1, len(smoothed) + 1) / len(smoothed)
    return coverage, selective, float(np.mean(1 - selective))


def action_effects(rows, mu_star=None):
    """Rescues/harms per action kind; with mu_star, also split by whether the frozen threshold takes the action."""
    effects = defaultdict(lambda: {"n": 0, "rescue": 0, "harm": 0, "voc": [], "gain": [], "cost": []})
    for r in rows:
        steps = r["anytime"]
        cut = stop_index(steps, mu_star) if mu_star is not None else None
        for k, (before, after) in enumerate(zip(steps[:-1], steps[1:])):
            chosen = before.get("chosen") or {}
            kind = chosen.get("kind")
            if not kind:
                continue
            keys = [kind] if cut is None else [kind, "within mu*" if k < cut else "beyond mu*"]
            for key in keys:
                e = effects[key]
                e["n"] += 1
                e["rescue"] += (not before["correct"]) and bool(after["correct"])
                e["harm"] += bool(before["correct"]) and not after["correct"]
                e["voc"].append(float(chosen.get("voc") or 0))
                e["gain"].append(int(bool(after["correct"])) - int(bool(before["correct"])))
                e["cost"].append(float(after["cost_usd"]) - float(before["cost_usd"]))
    return effects


CREATES = {"tool_alt": "tool_alt", "sample": "sample", "diversify": "method_diversification"}


def action_cases(res, registry):
    """Every decision an action changed, with a mechanical account of how it changed."""
    from mathmage.grading.equiv import cluster
    cases = []
    for r in res["rows"]:
        steps, trace, cands, graded = r["_steps"], r["_trace"], r["_candidates"], r["anytime"]
        if len(steps) < 2:
            continue
        grades, fmt = r["candidate_grades"], registry[r["item_id"]].answer_format
        queues = defaultdict(list)
        for c in cands:
            queues[c["phase"]].append(c)
        actions = [(t, trace[i + 1] if i + 1 < len(trace) and trace[i + 1]["event"] != "policy_action" else None)
                   for i, t in enumerate(trace) if t["event"] == "policy_action"]
        created = []
        for t, follow in actions:
            if t["kind"] in ("coach", "review"):                       # matched through the revised candidate
                phase = "coached_solve" if t["kind"] == "coach" else "rebuttal_revision"
                parent = (follow or {}).get("candidate_id")
                match = next((c for c in queues[phase] if c.get("parent_id") == parent), None)
                if match:
                    queues[phase].remove(match)
                created.append(match)
            else:
                queue = queues.get(CREATES.get(t["kind"]), [])
                created.append(queue.pop(0) if queue else None)
        seen = [c for c in cands if c["phase"] in ("stage0", "tool_forward")]
        right = sorted({str(c.get("answer")) for c in cands if grades.get(c["id"], {}).get("correct")})
        for k, ((t, follow), new) in enumerate(zip(actions, created)):
            if k + 1 >= len(steps):
                break
            before, after = bool(graded[k]["correct"]), bool(graded[k + 1]["correct"])
            if before != after:
                a_before, a_after = steps[k]["best_answer"], steps[k + 1]["best_answer"]
                answers = [a_after] + ([new["answer"]] if new and new.get("answer") else []) + \
                          [c["answer"] for c in seen if c.get("answer")]
                labels = cluster([str(a) for a in answers], fmt) if a_after is not None else []
                new_in_after = bool(new and new.get("answer")) and labels and labels[1] == labels[0]
                earlier = labels[2:] if new and new.get("answer") else labels[1:]
                after_seen = bool(labels) and labels[0] in earlier
                if t["kind"] == "judge":
                    how = "judge verdict"
                elif t["kind"] == "review":
                    how = "review / rebuttal"
                elif new_in_after and not after_seen:
                    how = "new answer introduced and adopted"
                elif new_in_after:
                    how = "corroborated an earlier answer"
                else:
                    how = "re-weighting after a dissenting candidate"
                case = {"study": res["study"].key, "item_id": r["item_id"], "benchmark": r["benchmark"],
                        "seed": r.get("seed", 0), "zone": r["zone"], "step": k, "kind": t["kind"], "label": t.get("label"),
                        "voc": t.get("voc"), "outcome": "rescue" if after else "harm", "mechanism": how,
                        "before": a_before, "after": a_after, "confidence": (steps[k]["confidence"], steps[k + 1]["confidence"]),
                        "new": None if not new else {"id": new["id"], "phase": new["phase"], "answer": new.get("answer"),
                                                     "correct": bool(grades.get(new["id"], {}).get("correct"))},
                        "correct_answers_in_pool": right, "problem": registry[r["item_id"]].problem}
                if t["kind"] == "judge" and follow:
                    options = follow.get("options", [])
                    choice = (follow.get("verdict") or {}).get("choice")
                    by_id = {c["id"]: c for c in cands}
                    case["judge"] = {"model": follow.get("judge"), "confidence": (follow.get("verdict") or {}).get("confidence"),
                                     "options": [(o, by_id.get(o, {}).get("answer")) for o in options],
                                     "picked": options[choice - 1] if isinstance(choice, int) and 0 < choice <= len(options) else None}
                if t["kind"] == "coach" and follow:
                    case["hint"] = {"level": follow.get("level"), "tutor": follow.get("tutor"),
                                    "leak_filtered": follow.get("answer_leak_filtered"),
                                    "accepted": (follow.get("audit") or {}).get("accepted")}
                cases.append(case)
            if new:
                seen.append(new)
    return cases


def write_cases(results, registry, out):
    cases = [c for res in results for c in action_cases(res, registry)]
    lines = ["# Decision changes made by actions (rescues and harms)", "",
             "Every step at which a chosen action flipped the team's answer between right and wrong, from the saved",
             "traces. 'Mechanism' is mechanical: whether the action's own new candidate carries the adopted answer,",
             "and whether that answer was already in the pool. Problem texts are truncated.", ""]
    summary = defaultdict(Counter)
    for c in cases:
        summary[(c["study"], c["outcome"])][c["mechanism"]] += 1
    lines += ["| Study | Outcome | Mechanism | Count |", "|---|---|---|---|"]
    for (study, outcome), counter in sorted(summary.items()):
        for how, n in counter.most_common():
            lines.append(f"| {study.upper()} | {outcome} | {how} | {n} |")
    for c in cases:
        lines += ["", f"## {c['study'].upper()} · {c['item_id']} · seed {c['seed']} · zone {c['zone']} · step {c['step']} · "
                      f"{c['outcome'].upper()} by {ACTION_LABEL.get(c['kind'], c['kind'])}",
                  "", f"> {c['problem'][:260].replace(chr(10), ' ')}{'…' if len(c['problem']) > 260 else ''}", "",
                  f"- decision {c['before']!r} (conf {c['confidence'][0]:.2f}) → {c['after']!r} (conf {c['confidence'][1]:.2f}); "
                  f"correct answers in final pool: {', '.join(c['correct_answers_in_pool']) or 'none'}",
                  f"- action `{c['label']}` (VOC {c['voc']:.3f}); mechanism: {c['mechanism']}"]
        if c["new"]:
            lines.append(f"- new candidate {c['new']['id']} ({c['new']['phase']}): {c['new']['answer']!r}, "
                         f"{'correct' if c['new']['correct'] else 'wrong'}")
        if "judge" in c:
            j = c["judge"]
            lines.append(f"- judge {j['model']} picked {j['picked']} (stated confidence {j['confidence']}); options: "
                         + ", ".join(f"{o}={a!r}" for o, a in j["options"]))
        if "hint" in c:
            h = c["hint"]
            lines.append(f"- hint level {h['level']} from {h['tutor']}; audit accepted {h['accepted']}; "
                         f"answer-leak filter fired {h['leak_filtered']}")
    (out / "cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cases, summary


def error_taxonomy(rows):
    out = defaultdict(Counter)
    for r in rows:
        for stage, correct, covered in [("stage 0", r["anytime"][0]["correct"], r["covered_with_tool"]),
                                        ("final", r["final_correct"], r["pool_covered"])]:
            kind = "correct" if correct else "selection failure" if covered else "coverage failure"
            out[(r["benchmark"], stage)][kind] += 1
            out[("all", stage)][kind] += 1
    return out


# ----------------------------------------------------------------------------- per-study analysis
def analyze_study(study, registry, samples):
    artifact = json.loads(study.artifact.read_text(encoding="utf-8"))
    mu_star = float(artifact["mu_star"])
    rows, manifest, complete = load_phase(study.root, study.phase)
    stage0, confidence, mismatches = stage0_analysis(rows, artifact["posterior_params"], registry)
    methods = method_rows(study, rows, mu_star, stage0)
    table = summarize(methods, samples)
    single = [name for name in methods if name in study.members.values() or name == "single sample"]
    best_single = max(single, key=lambda n: table[n]["macro"])          # chosen on test: descriptive only
    train_best = best_single
    if not study.homogeneous:                                          # chosen on training data (offline artifact)
        keys = {c["id"]: c["model_key"] for c in rows[0]["_candidates"] if c["id"] in study.members}
        train_best = next((study.members[i] for i, k in keys.items() if k == artifact.get("best_member")), best_single)
    references = ["vote", "weighted vote + program"] + list(dict.fromkeys([train_best, best_single]))
    contrast = contrasts(methods, references, ["posterior", "mu*", "maximal"], samples)
    # Contests released after every model's training cutoff: the contamination-free subset.
    fresh = {"aime_2026", "hmmt_feb_2026"}
    post_cutoff = None
    if any(r["benchmark"] in fresh for r in rows):
        subset = {name: [m for m in rs if m["benchmark"] in fresh] for name, rs in methods.items()}
        post_cutoff = {"n": len(subset["vote"]), "table": summarize({k: subset[k] for k in ["vote", "weighted vote + program",
                                                                                         "posterior", "mu*", "maximal"]}, samples),
                       "contrast": contrasts(subset, ["vote", "weighted vote + program"], ["posterior", "mu*", "maximal"], samples)}
    zone_stats = {}
    for z in sorted({r["zone"] for r in rows}):
        zr = [r for r in rows if r["zone"] == z]
        cut = [r["anytime"][stop_index(r["anytime"], mu_star)] for r in zr]
        zone_stats[z] = {"n": len(zr), "share": len(zr) / len(rows),
                         "vote": float(np.mean([bool(r["initial_correct"]) for r in zr])),
                         "posterior": float(np.mean([bool(r["anytime"][0]["correct"]) for r in zr])),
                         "mu*": float(np.mean([bool(s["correct"]) for s in cut])),
                         "maximal": float(np.mean([bool(r["final_correct"]) for r in zr])),
                         "oracle (stage 0)": float(np.mean([bool(r["covered_with_tool"]) for r in zr])),
                         "oracle (final pool)": float(np.mean([bool(r["pool_covered"]) for r in zr])),
                         "extra_cost_mu": float(np.mean([s["cost_usd"] - r["anytime"][0]["cost_usd"] for s, r in zip(cut, zr)])),
                         "extra_cost_max": float(np.mean([r["anytime"][-1]["cost_usd"] - r["anytime"][0]["cost_usd"] for r in zr])),
                         "actions_max": float(np.mean([len(r["anytime"]) - 1 for r in zr]))}
    component_rows = {**{k: v for k, v in stage0.items() if k.startswith("posterior")},
                      "vote + program": stage0["vote + program"],
                      "vote + program as tie-breaker": stage0["vote + program as tie-breaker"],
                      "weighted vote + program": stage0["weighted vote + program"], "vote (members)": methods["vote"]}
    components = summarize(component_rows, samples)
    component_contrast = {name: paired_contrast(rs, stage0["posterior (full)"], samples=samples)
                          for name, rs in component_rows.items() if name != "posterior (full)"}
    seeds = None
    if len({r.get("seed", 0) for r in rows}) > 1:
        seeds = {}
        for name in ["vote", "posterior", "mu*", "maximal"]:
            per = defaultdict(list)
            for r in methods[name]:
                if r["benchmark"] != "math500":
                    per[r["seed"]].append(float(r["final_correct"]))
            values = [float(np.mean(v)) for _, v in sorted(per.items())]
            seeds[name] = {"per_seed": values, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
    return {"study": study, "artifact": artifact, "mu_star": mu_star, "rows": rows, "manifest": manifest,
            "complete": complete, "methods": methods, "table": table, "contrast": contrast, "best_single": best_single,
            "train_best": train_best, "zone_stats": zone_stats, "post_cutoff": post_cutoff,
            "components": components, "component_contrast": component_contrast, "reproduction_mismatches": mismatches,
            "confidence": confidence, "calibration": {k: calibration(v) for k, v in confidence.items()},
            "aurc": {k: risk_coverage(v)[2] for k, v in confidence.items()},
            "actions": action_effects(rows), "actions_by_threshold": action_effects(rows, mu_star),
            "errors": error_taxonomy(rows), "seeds": seeds,
            "zones": Counter(r["zone"] for r in rows), "stops": Counter(r["anytime"][-1].get("stop_reason") for r in rows)}


def self_consistency(res, registry, draws=200, seed=0):
    """Matched-compute self-consistency on a homogeneous multi-seed study (no new calls).

    Each AIME item has 3 member samples per seed, so the seeds pool into a sample
    distribution. SC@k draws k natural-language samples; the program-inclusive variant
    pools whole seeds (3 samples + 1 program answer each). Expectations over draws.
    """
    import random

    from mathmage.grading.equiv import cluster
    if not res["study"].homogeneous or not res["seeds"]:
        return None
    rng = random.Random(seed)
    by_item = defaultdict(list)
    nl_cost, program_cost = [], []
    for r in res["rows"]:
        if r["benchmark"] == "math500":
            continue
        grades = r["candidate_grades"]
        nl = [(c.get("answer"), bool(grades.get(c["id"], {}).get("correct"))) for c in r["_candidates"]
              if c["id"] in ("s0-0", "s0-1", "s0-2")]
        program = next(((c.get("answer"), bool(grades.get(c["id"], {}).get("correct"))) for c in r["_candidates"]
                        if c["id"] == "s0-tool"), None)
        by_item[r["item_id"]].append({"nl": nl, "program": program})
        nl_cost += [c["resource_cost_usd"] for c in r["_calls"] if c["phase"] == "stage0"]
        program_cost += [c["resource_cost_usd"] for c in r["_calls"] if c["phase"] == "verify_forward"]
    u_nl, u_program = float(np.mean(nl_cost)), float(np.mean(program_cost))
    agents = res["artifact"]["posterior_params"]["agents"]
    member_key = next(c["model_key"] for c in res["rows"][0]["_candidates"] if c["id"] == "s0-0")
    w_nl, w_program = (float(1 / (1 + math.exp(-agents[k]["theta"]))) for k in (member_key, "tool"))

    def vote(pool, fmt, weights=(1.0, 1.0)):
        """Expected (weighted) plurality correctness with ties broken uniformly at random."""
        usable = [(a, ok, weights[program]) for a, ok, program in pool if a]
        if not usable:
            return 0.0
        labels = cluster([a for a, _, _ in usable], fmt)
        score = defaultdict(float)
        for (_, _, w), lab in zip(usable, labels):
            score[lab] += w
        best = max(score.values())
        return float(np.mean([any(ok for (_, ok, _), lab in zip(usable, labels) if lab == top)
                              for top, s in score.items() if s >= best - 1e-12]))

    def pooled(chosen):
        return [(a, ok, 0) for s in chosen for a, ok in s["nl"]] + \
               [(s["program"][0], s["program"][1], 1) for s in chosen if s["program"]]

    n_seeds = min(len(v) for v in by_item.values())
    nl_curve, program_curve = [], []
    for k in [1, 3, 5, 8, 12, 16, 3 * n_seeds]:
        accuracy = np.mean([np.mean([vote([(a, ok, 0) for a, ok in rng.sample([x for s in seeds for x in s["nl"]], k)],
                                          registry[item].answer_format)
                                     for _ in range(draws if k < 3 * n_seeds else 1)]) for item, seeds in by_item.items()])
        nl_curve.append({"k": k, "accuracy": float(accuracy), "cost": k * u_nl})
    for m in [1, 2, 4, n_seeds]:
        draws_m = [[rng.sample(seeds, m) for _ in range(draws if m < n_seeds else 1)] for seeds in by_item.values()]
        plain = np.mean([np.mean([vote(pooled(chosen), registry[item].answer_format) for chosen in d])
                         for item, d in zip(by_item, draws_m)])
        weighted = np.mean([np.mean([vote(pooled(chosen), registry[item].answer_format, (w_nl, w_program)) for chosen in d])
                            for item, d in zip(by_item, draws_m)])
        program_curve.append({"m": m, "accuracy": float(plain), "weighted_accuracy": float(weighted),
                              "cost": m * (3 * u_nl + u_program)})
    aime = [r for r in res["rows"] if r["benchmark"] != "math500"]
    points = {}
    for name, step_of in [("posterior", lambda s: s[0]), ("mu*", lambda s: s[stop_index(s, res["mu_star"])]),
                          ("maximal", lambda s: s[-1])]:
        chosen = [step_of(r["anytime"]) for r in aime]
        points[name] = {"accuracy": float(np.mean([bool(s["correct"]) for s in chosen])),
                        "cost": float(np.mean([s["cost_usd"] for s in chosen]))}
    weighted = [m for m in res["methods"]["weighted vote + program"] if m["benchmark"] != "math500"]
    points["weighted vote + program"] = {"accuracy": float(np.mean([m["final_correct"] for m in weighted])),
                                         "cost": float(np.mean([m["cost_usd"] for m in weighted]))}
    frontier_points = frontier(aime)
    return {"nl": nl_curve, "program": program_curve, "mathmage": points, "frontier": frontier_points,
            "unit_costs": {"nl_sample": u_nl, "program": u_program}, "weights": {"nl": w_nl, "program": w_program},
            "n_items": len(by_item), "n_seeds": n_seeds}


def fig_self_consistency(sc, out):
    import matplotlib.pyplot as plt
    P, M = style.P, style.METHOD
    if not sc:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.plot([p["cost"] for p in sc["nl"]], [p["accuracy"] for p in sc["nl"]], color=P["lav2"], marker="o", ms=3, lw=1.2,
            label="self-consistency, NL samples")
    for p in sc["nl"]:
        ax.annotate(f"{p['k']}", (p["cost"], p["accuracy"]), textcoords="offset points", xytext=(0, -9), ha="center",
                    fontsize=5.5, color=P["muted"])
    ax.plot([p["cost"] for p in sc["program"]], [p["accuracy"] for p in sc["program"]], color=P["cyan_deep"], marker="s",
            ms=3, lw=1.2, label="self-consistency, NL + program")
    for p in sc["program"]:
        ax.annotate(f"{3 * p['m']}+{p['m']}", (p["cost"], p["accuracy"]), textcoords="offset points", xytext=(0, -9),
                    ha="center", fontsize=5.5, color=P["cyan_deep"])
    ax.plot([p["cost"] for p in sc["program"]], [p["weighted_accuracy"] for p in sc["program"]], color=P["pink_deep"],
            marker="^", ms=3, lw=1.0, ls=(0, (4, 1.5)), label="ability-weighted SC, NL + program")
    front = pareto(sc["frontier"])
    ax.plot([p["cost_usd"] for p in front], [p["accuracy"] for p in front], color=M["mathmage"], lw=1.6,
            label="MathMage anytime frontier", zorder=4)
    for name, marker, color, label in [("posterior", "o", M["mathmage_light"], "MathMage posterior pick"),
                                       ("mu*", "*", P["pink_deep"], r"MathMage at dev $\mu^\star$"),
                                       ("maximal", "D", M["mathmage"], "MathMage maximal")]:
        p = sc["mathmage"][name]
        ax.scatter(p["cost"], p["accuracy"], marker=marker, s=40 if marker == "*" else 18, color=color, edgecolor=P["ink"],
                   lw=.4, zorder=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per problem (USD, log scale)")
    ax.set_ylabel("AIME 2024/2025 accuracy")
    ax.set_title(f"E2 · gpt-oss-120b, {sc['n_items']} problems × {sc['n_seeds']} seeds", fontsize=7.5, loc="left")
    ax.grid(True, color=P["grid"], lw=.5, which="both")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right")
    fig.tight_layout()
    style.save(fig, "e2_self_consistency", out)
    plt.close(fig)


def analyze_ablations(root, samples):
    rows = []
    for phase in ["ablations_primary", "ablations_secondary"]:
        if (root / phase / "manifest.json").exists() and (root / phase / "graded").exists():
            phase_rows, _, complete = load_phase(root, phase)
            for r in phase_rows:
                r["_phase_complete"] = complete
                r["_phase"] = phase
            rows.extend(phase_rows)
    by = defaultdict(list)
    for r in rows:
        if r["condition"] == "full" and r["_phase"] == "ablations_secondary":
            continue                                   # identical to the primary full condition (cache replay)
        by[r["condition"]].append(r)
    full = [base(r, bool(r["final_correct"]), r["cost_usd"]) for r in by.get("full", [])]
    out = {}
    for condition, rs in sorted(by.items()):
        method = [base(r, bool(r["final_correct"]), r["cost_usd"]) for r in rs]
        entry = {"n": len(rs), "accuracy": float(np.mean([m["final_correct"] for m in method])),
                 "macro": float(macro_accuracy(method)),
                 "cost": float(np.mean([r["cost_usd"] for r in rs])),
                 "marginal": float(np.mean([r["marginal_cost_usd"] for r in rs])),
                 "calls": float(np.mean([r["calls"] for r in rs])),
                 "complete": all(r["_phase_complete"] for r in rs) and len(rs) == len(full)}
        if condition != "full":
            shared = {(m["item_id"], m["seed"]) for m in method}
            reference = [m for m in full if (m["item_id"], m["seed"]) in shared]
            if len(reference) == len(method) and len(method) >= 5:
                entry["paired"] = paired_contrast(method, reference, samples=samples)
                entry["cost_delta"] = entry["cost"] - float(np.mean([m["cost_usd"] for m in reference]))
        out[condition] = entry
    return out


def analyze_mechanisms(root):
    from scipy.stats import binomtest
    out = {}
    for study in ["z6", "z3"]:
        directory = root / study / "graded"
        if not directory.exists():
            continue
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
        arms = defaultdict(list)
        for r in rows:
            arms[r["arm"]].append(r)
        entry = {"arms": {arm: {"n": len(rs), "accuracy": float(np.mean([bool(r["final_correct"]) for r in rs])),
                                "initial": float(np.mean([bool(r["initial_correct"]) for r in rs])),
                                "continuation_cost": float(np.mean([r.get("continuation_resource_cost_usd") or 0 for r in rs]))}
                          for arm, rs in arms.items()}}
        if study == "z6":
            gated = {r["item_id"]: r for r in arms.get("gated_review", [])}
            ungated = {r["item_id"]: r for r in arms.get("ungated_review", [])}
            pairs = [(bool(gated[i]["final_correct"]), bool(ungated[i]["final_correct"])) for i in gated if i in ungated]
            wins = sum(a and not b for a, b in pairs)
            losses = sum(b and not a for a, b in pairs)
            entry["paired"] = {"n": len(pairs), "gated_only": wins, "ungated_only": losses,
                               "mcnemar_exact_p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0}
        out[study] = entry
    return out


# ----------------------------------------------------------------------------- figures
def tag(ax, interim):
    if interim:
        ax.set_title("interim", fontsize=7, color=style.P["pink_deep"], loc="right")


def fig_frontiers(results, out):
    import matplotlib.pyplot as plt
    P, M = style.P, style.METHOD
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, res in zip(axes, results):
        rows, table = res["rows"], res["table"]
        points = frontier(rows)
        front = pareto(points)
        ax.plot([p["cost_usd"] for p in points], [p["accuracy"] for p in points], color=style.tint(M["mathmage"], .6), lw=.8)
        ax.plot([p["cost_usd"] for p in front], [p["accuracy"] for p in front], color=M["mathmage"], lw=1.5, marker="o",
                ms=2.2, label="MathMage anytime frontier", zorder=3)
        singles = [n for n in table if n in res["study"].members.values() or n == "single sample (mean of 3)"]
        for i, name in enumerate(singles):
            ax.scatter(table[name]["cost"], table[name]["macro"], s=15,
                       marker="o" if name == "single sample (mean of 3)" else "sD^"[i % 3],
                       color=[P["lav1"], P["lav2"], P["lav3"]][i % 3], edgecolor=P["ink"], lw=.4, zorder=4, label=name)
        ax.scatter(table["program"]["cost"], table["program"]["macro"], s=15, marker="v", color=P["cyan"], edgecolor=P["ink"],
                   lw=.4, zorder=4, label="program-of-thought")
        for name, color, marker, label in [("vote", M["member_vote"], "P", "majority vote"),
                                           ("weighted vote + program", P["cyan_deep"], "X", "weighted vote + program"),
                                           ("mu*", P["pink_deep"], "*", r"MathMage at dev $\mu^\star$")]:
            t = table[name]
            ax.errorbar(t["cost"], t["macro"], yerr=[[t["macro"] - t["low"]], [t["high"] - t["macro"]]], fmt=marker,
                        ms=6.5 if marker == "*" else 4.5, color=color, mec=P["ink"], mew=.4, elinewidth=.7, capsize=1.5,
                        zorder=5, label=label)
        for name, ls, va, dy in [("oracle (stage 0)", (0, (3, 2)), "top", -.004),
                                 ("oracle (final pool)", (0, (1, 1.4)), "bottom", .004)]:
            ax.axhline(table[name]["macro"], color=P["faint"], lw=.7, ls=ls, zorder=0)
            ax.text(.02, table[name]["macro"] + dy, name, ha="left", va=va, fontsize=6, color=P["muted"],
                    transform=ax.get_yaxis_transform())
        ax.set_title(res["study"].title, fontsize=8.5, loc="left")
        ax.set_xlabel("mean cost per problem (USD)")
        ax.grid(True, color=P["grid"], lw=.5)
        tag(ax, not res["complete"])
    axes[0].set_ylabel("macro accuracy")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(.5, -.02),
               handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(bottom=.31, top=.9, left=.08, right=.985, wspace=.22)
    style.save(fig, "frontiers", out)
    plt.close(fig)


def fig_components(results, out):
    import matplotlib.pyplot as plt
    P = style.P
    order = ["vote (members)", "vote + program", "vote + program as tie-breaker", "weighted vote + program",
             "posterior, no program answer",
             "posterior, no check evidence", "posterior, equal abilities", "posterior, difficulty fixed",
             "posterior, no trap mixture", "posterior (full)"]
    # Paired differences against the full posterior on identical saved answers: absolute
    # intervals would hide differences that are clear item by item.
    names = [n for n in order if n != "posterior (full)" and any(n in res["components"] for res in results)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, res in zip(axes, results):
        comps, deltas = res["components"], res["component_contrast"]
        full = comps["posterior (full)"]["macro"]
        for yi, n in zip(y, names):
            if n not in deltas:
                ax.text(.5, yi, "n/a (no trap type fitted)", transform=ax.get_yaxis_transform(), va="center",
                        ha="center", fontsize=6, color=P["muted"])
                continue
            d = deltas[n]
            color = P["pink_deep"] if d["high"] < 0 else P["cyan_deep"] if d["low"] > 0 else P["lav3"]
            ax.errorbar(100 * d["estimate"], yi, xerr=[[100 * (d["estimate"] - d["low"])], [100 * (d["high"] - d["estimate"])]],
                        fmt="o", ms=4, color=color, ecolor=color, elinewidth=1, capsize=2)
            ax.text(1.02, yi, f"{100 * comps[n]['macro']:.1f}", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=6.2, color=P["muted"], clip_on=False)
        ax.axvline(0, color=P["ink"], lw=.6)
        ax.set_yticks(y, names)
        ax.set_xlabel("change vs full posterior (pp, paired 95% CI)")
        ax.set_title(f"{res['study'].title} · full posterior {100 * full:.1f}", fontsize=8, loc="left")
        ax.grid(True, axis="x", color=P["grid"], lw=.5)
        tag(ax, not res["complete"])
    fig.tight_layout(w_pad=2.2)
    fig.subplots_adjust(right=.955)
    style.save(fig, "components", out)
    plt.close(fig)


def fig_calibration(results, out):
    import matplotlib.pyplot as plt
    P = style.P
    sources = [("posterior", style.METHOD["mathmage"]), ("vote share", style.METHOD["member_vote"]),
               ("weighted vote share", P["cyan_deep"])]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    for k, res in enumerate(results):
        a, b = axes[2 * k], axes[2 * k + 1]
        a.plot([0, 1], [0, 1], color=P["faint"], lw=.7, ls=(0, (3, 2)))
        for name, color in sources:
            cal = calibration(res["confidence"][name], bins=5)
            a.plot([r["confidence"] for r in cal["bins"]], [r["accuracy"] for r in cal["bins"]], color=color, lw=1.0,
                   marker="o", ms=2.6, label=f"{name} ({cal['ece']:.3f})")
            coverage, selective, aurc = risk_coverage(res["confidence"][name])
            b.plot(coverage, selective, color=color, lw=1.1, label=f"{name} ({aurc:.3f})")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        b.set_xlim(0, 1)
        b.set_ylim(.3, 1.02)
        a.set_xlabel("confidence")
        b.set_xlabel("coverage")
        a.set_ylabel("accuracy")
        b.set_ylabel("selective accuracy")
        a.set_title(res["study"].title.split(" · ")[0] + ": reliability (ECE)", fontsize=7, loc="left")
        b.set_title("risk–coverage (AURC)", fontsize=7, loc="left")
        a.legend(fontsize=5, frameon=False, loc="lower right", handlelength=1.2)
        b.legend(fontsize=5, frameon=False, loc="lower left", handlelength=1.2)
        for ax in (a, b):
            ax.grid(True, color=P["grid"], lw=.5)
            ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=.6)
    style.save(fig, "calibration", out)
    plt.close(fig)


def fig_actions(results, out):
    import matplotlib.pyplot as plt
    P = style.P
    kinds = [k for k in ACTIONS if any(res["actions"].get(k, {}).get("n") for res in results)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    y = np.arange(len(kinds))
    for ax, res in zip(axes, results):
        effects = res["actions"]
        rescue = [effects.get(k, {}).get("rescue", 0) for k in kinds]
        harm = [effects.get(k, {}).get("harm", 0) for k in kinds]
        n = [effects.get(k, {}).get("n", 0) for k in kinds]
        ax.barh(y, rescue, color=P["cyan"], edgecolor=P["cyan_deep"], lw=.5, label="rescued (wrong → right)")
        ax.barh(y, [-h for h in harm], color=P["pink"], edgecolor=P["pink_deep"], lw=.5, label="harmed (right → wrong)")
        top = max(rescue + harm + [1])
        for yi, count in zip(y, n):
            ax.text(top * 1.08, yi, f"n={count}", va="center", fontsize=6.2, color=P["muted"])
        ax.set_xlim(-top * 1.25, top * 1.55)
        ax.axvline(0, color=P["ink"], lw=.6)
        ax.set_yticks(y, [ACTION_LABEL[k] for k in kinds])
        ax.set_xlabel("decisions changed by the action")
        ax.set_title(res["study"].title, fontsize=8.5, loc="left")
        ax.grid(True, axis="x", color=P["grid"], lw=.5)
        tag(ax, not res["complete"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.3, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.01))
    fig.subplots_adjust(bottom=.26, top=.9, left=.15, right=.98, wspace=.12)
    style.save(fig, "actions", out)
    plt.close(fig)


def fig_voc(results, out):
    """Is the policy's value of computation calibrated? Predicted VOC vs realized change in correctness."""
    import matplotlib.pyplot as plt
    P = style.P
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    colors = [P["lav3"], P["cyan_deep"]]
    short = {"sample": "sample", "coach": "coach", "tool_alt": "program", "diversify": "diversify", "review": "review",
             "judge": "judge"}
    offsets = {("e1", "coach"): (0, -11, "center"), ("e1", "sample"): (6, -10, "left"), ("e1", "tool_alt"): (6, -9, "left"),
               ("e1", "judge"): (6, -5, "left"), ("e2", "review"): (2, 14, "left"), ("e2", "sample"): (5, 6, "left"),
               ("e2", "diversify"): (-5, 3, "right"),
               ("e2", "judge"): (-5, 5, "right")}
    right, low, high = .02, 0.0, .02
    for res, color, marker in zip(results, colors, ["o", "s"]):
        for kind, v in res["actions"].items():
            if v["n"] < 15:
                continue
            gain = np.array(v["gain"], float)
            x, y, se = float(np.mean(v["voc"])), float(gain.mean()), float(gain.std(ddof=1) / np.sqrt(len(gain)))
            right, low, high = max(right, x), min(low, y - 1.96 * se), max(high, y + 1.96 * se)
            ax.errorbar(x, y, yerr=1.96 * se, fmt=marker, ms=2.5 + 4 * np.sqrt(v["n"] / 213), color=color, mec=P["ink"],
                        mew=.4, elinewidth=.7, capsize=1.5, zorder=3)
            dx, dy, ha = offsets.get((res["study"].key, kind), (6, 1, "left"))
            ax.annotate(f"{short[kind]} ({v['n']})", (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=5.4, color=color)
        ax.scatter([], [], marker=marker, color=color, edgecolor=P["ink"], lw=.4, s=14, label=res["study"].title)
    lim = max(right * 1.3, high)
    ax.plot([0, lim], [0, lim], color=P["faint"], lw=.7, ls=(0, (3, 2)), zorder=0, label="perfectly calibrated")
    ax.axhline(0, color=P["ink"], lw=.5, zorder=0)
    ax.set_xlim(0, right * 1.3)
    ax.set_ylim(low * 1.1, high * 1.1)
    ax.set_xlabel("mean predicted value of computation (VOC)")
    ax.set_ylabel("realized change in correctness\nper action (95% CI)")
    ax.grid(True, color=P["grid"], lw=.5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    fig.tight_layout()
    style.save(fig, "voc_calibration", out)
    plt.close(fig)


def fig_errors(results, out):
    import matplotlib.pyplot as plt
    P = style.P
    kinds = [("correct", style.METHOD["mathmage"]), ("selection failure", P["cyan"]), ("coverage failure", P["peri_outer"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, res in zip(axes, results):
        benches = [b for b in BENCH if (b, "stage 0") in res["errors"]] + ["all"]
        labels, bars, xs = [], [], []
        for i, bench in enumerate(benches):
            for j, stage in enumerate(["stage 0", "final"]):
                counter = res["errors"][(bench, stage)]
                total = sum(counter.values())
                bars.append([counter[k] / total for k, _ in kinds])
                xs.append(i * 2.6 + j * 1.05)
                labels.append("stage 0" if j == 0 else "final")
        bars = np.array(bars)
        bottom = np.zeros(len(bars))
        for j, (kind, color) in enumerate(kinds):
            ax.bar(xs, bars[:, j], .95, bottom=bottom, color=color, edgecolor=P["ink"], lw=.25, label=kind,
                   hatch="////" if kind == "coverage failure" else None)
            bottom += bars[:, j]
        ax.set_xticks(xs, labels, fontsize=5.0)
        for i, bench in enumerate(benches):
            short = {"aime_2024": "AIME 24", "aime_2025": "AIME 25", "aime_2026": "AIME 26", "hmmt_feb_2026": "HMMT 26",
                     "tqa_phys": "TQA phys.", "math500": "MATH-500", "all": "All"}
            ax.text(i * 2.6 + .525, -.13, short[bench], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.2, color=P["ink"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of problems")
        ax.set_title(res["study"].title, fontsize=8.5, loc="left")
        tag(ax, not res["complete"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.3, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.01))
    fig.subplots_adjust(bottom=.3, top=.9, left=.07, right=.99, wspace=.18)
    style.save(fig, "errors", out)
    plt.close(fig)


def fig_ablations(ablations, out):
    import matplotlib.pyplot as plt
    P = style.P
    names = [c for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "C1", "C2", "R1"] if "paired" in ablations.get(c, {})]
    if not names:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 0.55 + .26 * len(names) + .55), sharey=True,
                               gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(names))[::-1]
    for yi, c in zip(y, names):
        d = ablations[c]["paired"]
        color = P["pink_deep"] if d["high"] < 0 else P["cyan_deep"] if d["low"] > 0 else P["lav3"]
        a.errorbar(100 * d["estimate"], yi, xerr=[[100 * (d["estimate"] - d["low"])], [100 * (d["high"] - d["estimate"])]],
                   fmt="o", ms=4, color=color, ecolor=color, elinewidth=1, capsize=2)
        a.text(100 * d["high"] + .8, yi, f"{d['wins']}/{d['losses']}", va="center", fontsize=6, color=P["muted"])
        b.barh(yi, ablations[c]["cost_delta"] * 1000, color=style.tint(color, .35), edgecolor=color, lw=.6)
    a.axvline(0, color=P["ink"], lw=.6)
    b.axvline(0, color=P["ink"], lw=.6)
    a.set_yticks(y, [f"{c} · {ABLATIONS[c]}" + ("" if ablations[c]["complete"] else " (interim)") for c in names],
                 fontsize=6.5)
    a.set_xlabel("accuracy change vs full MathMage (pp, 95% CI)")
    b.set_xlabel("cost change (USD / 1k problems)")
    for ax in (a, b):
        ax.grid(True, axis="x", color=P["grid"], lw=.5)
    fig.tight_layout(w_pad=.8)
    style.save(fig, "ablations", out)
    plt.close(fig)


def family_rate(trials, draws=4000, seed=0):
    """Success rate with a family-clustered bootstrap interval: trials sharing a problem family
    are resampled together instead of being treated as independent Bernoulli observations."""
    rng = np.random.default_rng(seed)
    fams = defaultdict(list)
    for t in trials:
        fams[t.get("family_id", t["item_id"])].append(float(bool(t["correct"])))
    keys = list(fams)
    if not keys:
        return {"k": 0, "n": 0, "rate": 0.0, "low": 0.0, "high": 0.0, "families": 0}
    boot = [np.mean([x for key in rng.choice(keys, len(keys)) for x in fams[key]]) for _ in range(draws)]
    k = int(sum(sum(v) for v in fams.values()))
    return {"k": k, "n": len(trials), "rate": k / len(trials), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975)), "families": len(keys)}


def fig_training(results, out):
    """Coaching evidence per roster and the budget-relative proximal zone."""
    import matplotlib.pyplot as plt
    from math import comb

    from scipy.special import expit
    P = style.P
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.1, 1]})
    x0, ticks, names = 0.0, [], []
    for res in results:
        evidence = json.loads((res["study"].artifact.parent / "training_evidence.json").read_text(encoding="utf-8"))
        trials = [t for t in evidence["tutoring_trials"] if t.get("correct") is not None]
        lifts = res["artifact"]["measurements"]["posterior_calibration"].get("coaching_lifts") or {}
        for j, (label, subset, fill, line, key) in enumerate([
                ("tutor right", [t for t in trials if t.get("tutor_correct")], P["cyan"], P["cyan_deep"], "coached_right"),
                ("tutor wrong", [t for t in trials if not t.get("tutor_correct")], P["pink"], P["pink_deep"], "coached_wrong")]):
            stats = family_rate(subset)
            n, k, rate, lo, hi = stats["n"], stats["k"], stats["rate"], stats["low"], stats["high"]
            x = x0 + j * .9
            a.bar(x, rate, .8, color=fill, edgecolor=line, lw=.6)
            a.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=line, elinewidth=.8, capsize=1.5)
            lift = lifts.get(key)
            a.text(x, min(.95, hi + .03), f"{k}/{n}" + (f"\n{lift:+.2f} logit" if lift is not None else ""),
                   ha="center", fontsize=5.8, color=P["ink"])
            ticks.append(x)
            names.append(f"{res['study'].title.split(' · ')[1]}\n{label}")
        x0 += 2.4
    a.set_xticks(ticks, names, fontsize=5.8)
    a.set_ylim(0, 1.08)
    a.set_ylabel("hinted re-solve correct")
    a.set_title("Coaching replays, training items (initially wrong solvers; family-clustered 95% CI)", fontsize=7,
                loc="left")
    a.grid(True, axis="y", color=P["grid"], lw=.5)
    Ks = [1, 2, 4, 8]
    grid = np.linspace(-7, 9, 1601)
    for res, palette in zip(results, [[P["lav1"], P["lav2"], P["lav4"]], [P["cyan_deep"]]]):
        pp = res["artifact"]["posterior_params"]
        prior = np.exp(-.5 * ((grid - pp["b_mean"]) / pp["b_sd"]) ** 2)
        prior /= prior.sum()
        rho, lift = pp["same_agent_correlation"], pp["lift"].get("coached_right", pp["lift"]["coached"])
        members = [k for k in pp["agents"] if "@" not in k and k != "tool"]
        for name, color in zip(members, palette):
            prof = pp["agents"][name]
            masses = []
            for K in Ks:
                conditional = expit(prof["a"] * (prof["theta"] - grid))
                p = (1 - prof["abstain"]) * conditional
                conc = (1 - rho) / rho
                none, wrong = np.zeros_like(p), np.ones_like(p)
                for usable in range(K + 1):
                    none += comb(K, usable) * (1 - prof["abstain"]) ** usable * prof["abstain"] ** (K - usable) * wrong
                    wrong *= (conc * (1 - conditional) + usable) / (conc + usable)
                c = (1 - prof["abstain"]) * expit(prof["a"] * (prof["theta"] + lift - grid))
                zone = np.where(p >= .5, 0, np.where(1 - none >= .5, 1, np.where(c >= .5, 2, 3)))
                masses.append(float(prior[zone == 2].sum()))
            b.plot(Ks, masses, marker="o", ms=3, lw=1.1, color=color, label=f"{name} (tutor-right lift {lift:+.2f})")
    b.set_xscale("log", base=2)
    b.set_xticks(Ks, [str(k) for k in Ks])
    b.set_xlabel("fresh attempts instead of one hint ($K$)")
    b.set_ylabel("proximal-zone share of problems")
    b.set_title("Budget-relative proximal zone", fontsize=7.5, loc="left")
    b.grid(True, color=P["grid"], lw=.5)
    b.legend(fontsize=5.6, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    style.save(fig, "training_coaching_zpd", out)
    plt.close(fig)


def fig_zones(results, out):
    import matplotlib.pyplot as plt
    P = style.P
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, res in zip(axes, results):
        rows = res["rows"]
        zones = [z for z in ["A", "L", "U", "P"] if any(r["zone"] == z for r in rows)]
        width = .2
        series = [("vote", lambda r: r["initial_correct"], style.METHOD["member_vote"]),
                  ("posterior", lambda r: r["anytime"][0]["correct"], style.METHOD["mathmage_light"]),
                  ("maximal", lambda r: r["final_correct"], style.METHOD["mathmage"]),
                  ("oracle (final pool)", lambda r: r["pool_covered"], P["peri_outer"])]
        for i, (label, fn, color) in enumerate(series):
            values = [np.mean([bool(fn(r)) for r in rows if r["zone"] == z]) for z in zones]
            ax.bar(np.arange(len(zones)) + (i - 1.5) * width, values, width, color=color, edgecolor=P["ink"], lw=.3, label=label)
        counts = [sum(r["zone"] == z for r in rows) for z in zones]
        extra = [np.mean([r["anytime"][-1]["cost_usd"] - r["anytime"][0]["cost_usd"] for r in rows if r["zone"] == z])
                 for z in zones]
        ax.set_xticks(range(len(zones)), [f"zone {z}: {n} ({100 * n / len(rows):.0f}%)\n+{1000 * c:.1f} USD per 1k"
                                          for z, n, c in zip(zones, counts, extra)], fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_title(res["study"].title, fontsize=8.5, loc="left")
        ax.grid(True, axis="y", color=P["grid"], lw=.5)
        tag(ax, not res["complete"])
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=6, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout(w_pad=1.0)
    style.save(fig, "zones", out)
    plt.close(fig)


def fig_seeds(res, out):
    import matplotlib.pyplot as plt
    P, M = style.P, style.METHOD
    if not res["seeds"]:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    colors = {"vote": M["member_vote"], "posterior": M["mathmage_light"], "mu*": P["pink_deep"], "maximal": M["mathmage"]}
    for i, (name, data) in enumerate(res["seeds"].items()):
        xs = i + np.linspace(-.18, .18, len(data["per_seed"]))
        ax.scatter(xs, data["per_seed"], s=10, color=colors[name], edgecolor=P["ink"], lw=.3, zorder=3)
        ax.errorbar(i, data["mean"], yerr=data["sd"], fmt="_", ms=14, color=P["ink"], elinewidth=.8, capsize=3, zorder=4)
    ax.set_xticks(range(len(res["seeds"])), [r"$\mu^\star$" if n == "mu*" else n for n in res["seeds"]])
    ax.set_ylabel("AIME accuracy per seed")
    ax.set_title("E2 · eight seeds (mean ± sample SD)", fontsize=8, loc="left")
    ax.grid(True, axis="y", color=P["grid"], lw=.5)
    fig.tight_layout()
    style.save(fig, "e2_seeds", out)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def pct(x):
    return f"{100 * x:.1f}"


def write_tables(results, ablations, mechanisms, out):
    lines = []
    for res in results:
        study, table = res["study"], res["table"]
        benches = [b for b in BENCH if any(b in t["by_benchmark"] for t in table.values())]
        order = ([n for n in study.members.values()] if not study.homogeneous else ["single sample (mean of 3)"]) + \
            ["program", "vote", "vote + program", "weighted vote + program", "posterior", "mu*", "maximal",
             "oracle (stage 0)", "oracle (final pool)"]
        label = {"vote": "Majority vote (members)", "vote + program": "Plurality vote incl. program",
                 "weighted vote + program": "Ability-weighted vote incl. program", "posterior": "MathMage posterior pick (stage 0)",
                 "mu*": "MathMage at dev threshold mu*", "maximal": "MathMage maximal (4 actions)",
                 "program": "Program-of-thought answer", "single sample (mean of 3)": "Single gpt-oss-120b sample (mean of 3)"}
        lines += [f"## {study.title}" + ("  (INTERIM)" if not res["complete"] else ""), "",
                  f"n = {len(res['rows'])} evaluations; macro accuracy over benchmarks with 95% family-clustered"
                  + (" and crossed-seed" if res["seeds"] else "") + " bootstrap CIs; cost = mean resource USD per problem.", "",
                  "| Method | " + " | ".join(BENCH[b] for b in benches) + " | Macro [95% CI] | $/problem |",
                  "|---" * (len(benches) + 3) + "|"]
        for name in order:
            t = table[name]
            cells = [pct(t["by_benchmark"][b]) if b in t["by_benchmark"] else "--" for b in benches]
            note = ""
            if not study.homogeneous and name == res["train_best"]:
                note = " (strongest on training data)"
            elif not study.homogeneous and name == res["best_single"]:
                note = " (best on test, hindsight)"
            lines.append(f"| {label.get(name, name)}{note} | " + " | ".join(cells)
                         + f" | {pct(t['macro'])} [{pct(t['low'])}, {pct(t['high'])}] | {t['cost']:.4f} |")
        lines += ["", "Paired differences (pp, clustered 95% CI; wins/losses; Holm-adjusted exact McNemar where pairs are independent):"]
        for key, c in res["contrast"].items():
            p = c.get("holm_p")
            lines.append(f"- {key}: {100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], "
                         f"{c['wins']}/{c['losses']}" + (f", Holm p = {p:.3g}" if p is not None else ""))
        if res.get("post_cutoff"):
            pc = res["post_cutoff"]
            lines += ["", f"Post-cutoff contests only (AIME 2026 + HMMT Feb 2026, n = {pc['n']}; contamination-free): "
                      + "; ".join(f"{name} {pct(t['macro'])} [{pct(t['low'])}, {pct(t['high'])}]" for name, t in pc["table"].items())]
            lines += [f"- {key}: {100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], {c['wins']}/{c['losses']}"
                      for key, c in pc["contrast"].items()]
        m = res["reproduction_mismatches"]
        lines += ["", f"Stage-0 component analysis on identical saved answers. Reconstruction vs the logged decision: "
                      f"chosen answer differs on {m['choice']}/{m['n']} items, correctness on {m['correctness']}/{m['n']}. "
                      f"The ability-weighted vote incl. program picks the same answer cluster as the full posterior on "
                      f"{m['weighted_vote_same_choice']}/{m['n']} items.", "",
                  "| Aggregation rule | Macro [95% CI] | vs full posterior (pp) | $/problem |", "|---|---|---|---|"]
        for name, t in res["components"].items():
            d = res["component_contrast"].get(name)
            lines.append(f"| {name} | {pct(t['macro'])} [{pct(t['low'])}, {pct(t['high'])}] | "
                         + ("—" if d is None else f"{100 * d['estimate']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]")
                         + f" | {t['cost']:.4f} |")
        lines += ["", "Zones (assigned by the stage-0 posterior; accuracy per method; extra spend after stage 0):", "",
                  "| Zone | n (share) | vote | posterior | mu* | maximal | oracle final | +$/1k at mu* | +$/1k maximal | actions (max) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for z, s in res["zone_stats"].items():
            lines.append(f"| {z} | {s['n']} ({pct(s['share'])}%) | {pct(s['vote'])} | {pct(s['posterior'])} | {pct(s['mu*'])} | "
                         f"{pct(s['maximal'])} | {pct(s['oracle (final pool)'])} | {1000 * s['extra_cost_mu']:.2f} | "
                         f"{1000 * s['extra_cost_max']:.2f} | {s['actions_max']:.2f} |")
        lines += ["", "Confidence quality (stage 0): " + "; ".join(
            f"{name}: ECE {res['calibration'][name]['ece']:.3f}, Brier {res['calibration'][name]['brier']:.3f}, AURC {res['aurc'][name]:.3f}"
            for name in res["calibration"]), ""]
        if res["seeds"]:
            lines.append("AIME accuracy across 8 seeds (mean ± sample SD): " + "; ".join(
                f"{name} {pct(d['mean'])} ± {pct(d['sd'])}" for name, d in res["seeds"].items()))
            lines.append("")
        lines.append("Actions (rescues/harms of the decision; n): " + "; ".join(
            f"{ACTION_LABEL[k]} {v['rescue']}/{v['harm']} (n={v['n']}, +{1000 * np.mean(v['cost']):.2f} USD per 1k)"
            for k, v in sorted(res["actions"].items(), key=lambda kv: -kv[1]["n"])))
        lines += ["", "Predicted value of computation vs realized change in correctness (per action; maximal runs):", "",
                  "| Action set | n | rescues | harms | net per action | mean predicted VOC | mean +$/action |",
                  "|---|---|---|---|---|---|---|"]
        split = res["actions_by_threshold"]
        for key in [k for k in ACTIONS if k in split] + ["within mu*", "beyond mu*"]:
            if key not in split:
                continue
            v = split[key]
            lines.append(f"| {ACTION_LABEL.get(key, key)} | {v['n']} | {v['rescue']} | {v['harm']} | "
                         f"{(v['rescue'] - v['harm']) / v['n']:+.3f} | {np.mean(v['voc']):.3f} | {np.mean(v['cost']):.5f} |")
        lines.append("")
        lines.append("Zones: " + ", ".join(f"{z} {n}" for z, n in sorted(res["zones"].items()))
                     + "; stop reasons: " + ", ".join(f"{k} {v}" for k, v in res["stops"].items()))
        errs = res["errors"]
        lines.append("Error taxonomy (all benchmarks): " + "; ".join(
            f"{stage}: " + ", ".join(f"{k} {errs[('all', stage)][k]}" for k in ["correct", "selection failure", "coverage failure"])
            for stage in ["stage 0", "final"]))
        lines.append("")
    if ablations:
        lines += ["## Ablations (SAT roster, 45 AIME 2026 / HMMT Feb 2026 problems, maximal runs)", "",
                  "Macro accuracy weights the two benchmarks equally, like the paired Δ; micro accuracy (all 45 items) "
                  "in parentheses. McNemar is the unweighted item-pair test.", "",
                  "| Condition | n | Macro (micro) acc. | Δ vs full [95% CI] (wins/losses) | McNemar p | $/problem | Δ cost |",
                  "|---|---|---|---|---|---|---|"]
        for condition, e in ablations.items():
            name = "full MathMage" if condition == "full" else f"{condition} · {ABLATIONS.get(condition, condition)}"
            d = e.get("paired")
            p = None if not d else d.get("mcnemar_exact_p")
            lines.append(f"| {name}{'' if e['complete'] else ' (interim)'} | {e['n']} | {pct(e['macro'])} ({pct(e['accuracy'])}) | "
                         + (f"{100 * d['estimate']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}] ({d['wins']}/{d['losses']})"
                            if d else "—")
                         + f" | {'' if p is None else format(p, '.3g')} | {e['cost']:.4f} | "
                         + ("" if "cost_delta" not in e else
                            f"{e['cost_delta']:+.4f} ({100 * e['cost_delta'] / (e['cost'] - e['cost_delta']):+.0f}%)") + " |")
        lines.append("")
    if mechanisms:
        lines += ["## Mechanism panels (10 untouched development families)", ""]
        for study, e in mechanisms.items():
            arms = "; ".join(f"{arm}: {pct(a['accuracy'])}% of {a['n']} (initial {pct(a['initial'])}%, "
                             f"continuation {1000 * a['continuation_cost']:.2f} USD per 1k)" for arm, a in e["arms"].items())
            lines.append(f"- {study.upper()}: {arms}" + (
                f"; paired discordant gated-only {e['paired']['gated_only']} vs ungated-only {e['paired']['ungated_only']}, "
                f"exact McNemar p = {e['paired']['mcnemar_exact_p']:.3g}" if "paired" in e else ""))
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


TEX_METHOD = {"vote": "Majority vote (members)", "vote + program": "Plurality incl.\\ program",
              "weighted vote + program": "Ability-weighted vote incl.\\ program",
              "posterior": "\\textsc{MathMage} posterior pick", "mu*": "\\textsc{MathMage} at dev $\\mu^\\star$",
              "maximal": "\\textsc{MathMage} maximal", "program": "Program-of-thought answer",
              "single sample (mean of 3)": "Single sample (mean of 3)", "oracle (stage 0)": "\\textit{Oracle: stage-0 pool}",
              "oracle (final pool)": "\\textit{Oracle: final pool}"}
TEX_BENCH = {"aime_2024": "AIME'24", "aime_2025": "AIME'25", "aime_2026": "AIME'26", "hmmt_feb_2026": "HMMT'26",
             "tqa_phys": "TQA-phys.", "math500": "MATH-500"}


def write_latex(results, ablations, out):
    """Booktabs versions of the main tables (requires \\usepackage{booktabs})."""
    def tex(name):
        return TEX_METHOD.get(name, name.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
                              .replace("×", "$\\times$"))
    component_label = {"posterior (full)": "Posterior (full)", "posterior, no program answer": "\\quad no program answer",
                       "posterior, no check evidence": "\\quad no partial-check evidence",
                       "posterior, equal abilities": "\\quad equal abilities",
                       "posterior, difficulty fixed": "\\quad difficulty fixed",
                       "posterior, no trap mixture": "\\quad no trap mixture", "vote (members)": "Majority vote (members)",
                       "vote + program": "Plurality incl.\\ program",
                       "vote + program as tie-breaker": "Plurality, program breaks ties",
                       "weighted vote + program": "Ability-weighted vote incl.\\ program"}

    blocks = ["% Auto-generated by analysis/results.py; regenerate instead of editing by hand.", ""]
    for res in results:
        study, table = res["study"], res["table"]
        benches = [b for b in BENCH if any(b in t["by_benchmark"] for t in table.values())]
        members = list(study.members.values()) if not study.homogeneous else ["single sample (mean of 3)"]
        methods = members + ["program", "vote", "vote + program", "weighted vote + program", "posterior", "mu*", "maximal"]
        best = max(table[m]["macro"] for m in methods)
        rows = []
        for name in methods + ["oracle (stage 0)", "oracle (final pool)"]:
            t = table[name]
            if name == "oracle (stage 0)":
                rows.append("\\midrule")
            label = tex(name)
            if not study.homogeneous and name == res["train_best"]:
                label += "$^\\dagger$"
            macro = f"{pct(t['macro'])}"
            macro = f"\\textbf{{{macro}}}" if abs(t["macro"] - best) < 1e-12 else macro
            cost = "--" if name.startswith("oracle") else f"{t['cost']:.4f}"
            cells = [pct(t["by_benchmark"][b]) if b in t["by_benchmark"] else "--" for b in benches]
            rows.append(f"{label} & " + " & ".join(cells) + f" & {macro} \\scriptsize[{pct(t['low'])}, {pct(t['high'])}] & {cost} \\\\")
        if study.homogeneous and "single sample (mean of 3)" in methods:
            rows[0] = rows[0].replace("Single sample", "Single gpt-oss-120b sample")
        caption = (f"{study.title.replace('·', '--')}: accuracy (\\%) per benchmark, macro accuracy with 95\\% family-clustered "
                   "bootstrap CI" + (" (crossed with seeds)" if res["seeds"] else "") + ", and mean cost per problem (USD). "
                   + ("$^\\dagger$Strongest member on the 16 training problems." if not study.homogeneous else ""))
        blocks += ["\\begin{table*}[t]", "\\centering", "\\small", "\\resizebox{\\textwidth}{!}{%",
                   f"\\begin{{tabular}}{{l{'c' * len(benches)}cc}}", "\\toprule",
                   "Method & " + " & ".join(TEX_BENCH[b] for b in benches) + " & Macro [95\\% CI] & \\$/problem \\\\",
                   "\\midrule", *rows, "\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}",
                   f"\\label{{tab:{study.key}}}", "\\end{table*}", ""]
    # Stage-0 decomposition, both rosters side by side.
    names = ["posterior (full)", "posterior, no program answer", "posterior, no check evidence", "posterior, equal abilities",
             "posterior, difficulty fixed", "posterior, no trap mixture", "vote (members)", "vote + program",
             "vote + program as tie-breaker", "weighted vote + program"]
    rows = []
    for name in names:
        cells = []
        for res in results:
            t, d = res["components"].get(name), res["component_contrast"].get(name)
            if t is None:
                cells += ["--", "--"]
                continue
            cells += [pct(t["macro"]), "--" if d is None else
                      f"{100 * d['estimate']:+.1f} \\scriptsize[{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]"]
        rows.append(("\\midrule\n" if name == "vote (members)" else "") + f"{component_label[name]} & "
                    + " & ".join(cells) + " \\\\")
    blocks += ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\columnwidth}{!}{%",
               "\\begin{tabular}{lcccc}", "\\toprule",
               " & \\multicolumn{2}{c}{E1 (SAT roster)} & \\multicolumn{2}{c}{E2 (open roster)} \\\\",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Stage-0 rule & Macro & $\\Delta$ vs full & Macro & $\\Delta$ vs full \\\\",
               "\\midrule", *rows, "\\bottomrule", "\\end{tabular}}",
               "\\caption{Stage-0 decomposition on identical saved answers: paired change (points, 95\\% clustered CI) "
               "relative to the full posterior.}", "\\label{tab:components}", "\\end{table}", ""]
    if ablations:
        rows = []
        full_cost = ablations.get("full", {}).get("cost")
        for condition, e in ablations.items():
            if condition == "full":
                continue
            d = e.get("paired")
            delta = (f"{100 * d['estimate']:+.1f} \\scriptsize[{100 * d['low']:+.1f}, {100 * d['high']:+.1f}] ({d['wins']}/{d['losses']})"
                     if d else "--")
            cost = f"{100 * e['cost_delta'] / (e['cost'] - e['cost_delta']):+.0f}\\%" if "cost_delta" in e else "--"
            rows.append(f"{condition} & {tex(ABLATIONS.get(condition, condition))}{'' if e['complete'] else ' (interim)'} & "
                        f"{pct(e['macro'])} & {delta} & {cost} \\\\")
        blocks += ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\columnwidth}{!}{%",
                   "\\begin{tabular}{llccc}", "\\toprule",
                   " & Condition & Macro & $\\Delta$ vs full (W/L) & $\\Delta$ cost \\\\", "\\midrule",
                   f" & Full \\textsc{{MathMage}} & {pct(ablations['full']['macro'])} & -- & \\${full_cost:.4f}/problem \\\\"
                   if full_cost else "", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}}",
                   "\\caption{Ablations on 45 AIME 2026 / HMMT Feb 2026 problems (SAT roster, maximal runs), paired with the "
                   "full system: macro accuracy, paired change with 95\\% clustered CI and wins/losses, and relative cost.}",
                   "\\label{tab:ablations}", "\\end{table}", ""]
    (out / "tables.tex").write_text("\n".join(blocks), encoding="utf-8")


def write_parameter_table(results, out):
    rows = ["| Parameter | " + " | ".join(r["study"].title for r in results) + " |", "|---" * (len(results) + 1) + "|"]

    def get(res, path, fmt="{:.3f}"):
        value = res["artifact"]
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return fmt.format(value) if isinstance(value, (int, float)) else "—"
    entries = [("item difficulty spread b_sd", ["posterior_params", "b_sd"]),
               ("wrong-answer collision kappa", ["posterior_params", "kappa"]),
               ("same-model collision kappa_same", ["posterior_params", "kappa_same"]),
               ("trap-item weight", ["posterior_params", "trap_weight"]),
               ("trap collision kappa_trap", ["posterior_params", "kappa_trap"]),
               ("same-model correlation rho", ["posterior_params", "same_agent_correlation"]),
               ("judge accuracy", ["posterior_params", "judge_accuracy"]),
               ("hint acceptance", ["posterior_params", "hint_acceptance"]),
               ("coaching lift, pooled (logit)", ["posterior_params", "lift", "coached"]),
               ("coaching lift, tutor right", ["posterior_params", "lift", "coached_right"]),
               ("coaching lift, tutor wrong", ["posterior_params", "lift", "coached_wrong"]),
               ("dev threshold mu*", ["mu_star"])]
    for label, path in entries:
        rows.append(f"| {label} | " + " | ".join(get(r, path) for r in results) + " |")
    # Raw coaching replays on training items, with family-clustered bootstrap intervals
    # (trials sharing a problem family are resampled together, not treated as independent).
    for subset, label in [(True, "hinted re-solve correct, tutor right"), (False, "hinted re-solve correct, tutor wrong")]:
        cells = []
        for res in results:
            evidence = json.loads((res["study"].artifact.parent / "training_evidence.json").read_text(encoding="utf-8"))
            trials = [t for t in evidence["tutoring_trials"] if t.get("correct") is not None and bool(t.get("tutor_correct")) == subset]
            s = family_rate(trials)
            cells.append(f"{s['k']}/{s['n']} = {pct(s['rate'])}% [{pct(s['low'])}, {pct(s['high'])}] ({s['families']} families)"
                         if s["n"] else "—")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    rows.append("")
    for res in results:
        thetas = ", ".join(f"{k} {v['theta']:+.2f}" for k, v in res["artifact"]["posterior_params"]["agents"].items())
        rows.append(f"- abilities theta, {res['study'].title}: {thetas}")
    (out / "parameters.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, Study):
        return value.title
    if isinstance(value, dict):
        return {(" / ".join(k) if isinstance(k, tuple) else str(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()
    tables, figures = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    style.use(publication=not args.draft)
    from mathmage.data.registry import load_registry

    registry = load_registry()
    results = [analyze_study(study, registry, args.samples) for study in STUDIES]
    ablations = analyze_ablations(RUNS / "program_v3", args.samples)
    mechanisms = analyze_mechanisms(RUNS / "program_v3")
    sc = self_consistency(results[1], registry)
    write_tables(results, ablations, mechanisms, tables)
    cases, case_summary = write_cases(results, registry, tables)
    with (tables / "results.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## How actions changed decisions (mechanical; details in cases.md)\n\n"
                     "| Study | Outcome | Mechanism | Count |\n|---|---|---|---|\n")
        for (study, outcome), counter in sorted(case_summary.items()):
            for how, n in counter.most_common():
                stream.write(f"| {study.upper()} | {outcome} | {how} | {n} |\n")
    if sc:
        lines = ["## E2 · matched-compute self-consistency (AIME 2024/2025, pooled seeds; no new calls)", "",
                 "| Method | Accuracy | $/problem |", "|---|---|---|"]
        lines += [f"| SC@{p['k']} (NL samples) | {pct(p['accuracy'])} | {p['cost']:.4f} |" for p in sc["nl"]]
        lines += [f"| SC over {3 * p['m']} NL + {p['m']} program | {pct(p['accuracy'])} | {p['cost']:.4f} |" for p in sc["program"]]
        lines += [f"| ability-weighted SC over {3 * p['m']} NL + {p['m']} program (weights {sc['weights']['nl']:.2f} / "
                  f"{sc['weights']['program']:.2f}) | {pct(p['weighted_accuracy'])} | {p['cost']:.4f} |" for p in sc["program"]]
        lines += [f"| {name} | {pct(p['accuracy'])} | {p['cost']:.4f} |" for name, p in
                  [("ability-weighted vote incl. program", sc["mathmage"]["weighted vote + program"]),
                   ("MathMage posterior pick", sc["mathmage"]["posterior"]), ("MathMage at dev mu*", sc["mathmage"]["mu*"]),
                   ("MathMage maximal", sc["mathmage"]["maximal"])]]
        with (tables / "results.md").open("a", encoding="utf-8") as stream:
            stream.write("\n" + "\n".join(lines) + "\n")
    write_parameter_table(results, tables)
    write_latex(results, ablations, tables)
    summary = {}
    for res in results:
        entry = {k: to_jsonable(v) for k, v in res.items()
                 if k in ("table", "contrast", "components", "component_contrast", "calibration", "aurc", "seeds", "zones",
                          "stops", "mu_star", "complete", "reproduction_mismatches", "best_single", "train_best", "zone_stats",
                          "post_cutoff")}
        entry["actions"] = {k: {"n": v["n"], "rescue": v["rescue"], "harm": v["harm"], "mean_voc": float(np.mean(v["voc"])),
                                "mean_gain": float(np.mean(v["gain"])), "mean_cost": float(np.mean(v["cost"]))}
                            for k, v in res["actions"].items()}
        entry["errors"] = {f"{b} / {s}": dict(c) for (b, s), c in res["errors"].items()}
        summary[res["study"].key] = entry
    summary["ablations"] = to_jsonable(ablations)
    summary["mechanisms"] = mechanisms
    summary["self_consistency_e2"] = to_jsonable({k: v for k, v in (sc or {}).items() if k != "frontier"})
    summary["action_cases"] = {f"{s} / {o}": dict(c) for (s, o), c in case_summary.items()}
    (tables / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    fig_frontiers(results, figures)
    fig_components(results, figures)
    fig_calibration(results, figures)
    fig_actions(results, figures)
    fig_errors(results, figures)
    fig_ablations(ablations, figures)
    fig_training(results, figures)
    fig_zones(results, figures)
    fig_seeds(results[1], figures)
    fig_self_consistency(sc, figures)
    fig_voc(results, figures)
    print((tables / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
