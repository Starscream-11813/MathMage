"""v6 analysis: fresh post-cutoff contests, the mechanism layer and MathMage-R (read-only; no API calls).

    python analysis/v6_results.py            # tables + figures (LuaLaTeX)
    python analysis/v6_results.py --draft    # fast previews (fewer bootstrap draws)

Implements the analysis plan frozen in runs/live_20261018/program_v6/plan.json (stage 1) and
plan_sat26.json (stage 2), both written before any v6 result of their package existed:

  Primary (Holm within each team's five contrasts)
    H9   sat26 team (GPT-5-mini low, Claude Haiku 4.5, DeepSeek-V3), 187 fresh problems, one seed:
         MathMage-16 at mu* vs majority vote, DeepSeek-V3 SC@10, USC, one-round debate, MoA
    H10  open roster (3 x gpt-oss-120b), 187 fresh problems x 8 seeds: mu* vs majority vote, SC@5 from the 24
         pooled stage-0 samples, USC, debate, MoA
  Secondary
    S6   MathMage-R vs MathMage-16 (open roster), each at its own knee-rule price and at the maximal run
    S7   sat26 calibration beside the published SAT roster's (descriptive)
    S8   hint trial: rescue after a right / wrong tutor's hint vs fresh attempts and a placebo re-solve; harm
    S9   VOC calibration: predicted value of computation vs realized change in decision correctness
    S11  dense ZPD sub-study (open roster, if run)
    S12  sat26 strongest-member self-consistency (GPT-5-mini SC@k) at matched cost and at k = 5
A job that is incomplete is reported as such, never silently pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v4_results as V4R  # noqa: E402
import v5_results as V5R  # noqa: E402
from mathmage.analysis.stats import holm  # noqa: E402
from mathmage.data.registry import Registry  # noqa: E402


def install_grading_guard():
    """Answer equivalence through the fourth wave's killable grading child (scripts/v6_worker_guarded.py).

    The frozen thread-based timeout cannot stop exact evaluations that hold the interpreter lock (a fresh-set
    answer contains 5^(2^32)); the guard returns the frozen helper's timeout value instead of hanging. The
    grading code is identical in every package copy, and the guard agrees with it on every pair it finishes.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("v6_worker_guarded", REPO / "scripts" / "v6_worker_guarded.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    import functools
    memo = functools.lru_cache(maxsize=None)(guard.guarded_exact_correct)     # a pure function of two strings
    guard.EQ.exact_correct = memo
    guard.VERIFY.exact_correct = memo
    return guard


GUARD = install_grading_guard()

_BOOT_CACHE = {}


def cached_bootstrap(rows, field, samples, seed=0):
    """analysis/v4_results.bootstrap with its resampling indices cached per unit structure.

    The draws depend only on the rows' (family, benchmark, seed) cells and their order, so every method row and
    paired contrast over the same units reuses one set of index arrays; the values are then exactly those of
    v4's function (same generator, same cells, same order)."""
    names = sorted({r["benchmark"] for r in rows})
    key = (tuple((r["family_id"], r["benchmark"], int(r["seed"])) for r in rows), samples, seed)
    if key not in _BOOT_CACHE:
        code = {b: i for i, b in enumerate(names)}
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
        draws = []
        for _ in range(samples):
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
            draws.append(np.concatenate(parts))
        _BOOT_CACHE[key] = (benches, draws)
    benches, draws = _BOOT_CACHE[key]
    values = np.array([float(r[field]) for r in rows])
    out = np.empty(len(draws))
    for d, index in enumerate(draws):
        out[d] = V4R._macro(values[index], benches[index], len(names))
    return V4R._macro(values, benches, len(names)), out


ORIGINAL_BOOTSTRAP = V4R.bootstrap
V4R.bootstrap = cached_bootstrap

R = V4R.R
V6 = R.RUNS / "program_v6"
REG6 = REPO / "data" / "registry_v6"
OUT_T, OUT_F = REPO / "results/tables", REPO / "results/figures"
CELLS = ("hmmt_nov_2025_guts", "pumac_2025", "hmmt_feb_2026_guts", "cmimc_2026", "brumo_2026")
LABEL = {"hmmt_nov_2025_guts": "HMMT Nov 2025", "pumac_2025": "PUMaC 2025", "hmmt_feb_2026_guts": "HMMT Feb 2026",
         "cmimc_2026": "CMIMC 2026", "brumo_2026": "BrUMO 2026"}
SAT26 = R.Study("s26", "sat26 team", V6 / "sat26", "fresh_mm", V6 / "offline_sat26" / "artifacts.json",
                {"s0-0": "GPT-5-mini (low)", "s0-1": "Claude Haiku 4.5", "s0-2": "DeepSeek-V3"}, False)
OPEN = R.STUDIES[1]
ALL_CORRECT_WEIGHT = {"sat26": 1, "lane": 4}
TAU_ZONE = 0.5                                   # Definition 1's threshold, as in the paper's ZPD figure


class Registry6(V4R.Combined):
    """Item lookup across the frozen registry, the v4 extension and the v6 fresh registry."""

    def __init__(self):
        super().__init__()
        self.v6 = Registry.from_frame(pd.read_parquet(REG6 / "items.parquet"))

    def __getitem__(self, item_id):
        return self.v6[item_id] if item_id in self.v6 else super().__getitem__(item_id)


# ----------------------------------------------------------------------------- loading
def folder(package, job):
    return V6 / package / job


def graded(package, job):
    directory = folder(package, job) / "graded"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))] if directory.exists() else []


def mathmage_rows(package, job):
    """Graded MathMage rows with the call, candidate, step and trace fields the v3 analysis reads."""
    rows = []
    for path in sorted((folder(package, job) / "graded").glob("*.json")) if (folder(package, job) / "graded").exists() else []:
        row = json.loads(path.read_text(encoding="utf-8"))
        record = json.loads((folder(package, job) / "agent" / path.name).read_text(encoding="utf-8"))
        row["_calls"] = [{k: c.get(k) for k in ("phase", "role", "model_key", "resource_cost_usd")}
                         for c in record.get("meta", {}).get("calls", [])]
        row["_candidates"] = [{k: c.get(k) for k in ("id", "phase", "model_key", "answer", "valid", "evidence",
                                                     "parent_id")} for c in record.get("candidates", [])]
        row["_steps"] = [{k: s.get(k) for k in ("best_candidate", "best_answer", "confidence", "chosen")}
                         for s in record.get("meta", {}).get("anytime", [])]
        row["_trace"] = [t for t in record.get("trace", []) if t.get("event") in ("policy_action", "judge", "hint", "review")]
        rows.append(row)
    return rows


def plan_units(package):
    name = "plan.json" if package == "lane" else f"plan_{package}.json"
    path = V6 / name
    if not path.exists():
        return {}
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {j["id"]: j["n_units"] for j in plan["jobs"][package]}


def status(package):
    units = plan_units(package)
    return {job: {"graded": len(graded(package, job)), "planned": n} for job, n in units.items()}


def complete(package, job):
    units = plan_units(package)
    return job in units and len(graded(package, job)) == units[job]


# ----------------------------------------------------------------------------- fresh-set methods
def summarize(rows, samples):
    """v4's summary plus the bootstrap standard deviation of the macro estimate (the paper's subscripts)."""
    estimate, draws = V4R.bootstrap(rows, "final_correct", samples)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"macro": estimate, "low": float(lo), "high": float(hi), "sd": float(np.std(draws, ddof=1)),
            "cost": float(np.mean([r["cost_usd"] for r in rows])), "n": len(rows),
            "by_benchmark": {b: float(np.mean([r["final_correct"] for r in rows if r["benchmark"] == b]))
                             for b in sorted({r["benchmark"] for r in rows})}}


def team_methods(package, registry, samples):
    """MathMage-16 operating points and every baseline of one team on the fresh set (unit = problem x seed)."""
    study = SAT26 if package == "sat26" else OPEN
    rows = mathmage_rows(package, "fresh_mm")
    if not rows or not study.artifact.exists():
        return None
    artifact = json.loads(study.artifact.read_text(encoding="utf-8"))
    m = V4R.mathmage_methods(rows, study, artifact, registry)
    methods = {name: m[name] for name in m}
    methods["debate"] = V4R.debate_outcomes(graded(package, "fresh_debate"), registry)
    methods["USC"] = V4R.usc_outcomes(graded(package, "fresh_usc"))
    methods["MoA"] = V5R.moa_outcomes(graded(package, "fresh_moa"))
    curves = {}
    if package == "sat26":
        sc = V4R.sc_outcomes(graded(package, "fresh_sc"), registry, list(range(1, 11)))
        best = V4R.sc_outcomes(graded(package, "fresh_sc_best"), registry, list(range(1, 6)))
        methods.update({f"SC@{k} (DeepSeek-V3)": v for k, v in sc.items()})
        methods.update({f"SC@{k} (GPT-5-mini)": v for k, v in best.items()})
        curves = {"DeepSeek-V3": sc, "GPT-5-mini": best}
        primary = ["vote", "SC@10 (DeepSeek-V3)", "USC", "debate", "MoA"]
    else:
        pooled = V4R.pooled_sc(rows, registry, [1, 3, 5, 8, 12, 24])
        methods.update({f"SC@{k} (pooled seeds)": v for k, v in pooled.items()})
        curves = {"pooled": pooled}
        primary = ["vote", "SC@5 (pooled seeds)", "USC", "debate", "MoA"]
    return {"study": study, "artifact": artifact, "rows": rows, "methods": methods, "curves": curves,
            "primary": primary, "mu_star": float(artifact["mu_star"])}


def primary_block(team, samples):
    methods = team["methods"]
    contrasts = {f"mu* vs {b}": V4R.contrast(methods["mu*"], methods[b], samples) for b in team["primary"]
                 if methods.get(b)}
    keys = [k for k, c in contrasts.items() if c]
    for key, adjusted in zip(keys, holm([contrasts[k]["p"] for k in keys])):
        contrasts[key]["holm_p"] = adjusted
    secondary = {f"maximal vs {b}": V4R.contrast(methods["maximal"], methods[b], samples) for b in team["primary"]
                 if methods.get(b)}
    return {"contrasts": contrasts, "secondary": secondary}


def matched_sc(team, family, samples):
    """S12: the largest k whose mean SC@k cost does not exceed MathMage-16's at mu*, and the largest k run."""
    curve = team["curves"].get(family) or {}
    if not curve:
        return None
    target = float(np.mean([r["cost_usd"] for r in team["methods"]["mu*"]]))
    costs = {k: float(np.mean([r["cost_usd"] for r in v])) for k, v in curve.items() if v}
    if not costs:
        return None
    under = [k for k, c in costs.items() if c <= target]
    k_match = max(under) if under else min(costs)
    k_max = max(costs)
    out = {"mu_star_cost": target, "costs": costs, "k_matched": k_match, "k_max": k_max,
           "contrasts": {f"mu* vs SC@{k_match} ({family}, matched cost)": V4R.contrast(team["methods"]["mu*"], curve[k_match], samples),
                         f"mu* vs SC@{k_max} ({family})": V4R.contrast(team["methods"]["mu*"], curve[k_max], samples)}}
    return out


# ----------------------------------------------------------------------------- MathMage-R
def mathmage_r(registry, samples):
    """S6: MathMage-R (refitted artifact, VOC factors, its own price) vs MathMage-16 on the open roster."""
    price_path, artifact_path = V6 / "price_r_lane.json", V6 / "artifacts_r_lane.json"
    rows = mathmage_rows("lane", "fresh_r")
    if not rows or not price_path.exists():
        return None
    price = json.loads(price_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    m_r = V4R.mathmage_methods(rows, OPEN, {**artifact, "mu_star": price["mu_star_r"]}, registry)
    base_rows = mathmage_rows("lane", "fresh_mm")
    m_16 = V4R.mathmage_methods(base_rows, OPEN, json.loads(OPEN.artifact.read_text(encoding="utf-8")), registry)
    return {"mu_star_r": price["mu_star_r"], "dev_operating_point": price.get("dev_operating_point"),
            "voc_factors": artifact.get("voc_factors"),
            "methods": {"MathMage-16 mu*": summarize(m_16["mu*"], samples),
                        "MathMage-16 maximal": summarize(m_16["maximal"], samples),
                        "MathMage-R mu*_R": summarize(m_r["mu*"], samples),
                        "MathMage-R maximal": summarize(m_r["maximal"], samples),
                        "MathMage-R posterior (stage 0)": summarize(m_r["posterior"], samples),
                        "MathMage-16 posterior (stage 0)": summarize(m_16["posterior"], samples)},
            "contrasts": {"R mu*_R vs 16 mu*": V4R.contrast(m_r["mu*"], m_16["mu*"], samples),
                          "R maximal vs 16 maximal": V4R.contrast(m_r["maximal"], m_16["maximal"], samples),
                          "R posterior vs 16 posterior": V4R.contrast(m_r["posterior"], m_16["posterior"], samples)}}


# ----------------------------------------------------------------------------- mechanism layer
def mech_records(package):
    """Graded wave-M rows joined with the hint events of their agent records (audits, delivery)."""
    out = graded(package, "mech")
    agent = folder(package, "mech") / "agent"
    events = {}
    for path in sorted(agent.glob("*.json")) if agent.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["item_id"], int(record["seed"]))
        events[key] = []
        for branch in record["arms"]:
            hint = next((t for t in branch["trace"] if t.get("event") == "mech_hint"), None)
            events[key].append({"arm": branch["arm"], "hint": hint})
    for row in out:
        row["_events"] = events.get((row["item_id"], int(row["seed"])), [])
    return out


def _cluster_mean(values, clusters, draws=2000, seed=0):
    """Mean with a problem-clustered bootstrap 95% CI."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if not len(values):
        return None
    names = sorted(set(clusters))
    index = {c: np.flatnonzero(clusters == c) for c in names}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        pick = rng.integers(0, len(names), len(names))
        idx = np.concatenate([index[names[j]] for j in pick])
        boot.append(values[idx].mean())
    lo, hi = np.quantile(boot, [.025, .975])
    return {"mean": float(values.mean()), "low": float(lo), "high": float(hi), "n": int(len(values)),
            "clusters": len(names)}


def hint_trial(package, draws):
    """S8. Unit: an initially wrong member of a recorded state (wrong-tutor harm: an initially right member).

    Rescue = the arm's new answer is correct. An undelivered hint (screened out by the structural audit and
    answer filter) yields no new answer and counts as no rescue (intention to treat); delivered-only rates
    are reported beside it. Paired differences are taken within the same member and state."""
    rows = mech_records(package)
    per = defaultdict(lambda: defaultdict(list))
    clusters = defaultdict(list)
    harm = {"itt": [], "delivered": [], "clusters": []}
    for row in rows:
        correct0 = row["stage0_member_correct"]
        by_arm = {}
        for graded_arm, event in zip(row["arms"], row["_events"]):
            arm = graded_arm["arm"]
            by_arm[json.dumps(arm, sort_keys=True)] = (graded_arm, event.get("hint"))
        cluster = row["family_id"]
        for s, ok in enumerate(correct0):
            if ok:
                continue
            outcome = {}
            for key, (g, hint) in by_arm.items():
                arm = json.loads(key)
                new = g["new_correct"]
                if arm["kind"] == "sample" and arm["member"] == s:
                    outcome[f"fresh k={arm['k']}"] = bool(new and new[0])
                elif arm["kind"] == "placebo" and arm["solver"] == s:
                    outcome["placebo"] = bool(new and new[0])
                elif arm["kind"] == "hint" and arm["solver"] == s and arm["arm"] in ("right", "wrong"):
                    delivered = bool(hint and hint.get("delivered"))
                    residual = bool(hint and (hint.get("residual_audit") or {}).get("residual_pass"))
                    outcome[f"{arm['arm']} hint"] = bool(new and new[0])
                    outcome[f"{arm['arm']} delivered"] = delivered
                    outcome[f"{arm['arm']} residual pass"] = residual
                    outcome[f"{arm['arm']} level"] = arm["level"]
            if "fresh k=1" in outcome and "fresh k=2" in outcome:
                outcome["best of two fresh"] = outcome["fresh k=1"] or outcome["fresh k=2"]
            for key, value in outcome.items():
                per[key][len(clusters[key])] = value
                clusters[key].append(cluster)
            per["_units"][len(clusters["_units"])] = outcome
            clusters["_units"].append(cluster)
        for key, (g, hint) in by_arm.items():
            arm = json.loads(key)
            if arm["kind"] == "hint" and arm["arm"] == "harm":
                new = g["new_correct"]
                delivered = bool(hint and hint.get("delivered"))
                harm["itt"].append(float(delivered and not (new and new[0])))
                harm["delivered"].append(float(not (new and new[0])) if delivered else np.nan)
                harm["clusters"].append(cluster)
    units = list(per["_units"].values())
    unit_clusters = clusters["_units"]

    def rate(field, condition=None):
        values, cl = [], []
        for u, c in zip(units, unit_clusters):
            if field in u and (condition is None or condition(u)):
                values.append(float(u[field]))
                cl.append(c)
        return _cluster_mean(values, cl, draws)

    def paired(a, b, condition=None):
        values, cl = [], []
        for u, c in zip(units, unit_clusters):
            if a in u and b in u and (condition is None or condition(u)):
                values.append(float(u[a]) - float(u[b]))
                cl.append(c)
        return _cluster_mean(values, cl, draws)

    out = {"n_states": len(rows), "n_wrong_members": len(units),
           "rates": {"fresh attempt (k=1)": rate("fresh k=1"), "best of two fresh attempts": rate("best of two fresh"),
                     "placebo re-solve": rate("placebo"), "right tutor's hint (ITT)": rate("right hint"),
                     "right tutor's hint (delivered)": rate("right hint", lambda u: u.get("right delivered")),
                     "right tutor's hint (residual-audit pass)": rate("right hint", lambda u: u.get("right residual pass")),
                     "wrong tutor's hint (ITT)": rate("wrong hint"),
                     "wrong tutor's hint (delivered)": rate("wrong hint", lambda u: u.get("wrong delivered"))},
           "paired": {"right hint - fresh (k=1)": paired("right hint", "fresh k=1"),
                      "right hint - best of two fresh": paired("right hint", "best of two fresh"),
                      "right hint - wrong hint": paired("right hint", "wrong hint"),
                      "right hint - placebo": paired("right hint", "placebo"),
                      "placebo - fresh (k=1)": paired("placebo", "fresh k=1"),
                      "right hint - fresh (k=1), residual pass": paired("right hint", "fresh k=1",
                                                                        lambda u: u.get("right residual pass"))},
           "delivery": {"right": rate("right delivered"), "wrong": rate("wrong delivered"),
                        "right residual pass": rate("right residual pass")},
           "harm": {"wrong tutor's hint on a right member (ITT)": _cluster_mean(harm["itt"], harm["clusters"], draws),
                    "delivered only": _cluster_mean([v for v in harm["delivered"] if v == v],
                                                    [c for v, c in zip(harm["delivered"], harm["clusters"]) if v == v],
                                                    draws)}}
    by_level = {}
    for level in sorted({u.get("right level") for u in units if u.get("right level")}):
        by_level[level] = paired("right hint", "fresh k=1", lambda u, lv=level: u.get("right level") == lv)
    out["right hint - fresh by format"] = by_level
    # Descriptive, beside the prespecified rows: every arm on the units that have a right tutor (a teammate was
    # right), so the arms compare like with like; and the paired contrasts on delivered hints only (delivery is
    # decided after assignment, so this subset is selected and the contrast is not randomized).
    has_right = lambda u: "right hint" in u  # noqa: E731
    out["rates (right-tutor units)"] = {
        "fresh attempt (k=1)": rate("fresh k=1", has_right), "best of two fresh attempts": rate("best of two fresh", has_right),
        "placebo re-solve": rate("placebo", has_right), "wrong tutor's hint (ITT)": rate("wrong hint", has_right),
        "right tutor's hint (ITT)": rate("right hint"),
        "right tutor's hint (delivered)": rate("right hint", lambda u: u.get("right delivered"))}
    delivered = lambda u: bool(u.get("right delivered"))  # noqa: E731
    out["paired (delivered right hints)"] = {
        "right hint - fresh (k=1)": paired("right hint", "fresh k=1", delivered),
        "right hint - best of two fresh": paired("right hint", "best of two fresh", delivered),
        "right hint - placebo": paired("right hint", "placebo", delivered)}
    return out


def voc_calibration(package, draws):
    """S9. Swept arms at stage-0 states: frozen predicted VOC vs realized change in decision correctness.

    Realized = post-arm posterior pick correct minus stage-0 pick correct, both under the frozen parameters.
    Open-roster all-correct states carry their inverse sampling weight (4)."""
    rows = mech_records(package)
    kinds = {"sample": lambda a: a["kind"] == "sample" and a.get("k") == 1, "tool_alt": lambda a: a["kind"] == "tool_alt",
             "diversify": lambda a: a["kind"] == "diversify", "review": lambda a: a["kind"] == "review",
             "judge": lambda a: a["kind"] == "judge", "coach (right tutor)": lambda a: a["kind"] == "hint" and a["arm"] == "right"}
    out = {}
    for kind, test in kinds.items():
        predicted, realized, weights, cl = [], [], [], []
        for row in rows:
            weight = 1 if not all(row["stage0_member_correct"]) else ALL_CORRECT_WEIGHT[package]
            for g in row["arms"]:
                if test(g["arm"]) and g["predicted_voc"] is not None:
                    predicted.append(float(g["predicted_voc"]))
                    realized.append(float(g["post_correct"]) - float(row["stage0_correct"]))
                    weights.append(weight)
                    cl.append(row["family_id"])
        if not predicted:
            continue
        w = np.asarray(weights, float)
        out[kind] = {"n": len(predicted), "mean_predicted": float(np.average(predicted, weights=w)),
                     "mean_realized": float(np.average(realized, weights=w)),
                     "realized_ci": _cluster_mean(np.repeat(realized, w.astype(int)),
                                                  np.repeat(cl, w.astype(int)), draws),
                     "share_helped": float(np.average(np.asarray(realized) > 0, weights=w)),
                     "share_hurt": float(np.average(np.asarray(realized) < 0, weights=w))}
    return out


def dense(draws):
    """S11. Per problem: the solver's success rate over eight fresh attempts (p) and over its hinted re-solves (c),
    the latter split by whether the hint's tutor was right at stage 0 (from the saved record), by intention to treat
    (an undelivered hint is a failure) and on delivered hints. Zones follow Definition 1 at tau = 1/2 with plug-in
    rates: autonomous p >= tau; latent p < tau <= r_K, r_K = 1 - (1 - p)^K; proximal r_K < tau <= c (right tutors);
    unreachable otherwise. Eight draws per arm make every per-problem label noisy; the shares are descriptive."""
    rows = graded("lane", "dense")
    if not rows:
        return None
    agent = folder("lane", "dense") / "agent"
    records = {}
    for path in sorted(agent.glob("*.json")) if agent.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        records[(record["item_id"], int(record["seed"]))] = record
    units = []
    for row in rows:
        record = records.get((row["item_id"], int(row["seed"])), {})
        member_correct = record.get("stage0_member_correct") or []
        hints = record.get("hints") or []
        right_itt, right_del, wrong_itt = [], [], []
        for h, ok in zip(hints, row["hint_correct"]):
            tutor_right = bool(member_correct[h["tutor"]]) if h.get("tutor") is not None and h["tutor"] < len(member_correct) else None
            success = bool(ok) if ok is not None else False
            if tutor_right:
                right_itt.append(success)
                if ok is not None:
                    right_del.append(bool(ok))
            elif tutor_right is False:
                wrong_itt.append(success)
        units.append({"family": row["family_id"], "p": float(np.mean(row["retry_correct"])),
                      "c_right_itt": float(np.mean(right_itt)) if right_itt else None,
                      "c_right_del": float(np.mean(right_del)) if right_del else None,
                      "c_wrong_itt": float(np.mean(wrong_itt)) if wrong_itt else None})

    def zone_shares(field, K):
        counts = Counter()
        for u in units:
            p, c = u["p"], u[field]
            if p >= TAU_ZONE:
                counts["autonomous"] += 1
            elif 1 - (1 - p) ** K >= TAU_ZONE:
                counts["latent"] += 1
            elif c is not None and c >= TAU_ZONE:
                counts["proximal"] += 1
            else:
                counts["unreachable"] += 1
        total = sum(counts.values())
        return {k: counts[k] / total for k in ("autonomous", "latent", "proximal", "unreachable")}

    fam = [u["family"] for u in units]
    out = {"n": len(units), "retry_rate": _cluster_mean([u["p"] for u in units], fam, draws)}
    for field in ("c_right_itt", "c_right_del", "c_wrong_itt"):
        vals = [(u[field], u["family"]) for u in units if u[field] is not None]
        out[f"hint_rate ({field})"] = _cluster_mean([v for v, _ in vals], [f for _, f in vals], draws) if vals else None
    out["zones"] = {f"K={K}, {field}": zone_shares(field, K) for K in (1, 2) for field in ("c_right_itt", "c_right_del")}
    paired = [(u["c_right_itt"] - u["p"], u["family"]) for u in units if u["c_right_itt"] is not None]
    out["right hint (ITT) - retry"] = _cluster_mean([v for v, _ in paired], [f for _, f in paired], draws) if paired else None
    return out


def calibration_transfer():
    """S7: the sat26 calibration beside the published SAT roster's (both frozen offline artifacts)."""
    out = {}
    for name, path in [("SAT roster (published)", R.STUDIES[0].artifact), ("sat26 team", SAT26.artifact)]:
        if not path.exists():
            continue
        a = json.loads(path.read_text(encoding="utf-8"))
        p = a["posterior_params"]
        out[name] = {"theta": {k: v["theta"] for k, v in p["agents"].items() if "@" not in k},
                     "b_sd": p.get("b_sd"), "rho": p.get("same_agent_correlation"), "kappa": p.get("kappa"),
                     "kappa_same": p.get("kappa_same"), "trap_weight": p.get("trap_weight"),
                     "lift": p.get("lift"), "judge_accuracy": p.get("judge_accuracy"), "mu_star": a.get("mu_star")}
    return out


# ----------------------------------------------------------------------------- analysis
def analyze(samples):
    registry = Registry6()
    out = {"status": {"sat26": status("sat26"), "lane": status("lane")}}
    for package, key in (("sat26", "H9"), ("lane", "H10")):
        team = team_methods(package, registry, samples)
        if team is None:
            continue
        out[key] = {"methods": {k: summarize(v, samples) for k, v in team["methods"].items() if v},
                    **primary_block(team, samples), "mu_star": team["mu_star"],
                    "complete": {job: complete(package, job) for job in plan_units(package) if job.startswith("fresh_")}}
        if package == "sat26":
            out["S12"] = matched_sc(team, "GPT-5-mini", samples)
            out["S12_deepseek"] = matched_sc(team, "DeepSeek-V3", samples)
        out[f"_curves_{package}"] = {fam: {k: (float(np.mean([r["cost_usd"] for r in v])), V4R.macro_accuracy(v))
                                           for k, v in curve.items() if v} for fam, curve in team["curves"].items()}
        out[f"_frontier_{package}"] = V4R.frontier(team["rows"])
        # Descriptive price curve for the figure: every distinct prefix of the maximal run (macro accuracy, mean
        # cost), thinned to at most 60 points evenly spaced in log cost; selecting a price on it would be an oracle.
        distinct = {}
        for p in out[f"_frontier_{package}"]:
            distinct.setdefault((round(p["cost_usd"], 7), p["accuracy"]), p["mu"])
        curve = sorted((c, a, m) for (c, a), m in distinct.items())
        if len(curve) > 60:
            costs = np.log([c for c, _, _ in curve])
            grid = np.linspace(costs[0], costs[-1], 60)
            keep = sorted({int(np.argmin(np.abs(costs - g))) for g in grid} | {0, len(curve) - 1})
            curve = [curve[i] for i in keep]
        out[f"_price_curve_{package}"] = [{"cost": c, "accuracy": a, "mu": m} for c, a, m in curve]
    out["S6"] = mathmage_r(registry, samples)
    out["S7"] = calibration_transfer()
    draws = min(samples, 2000)
    out["S8"] = {p: hint_trial(p, draws) for p in ("sat26", "lane") if graded(p, "mech")}
    out["S9"] = {p: voc_calibration(p, draws) for p in ("sat26", "lane") if graded(p, "mech")}
    out["S11"] = dense(draws)
    return out


# ----------------------------------------------------------------------------- output
def fmt_rate(r):
    return "--" if not r else f"{100 * r['mean']:.1f} [{100 * r['low']:.1f}, {100 * r['high']:.1f}] (n = {r['n']})"


def write(out, path):
    lines = ["# v6 results (pre-registered; see runs/live_20261018/program_v6/PREREGISTRATION*.md)", ""]
    for package, s in out["status"].items():
        lines.append(f"- {package}: " + ", ".join(f"{job} {v['graded']}/{v['planned']}" for job, v in s.items()))
    lines.append("")
    for key, title in (("H9", "sat26 team, 187 fresh problems (one seed)"),
                       ("H10", "open roster, 187 fresh problems x 8 seeds")):
        if key not in out:
            continue
        h = out[key]
        lines += [f"## {key} · {title}", ""]
        lines += V4R.method_table(h["methods"], f"Accuracy and cost (mu* = {h['mu_star']:.4g})")
        lines += ["Primary contrasts (paired macro difference, pp; 95% clustered CI; wins/losses):"]
        lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in h["contrasts"].items()]
        lines += ["", "Secondary (maximal run):"] + [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in h["secondary"].items()]
        lines += ["", "Jobs complete: " + ", ".join(f"{j} {'yes' if v else 'NO'}" for j, v in h["complete"].items()), ""]
    for key in ("S12", "S12_deepseek"):
        if out.get(key):
            s = out[key]
            lines += [f"## {key} · self-consistency at matched cost", "",
                      f"MathMage-16 mu* cost ${s['mu_star_cost']:.4f}; SC@k cost: " +
                      ", ".join(f"k={k} ${c:.4f}" for k, c in sorted(s["costs"].items())), ""]
            lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in s["contrasts"].items()] + [""]
    if out.get("S6"):
        s = out["S6"]
        lines += ["## S6 · MathMage-R vs MathMage-16 (open roster)", "", f"mu*_R = {s['mu_star_r']:.4g}; VOC factors: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in (s["voc_factors"] or {}).items()), ""]
        lines += [f"- {k}: {V4R.pct(t['macro'])} [{V4R.pct(t['low'])}, {V4R.pct(t['high'])}], ${t['cost']:.4f}"
                  for k, t in s["methods"].items()]
        lines += [f"- {k}: {V4R.fmt_contrast(c)}" for k, c in s["contrasts"].items()] + [""]
    if out.get("S7"):
        lines += ["## S7 · calibration transfer (descriptive)", ""]
        for name, s in out["S7"].items():
            lines.append(f"- {name}: theta " + ", ".join(f"{k} {v:+.2f}" for k, v in s["theta"].items()) +
                         f"; b_sd {s['b_sd']:.2f}, rho {s['rho']:.2f}, kappa {s['kappa']:.3f}, kappa_same "
                         f"{s['kappa_same']:.3f}, trap {s['trap_weight']:.2f}, judge {s['judge_accuracy']:.2f}, "
                         f"mu* {s['mu_star']:.4g}")
        lines.append("")
    for package, s in (out.get("S8") or {}).items():
        lines += [f"## S8 · hint trial, {package} ({s['n_states']} states, {s['n_wrong_members']} wrong members)", ""]
        lines += [f"- {k}: {fmt_rate(r)}" for k, r in s["rates"].items()]
        lines += ["", "Paired (pp):"] + [f"- {k}: {fmt_rate(r)}" for k, r in s["paired"].items()]
        lines += ["", "Delivery:"] + [f"- {k}: {fmt_rate(r)}" for k, r in s["delivery"].items()]
        lines += ["", "Harm:"] + [f"- {k}: {fmt_rate(r)}" for k, r in s["harm"].items()]
        lines += ["", "Right hint minus one fresh attempt, by hint format:"]
        lines += [f"- {k}: {fmt_rate(r)}" for k, r in s["right hint - fresh by format"].items()] + [""]
        lines += ["Descriptive: every arm on right-tutor units (like with like):"]
        lines += [f"- {k}: {fmt_rate(r)}" for k, r in s.get("rates (right-tutor units)", {}).items()]
        lines += ["", "Descriptive: paired on delivered right hints only (selected subset):"]
        lines += [f"- {k}: {fmt_rate(r)}" for k, r in s.get("paired (delivered right hints)", {}).items()] + [""]
    for package, s in (out.get("S9") or {}).items():
        lines += [f"## S9 · VOC calibration, {package}", "", "| Action | n | mean predicted | mean realized [95% CI] | helped | hurt |",
                  "|---|---|---|---|---|---|"]
        for kind, v in s.items():
            ci = v["realized_ci"]
            lines.append(f"| {kind} | {v['n']} | {100 * v['mean_predicted']:.2f} | {100 * v['mean_realized']:.2f} "
                         f"[{100 * ci['low']:.2f}, {100 * ci['high']:.2f}] | {100 * v['share_helped']:.1f}% | "
                         f"{100 * v['share_hurt']:.1f}% |")
        lines.append("")
    if out.get("S11"):
        s = out["S11"]
        lines += [f"## S11 · dense ZPD (open roster, {s['n']} problems)", "", f"- retry rate: {fmt_rate(s['retry_rate'])}"]
        lines += [f"- {k}: {fmt_rate(v)}" for k, v in s.items() if k.startswith("hint_rate") or k.startswith("right hint")]
        lines += [f"- zones {k}: " + ", ".join(f"{z} {100 * v:.1f}%" for z, v in shares.items())
                  for k, shares in s["zones"].items()] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, R.Study):
        return value.key
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    samples = args.samples or (300 if args.draft else 2000)
    out = analyze(samples)
    OUT_T.mkdir(parents=True, exist_ok=True)
    write(out, OUT_T / "v6_results.md")
    (OUT_T / "v6_results.json").write_text(json.dumps(to_jsonable({k: v for k, v in out.items()
                                                                   if not k.startswith("_frontier")}), indent=1),
                                           encoding="utf-8")
    print((OUT_T / "v6_results.md").read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
