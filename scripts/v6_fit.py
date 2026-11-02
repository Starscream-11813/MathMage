"""MathMage-R: the team posterior refitted on a larger calibration pool (pre-registered with the v6 plan).

The estimators and priors are those of the published calibration (mathmage/psychometrics/calibrate.py),
applied to more data:

  * the original 16 training problems: stage-0 panels, extra samples, public MathArena anchors and collision
    panels, review, discernment and hint-replay trials, rebuilt from the frozen offline observations (the
    rebuilt response count is checked against the frozen calibration's own count);
  * the recorded non-fresh evaluation runs of the open roster (E2 and the 2025-26 contests, 2,308
    problem-seed runs): stage-0 panels (members and program),
    certificate-check outcomes, and the reviews and judge picks of their maximal runs;
  * the v6 mechanism layer (wave M): bank attempts and alternative programs as repeated responses, sweep
    reviews and judges, the crossed hint trial (tutor-dependent lifts) and the diversified re-solves
    (diversify lift, previously a prior).

No fresh (registry_v6) outcome is read. Abilities, difficulty spread, same-author correlation and collision
rates use one record per problem (seeds and repeated responses merged); lifts are scored against each state's
own stage-0 panel, as at test time. VOC factors per action type come from the non-adaptive wave-M sweep under
the refitted parameters: (sum realized + 5 m) / (sum predicted + 5 m), clipped to [0.1, 1], with m the mean
predicted value; coaching takes the fresh-attempt factor (no outcome-independent coaching arm exists).

MathMage-R is pre-registered for the open roster only (the sat26 team has no recorded non-fresh runs).

    python scripts/v6_fit.py fit --package lane      # writes program_v6/artifacts_r_lane.json (no calls)
    python scripts/v6_fit.py price --package lane    # after wave D: mu*_R by the frozen knee rule (no calls)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v6_followup", REPO / "scripts" / "v6_followup.py")
F6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F6)

from mathmage.analysis.frontier import frontier, operating_point  # noqa: E402
from mathmage.config import load_models  # noqa: E402
from mathmage.data.registry import load_registry  # noqa: E402
from mathmage.experiments.artifacts import digest, file_digest, freeze, read_frozen, read_record  # noqa: E402
from mathmage.grading.equiv import normalize  # noqa: E402
from mathmage.prompts.templates import SOLVE  # noqa: E402
from mathmage.psychometrics.calibrate import (CoachingRecord, EvidenceRecord, ItemRecord, Response, _abstain,  # noqa: E402
                                              _observations, fit_coaching_lifts, fit_evidence, fit_irt,
                                              fit_kappa_mixture)
from mathmage.team.engine import extract_answer  # noqa: E402
from mathmage.team.policy import Action, Policy, PolicyConfig  # noqa: E402
from mathmage.team.posterior import AgentProfile, Obs, Evidence, PosteriorParams, TeamPosterior  # noqa: E402

V = F6.V
RUN_SOURCES = {"lane": ["primary", "new_e2_mathmage_nov", "new_e2_mathmage_apr"]}
TOOL_PHASES = {"tool_forward", "verify_forward", "tool_alt"}
FACTOR_KINDS = ("sample", "tool_alt", "diversify", "review", "judge")
PSEUDO, FLOOR = 5.0, 0.1


def agent_of(candidate):
    return "tool" if candidate.get("phase") in TOOL_PHASES else candidate["model_key"]


def usable(candidate):
    return bool(candidate.get("valid", True) and not candidate.get("error") and candidate.get("answer"))


# ----------------------------------------------------------------------------- the original training data
def training_data(package):
    """Rebuild the frozen offline calibration's records (checked against its recorded response count)."""
    root = F6.offline_root(package)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evidence = read_record(root / "training_evidence.json", plan_id=manifest["plan_id"], mode=manifest["mode"])
    registry = load_registry()
    roster = load_models().rosters[F6.SPECS[package]["roster"]]

    def observation(name):
        path = root / "observations" / (digest(name)[:24] + ".json")
        return read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"],
                           identity={"observation_id": digest(name)})["data"]

    items = {}
    for item_id in manifest["train_ids"]:
        item = registry[item_id]
        grade = F6.correct_of(item_id)
        base = observation(["base", item_id])
        responses = [Response(agent_of(c), c["answer"] if usable(c) else None, bool(usable(c) and grade(c["answer"])))
                     for c in base["stage0"]]
        kw = {"problem": item.problem, "format_instruction": item.view().format_instruction, "cards": ""}
        for k in range(manifest["options"]["extra_samples"]):
            for i, member in enumerate(roster.members):
                r = observation([item_id, member, SOLVE.id, f"extra:{k}:member={i}", kw])
                raw = extract_answer(r["text"]) if not r.get("error") and not r.get("truncated") else None
                answer = normalize(raw, item.answer_format) if raw else None
                responses.append(Response(member, answer, bool(answer and grade(answer))))
        items[item_id] = ItemRecord(item_id, item.answer_format, responses, item.family_id)
    for record in evidence["public_anchor_snapshot"]:
        target = items.setdefault(record["item_id"], ItemRecord(record["item_id"], record["answer_format"], [],
                                                               record.get("family_id", "")))
        target.responses.extend(Response(**r) for r in record["responses"])
    frozen = read_frozen(F6.SPECS[package]["artifact"], live=True)
    expected = frozen["measurements"]["posterior_calibration"]["n_responses"]
    rebuilt = sum(len(it.responses) for it in items.values())
    if rebuilt != expected:
        raise ValueError(f"rebuilt training panels have {rebuilt} responses; the frozen calibration used {expected}")
    kappa_panels = [ItemRecord(r["item_id"], r["answer_format"], [Response(**x) for x in r["responses"]],
                               r.get("family_id", "")) for r in evidence["public_kappa_snapshot"]]
    reviews = [EvidenceRecord(r["kind"], r["target_correct"]) for r in evidence["review_trials"]]
    judges = [bool(t["correct_choice"]) for t in evidence["discernment_trials"]]
    coaching = [CoachingRecord(r["item_id"], r["solver"], bool(r["correct"]), bool(r["tutor_correct"]))
                for r in evidence["tutoring_trials"] if r["correct"] is not None]
    hints = list({(r["item_id"], r["tutor_instance"], r["hint_id"]): r["audit_accepted"]
                  for r in evidence["tutoring_trials"]}.values())
    return {"items": items, "kappa_panels": kappa_panels, "reviews": reviews, "judges": judges,
            "coaching": coaching, "hints": hints, "train_ids": manifest["train_ids"]}


# ----------------------------------------------------------------------------- the recorded evaluation runs
def recorded_runs(package):
    """Stage-0 panels, certificate checks, reviews and judge picks of the frozen non-fresh runs."""
    panels, states, checks, reviews, judges = defaultdict(list), {}, [], [], []
    formats, families = {}, {}
    for source in RUN_SOURCES[package]:
        folder = F6.recorded_folder(package, source)
        for path in sorted((folder / "graded").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            item_id, seed = row["item_id"], int(row.get("seed", 0))
            record = F6.recorded_record(package, source, item_id, seed)
            grades = {cid: bool(g["correct"]) for cid, g in row["candidate_grades"].items()}
            item = F6.item_of(item_id)
            formats[item_id], families[item_id] = item.answer_format, item.family_id
            stage0 = record["checkpoint"]["stage0"]
            responses = [Response(agent_of(c), c["answer"] if usable(c) else None, bool(usable(c) and grades.get(c["id"])))
                         for c in stage0]
            panels[item_id].extend(responses)
            states[f"{item_id}#s{seed}"] = ItemRecord(f"{item_id}#s{seed}", item.answer_format, list(responses),
                                                      item.family_id)
            for c in stage0:
                ev = c.get("evidence") or {}
                if usable(c) and ev.get("source") == "certificate_program" and ev.get("status") in ("failed", "partial"):
                    checks.append(EvidenceRecord("check_failed" if ev["status"] == "failed" else "check_passed",
                                                 bool(grades.get(c["id"]))))
            for event in record.get("trace", []):
                if event.get("event") == "review" and event.get("evidence") and event.get("candidate_id") in grades:
                    reviews.append(EvidenceRecord(event["evidence"], grades[event["candidate_id"]]))
                elif event.get("event") == "judge":
                    options = [o for o in event.get("options", []) if o in grades]
                    if any(grades[o] for o in options):
                        choice = (event.get("verdict") or {}).get("choice")
                        valid = event.get("valid_choice") and isinstance(choice, int) and 1 <= choice <= len(options)
                        judges.append(bool(valid and grades[options[choice - 1]]))
    items = {i: ItemRecord(i, formats[i], r, families[i]) for i, r in panels.items()}
    return {"items": items, "states": states, "checks": checks, "reviews": reviews, "judges": judges}


# ----------------------------------------------------------------------------- the v6 mechanism layer
def mechanism_data(package, plan):
    """Bank and program responses, sweep reviews and judges, hint and diversify records from wave M."""
    folder = F6.job_dir(package, "mech") / "agent"
    bank, reviews, judges, coaching, diversify, hints, records = defaultdict(list), [], [], [], [], [], []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        stored = json.loads(path.read_text(encoding="utf-8"))
        record = read_record(path, plan_id=F6.job_plan_id(plan, package, "mech"), mode="live",
                             identity={"item_id": stored["item_id"], "seed": stored["seed"]})
        records.append(record)
        item_id, seed = record["item_id"], int(record["seed"])
        grade = F6.correct_of(item_id)
        state = f"{item_id}#s{seed}"
        member_keys = [m.split("#")[0] for m in record["members"]]
        stage0 = {c["id"]: bool(usable(c) and grade(c["answer"])) for c in record["stage0_candidates"]}
        for branch in record["arms"]:
            arm, new = branch["arm"], branch["new_candidates"]
            if arm["kind"] in ("sample", "tool_alt"):
                for c in new:
                    bank[item_id].append(Response(agent_of(c), c["answer"] if usable(c) else None,
                                                  bool(usable(c) and grade(c["answer"]))))
            elif arm["kind"] == "review":
                for kind, target, _ in branch["new_evidence"]:
                    if target in stage0:
                        reviews.append(EvidenceRecord(kind, stage0[target]))
            elif arm["kind"] == "judge":
                for event in branch["trace"]:
                    if event.get("event") == "judge":
                        options = [o for o in event.get("options", []) if o in stage0]
                        if any(stage0[o] for o in options):
                            choice = (event.get("verdict") or {}).get("choice")
                            valid = event.get("valid_choice") and isinstance(choice, int) and 1 <= choice <= len(options)
                            judges.append(bool(valid and stage0[options[choice - 1]]))
            elif arm["kind"] == "diversify":
                for c in new:
                    diversify.append(CoachingRecord(state, c["model_key"], bool(usable(c) and grade(c["answer"])), False))
            elif arm["kind"] == "hint":
                event = next((e for e in branch["trace"] if e.get("event") == "mech_hint"), None)
                if event is not None:
                    hints.append(bool(event["delivered"]))
                for c in new:
                    tutor_right = bool(record["stage0_member_correct"][arm["tutor"]])
                    coaching.append(CoachingRecord(state, member_keys[arm["solver"]],
                                                   bool(usable(c) and grade(c["answer"])), tutor_right))
    return {"bank": bank, "reviews": reviews, "judges": judges, "coaching": coaching, "diversify": diversify,
            "hints": hints, "records": records}


# ----------------------------------------------------------------------------- fit
def fit(package, dry=False):
    """Write MathMage-R's artifact. ``dry`` (testing only) skips the plan check and writes to a temporary file."""
    plan = {"sha256": "dry-run"} if dry else F6.load_plan(package)
    frozen = read_frozen(F6.SPECS[package]["artifact"], live=True)
    base = PosteriorParams.from_dict(frozen["posterior_params"])
    train = training_data(package)
    runs = recorded_runs(package)
    mech = mechanism_data(package, plan)
    merged = {}
    for source in (train["items"], runs["items"]):
        for item_id, record in source.items():
            target = merged.setdefault(item_id, ItemRecord(item_id, record.answer_format, [], record.family_id))
            target.responses.extend(record.responses)
    for item_id, responses in mech["bank"].items():
        merged[item_id].responses.extend(responses)
    items, dropped = [], []
    for item_id in sorted(merged):
        try:
            _observations(merged[item_id])        # one correct cluster per item, consistent labels
            items.append(merged[item_id])
        except ValueError as exc:                 # pre-registered: such panels are dropped and counted
            dropped.append({"item_id": item_id, "reason": str(exc)})
    irt = fit_irt(items, grid=61)
    if not irt["converged"]:
        raise ValueError("IRT optimization did not converge")
    groups = defaultdict(list)
    for item in [*items, *train["kappa_panels"]]:
        groups[item.item_id].append(item)
    mixture = fit_kappa_mixture(groups)
    abstain = _abstain(items)
    state_items = {**train["items"], **runs["states"]}
    coaching = fit_coaching_lifts(train["coaching"] + mech["coaching"], state_items, irt["theta"], irt["b_sd"],
                                  irt["same_agent_correlation"])
    diversify = (fit_coaching_lifts(mech["diversify"], state_items, irt["theta"], irt["b_sd"],
                                    irt["same_agent_correlation"]) if mech["diversify"] else None)
    evidence = fit_evidence(train["reviews"] + runs["reviews"] + runs["checks"] + mech["reviews"])
    judge_trials = train["judges"] + runs["judges"] + mech["judges"]
    hint_trials = train["hints"] + mech["hints"]
    lift = dict(base.lift)
    lift.update({key: coaching[key] for key in ("coached", "coached_right", "coached_wrong")})
    if diversify is not None:
        lift["diversify"] = diversify["coached"]
    pool_ids = sorted(set(runs["items"]) - set(train["train_ids"]))
    agents = {a: AgentProfile(theta=irt["theta"][a], a=1.0, abstain=abstain.get(a, 0.03)) for a in irt["agents"]}
    params = PosteriorParams(agents=agents, b_mean=0.0, b_sd=irt["b_sd"], kappa=mixture["kappa"],
                             kappa_same=mixture["kappa_same"], trap_weight=mixture["trap_weight"],
                             kappa_trap=mixture["kappa_trap"], none_weight=1.0,
                             same_agent_correlation=irt["same_agent_correlation"],
                             hint_acceptance=(sum(hint_trials) + 4.0) / (len(hint_trials) + 5.0),
                             lift=lift, evidence=evidence,
                             judge_accuracy=float((sum(judge_trials) + 7.0) / (len(judge_trials) + 10.0)),
                             grid=base.grid, calibrated=True,
                             provenance={"train_ids": sorted(set(train["train_ids"]) | set(pool_ids)),
                                         "public_log_items": base.provenance.get("public_log_items", []),
                                         "public_log_agents": base.provenance.get("public_log_agents", []),
                                         "calibration_scope": "v6 MathMage-R: training + recorded runs + wave M"})
    factors, factor_rows = voc_factors(package, params, mech["records"])
    counts = {"items": len(items), "responses": sum(len(i.responses) for i in items),
              "training_items": len(train["items"]), "recorded_items": len(runs["items"]),
              "recorded_states": len(runs["states"]), "bank_responses": sum(len(v) for v in mech["bank"].values()),
              "checks": len(runs["checks"]), "reviews": len(train["reviews"]) + len(runs["reviews"]) + len(mech["reviews"]),
              "judge_trials": len(judge_trials), "coaching_records": len(train["coaching"]) + len(mech["coaching"]),
              "diversify_records": len(mech["diversify"]), "hint_trials": len(hint_trials),
              "mechanism_states": len(mech["records"]), "dropped_items": dropped}
    artifact = {k: v for k, v in frozen.items() if k not in ("sha256", "mu_star")}
    artifact["posterior_params"] = params.to_dict()
    artifact["provenance"] = {**frozen["provenance"], "train_ids": params.provenance["train_ids"]}
    artifact["voc_factors"] = factors
    artifact["v6_fit"] = {"fit_sha256": file_digest(Path(__file__)), "plan_sha256": plan["sha256"],
                          "counts": counts, "irt": irt, "kappa_mixture": mixture, "coaching_lifts": coaching,
                          "diversify_lift": diversify, "evidence": {k: list(v) for k, v in evidence.items()},
                          "voc_factor_rows": factor_rows,
                          "note": "No registry_v6 outcome read. Price chosen in wave D (scripts/v6_fit.py price)."}
    artifact["measurements"] = {**frozen["measurements"], "v6_fit_counts": counts}
    target = ((Path(F6.tempfile.mkdtemp(prefix="v6_fit_dry_")) / f"artifacts_r_{package}.json") if dry
              else F6.r_artifact_path(package))
    frozen_r = freeze(artifact, target)
    print(json.dumps({"artifact": str(target), "sha256": frozen_r["sha256"], "counts": counts,
                      "theta": irt["theta"], "b_sd": irt["b_sd"], "rho": irt["same_agent_correlation"],
                      "kappa": mixture, "lift": lift, "judge_accuracy": params.judge_accuracy,
                      "voc_factors": factors}, indent=1, default=float))


def lifted(obs_rows, params):
    out = []
    for o in obs_rows:
        kind = o["lift_kind"]
        lift = {"none": 0.0, "coached": params.lift.get("coached_wrong", params.lift.get("coached", 0.0)),
                "diversify": params.lift.get("diversify", 0.0), "revision": params.lift.get("revision", 0.0)}[kind]
        source = o["source"] if kind == "coached" else -1
        source_lift = params.lift.get("coached_right") if source >= 0 else None
        out.append(Obs(o["agent"], o["cluster"], lift, source, source_lift))
    return out


def decide(summary, params):
    obs = lifted(summary["observations"], params)
    evidence = [Evidence(e["kind"], e["cluster"], tuple(e["options"])) for e in summary["evidence"]]
    C = summary["n_clusters"]
    tp = TeamPosterior(params)
    result = tp.compute(obs, evidence, C, selectable=summary["selectable"] or None) if C else None
    return obs, evidence, C, result


def voc_factors(package, params, records):
    """VOC factor per action type from the non-adaptive sweep (weights undo the open roster's 1-in-4 sampling)."""
    sums = defaultdict(lambda: [0.0, 0.0, 0.0])      # weight, weighted predicted, weighted realized
    policy = Policy(params, PolicyConfig(mu=0.0))
    for record in records:
        item_id = record["item_id"]
        grade = F6.correct_of(item_id)
        weight = F6.mech_weight(package, record)
        obs, evidence, C, result = decide(record["stage0"], params)
        if result is None:
            continue
        answer_before = record["stage0"]["cluster_answers"].get(str(int(result.best)))
        correct_before = grade(answer_before)
        members = [m.split("#")[0] for m in record["members"]]
        for branch in record["arms"]:
            arm = branch["arm"]
            if arm["kind"] not in FACTOR_KINDS or (arm["kind"] == "sample" and arm["k"] != 1):
                continue
            action = {"sample": lambda: Action("sample", member=arm["member"], cost=1.0),
                      "tool_alt": lambda: Action("tool_alt", cost=1.0),
                      "diversify": lambda: Action("diversify", member=(branch["predicted"] or {}).get("member") or 0, cost=1.0),
                      "review": lambda: Action("review", member=arm["member"], cluster=arm["cluster"], cost=1.0),
                      "judge": lambda: Action("judge", options=tuple(arm["options"]), cost=1.0)}[arm["kind"]]()
            if arm["kind"] == "review" and arm["cluster"] >= C:
                continue
            scored = policy.score([action], obs, evidence, C, lambda m: members[m], lambda s, t: (0.0, -1, None), result)
            predicted = float(scored[0].voc)
            _, _, C_after, after = decide(branch["post"], params)
            if after is None:
                continue
            correct_after = grade(branch["post"]["cluster_answers"].get(str(int(after.best))))
            row = sums[arm["kind"]]
            row[0] += weight
            row[1] += weight * predicted
            row[2] += weight * (float(correct_after) - float(correct_before))
    factors, rows = {}, {}
    for kind in FACTOR_KINDS:
        w, predicted, realized = sums[kind]
        if w <= 0:
            factors[kind] = 1.0
            continue
        mean = predicted / w
        value = (realized + PSEUDO * mean) / (predicted + PSEUDO * mean) if predicted + PSEUDO * mean > 0 else 1.0
        factors[kind] = float(min(1.0, max(FLOOR, value)))
        rows[kind] = {"weight": w, "mean_predicted": mean, "mean_realized": realized / w, "factor": factors[kind]}
    factors["coach"] = factors["sample"]
    return factors, rows


# ----------------------------------------------------------------------------- price
def price(package):
    plan = F6.load_plan(package)
    folder = F6.job_dir(package, "dev_r") / "graded"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    if len(rows) != len(F6.dev_units(package)):
        raise SystemExit(f"wave D incomplete: {len(rows)} of {len(F6.dev_units(package))} development runs")
    points = frontier(rows)
    top = max(p["accuracy"] for p in points)
    knee = min((p for p in points if p["accuracy"] >= top - 0.01), key=lambda p: p["cost_usd"])
    out = {"package": package, "plan_sha256": plan["sha256"], "mu_star_r": float(knee["mu"]),
           "rule": "cheapest threshold within 1 point of the best development accuracy (frozen v3 rule)",
           "dev_frontier": points, "dev_operating_point": operating_point(rows, float(knee["mu"])),
           "artifact_sha256": file_digest(F6.r_artifact_path(package)), "fit_sha256": file_digest(Path(__file__))}
    F6.atomic_json(F6.OUT / f"price_r_{package}.json", out)
    print(json.dumps({k: out[k] for k in ("package", "mu_star_r", "dev_operating_point")}, indent=1))


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if F6.PACKAGE not in RUN_SOURCES:
        raise SystemExit("MathMage-R is pre-registered for the open roster only (--package lane)")
    if command == "fit":
        fit(F6.PACKAGE, dry="--dry" in sys.argv)
    elif command == "price":
        price(F6.PACKAGE)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
