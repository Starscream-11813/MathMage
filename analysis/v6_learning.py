"""S10 (pre-registered with v6): calibration-scale learning curves from recorded runs (read-only; no API calls).

Does the 16-problem calibration limit the team posterior? Cross-fitted by problem family (five folds), the
stage-0 posterior pick is evaluated on held-out recorded runs while the calibration pool grows from the 16
labelled training problems to every recorded problem outside the held-out fold. Abilities, difficulty spread,
same-author correlation, abstention and the collision mixture are refitted with the published estimators
(mathmage/psychometrics/calibrate.py: fit_irt on a 61-point grid, fit_kappa_mixture, _abstain), exactly as
scripts/v6_fit.py applies them; certificate-check evidence likelihoods, lifts and judge accuracy keep their
frozen values. Pool order is a hash draw (three draws per size). Reported: decision accuracy, ECE (ten bins)
and Brier score of the posterior's confidence, pooled over folds.

    python analysis/v6_learning.py --roster sat     # published SAT roster: E1 and its 2025-26 contests (390 problems)
    python analysis/v6_learning.py --roster open    # open roster: E2 and its 2025-26 contests (2,308 states)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROSTER = sys.argv[sys.argv.index("--roster") + 1] if "--roster" in sys.argv else "sat"
sys.path.insert(0, str(REPO / "lanes" / "open_weight" if ROSTER == "open" else REPO))
import mathmage  # noqa: E402,F401  (binds the roster's own package copy before the analysis helpers load)
sys.path.insert(1, str(REPO / "analysis"))

import numpy as np  # noqa: E402

import results as R  # noqa: E402
import v4_results as V4R  # noqa: E402
from mathmage.config import load_models  # noqa: E402
from mathmage.experiments.artifacts import digest, read_frozen, read_record  # noqa: E402
from mathmage.grading.equiv import normalize  # noqa: E402
from mathmage.grading.vault import load_vault  # noqa: E402
from mathmage.grading.verify import grade_answer  # noqa: E402
from mathmage.prompts.templates import SOLVE  # noqa: E402
from mathmage.psychometrics.calibrate import ItemRecord, Response, _abstain, _observations, fit_irt, fit_kappa_mixture  # noqa: E402
from mathmage.team.engine import extract_answer  # noqa: E402
from mathmage.team.posterior import AgentProfile, PosteriorParams  # noqa: E402

TOOL_PHASES = {"tool_forward", "verify_forward", "tool_alt"}
SPEC = {"sat": {"root": R.RUNS / "program_v3", "phase": "e1_primary", "offline": R.RUNS / "program_v3" / "offline_sat",
                "roster": "sat", "jobs": ["new_e1_mathmage_nov", "new_e1_mathmage_apr"]},
        "open": {"root": R.RUNS / "program_open", "phase": "e2", "offline": R.RUNS / "program_open" / "offline_open",
                 "roster": "open", "jobs": ["new_e2_mathmage_nov", "new_e2_mathmage_apr"]}}
FOLDS, DRAWS = 5, 3
SIZES = (0, 16, 32, 64, 128, 256, 10 ** 6)
OUT = REPO / "results" / "tables"


def usable(c):
    return bool(c.get("valid", True) and not c.get("error") and c.get("answer"))


def training(spec, registry):
    """The frozen offline calibration's training panels and public collision panels (as scripts/v6_fit.py)."""
    root = spec["offline"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evidence = read_record(root / "training_evidence.json", plan_id=manifest["plan_id"], mode=manifest["mode"])
    members = load_models().rosters[spec["roster"]].members
    vault = load_vault()

    def observation(name):
        path = root / "observations" / (digest(name)[:24] + ".json")
        return read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"],
                           identity={"observation_id": digest(name)})["data"]

    def correct(item_id, answer):
        return bool(answer) and grade_answer(item_id, answer, vault).correct

    items = {}
    for item_id in manifest["train_ids"]:
        item = registry[item_id]
        base = observation(["base", item_id])
        responses = [Response("tool" if c.get("phase") in TOOL_PHASES else c["model_key"],
                              c["answer"] if usable(c) else None, usable(c) and correct(item_id, c["answer"]))
                     for c in base["stage0"]]
        kw = {"problem": item.problem, "format_instruction": item.view().format_instruction, "cards": ""}
        for k in range(manifest["options"]["extra_samples"]):
            for i, member in enumerate(members):
                r = observation([item_id, member, SOLVE.id, f"extra:{k}:member={i}", kw])
                raw = extract_answer(r["text"]) if not r.get("error") and not r.get("truncated") else None
                answer = normalize(raw, item.answer_format) if raw else None
                responses.append(Response(member, answer, correct(item_id, answer)))
        items[item_id] = ItemRecord(item_id, item.answer_format, responses, item.family_id)
    for record in evidence["public_anchor_snapshot"]:
        target = items.setdefault(record["item_id"], ItemRecord(record["item_id"], record["answer_format"], [],
                                                               record.get("family_id", "")))
        target.responses.extend(Response(**r) for r in record["responses"])
    frozen = read_frozen(root / "artifacts.json", live=True)
    expected = frozen["measurements"]["posterior_calibration"]["n_responses"]
    if sum(len(it.responses) for it in items.values()) != expected:
        raise ValueError("rebuilt training panels differ from the frozen calibration's")
    kappa_panels = [ItemRecord(r["item_id"], r["answer_format"], [Response(**x) for x in r["responses"]],
                               r.get("family_id", "")) for r in evidence["public_kappa_snapshot"]]
    return list(items.values()), kappa_panels, frozen, set(manifest["train_ids"]) | set(manifest["dev_ids"])


def recorded(spec):
    rows, _, _ = R.load_phase(spec["root"], spec["phase"])
    rows = [r for r in rows if r.get("condition", "full") == "full"]
    return rows + V4R.mathmage_rows(spec["jobs"])


def pool_items(rows, registry):
    """One record per problem (seeds merged), from the saved stage-0 member and program answers."""
    panels = defaultdict(list)
    for row in rows:
        grades = row["candidate_grades"]
        for c in row["_candidates"]:
            if c["id"].startswith("s0-"):
                panels[row["item_id"]].append(Response("tool" if c.get("phase") in TOOL_PHASES else c["model_key"],
                                                       c["answer"] if usable(c) else None,
                                                       bool(usable(c) and grades.get(c["id"], {}).get("correct"))))
    items, dropped = {}, []
    for item_id, responses in sorted(panels.items()):
        item = registry[item_id]
        record = ItemRecord(item_id, item.answer_format, responses, item.family_id)
        try:
            _observations(record)
            items[item_id] = record
        except ValueError:
            dropped.append(item_id)
    return items, dropped


def refit(items, kappa_panels, frozen):
    base = PosteriorParams.from_dict(frozen["posterior_params"])
    irt = fit_irt(items, grid=61)
    groups = defaultdict(list)
    for item in [*items, *kappa_panels]:
        groups[item.item_id].append(item)
    mixture = fit_kappa_mixture(groups)
    abstain = _abstain(items)
    agents = {a: AgentProfile(theta=irt["theta"][a], a=1.0, abstain=abstain.get(a, 0.03)) for a in irt["agents"]}
    params = PosteriorParams(agents=agents, b_mean=0.0, b_sd=irt["b_sd"], kappa=mixture["kappa"],
                             kappa_same=mixture["kappa_same"], trap_weight=mixture["trap_weight"],
                             kappa_trap=mixture["kappa_trap"], none_weight=1.0,
                             same_agent_correlation=irt["same_agent_correlation"],
                             hint_acceptance=base.hint_acceptance, lift=dict(base.lift), evidence=base.evidence,
                             judge_accuracy=base.judge_accuracy, grid=base.grid, calibrated=True,
                             provenance=base.provenance)
    return params, irt, mixture


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster", choices=["sat", "open"], default="sat")
    parser.add_argument("--quick", action="store_true", help="one draw and three sizes (testing)")
    parser.add_argument("--draws", type=int, default=DRAWS, help="hash draws of the pool order per size")
    parser.add_argument("--package", help="read by scripts/v6_guarded.py (grading guard); ignored here")
    args = parser.parse_args()
    spec = SPEC[args.roster]
    registry = V4R.Combined()
    train, kappa_panels, frozen, excluded = training(spec, registry)
    rows = [r for r in recorded(spec) if r["item_id"] not in excluded]
    pool, dropped = pool_items(rows, registry)
    started = time.time()
    reference, _, _ = refit(train, kappa_panels, frozen)
    frozen_theta = {k: v["theta"] for k, v in frozen["posterior_params"]["agents"].items()}
    check = {k: (round(reference.agents[k].theta, 3), round(frozen_theta.get(k, float("nan")), 3))
             for k in reference.agents}
    print(json.dumps({"roster": args.roster, "training_items": len(train), "pool_items": len(pool),
                      "states": len(rows), "dropped": len(dropped), "refit_vs_frozen_theta": check,
                      "seconds_per_fit": round(time.time() - started, 1)}), flush=True)
    family = {it.item_id: it.family_id for it in pool.values()}
    fold = {f: int(digest(["s10-fold", f])[:8], 16) % FOLDS for f in sorted(set(family.values()))}
    sizes = (0, 64, 10 ** 6) if args.quick else SIZES
    draws = 1 if args.quick else args.draws
    results = defaultdict(list)
    frozen_pairs = []
    for f in range(FOLDS):
        test = [r for r in rows if fold.get(family.get(r["item_id"], ""), -1) == f]
        if not test:
            continue
        _, conf, _ = R.stage0_analysis(test, frozen["posterior_params"], registry)
        frozen_pairs += conf["posterior"]
        candidates = sorted(i for i, fam in family.items() if fold[fam] != f)
        for draw in range(draws):
            order = sorted(candidates, key=lambda i: digest(["s10-order", draw, i]))
            for n in sizes:
                if n > len(order) and n != 10 ** 6:
                    continue
                if n == 10 ** 6 and draw > 0:
                    continue                    # the full pool has no draw-to-draw variation
                chosen = order[:min(n, len(order))]
                params, _, _ = refit(train + [pool[i] for i in chosen], kappa_panels, frozen)
                _, conf, _ = R.stage0_analysis(test, params.to_dict(), registry)
                results[(min(n, len(order)) if n != 10 ** 6 else "all", draw)].extend(conf["posterior"])
                print(json.dumps({"fold": f, "draw": draw, "n": n if n != 10 ** 6 else "all",
                                  "pairs": len(conf["posterior"]), "elapsed_s": round(time.time() - started)}),
                      flush=True)
    summary = {"roster": args.roster, "training_items": len(train), "pool_items": len(pool), "states": len(rows),
               "dropped_items": dropped, "folds": FOLDS, "frozen_16": {**R.calibration(frozen_pairs),
                                                                     "accuracy": float(np.mean([y for _, y in frozen_pairs]))},
               "curve": {}}
    by_size = defaultdict(list)
    for (n, draw), pairs in results.items():
        cal = R.calibration(pairs)
        by_size[str(n)].append({"accuracy": float(np.mean([y for _, y in pairs])), "ece": cal["ece"],
                                "brier": cal["brier"], "n_pairs": len(pairs)})
    for n, entries in by_size.items():
        summary["curve"][n] = {k: float(np.mean([e[k] for e in entries])) for k in ("accuracy", "ece", "brier")}
        summary["curve"][n]["draws"] = len(entries)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"v6_learning_{args.roster}.json"
    path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("frozen_16", "curve")}, indent=1, default=float))


if __name__ == "__main__":
    main()
