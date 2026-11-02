"""Default-offline study runner with immutable manifests and per-item checkpoints.

Agent output and gold-attached evaluation are separate files. Resuming a run checks
the manifest and artifact hashes before skipping any completed observation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from mathmage import paths
from mathmage.config import load_budget, load_models
from mathmage.data.registry import load_registry
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, read_frozen, seal_record, read_record
from mathmage.llm.locking import file_lock

STUDIES = {
    "pilot": ("pilot", "pilot_20", "sat", ["full"], [0]),
    "e1": ("e1_sat", "sat_suite", "sat", ["full"], [0]),
    "e2": ("e2_xolver", "xolver_suite", "xolver", ["full"], list(range(8))),
    "ablations": ("e4_ablations", "ablation_45", "sat",
                  ["full", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "C1", "C2", "R1"], [0]),
}


def study_items(study: str, limit: int | None = None):
    reg = load_registry()
    subsets = json.loads((paths.REGISTRY / "subsets.json").read_text(encoding="utf-8"))
    name = STUDIES[study][1]
    ids = (subsets["xolver_aime"] + subsets["math500_100"]) if name == "xolver_suite" else subsets[name]
    if len(ids) != len(set(ids)):
        raise ValueError("subset contains duplicate items")
    items = [reg[i] for i in ids]
    if study == "pilot":
        if any(i.split != "train" for i in items):
            raise ValueError("pilot must use training data only")
    elif any(i.split != "test" for i in items):
        raise ValueError("evaluation subset contains non-test data")
    return items[:limit] if limit is not None else items


def plan(study: str, items: list, roster: str, conditions: list[str], seeds: list[int], artifacts=None,
         policy=None, runtime=None) -> dict:
    """Bind prices and request ceilings; keep utilization assumptions explicitly heuristic."""
    from mathmage.team.types import POLICY_DEFAULTS
    models = load_models()
    rs = models.rosters[roster]
    budget = load_budget()
    options = {**POLICY_DEFAULTS, "router": "posterior", "mu": 0.0,
               **(artifacts or {}).get("policy_config", {}), **(policy or {})}
    if options["mu"] is None:
        options["mu"] = (artifacts or {}).get("mu_star", 0.0)
    tools = bool(runtime and runtime.get("available"))
    lines, ceilings, used_models = [], [], set()
    n = sum(1 if study == "e2" and item.benchmark == "math500" else len(seeds) for item in items)
    best = (artifacts or {}).get("best_member", rs.members[0])
    saved_checkpoint = False
    saved_stage0_floor = 0
    for condition in conditions:
        members = [best] * len(rs.members) if condition in {"C1", "C2"} else rs.members
        checker = best if condition == "C1" else rs.checker
        discernment = (artifacts or {}).get("discernment", {})
        judge = (best if condition == "C1" else rs.default_judge if condition == "A2" or not discernment else
                 max(members, key=lambda key: float(discernment.get(key, 0))))
        active_tools = tools and condition != "A7"
        used_models.update(members + [checker, judge])
        share = saved_checkpoint and condition not in {"A8", "C1", "C2"}
        stage0_floor = min(len(members), options["max_calls"])
        stage0_upper = min(options["max_calls"], len(members) + (1 + len(members) if active_tools else 0))
        router = "rule" if condition == "R1" else options["router"]
        terminal = int(condition == "A2" and router == "posterior")
        request_cap = (min(options["max_calls"], stage0_upper + 3 * options["max_policy_actions"] + terminal)
                       if router == "posterior" else options["max_calls"])
        # Inherited stage0 consumes the condition's logical call limit but causes
        # no new dispatch. Only subtract its known member-call floor for a bound.
        dispatch_cap = max(0, request_cap - saved_stage0_floor) if share else request_cap
        allowed = sorted(set(members + [checker, judge]))
        max_output_cost = max(models.model(key).price.cost(0, models.model(key).max_tokens) for key in allowed)
        ceilings.append({"condition": condition, "router": router, "reuses_full_checkpoint": share,
                         "stage0_member_call_floor": stage0_floor, "stage0_call_ceiling": stage0_upper,
                         "max_policy_actions": options["max_policy_actions"] if router == "posterior" else None,
                         "max_calls_per_method": options["max_calls"],
                         "logical_calls_per_item_seed_ceiling": request_cap,
                         "additional_dispatches_per_item_seed_ceiling": dispatch_cap,
                         "total_additional_dispatch_ceiling": n * dispatch_cap,
                         "possible_models": allowed,
                         "output_component_usd_ceiling": n * dispatch_cap * max_output_cost,
                         "total_usd_ceiling": None})

        def add(role, key, count, input_tokens, output_kind):
            spec = models.model(key)
            expected = sum(count * (1 if study == "e2" and it.benchmark == "math500" else len(seeds)) *
                           spec.price.cost(max(input_tokens, len(it.problem) // 3 + 350), spec.expected_out(output_kind or it.kind))
                           for it in items)
            row = {"condition": condition, "role": role, "model": key, "calls": n * count,
                   "expected_usd": expected, "input_tokens_per_call_assumption": input_tokens,
                   "output_kind_assumption": output_kind or "item kind"}
            lines.append(row)
            return expected / n

        stage0_cost = 0.0
        if not share:
            for member in members[:stage0_floor]:
                stage0_cost += add("independent proposal", member, 1.0, 700, None)
            if active_tools:
                count = min(1 + len(members) * 0.25, max(0, options["max_calls"] - stage0_floor))
                stage0_cost += add("program/check planning allowance", checker, count, 1500, "code")
        else:
            # Reused work still consumes logical resources for the stopping rule.
            stage0_cost = sum(models.model(key).price.cost(700, models.model(key).expected_out("olympiad")) for key in members)
            if active_tools:
                stage0_cost += 1.75 * models.model(checker).price.cost(1500, models.model(checker).expected_out("code"))
        expected_stage0_calls = min(options["max_calls"], stage0_floor + (1 + len(members) * .25 if active_tools else 0))
        # No measured stopping distribution exists yet. Use half the declared
        # horizon, with a disclosed mixture of solver/short/auxiliary calls.
        actions = options["max_policy_actions"] * .5 if router == "posterior" else 4.0
        mixture = [(key, actions * .8 / len(members), 1800, None) for key in members]
        mixture += [(key, actions * .5 / len(members), 2200, "short") for key in members]
        if active_tools:
            mixture.append((checker, actions * .1, 1500, "code"))
        if router == "rule" or terminal:
            mixture.append((judge, 1.0, 3500, "short"))
        proposed_calls = sum(count for _, count, _, _ in mixture)
        call_room = max(0, request_cap - expected_stage0_calls)
        scale = min(1.0, call_room / proposed_calls) if proposed_calls else 0.0
        if router == "posterior":
            proposed_cost = sum(count * models.model(key).price.cost(tin, models.model(key).expected_out(kind or "olympiad"))
                                for key, count, tin, kind in mixture)
            scale = min(scale, max(0, options["max_item_cost_usd"] - stage0_cost) / proposed_cost) if proposed_cost else scale
        for key, count, tin, kind in mixture:
            if count * scale:
                add("continuation utilization heuristic", key, count * scale, tin, kind)
        if condition == "full":
            saved_checkpoint = True
            saved_stage0_floor = stage0_floor
    experiment = STUDIES[study][0]
    expected_usd = sum(row["expected_usd"] for row in lines)
    line_cap = budget.experiments[experiment] * (1 + budget.overrun_tolerance)
    warnings = []
    if expected_usd > line_cap:
        warnings.append("Heuristic expected spend exceeds the configured experiment line including tolerance; reduce scope or explicitly reallocate within the global cap.")
    return {"schema_version": 1, "study": study, "experiment": STUDIES[study][0], "roster": roster,
            "item_ids": [i.item_id for i in items], "n_item_seed_pairs": n,
            "conditions": conditions, "seeds": seeds,
            "benchmark_repetitions": {benchmark: 1 if study == "e2" and benchmark == "math500" else len(seeds)
                                      for benchmark in sorted({i.benchmark for i in items})},
            "registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
            "answer_vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
            "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
            "budget_sha256": file_digest(paths.CONFIGS / "budget.yaml"),
            "artifact_sha256": artifacts.get("sha256") if artifacts else None,
            "pricing": {key: {"slug": models.model(key).slug, "provider": models.model(key).provider,
                              "price_usd_per_million": models.model(key).price.model_dump(by_alias=True),
                              "max_output_tokens": models.model(key).max_tokens,
                              "expected_output_tokens": models.model(key).expected_out_tokens}
                        for key in sorted(used_models)},
            "cost_estimate": {"lines": lines, "expected_usd": expected_usd,
                "request_ceilings": ceilings,
                "additional_dispatch_call_ceiling": sum(row["total_additional_dispatch_ceiling"] for row in ceilings),
                "output_component_usd_ceiling": sum(row["output_component_usd_ceiling"] for row in ceilings),
                "total_usd_ceiling": None,
                "budget_limits": {"experiment_line_usd": budget.experiments[experiment],
                                  "experiment_cap_with_tolerance_usd": line_cap,
                                  "global_cap_usd": budget.global_cap_usd,
                                  "offline_shared_cap_with_tolerance_usd": budget.experiments["offline"] * (1 + budget.overrun_tolerance)},
                "warnings": warnings,
                "assumptions": {"posterior_horizon_utilization": 0.5, "rule_reference_action_units": 4,
                                "solver_calls_per_action": .8, "short_calls_per_action": .5,
                                "checker_calls_per_action_when_available": .1,
                                "stage0_partial_check_fraction": .25,
                                "mu_effect": "not forecast without measured development stopping frequencies",
                                "global_cache_savings": "not assumed; explicit full-checkpoint reuse only"},
                "note": "Expected dollars/calls are an uncalibrated utilization heuristic. Request ceilings bound logical completions, not provider retries. Output-only dollar ceilings omit unbounded adaptive inputs and are not total-cost guarantees. The item cap uses estimated action costs and may overshoot; only per-call ledger reservations enforce spending. Both rosters share the offline line. No batch discount assumed."}}


def grade_result(result: dict, item, *, mode: str, vault=None) -> dict:
    from mathmage.grading.verify import grade_answer
    from mathmage.grading.equiv import equivalent

    candidates = result["candidates"]
    by_id = {c["id"]: c for c in candidates}
    if result.get("item_id") != item.item_id:
        raise ValueError("result belongs to a different item")
    if (len(by_id) != len(candidates) or len(result["stage0"]) != len(set(result["stage0"]))
            or any(cid not in by_id for cid in result["stage0"])):
        raise ValueError("candidate ids must be unique and cover stage0")
    if result.get("final_id") is not None and result["final_id"] not in by_id:
        raise ValueError("final candidate is missing")
    valid = lambda c: c.get("valid", True) and not c.get("error")
    grades = {c["id"]: asdict(grade_answer(item.item_id, c.get("answer") if valid(c) else None, vault))
              for c in candidates}
    initial = [by_id[cid] for cid in result["stage0"]]
    # Vote only on answer equivalence, never reference correctness.
    clusters = []
    for candidate in initial:
        if not valid(candidate) or not candidate.get("answer"):
            continue
        for group in clusters:
            if equivalent(candidate["answer"], group[0]["answer"], item.answer_format):
                group.append(candidate)
                break
        else:
            clusters.append([candidate])
    winner = max(clusters, key=len)[0]["id"] if clusters else None
    final_id = result.get("final_id")
    costs = result.get("costs", {})
    zone = result.get("zone", {})
    meta = result.get("meta", {})
    tool_ids = [c for c in meta.get("stage0_tool", []) if c in by_id]
    anytime = []
    for step in meta.get("anytime", []):
        cid = step.get("best_candidate")
        chosen = step.get("chosen")
        anytime.append({"step": step.get("step"), "calls": step.get("calls"), "cost_usd": step.get("cost_usd"),
                        "marginal_cost_usd": step.get("marginal_cost_usd"), "confidence": step.get("confidence"),
                        "coverage": step.get("coverage"), "stop_reason": step.get("stop_reason"),
                        "chosen": {k: chosen.get(k) for k in ("kind", "label", "voc", "cost", "ratio")} if chosen else None,
                        "correct": bool(cid in grades and grades[cid]["correct"])})
    return {"schema_version": 1, "mode": mode, "item_id": item.item_id,
            "benchmark": item.benchmark, "family_id": item.family_id, "split": item.split,
            "item_date": item.item_date, "seed": result["seed"], "condition": result["condition"],
            "zone": zone.get("zone", zone.get("label", "unknown")),
            "zone_uncertain": zone.get("uncertain", True),
            "covered": any(grades[c]["correct"] for c in result["stage0"]),
            "covered_with_tool": any(grades[c]["correct"] for c in list(result["stage0"]) + tool_ids),
            "pool_covered": any(g["correct"] for g in grades.values()),
            "router": meta.get("router", "rule"), "anytime": anytime,
            "frontier_replay_valid": meta.get("frontier_replay_valid", False),
            "mu": meta.get("mu"),
            "posterior_calibrated": meta.get("posterior_calibrated", False),
            "sandbox_backend": meta.get("sandbox_backend", "unknown"),
            "zone_confidence": zone.get("signals", {}).get("confidence"),
            "zone_coverage": zone.get("signals", {}).get("coverage"),
            "initial_correct": bool(winner and grades[winner]["correct"]),
            "initial_vote_id": winner,
            "final_correct": bool(final_id and grades[final_id]["correct"]),
            "strict_final_correct": bool(grades[final_id]["strict_correct"] if grades[final_id]["strict_correct"] is not None
                                         else grades[final_id]["correct"]) if final_id else False,
            "stage0_correct": {by_id[c]["member"]: grades[c]["correct"] for c in result["stage0"]},
            "candidate_grades": grades,
            "cost_usd": costs.get("resource_cost_usd", costs.get("cost_usd", costs.get("cost_usd_total", 0.))),
            "marginal_cost_usd": costs.get("marginal_cost_usd", costs.get("cost_usd", 0.)),
            "calls": costs.get("calls", 0), "tokens_in": costs.get("tokens_in", 0),
            "tokens_out": costs.get("tokens_out", 0),
            "tokens_reasoning": costs.get("tokens_reasoning", 0),
            "latency_s": costs.get("resource_latency_s", costs.get("latency_s", 0)),
            "resource_latency_s": costs.get("resource_latency_s", costs.get("latency_s", 0)),
            "wall_latency_s": costs.get("wall_latency_s", 0),
            "marginal_summed_latency_s": costs.get("summed_latency_s", costs.get("latency_s", 0)),
            "marginal_latency_s": costs.get("latency_s", 0),
            "backend_calls": costs.get("backend_calls", 0),
            "tools_status": result.get("meta", {}).get("tools_status", "unknown"),
            "tools_reason": result.get("meta", {}).get("tools_reason", "")}


# Figure and atlas code reads saved results only and is never imported by collection,
# grading, calibration, or operating-point selection (see tests/test_runner.py), so it
# can change while paid collection runs. Everything else in the package is protocol.
PRESENTATION_ONLY = frozenset({"analysis/style.py", "analysis/diagram.py", "analysis/concepts.py",
                               "analysis/figures.py", "analysis/atlas.py"})


def _protocol_digest() -> str:
    """Changing agent, harness, or grading code requires a separate run directory."""
    package = Path(__file__).resolve().parents[1]
    return digest({p.relative_to(package).as_posix(): file_digest(p) for p in sorted(package.rglob("*.py"))
                   if p.relative_to(package).as_posix() not in PRESENTATION_ONLY})


def prompt_hashes() -> dict:
    from mathmage.prompts.templates import TEMPLATES
    from mathmage.team.engine import TEAM_TEMPLATES
    return {key: template.id for key, template in {**TEMPLATES, **TEAM_TEMPLATES}.items()}


def tools_runtime(conditions, sandbox="auto") -> dict:
    from mathmage.tools import DockerSandbox
    from mathmage.tools.local_sandbox import LocalSandbox
    if sandbox not in {"auto", "docker", "local"}:
        raise ValueError("sandbox must be auto, docker, or local")
    requested = any(condition != "A7" for condition in conditions)
    executor = LocalSandbox() if sandbox == "local" else DockerSandbox()
    available = executor.available if requested else False
    runtime = {"requested": requested, "available": available, "selection": sandbox,
               "backend": "local-process" if sandbox == "local" else "docker",
               "sandbox_image": executor.image, "image_id": getattr(executor, "_image_id", None),
               "reason": executor.reason if requested else "tools disabled by A7"}
    if sandbox == "local":
        from importlib.metadata import version
        runtime.update({"python_version": sys.version, "python_sha256": file_digest(Path(executor.python)),
                        "libraries": {name: version(name) for name in ("numpy", "sympy")},
                        "security_boundary": "limited process isolation; not a hardened security boundary"})
    return runtime


def require_live_tools(live: bool, conditions, sandbox="auto") -> dict:
    runtime = tools_runtime(conditions, sandbox)
    if live and runtime["requested"] and not runtime["available"]:
        kind = "explicit local process sandbox" if sandbox == "local" else "prepared Docker sandbox"
        raise ValueError("live tool-enabled method requires a " + kind + ": " + runtime["reason"])
    return runtime


def validate_artifact_runtime(artifact: dict, roster: str, runtime: dict | None = None) -> None:
    provenance = artifact["provenance"]
    for key, expected in {"registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
                          "answer_vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
                          "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
                          "prompt_hashes": prompt_hashes(), "protocol_sha256": _protocol_digest()}.items():
        if provenance.get(key) != expected:
            raise ValueError(f"offline artifact {key} changed; remeasure and freeze the offline policy")
    if artifact.get("roster", roster) != roster:
        raise ValueError("offline artifacts belong to a different roster")
    if artifact.get("mode") == "live" and runtime is not None and runtime.get("requested"):
        frozen = provenance.get("tools_runtime")
        if not frozen:
            raise ValueError("live offline artifact has no frozen tools_runtime; remeasure using the selected sandbox")
        for key in ("available", "backend", "image_id", "sandbox_image", "python_version", "python_sha256", "libraries"):
            if frozen.get(key) != runtime.get(key):
                raise ValueError(f"offline artifact tools_runtime {key} changed; remeasure with the evaluation sandbox")


def budget_review(experiment: str, expected_usd: float) -> dict:
    """Read an advisory ledger snapshot without locks, writes, or an API client.

    This snapshot is deliberately outside the immutable plan: a run's own spending
    changes the ledger, and must not invalidate its otherwise resumable manifest.
    Dispatch always rechecks the authoritative ledger under its normal lock.
    """
    from mathmage.llm.ledger import DEFAULT_LEDGER
    budget = load_budget()
    spent, pending = {}, {}
    try:
        raw = DEFAULT_LEDGER.read_bytes() if DEFAULT_LEDGER.exists() else b""
        for line in raw.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event", "commit")
            if event == "reserve":
                amount = float(row["amount"])
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError("invalid reserve")
                pending[row["token"]] = (row["experiment"], amount)
            elif event == "release":
                pending.pop(row["token"], None)
            elif event == "commit":
                pending.pop(row.get("token"), None)
                amount = float(row["cost_usd"])
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError("invalid commit")
                spent[row["experiment"]] = spent.get(row["experiment"], 0.0) + amount
            else:
                raise ValueError("unknown event")
        total = sum(spent.values()) + sum(amount for _, amount in pending.values())
        cap = budget.experiments[experiment] * (1 + budget.overrun_tolerance)
        experiment_used = spent.get(experiment, 0.0) + sum(amount for exp, amount in pending.values() if exp == experiment)
        global_remaining, experiment_remaining = budget.global_cap_usd - total, cap - experiment_used
        offline_remaining = budget.experiments["offline"] * (1 + budget.overrun_tolerance) - spent.get("offline", 0.0) - sum(amount for exp, amount in pending.values() if exp == "offline")
        return {"status": "advisory snapshot", "spent_plus_reserved_usd": total,
                "global_remaining_usd": global_remaining, "experiment_remaining_usd": experiment_remaining,
                "offline_shared_remaining_usd": offline_remaining,
                "heuristic_exceeds_remaining": expected_usd > min(global_remaining, experiment_remaining),
                "warning": "Planning heuristic exceeds the remaining shared ledger capacity." if expected_usd > min(global_remaining, experiment_remaining) else None,
                "note": "Read-only snapshot; concurrent changes are possible. Remaining balances do not certify feasibility. Actual dispatch reserves full output ceilings under a lock. Both rosters debit the same offline line."}
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return {"status": "unknown", "warning": "Ledger snapshot unavailable or changing; authoritative dispatch will fail closed on invalid ledger data."}


def _observation(payload: dict, manifest: dict) -> dict:
    return seal_record(payload, plan_id=manifest["plan_id"], mode=manifest["mode"])


def _read_observation(path: Path, manifest: dict, item_id: str, seed: int, condition: str) -> dict:
    return read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"],
                       identity={"item_id": item_id, "seed": seed, "condition": condition})


def run_study(study: str, *, live=False, artifacts_path=None, limit=None, roster=None,
              conditions=None, seeds=None, output=None, estimate=False, approved_plan=None,
              sandbox="auto", router="posterior", mu=None, max_policy_actions=None,
              max_item_cost_usd=None, voc_epsilon=None, hint_audit=None, max_calls=None):
    from mathmage.llm.client import LLMClient
    from mathmage.team import TeamConfig, TeamEngine

    exp, _, default_roster, default_conditions, default_seeds = STUDIES[study]
    roster = roster or default_roster
    conditions = list(default_conditions if conditions is None else conditions)
    seeds = list(default_seeds if seeds is None else seeds)
    if not conditions or not seeds or len(conditions) != len(set(conditions)) or len(seeds) != len(set(seeds)):
        raise ValueError("conditions and seeds must be nonempty and unique")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    items = study_items(study, limit)
    if not items:
        raise ValueError("empty study")
    artifacts = read_frozen(Path(artifacts_path), live=live) if artifacts_path else None
    # Frozen development settings define the deployed method; explicit CLI
    # overrides create a different immutable plan instead of silently extending it.
    from mathmage.team.types import POLICY_DEFAULTS
    requested = {"max_policy_actions": max_policy_actions, "max_item_cost_usd": max_item_cost_usd,
                 "voc_epsilon": voc_epsilon, "hint_audit": hint_audit, "max_calls": max_calls}
    frozen_policy = (artifacts or {}).get("policy_config", {})
    resolved = {key: value if value is not None else frozen_policy.get(key, POLICY_DEFAULTS[key])
                for key, value in requested.items()}
    policy_options = {"sandbox": sandbox, "router": router, "mu": mu, **resolved}
    for condition in conditions:
        TeamConfig(roster=roster, condition=condition, **policy_options)
    if live and study != "pilot" and artifacts is None:
        raise ValueError("live evaluation requires frozen offline artifacts")
    if live and study == "pilot" and artifacts is None and router != "rule":
        raise ValueError("an uncalibrated live pilot requires --router rule; posterior evaluation requires offline calibration")
    runtime = tools_runtime(conditions, sandbox)
    if artifacts:
        validate_artifact_runtime(artifacts, roster, runtime)
        provenance = artifacts["provenance"]
        for key, split in [("train_ids", "train"), ("dev_ids", "dev")]:
            if any(load_registry()[i].split != split for i in provenance[key]):
                raise ValueError("artifact provenance does not match frozen registry splits")
        if artifacts["provenance"]["registry_sha256"] != file_digest(paths.REGISTRY / "items.parquet"):
            raise ValueError("artifact registry changed")
        if artifacts.get("roster", roster) != roster:
            raise ValueError("offline artifacts belong to a different roster")
    manifest = {}
    manifest["prompt_hashes"] = prompt_hashes()
    manifest["protocol_sha256"] = _protocol_digest()
    manifest["mode"] = "live" if live else "mock"
    manifest["tools_runtime"] = runtime
    effective_mu = mu if mu is not None else (artifacts or {}).get("mu_star", 0.0)
    if not isinstance(effective_mu, (int, float)) or not math.isfinite(effective_mu) or effective_mu < 0:
        raise ValueError("operating threshold must be finite and nonnegative")
    manifest["policy"] = {**policy_options, "mu": float(effective_mu),
                          "mu_source": "explicit" if mu is not None else "frozen development operating point" if artifacts and "mu_star" in artifacts else "mock maximal default",
                          "setting_sources": {key: "explicit" if requested[key] is not None else
                                              "frozen development setting" if key in frozen_policy else "default"
                                              for key in requested},
                          "condition_routers": {c: "rule" if c == "R1" else router for c in conditions}}
    manifest = {**plan(study, items, roster, conditions, seeds, artifacts,
                       policy=manifest["policy"], runtime=runtime), **manifest}
    manifest["plan_id"] = digest(manifest)
    if estimate:
        review = {**manifest, "budget_review": budget_review(manifest["experiment"], manifest["cost_estimate"]["expected_usd"])}
        print(json.dumps(review, indent=2))
        return review
    if live and approved_plan != manifest["plan_id"]:
        raise ValueError("Paid runs require --approved-plan with the plan_id from --estimate --live after explicit review")
    checked_runtime = require_live_tools(live, conditions, sandbox)
    if live and checked_runtime != runtime:
        raise ValueError("sandbox runtime changed after planning; review a fresh plan")
    target = Path(output) if output else paths.RUNS / f"{study}_{manifest['mode']}"
    target.mkdir(parents=True, exist_ok=True)
    with file_lock(target / ".run.lock"):
        return _execute_study(study, target, manifest, items, roster, conditions, seeds, artifacts, live)


def _execute_study(study, target, manifest, items, roster, conditions, seeds, artifacts, live):
    from mathmage.llm.client import LLMClient
    from mathmage.team import TeamConfig, TeamEngine

    manifest_file = target / "manifest.json"
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("run manifest changed; use a new output directory")
    else:
        atomic_json(manifest_file, manifest)
    kwargs = {} if live else {"log_directory": target / "calls", "cache_path": target / "cache.sqlite",
                              "ledger_path": target / "ledger.jsonl"}
    client = LLMClient(experiment=manifest["experiment"], live=live, **kwargs)
    results = []
    try:
        for seed in seeds:
            for item in items:
                if study == "e2" and item.benchmark == "math500" and seed != seeds[0]:
                    continue
                saved_stage0 = None
                for condition in conditions:
                    name = digest([item.item_id, seed, condition])[:24] + ".json"
                    output_path = target / "agent" / name
                    if output_path.exists():
                        payload = _read_observation(output_path, manifest, item.item_id, seed, condition)
                    else:
                        policy = manifest["policy"]
                        config = TeamConfig(roster=roster, condition=condition, seed=seed, offline_artifacts=artifacts,
                            **{key: policy[key] for key in ("sandbox", "router", "mu", "max_policy_actions",
                                                           "max_item_cost_usd", "voc_epsilon", "hint_audit", "max_calls")})
                        engine = TeamEngine(client, config)
                        payload = asdict(engine.run(item, checkpoint=None if condition in {"A8", "C1", "C2"} else saved_stage0))
                        atomic_json(output_path, _observation(payload, manifest))
                    if condition == "full":
                        saved_stage0 = payload.get("checkpoint")
                    graded = grade_result(payload, item, mode=manifest["mode"])
                    atomic_json(target / "graded" / name, graded)
                    results.append(graded)
                    print(f"{study} {condition} seed={seed} {item.item_id} saved ({len(results)} observations)", flush=True)
        from mathmage.analysis.make_tables import build_report
        build_report(results, target / "analysis")
        atomic_json(target / "execution.json", {"status": "complete", "observations": len(results),
                                               "tools_runtime": manifest["tools_runtime"], "client": client.summary()})
    except BaseException as exc:
        atomic_json(target / "execution.json", {"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                                               "observations": len(results), "error_type": type(exc).__name__,
                                               "tools_runtime": manifest["tools_runtime"],
                                               "client": client.summary()})
        raise
    finally:
        client.close()
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--live", action="store_true", help="use paid OpenRouter calls (requires reviewed plan ID)")
    parser.add_argument("--estimate", action="store_true", help="print itemized plan without model calls")
    parser.add_argument("--approved-plan")
    parser.add_argument("--artifacts")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--roster", choices=["sat", "xolver"])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sandbox", choices=["auto", "docker", "local"], default="auto")
    parser.add_argument("--router", choices=["posterior", "rule"], default="posterior")
    parser.add_argument("--mu", type=float, help="VOC per dollar threshold; defaults to frozen development choice (0 requests a maximal run)")
    parser.add_argument("--max-policy-actions", type=int)
    parser.add_argument("--max-item-cost-usd", type=float)
    parser.add_argument("--voc-epsilon", type=float)
    parser.add_argument("--hint-audit", choices=["structural", "residual"])
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run_study(args.study, live=args.live, artifacts_path=args.artifacts, limit=args.limit,
              roster=args.roster, conditions=args.conditions, seeds=args.seeds, output=args.output,
              estimate=args.estimate, approved_plan=args.approved_plan,
              sandbox=args.sandbox, router=args.router, mu=args.mu,
              max_policy_actions=args.max_policy_actions, max_item_cost_usd=args.max_item_cost_usd,
              voc_epsilon=args.voc_epsilon, hint_audit=args.hint_audit, max_calls=args.max_calls)


if __name__ == "__main__":
    main()
