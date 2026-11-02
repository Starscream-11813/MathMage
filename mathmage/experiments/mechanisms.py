"""Preassigned randomized continuations for Z3 (coaching/search) and Z6 (review gate).

Default runs are synthetic, use development items, and select one arm per family/seed.
Assignments and answer-free checkpoints are saved before attaching any gold labels.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mathmage import paths
from mathmage.config import load_budget, load_models
from mathmage.data.registry import load_registry
from mathmage.data.splits import stratified_sample
from mathmage.experiments.artifacts import (atomic_json, digest, file_digest, read_frozen,
                                            read_record, seal_record)
from mathmage.experiments.runner import (_protocol_digest, grade_result, prompt_hashes,
                                        validate_artifact_runtime, tools_runtime, require_live_tools)
from mathmage.llm.locking import file_lock

ARMS = {"z3": ["graded_coaching", "fresh_verified_search"], "z6": ["gated_review", "ungated_review"]}


def assigned_arm(study: str, family_id: str, seed: int, assignment_seed: int = 0) -> str:
    """Stable 1:1 Bernoulli assignment; related items share a family/seed assignment."""
    bit = int(digest(["mechanisms-v1", study, assignment_seed, family_id, seed])[:16], 16) % 2
    return ARMS[study][bit]


def select_items(study: str, limit: int = 8, subset: str | None = None,
                 item_ids: list[str] | None = None):
    """Select a default sample, or the complete explicitly ordered population.

    ``limit`` applies only to automatic sampling. Explicit IDs are never sampled
    or truncated, and must all belong to the declared development/test pool.
    """
    if study not in ARMS or not isinstance(limit, int) or limit < 1:
        raise ValueError("z3/z6 and a positive item limit are required")
    registry = load_registry()
    if subset is not None:
        if study != "z3" or subset != "ablation_45":
            raise ValueError("only Z3 may use the named ablation_45 test subset")
        ids = json.loads((paths.REGISTRY / "subsets.json").read_text(encoding="utf-8"))[subset]
        items = [registry[iid] for iid in ids]
        if any(item.split != "test" for item in items):
            raise ValueError("named ablation subset must contain test items only")
    else:
        items = registry.select(split="dev")
        if any(item.split != "dev" for item in items):
            raise ValueError("mechanism development sample contains non-dev items")
    if not items:
        raise ValueError("no mechanism-study items")
    if item_ids is not None:
        if (not isinstance(item_ids, list) or not item_ids
                or any(not isinstance(iid, str) or not iid for iid in item_ids)
                or len(item_ids) != len(set(item_ids))):
            raise ValueError("explicit mechanism item IDs must be a nonempty unique list of strings")
        allowed = {item.item_id for item in items}
        if any(iid not in allowed for iid in item_ids):
            raise ValueError("explicit mechanism item IDs must all belong to the declared development split or named test subset")
        return [registry[iid] for iid in item_ids]
    ids = stratified_sample(items, min(limit, len(items)), key=lambda item: item.benchmark,
                            salt=f"mechanisms-{study}-{subset or 'dev'}")
    return [registry[iid] for iid in ids]


def mechanism_plan(study, items, *, roster="sat", seeds=(0,), assignment_seed=0, paired=False,
                   max_calls=20, stage0_max_calls=20, artifacts=None, live=False, subset=None, sandbox="auto",
                   explicit_item_ids=False):
    if study not in ARMS or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("study and nonempty unique seeds required")
    if max_calls < 1 or stage0_max_calls < 1:
        raise ValueError("call ceilings must be positive")
    mode = "live" if live else "mock"
    models = load_models()
    rs = models.rosters[roster]
    used = set(rs.members + [rs.checker, rs.default_judge])
    max_output = max(models.model(key).max_tokens for key in used)
    max_output_cost = max(models.model(key).price.cost(0, models.model(key).max_tokens) for key in used)
    assignments = [{"item_id": item.item_id, "family_id": item.family_id, "seed": seed,
                    "assigned_arm": assigned_arm(study, item.family_id, seed, assignment_seed),
                    "executed_arms": list(ARMS[study]) if paired else [assigned_arm(study, item.family_id, seed, assignment_seed)]}
                   for item in items for seed in seeds]
    calls = sum(stage0_max_calls + len(a["executed_arms"]) * max_calls for a in assignments)
    budget = load_budget()
    ordered_ids = [item.item_id for item in items]
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("mechanism population must contain nonempty unique item IDs")
    provenance = (artifacts or {}).get("provenance", {})
    tuning_ids = set(provenance.get("train_ids", []) + provenance.get("dev_ids", []))
    tuning_family_ids = set()
    if tuning_ids:
        registry = load_registry()
        tuning_family_ids = {registry[iid].family_id for iid in tuning_ids}
    overlap_ids = [iid for iid in ordered_ids if iid in tuning_ids]
    overlap_families = sorted({item.family_id for item in items} & tuning_family_ids)
    manifest = {"schema_version": 1, "study": study, "experiment": "e5_zpd", "mode": mode,
                "roster": roster, "seeds": list(seeds), "assignment_seed": assignment_seed,
                "design": "paired checkpoint replay" if paired else "family-seed randomized parallel arms",
                "assignment_probability": 0.5, "randomization_unit": "family_id x seed",
                "selection": subset or "development split", "paired": paired, "assignments": assignments,
                "selection_provenance": {
                    "source": "explicit ordered item IDs" if explicit_item_ids else "automatic stratified sample",
                    "item_ids": ordered_ids, "item_ids_sha256": digest(ordered_ids),
                    "limit_applies": not explicit_item_ids,
                    "offline_learning_overlap_item_ids": overlap_ids,
                    "offline_learning_overlap_family_ids": overlap_families,
                    "interpretation": (
                        "exploratory: population overlaps offline training or development tuning families"
                        if overlap_families else
                        "population is separate from frozen offline training and development families; "
                        "confirmatory status additionally requires selection and hypotheses fixed before outcomes")
                        if artifacts else "offline-artifact independence not assessed without frozen artifacts"},
                "pre_treatment_policy": "all selected items assigned before gold attachment; correctness strata attached posthoc",
                "registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
                "answer_vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
                "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
                "budget_sha256": file_digest(paths.CONFIGS / "budget.yaml"),
                "protocol_sha256": _protocol_digest(), "artifact_sha256": artifacts.get("sha256") if artifacts else None,
                "prompt_hashes": prompt_hashes(),
                "tools_runtime": tools_runtime(["full"], sandbox),
                "policy": {"router": "rule", "sandbox": sandbox,
                           "reason": "forced named strategy continuations from the same pre-treatment checkpoint"},
                "resource_limits": {"stage0_calls_per_item_seed": stage0_max_calls,
                    "continuation_calls_per_arm": max_calls, "total_call_ceiling": calls,
                    "continuation_output_token_ceiling_per_arm": max_calls * max_output,
                    "continuation_output_cost_ceiling_usd_per_arm": max_calls * max_output_cost,
                    "max_output_tokens_per_call": max_output, "total_output_token_ceiling": calls * max_output,
                    "output_component_cost_ceiling_usd": calls * max_output_cost,
                    "input_token_ceiling": None, "total_cost_ceiling_usd": None,
                    "experiment_ledger_cap_usd": budget.experiments["e5_zpd"] * (1 + budget.overrun_tolerance),
                    "global_ledger_cap_usd": budget.global_cap_usd,
                    "note": "Equal request ceilings do not match realized tokens, dollars, or tools. Input sizes are adaptive; the ledger separately enforces hard spending caps."},
                "unmeasured": ["Z1/Z4/Z5/Z7 require their separately declared measurements; this runner does not fabricate them"]}
    manifest["plan_id"] = digest(manifest)
    return manifest


def arm_config(study, arm, *, roster, seed, artifacts, max_calls, stage0=False, sandbox="auto"):
    from mathmage.team import TeamConfig
    if stage0:
        return TeamConfig(roster=roster, seed=seed, offline_artifacts=artifacts,
                          strategy_override=[], max_calls=max_calls, sandbox=sandbox)
    if arm not in ARMS[study]:
        raise ValueError("unknown study arm")
    condition = "A3" if study == "z6" and arm == "ungated_review" else "full"
    strategy = arm if study == "z3" else "evidence_gated_review"
    return TeamConfig(roster=roster, seed=seed, condition=condition, offline_artifacts=artifacts,
                      strategy_override=[strategy], max_strategies=1, max_calls=max_calls, sandbox=sandbox)


def _summarize(rows, manifest):
    from mathmage.analysis.metrics import decomposition
    arms = ARMS[manifest["study"]]
    strata = {"all": lambda r: True, "initially_correct": lambda r: r["initial_correct"],
              "initially_wrong": lambda r: not r["initial_correct"], "covered": lambda r: r["covered"],
              "blocked": lambda r: not r["covered"]}
    comparisons = {}
    for name, predicate in strata.items():
        selected = [r for r in rows if predicate(r)]
        arm_rows = {arm: [r for r in selected if r["arm"] == arm] for arm in arms}
        summary = {arm: decomposition(block) for arm, block in arm_rows.items()}
        # Explicitly cluster-weighted differences; no item-independent p-values.
        units = {}
        for arm, block in arm_rows.items():
            grouped = defaultdict(list)
            for row in block:
                grouped[(row["family_id"], row["seed"])].append(float(row["final_correct"]))
            units[arm] = {key: float(np.mean(values)) for key, values in grouped.items()}
        means = {arm: float(np.mean(list(unit.values()))) if unit else None for arm, unit in units.items()}
        delta = means[arms[0]] - means[arms[1]] if all(v is not None for v in means.values()) else None
        comparisons[name] = {"arms": summary, "family_seed_mean_accuracy": means,
                             "risk_difference_first_minus_second": delta,
                             "inference_status": "descriptive; cluster-aware confirmatory inference not run"}
    diagnostics = {}
    for arm in arms:
        block = [r for r in rows if r["arm"] == arm]
        diagnostics[arm] = {"n": len(block), **{field: sum(float(r.get(field, 0)) for r in block)
                            for field in ["calls", "tokens_in", "tokens_out", "cost_usd", "marginal_cost_usd",
                                          "continuation_calls", "continuation_tokens_in", "continuation_tokens_out",
                                          "continuation_resource_cost_usd", "continuation_marginal_cost_usd"]}}
    return {"mode": manifest["mode"], "study": manifest["study"], "plan_id": manifest["plan_id"],
            "status": "synthetic validation only" if manifest["mode"] == "mock" else "collected observations; confirmatory inference pending",
            "design": manifest["design"], "comparisons": comparisons, "resource_diagnostics": diagnostics,
            "matching_claim": "Request ceilings are equal; realized token, dollar, and tool usage are reported and are not claimed to be matched.",
            "estimand": "family-seed mean final correctness; pre-treatment strata attached only after continuation",
            "caution": "Development results are exploratory; paired replay is a paired design, not an independent-arm randomized trial."}


def run_mechanism(study, *, roster="sat", live=False, limit=8, subset=None, item_ids=None, seeds=None, assignment_seed=0,
                  paired=False, max_calls=20, stage0_max_calls=20, artifacts_path=None, output=None,
                  estimate=False, approved_plan=None, sandbox="auto"):
    from mathmage.llm import LLMClient
    from mathmage.team import TeamEngine

    seeds = [0] if seeds is None else list(seeds)
    items = select_items(study, limit, subset, item_ids)
    artifacts = read_frozen(Path(artifacts_path), live=live) if artifacts_path else None
    if live and artifacts is None:
        raise ValueError("live mechanism studies require frozen live offline artifacts")
    if artifacts:
        validate_artifact_runtime(artifacts, roster, tools_runtime(["full"], sandbox))
        if artifacts.get("roster", roster) != roster:
            raise ValueError("artifact roster does not match")
        if artifacts["provenance"].get("registry_sha256") != file_digest(paths.REGISTRY / "items.parquet"):
            raise ValueError("artifact registry does not match")
        registry = load_registry()
        for field, split in [("train_ids", "train"), ("dev_ids", "dev")]:
            if any(registry[iid].split != split for iid in artifacts["provenance"].get(field, [])):
                raise ValueError("artifact split provenance does not match")
    manifest = mechanism_plan(study, items, roster=roster, seeds=seeds, assignment_seed=assignment_seed,
                              paired=paired, max_calls=max_calls, stage0_max_calls=stage0_max_calls,
                              artifacts=artifacts, live=live, subset=subset, sandbox=sandbox,
                              explicit_item_ids=item_ids is not None)
    if estimate:
        print(json.dumps(manifest, indent=2))
        return manifest
    if live and approved_plan != manifest["plan_id"]:
        raise ValueError("live mechanism runs require --approved-plan from --estimate --live")
    runtime = require_live_tools(live, ["full"], sandbox)
    if live and runtime != manifest["tools_runtime"]:
        raise ValueError("sandbox runtime changed after planning; review a fresh mechanism plan")
    target = Path(output) if output else paths.RUNS / f"{study}_{manifest['mode']}"
    target.mkdir(parents=True, exist_ok=True)
    with file_lock(target / ".run.lock"):
        plan_path = target / "manifest.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("mechanism manifest changed; use a new output directory")
        else:
            atomic_json(plan_path, manifest)
        # Persist the complete assignment list before constructing a client or attaching labels.
        assignment_path = target / "assignments.json"
        assignment_payload = {"assignments": manifest["assignments"]}
        if assignment_path.exists():
            if read_record(assignment_path, plan_id=manifest["plan_id"], mode=manifest["mode"]) != assignment_payload:
                raise ValueError("saved assignments changed")
        else:
            atomic_json(assignment_path, seal_record(assignment_payload, plan_id=manifest["plan_id"], mode=manifest["mode"]))
        client_kwargs = {} if live else {"cache_path": target / "cache.sqlite", "ledger_path": target / "ledger.jsonl",
                                        "log_directory": target / "calls"}
        client = LLMClient("e5_zpd", live=live, **client_kwargs)
        by_id, rows = {item.item_id: item for item in items}, []
        checkpoint_costs = []
        try:
            for assignment in manifest["assignments"]:
                item, seed = by_id[assignment["item_id"]], assignment["seed"]
                ident = digest([item.item_id, seed])[:24]
                checkpoint_path = target / "checkpoints" / f"{ident}.json"
                identity = {"item_id": item.item_id, "seed": seed}
                if checkpoint_path.exists():
                    saved = read_record(checkpoint_path, plan_id=manifest["plan_id"], mode=manifest["mode"], identity=identity)
                    checkpoint = saved["checkpoint"]
                else:
                    config = arm_config(study, None, roster=roster, seed=seed, artifacts=artifacts,
                                        max_calls=stage0_max_calls, stage0=True, sandbox=sandbox)
                    checkpoint = TeamEngine(client, config).create_checkpoint(item)
                    saved = {**identity, "checkpoint": checkpoint}
                    atomic_json(checkpoint_path, seal_record(saved, plan_id=manifest["plan_id"], mode=manifest["mode"]))
                checkpoint_costs.append(checkpoint.get("stage0_costs", {}))
                # All requested potential continuations are sealed before any label is attached.
                pending = []
                for arm in assignment["executed_arms"]:
                    raw_path = target / "agent" / f"{ident}-{arm}.json"
                    arm_identity = {**identity, "arm": arm}
                    if raw_path.exists():
                        saved_arm = read_record(raw_path, plan_id=manifest["plan_id"], mode=manifest["mode"], identity=arm_identity)
                        result = saved_arm["result"]
                    else:
                        config = arm_config(study, arm, roster=roster, seed=seed, artifacts=artifacts,
                                            max_calls=max_calls, sandbox=sandbox)
                        result = asdict(TeamEngine(client, config).run(item, checkpoint=checkpoint))
                        saved_arm = {**arm_identity, "assigned_arm": assignment["assigned_arm"], "result": result,
                                     "checkpoint_sha256": checkpoint["sha256"]}
                        atomic_json(raw_path, seal_record(saved_arm, plan_id=manifest["plan_id"], mode=manifest["mode"]))
                    if saved_arm["checkpoint_sha256"] != checkpoint["sha256"]:
                        raise ValueError("continuation uses a different checkpoint")
                    pending.append((arm, result))
                for arm, result in pending:
                    graded = grade_result(result, item, mode=manifest["mode"])
                    base = checkpoint.get("stage0_costs", {})
                    costs = result.get("costs", {})
                    graded.update({"arm": arm, "assigned_arm": assignment["assigned_arm"], "study": study,
                        "pre_blocked": not graded["covered"],
                        "continuation_calls": costs.get("dispatch_calls", costs.get("calls", 0)),
                        "continuation_tokens_in": costs.get("tokens_in", 0) - base.get("tokens_in", 0),
                        "continuation_tokens_out": costs.get("tokens_out", 0) - base.get("tokens_out", 0),
                        "continuation_resource_cost_usd": max(0, costs.get("resource_cost_usd", 0) - base.get("resource_cost_usd", 0)),
                        "continuation_marginal_cost_usd": costs.get("marginal_cost_usd", costs.get("cost_usd", 0)),
                        "continuation_call_ceiling": max_calls})
                    atomic_json(target / "graded" / f"{ident}-{arm}.json",
                                seal_record(graded, plan_id=manifest["plan_id"], mode=manifest["mode"]))
                    rows.append(graded)
            report = _summarize(rows, manifest)
            report["checkpoint_creation"] = {"calls": sum(c.get("calls", 0) for c in checkpoint_costs),
                "resource_cost_usd": sum(c.get("resource_cost_usd", 0) for c in checkpoint_costs),
                "original_marginal_cost_usd": sum(c.get("marginal_cost_usd", c.get("cost_usd", 0)) for c in checkpoint_costs)}
            atomic_json(target / "analysis.json", report)
            atomic_json(target / "execution.json", {"status": "complete", "observations": len(rows),
                                                    "tools_runtime": manifest["tools_runtime"], "client": client.summary()})
        except BaseException as exc:
            atomic_json(target / "execution.json", {"status": "failed", "error_type": type(exc).__name__,
                                                    "tools_runtime": manifest["tools_runtime"],
                                                    "observations": len(rows), "client": client.summary()})
            raise
        finally:
            client.close()
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=ARMS)
    parser.add_argument("--roster", choices=["sat", "xolver"], default="sat")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--approved-plan")
    parser.add_argument("--artifacts", dest="artifacts_path")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--subset", choices=["ablation_45"])
    parser.add_argument("--item-ids-json", type=Path,
                        help="JSON list of explicit ordered IDs; runs the complete list without --limit truncation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--assignment-seed", type=int, default=0)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--stage0-max-calls", type=int, default=20)
    parser.add_argument("--sandbox", choices=["auto", "docker", "local"], default="auto")
    parser.add_argument("--output", type=Path)
    options = vars(parser.parse_args(argv))
    item_ids_path = options.pop("item_ids_json")
    if item_ids_path is not None:
        try:
            options["item_ids"] = json.loads(item_ids_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --item-ids-json: {exc}")
    run_mechanism(**options)


if __name__ == "__main__":
    main()
