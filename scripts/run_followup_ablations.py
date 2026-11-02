"""Follow-up run of the frozen v3 secondary ablations (optional panel).

Why a follow-up: the v3 driver gates optional panels on "both primary lanes finish",
and its Qwen lane can never finish (sustained upstream rate limits). E2 was completed
by the separately frozen open-weight lane. This runner executes exactly the planned
secondary ablation panel with the frozen v3 code, artifact, manifest and guards.

Integrity: the program plan hash, package protocol, models, budget, driver and
operational-dependency hashes must equal the frozen v3 plan; the rebuilt study plan
must equal any saved manifest; the plan's own affordability rule must pass.
Ordering: the shared ``full`` checkpoint first, then conditions in decreasing value
(C2, A8, C1, A5, A6), so a budget-line stop can only cut the least informative ones.
Work is split across worker processes by item; a final pass through the frozen
runner reads every record, computes anything missing, and writes the report.
Guards: STOP, the $73 soft stop, unresolved billing; a provider fault stops this run only.

    python scripts/run_followup_ablations.py --root runs/live_20261018/program_v3 [--workers 4]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathmage import paths  # noqa: E402
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, read_frozen  # noqa: E402
from mathmage.experiments.runner import (_observation, _protocol_digest, _read_observation, grade_result,  # noqa: E402
                                         run_study, study_items)
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402
from mathmage.team import TeamConfig, TeamEngine  # noqa: E402
from scripts.run_paid_program import guarded_completion  # noqa: E402

PHASE = "ablations_secondary"
ORDER = ["full", "C2", "A8", "C1", "A5", "A6"]
POLICY_KEYS = ("sandbox", "router", "mu", "max_policy_actions", "max_item_cost_usd", "voc_epsilon", "hint_audit",
               "max_calls")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_plan(root: Path) -> dict:
    plan = json.loads((root / "program_plan.json").read_text(encoding="utf-8"))
    if plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("program plan hash changed")
    checks = [("protocol_sha256", _protocol_digest()),
              ("models_sha256", file_digest(paths.CONFIGS / "models.yaml")),
              ("budget_sha256", file_digest(paths.CONFIGS / "budget.yaml")),
              ("driver_sha256", file_digest(Path(__file__).with_name("run_paid_program.py")))]
    for name, expected in (plan.get("operational_dependencies_sha256") or {}).items():
        checks.append((f"operational:{name}", file_digest(Path(__file__).with_name(name)), expected))
    for check in checks:
        key, current = check[0], check[1]
        expected = check[2] if len(check) == 3 else plan[key]
        if current != expected:
            raise ValueError(f"{key} differs from the frozen v3 plan; the follow-up must not run")
    if sorted(plan["secondary_ablations"]) != sorted(ORDER):
        raise ValueError("secondary ablation conditions differ from the frozen plan")
    return plan


def study_options(root: Path, plan: dict) -> dict:
    return {"study": "ablations", "live": True, "sandbox": "local",
            "artifacts_path": root / "offline_sat" / "artifacts.json", "output": root / PHASE,
            "conditions": plan["secondary_ablations"],
            **plan.get("study_policy_overrides", {}).get("ablations", {})}


class Guard:
    def __init__(self, root: Path, plan: dict):
        self.root, self.plan = root, plan
        self.ledger = BudgetLedger()
        self.stopped, self.everything = threading.Event(), threading.Event()
        self.lock = threading.RLock()

    def reason(self):
        if (self.root / "STOP").exists() or (self.root / "followup" / "STOP").exists():
            return "STOP requested"
        if (self.root / "followup" / "provider_attention_followup.json").exists():
            return "follow-up provider attention recorded"
        if self.stopped.is_set() or self.everything.is_set():
            return "follow-up guard stopped"
        balance = self.ledger.summary()
        if balance["unreconciled_calls"] or balance["spent_total"] + balance["pending_total"] >= self.plan["soft_stop_usd"]:
            return "global spending or billing guard"
        return None

    def install(self):
        original = LLMClient.complete
        guard = self

        def guarded(client, request):
            return guarded_completion(original, client, request, root=guard.root, stopped=guard.stopped,
                                      global_stopped=guard.everything,
                                      attention_name="followup/provider_attention_followup.json",
                                      ledger=guard.ledger, soft_stop_usd=guard.plan["soft_stop_usd"],
                                      fault_lock=guard.lock, recovery_waves=0)
        LLMClient.complete = guarded
        return original


def name_of(item, seed, condition):
    return digest([item.item_id, seed, condition])[:24] + ".json"


def worker(root: Path, index: int, count: int):
    plan = load_plan(root)
    guard = Guard(root, plan)
    target = root / PHASE
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    artifacts = read_frozen(root / "offline_sat" / "artifacts.json", live=True)
    items = [item for position, item in enumerate(study_items("ablations")) if position % count == index]
    seed = manifest["seeds"][0]
    state = {"index": index, "of": count, "pid": os.getpid(), "assigned": len(items) * (len(ORDER) - 1), "done": 0,
             "status": "running", "started_at_utc": now()}
    status_path = root / "followup" / f"worker_{index}.json"
    original = guard.install()
    client = LLMClient(experiment=manifest["experiment"], live=True)
    try:
        for condition in ORDER[1:]:
            for item in items:
                why = guard.reason()
                if why:
                    state["status"] = f"stopped: {why}"
                    return
                output = target / "agent" / name_of(item, seed, condition)
                if not output.exists():
                    full_path = target / "agent" / name_of(item, seed, "full")
                    checkpoint = None
                    if condition not in {"A8", "C1", "C2"}:
                        checkpoint = _read_observation(full_path, manifest, item.item_id, seed, "full").get("checkpoint")
                    policy = manifest["policy"]
                    config = TeamConfig(roster=manifest["roster"], condition=condition, seed=seed,
                                        offline_artifacts=artifacts, **{key: policy[key] for key in POLICY_KEYS})
                    try:
                        payload = asdict(TeamEngine(client, config).run(item, checkpoint=checkpoint))
                    except BudgetExceeded as exc:
                        state["status"] = f"stopped: {exc}"
                        return
                    if not output.exists():
                        atomic_json(output, _observation(payload, manifest))
                        atomic_json(target / "graded" / output.name, grade_result(payload, item, mode=manifest["mode"]))
                state["done"] += 1
                state["current"] = f"{condition} {item.item_id}"
                state["updated_at_utc"] = now()
                atomic_json(status_path, state)
        state["status"] = "complete"
    finally:
        state["updated_at_utc"] = now()
        atomic_json(status_path, state)
        LLMClient.complete = original
        client.close()


def update_status(root: Path, **phase):
    path = root / "status.json"
    with file_lock(root / ".followup_status.lock"):
        status = json.loads(path.read_text(encoding="utf-8"))
        entry = status.setdefault("phases", {}).setdefault(PHASE, {})
        entry.update(phase)
        active = [p for p in status.get("active_phases", []) if p != PHASE]
        if phase.get("status") == "running":
            active.append(PHASE)
        status["active_phases"] = active
        if phase.get("status") == "running":
            status["status"] = "follow-up: secondary ablations running"
        elif phase.get("status") == "complete":
            status["status"] = "primary and secondary panels complete; optional extra seeds not run"
        status["updated_at_utc"] = now()
        try:
            status["budget"] = BudgetLedger().summary()
        except Exception:
            pass
        atomic_json(path, status)
        atomic_json(paths.OUTPUTS / "live_status.json", status)


def coordinate(root: Path, workers: int):
    plan = load_plan(root)
    ledger = BudgetLedger()
    balance = ledger.summary()
    if balance["pending_calls"] or balance["unreconciled_calls"]:
        raise ValueError("reconcile outstanding reservations or billing before the follow-up")
    options = study_options(root, plan)
    with contextlib.redirect_stdout(io.StringIO()):
        review = run_study(**options, estimate=True)
    manifest = {k: v for k, v in review.items() if k != "budget_review"}
    target = root / PHASE
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        if json.loads((target / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("saved secondary-ablation manifest differs from the rebuilt plan")
    else:
        atomic_json(target / "manifest.json", manifest)
    # The plan's own resource rule for this optional panel (see run_paid_program.affordable).
    allowance = plan["secondary_ablations_allowance_usd"] * 1.35
    capacity = plan["soft_stop_usd"] - balance["spent_total"] - balance["pending_total"]
    line_capacity = ledger.line("e4_ablations") - balance["by_experiment"].get("e4_ablations", 0)
    if capacity < allowance + plan["optional_reserve_usd"] or line_capacity < allowance:
        raise ValueError("the frozen resource rule does not permit the secondary panel")
    (root / "followup").mkdir(exist_ok=True)
    atomic_json(root / "followup" / "amendment.json", {
        "time_utc": now(), "phase": PHASE, "program_plan_sha256": plan["sha256"], "plan_id": manifest["plan_id"],
        "reason": ("The v3 driver defers optional panels until both primary lanes finish; its Qwen lane cannot finish "
                   "(sustained upstream rate limits) and E2 was completed by the separately frozen open-weight lane. "
                   "This follow-up runs exactly the planned secondary ablation panel with the frozen v3 code, "
                   "artifact, manifest and guards. Extra E1 seeds are not run (user decision, 2026-10-19)."),
        "order": ORDER, "workers": workers, "affordability": {"allowance_x1_35": allowance, "capacity": capacity,
                                                              "line_capacity": line_capacity},
        "no_test_label_use": "Scope fixed by the frozen plan; ordering by declared condition value, not by results."})
    atomic_json(root / f"{PHASE}_review.json", {"reviewed_at_utc": now(), "program_plan_sha256": plan["sha256"],
                                                "plan": review, "followup": True})
    update_status(root, status="running", started_at_utc=now(), plan_id=manifest["plan_id"], followup=True,
                  workers=workers)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            try:
                update_status(root, status="running")
            except Exception:
                pass
    threading.Thread(target=heartbeat, daemon=True).start()
    guard = Guard(root, plan)
    original = guard.install()
    try:
        # The shared full checkpoint (identical requests to E1/primary ablations: cache hits).
        artifacts = read_frozen(options["artifacts_path"], live=True)
        seed = manifest["seeds"][0]
        client = LLMClient(experiment=manifest["experiment"], live=True)
        try:
            for item in study_items("ablations"):
                output = target / "agent" / name_of(item, seed, "full")
                if output.exists():
                    continue
                policy = manifest["policy"]
                config = TeamConfig(roster=manifest["roster"], condition="full", seed=seed, offline_artifacts=artifacts,
                                    **{key: policy[key] for key in POLICY_KEYS})
                payload = asdict(TeamEngine(client, config).run(item, checkpoint=None))
                atomic_json(output, _observation(payload, manifest))
                atomic_json(target / "graded" / output.name, grade_result(payload, item, mode=manifest["mode"]))
        finally:
            client.close()
        processes = []
        for index in range(workers):
            log = open(root / "followup" / f"worker_{index}.log", "a", encoding="utf-8")
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--root", str(root),
                                               "--worker", str(index), "--workers", str(workers)],
                                              cwd=str(paths.ROOT), stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            time.sleep(10)
        why = guard.reason()
        if why:
            raise BudgetExceeded(f"follow-up stopped before the final pass: {why}")
        load_plan(root)
        run_study(**options, approved_plan=manifest["plan_id"])
        update_status(root, status="complete", completed_at_utc=now(), output=str(target))
    except BaseException as exc:
        update_status(root, status="stopped", error_type=type(exc).__name__, error=str(exc)[:300], ended_at_utc=now())
        raise
    finally:
        stop.set()
        LLMClient.complete = original


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--check", action="store_true", help="verify hashes, plan and affordability; no requests")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker is not None:
        worker(root, args.worker, args.workers)
    elif args.check:
        plan = load_plan(root)
        with contextlib.redirect_stdout(io.StringIO()):
            review = run_study(**study_options(root, plan), estimate=True)
        balance = BudgetLedger().summary()
        print(json.dumps({"plan": plan["sha256"], "study_plan_id": review["plan_id"],
                          "planned_expected_usd": review["cost_estimate"]["expected_usd"],
                          "observations": review.get("n_item_seed_pairs"), "conditions": review["conditions"],
                          "spent_total": balance["spent_total"], "e4_spent": balance["by_experiment"].get("e4_ablations"),
                          "pending": balance["pending_calls"]}, indent=1))
    else:
        with file_lock(paths.RUNS / ".paid_program.lock"):
            coordinate(root, args.workers)


if __name__ == "__main__":
    main()
