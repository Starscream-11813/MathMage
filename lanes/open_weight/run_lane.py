"""Open-weight lane: gpt-oss-120b calibration, then E2 with parallel workers.

Runs from this lane's frozen code copy and configuration and shares the main
repository's ledger, response cache, call logs and runs (see PROTOCOL.md).

    python run_lane.py --root <lane program root> --freeze     # write the frozen lane plan
    python run_lane.py --root <lane program root>              # run (resumable)

Guards shared with the main program: either program's STOP file, the $73 soft stop on
the shared ledger, and unresolved billing stop this lane. A provider fault stops only
this lane (``provider_attention_open.json``); the main program is never touched.
E2 work is split into disjoint (seed, item) partitions across worker processes that use
the same sealed manifest, engine, sealing and grading code; a final pass through the
frozen runner then reads every record, computes anything missing, and writes the report.
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

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))

from mathmage import paths  # noqa: E402
from mathmage.data.registry import load_registry  # noqa: E402
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, read_frozen  # noqa: E402
from mathmage.experiments.offline import _sample_split, offline_phase  # noqa: E402
from mathmage.experiments.runner import _observation, _protocol_digest, grade_result, run_study, study_items  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402
from mathmage.team import TeamConfig, TeamEngine  # noqa: E402

MAIN_ROOT = paths.RUNS / "live_20261018" / "program_v3"
ROSTER = "open"
POLICY_KEYS = ("sandbox", "router", "mu", "max_policy_actions", "max_item_cost_usd", "voc_epsilon", "hint_audit",
               "max_calls")
ATTENTION = "provider_attention_open.json"


def now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- plan
def plan_body(root: Path) -> dict:
    main_plan = json.loads((MAIN_ROOT / "program_plan.json").read_text(encoding="utf-8"))
    pilot = LANE / "pilot_effort.json"
    pilot_record = json.loads(pilot.read_text(encoding="utf-8"))
    if pilot_record.get("chosen") != "gpt-oss-120b-low":
        raise ValueError("the frozen roster uses the pilot-selected effort; rebuild the roster if it changed")
    return {"mode": "live", "lane": "open_weight", "roster": ROSTER,
            "authorization": {"cap_usd": 75, "source": main_plan["authorization"]["source"]
                              + " On 2026-10-19 the user chose gpt-oss-120b to replace the rate-limited Qwen roster."},
            "soft_stop_usd": 73, "protocol_sha256": _protocol_digest(),
            "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
            "budget_sha256": file_digest(paths.CONFIGS / "budget.yaml"),
            "driver_sha256": file_digest(Path(__file__)),
            "protocol_record_sha256": file_digest(LANE / "PROTOCOL.md"),
            "effort_pilot": {"path": "lanes/open_weight/pilot_effort.json", "sha256": file_digest(pilot),
                             "chosen": pilot_record["chosen"], "reason": pilot_record["reason"],
                             "split": "training only"},
            "calibration": main_plan["calibration"], "deployment_policy": main_plan["deployment_policy"],
            "training_ids": main_plan["training_ids"], "tuning_ids": main_plan["tuning_ids"],
            "e2": {"study": "e2", "seeds": list(range(8)), "mu": 0.0,
                   "scope": "All 60 AIME 2024/2025 items at eight seeds plus MATH-100 once (580 evaluations)."},
            "workers": 8, "main_program_plan_sha256": main_plan["sha256"],
            "frozen_at_utc": now()}


def load_plan(root: Path) -> dict:
    plan = json.loads((root / "program_plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != digest(body):
        raise ValueError("lane plan hash changed")
    return plan


def verify_sources(plan: dict):
    for key, value in [("protocol_sha256", _protocol_digest()),
                       ("models_sha256", file_digest(paths.CONFIGS / "models.yaml")),
                       ("budget_sha256", file_digest(paths.CONFIGS / "budget.yaml")),
                       ("driver_sha256", file_digest(Path(__file__))),
                       ("protocol_record_sha256", file_digest(LANE / "PROTOCOL.md"))]:
        if plan[key] != value:
            raise ValueError(f"{key} changed; the open lane must not run under a different protocol")
    if plan["effort_pilot"]["sha256"] != file_digest(LANE / "pilot_effort.json"):
        raise ValueError("effort pilot record changed")


# ----------------------------------------------------------------------------- guards
class Guard:
    def __init__(self, root: Path, plan: dict):
        self.root, self.plan = root, plan
        self.ledger = BudgetLedger()
        self.stopped = threading.Event()
        self.lock = threading.RLock()

    def reason(self):
        if (self.root / "STOP").exists() or (MAIN_ROOT / "STOP").exists():
            return "STOP requested"
        if (self.root / ATTENTION).exists():
            return "open-lane provider attention recorded"
        if self.stopped.is_set():
            return "lane stopped"
        balance = self.ledger.summary()
        if balance["unreconciled_calls"] or balance["spent_total"] + balance["pending_total"] >= self.plan["soft_stop_usd"]:
            self.stopped.set()
            return "global spending or billing guard"
        return None

    def install(self):
        original = LLMClient.complete
        guard = self

        def guarded(client, request):
            if client.live:
                why = guard.reason()
                if why:
                    raise BudgetExceeded(f"open lane stopped: {why}")
            response = original(client, request)
            if client.live and not response.cached and (response.error or response.billing_uncertain):
                if (response.error or "").startswith("BudgetExceeded"):
                    return response
                with guard.lock:
                    guard.stopped.set()
                    if not (guard.root / ATTENTION).exists():
                        atomic_json(guard.root / ATTENTION, {
                            "time_utc": now(), "model": request.model_key, "error": response.error,
                            "billing_uncertain": response.billing_uncertain, "generation_id": response.generation_id,
                            "provider": response.provider, "worker_pid": os.getpid(),
                            "note": "Open lane only. No automatic replay; review before restarting."})
                raise BudgetExceeded(f"provider health gate stopped the open lane; see {ATTENTION}")
            return response

        LLMClient.complete = guarded
        return original


# ----------------------------------------------------------------------------- phases
def calibration_options(root: Path, plan: dict) -> dict:
    return {**plan["calibration"], "roster": ROSTER, "live": True, "sandbox": "local", "output": root / "offline_open"}


def e2_options(root: Path, plan: dict, artifact: Path) -> dict:
    return {"study": "e2", "roster": ROSTER, "live": True, "sandbox": "local", "artifacts_path": artifact,
            "seeds": plan["e2"]["seeds"], "mu": plan["e2"]["mu"], "output": root / "e2"}


def review(function, options):
    with contextlib.redirect_stdout(io.StringIO()):
        result = function(**options, estimate=True)
    return result


def pairs(manifest, items):
    by_id = {item.item_id: item for item in items}
    out = []
    seeds = manifest["seeds"]
    for seed in seeds:
        for item_id in manifest["item_ids"]:
            item = by_id[item_id]
            if item.benchmark == "math500" and seed != seeds[0]:
                continue
            out.append((seed, item))
    return out


def worker(root: Path, index: int, count: int):
    plan = load_plan(root)
    verify_sources(plan)
    guard = Guard(root, plan)
    target = root / "e2"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    artifact_path = root / "offline_open" / "artifacts.json"
    rebuilt = review(run_study, e2_options(root, plan, artifact_path))
    if {k: v for k, v in rebuilt.items() if k != "budget_review"} != manifest:
        raise ValueError("rebuilt E2 plan differs from the sealed manifest")
    artifacts = read_frozen(artifact_path, live=True)
    items = study_items("e2")
    mine = [pair for position, pair in enumerate(pairs(manifest, items)) if position % count == index]
    status_path = root / "workers" / f"worker_{index}.json"
    state = {"index": index, "of": count, "pid": os.getpid(), "assigned": len(mine), "done": 0,
             "started_at_utc": now(), "status": "running"}
    original = guard.install()
    client = LLMClient(experiment=manifest["experiment"], live=True)
    try:
        for seed, item in mine:
            why = guard.reason()
            if why:
                state.update(status=f"stopped: {why}")
                break
            name = digest([item.item_id, seed, "full"])[:24] + ".json"
            output = target / "agent" / name
            if not output.exists():
                policy = manifest["policy"]
                config = TeamConfig(roster=manifest["roster"], condition="full", seed=seed, offline_artifacts=artifacts,
                                    **{key: policy[key] for key in POLICY_KEYS})
                try:
                    payload = asdict(TeamEngine(client, config).run(item, checkpoint=None))
                except BudgetExceeded as exc:
                    state.update(status=f"stopped: {exc}")
                    break
                atomic_json(output, _observation(payload, manifest))
                atomic_json(target / "graded" / name, grade_result(payload, item, mode=manifest["mode"]))
            state["done"] += 1
            state["updated_at_utc"] = now()
            atomic_json(status_path, state)
        else:
            state.update(status="complete")
    finally:
        state["updated_at_utc"] = now()
        atomic_json(status_path, state)
        LLMClient.complete = original
        client.close()


def publish(root: Path, state: dict):
    state["updated_at_utc"] = now()
    try:
        state["budget"] = BudgetLedger().summary()
    except Exception:                                  # status must never stop collection
        pass
    atomic_json(root / "status.json", state)


def drive(root: Path):
    plan = load_plan(root)
    verify_sources(plan)
    guard = Guard(root, plan)
    state = {"lane": "open_weight", "status": "running", "started_at_utc": now(), "phases": {},
             "program_plan_sha256": plan["sha256"], "classification": "LIVE COLLECTION; NOT SIMULATED"}
    publish(root, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            publish(root, state)
    threading.Thread(target=heartbeat, daemon=True).start()
    original = guard.install()
    try:
        # ---- calibration (training + development, roster open) ----
        options = calibration_options(root, plan)
        offline_review = review(offline_phase, options)
        atomic_json(root / "offline_open_review.json", {"reviewed_at_utc": now(), "program_plan_sha256": plan["sha256"],
                                                         "plan": offline_review})
        state["phases"]["offline_open"] = {"status": "running", "started_at_utc": now(), "plan_id": offline_review["plan_id"]}
        artifact = offline_phase(**options, approved_plan=offline_review["plan_id"])
        state["phases"]["offline_open"].update(status="complete", completed_at_utc=now(), output=str(artifact))
        frozen = read_frozen(artifact, live=True)
        if frozen["policy_config"] != plan["deployment_policy"]:
            raise ValueError("calibration/deployment policy mismatch")
        verify_sources(plan)
        # ---- E2: sealed manifest, parallel workers, final runner pass ----
        e2 = e2_options(root, plan, artifact)
        e2_review = review(run_study, e2)
        manifest = {k: v for k, v in e2_review.items() if k != "budget_review"}
        target = root / "e2"
        target.mkdir(parents=True, exist_ok=True)
        if (target / "manifest.json").exists():
            if json.loads((target / "manifest.json").read_text(encoding="utf-8")) != manifest:
                raise ValueError("saved E2 manifest differs from the reviewed plan")
        else:
            atomic_json(target / "manifest.json", manifest)
        atomic_json(root / "e2_review.json", {"reviewed_at_utc": now(), "program_plan_sha256": plan["sha256"],
                                               "plan": e2_review})
        state["phases"]["e2"] = {"status": "running", "started_at_utc": now(), "plan_id": manifest["plan_id"],
                                 "workers": plan["workers"]}
        (root / "workers").mkdir(exist_ok=True)
        processes = []
        for index in range(plan["workers"]):
            log = open(root / "workers" / f"worker_{index}.log", "a", encoding="utf-8")
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--root", str(root),
                                               "--worker", str(index), "--workers", str(plan["workers"])],
                                              cwd=str(LANE), stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            time.sleep(10)
        if any(p.returncode for p in processes):
            state["phases"]["e2"]["worker_failures"] = [p.returncode for p in processes]
        why = guard.reason()
        if why:
            raise BudgetExceeded(f"open lane stopped before the final pass: {why}")
        verify_sources(plan)
        run_study(**e2, approved_plan=manifest["plan_id"])
        state["phases"]["e2"].update(status="complete", completed_at_utc=now())
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="stopped; review required", error_type=type(exc).__name__, error=str(exc)[:300],
                     ended_at_utc=now())
        raise
    finally:
        stop.set()
        publish(root, state)
        LLMClient.complete = original


def freeze(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    body = plan_body(root)
    target = root / "program_plan.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        same = {k: v for k, v in existing.items() if k not in ("sha256", "frozen_at_utc")} == \
            {k: v for k, v in body.items() if k != "frozen_at_utc"}
        if not same:
            raise ValueError("a different lane plan is already frozen here; use a new root")
        print("lane plan already frozen:", existing["sha256"])
        return
    body["sha256"] = digest(body)
    atomic_json(target, body)
    print("frozen lane plan:", body["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify plan, sources and the calibration plan; no requests")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.freeze:
        freeze(root)
    elif args.check:
        plan = load_plan(root)
        verify_sources(plan)
        offline_review = review(offline_phase, calibration_options(root, plan))
        print(json.dumps({"plan": plan["sha256"], "offline_plan_id": offline_review["plan_id"],
                          "planning_expected_usd": offline_review["planning_expected_usd"],
                          "public_anchor_snapshot": offline_review["public_anchor_snapshot"],
                          "guard": Guard(root, plan).reason()}, indent=1))
    elif args.worker is not None:
        worker(root, args.worker, args.workers)
    else:
        with file_lock(root / ".lane.lock"):
            drive(root)


if __name__ == "__main__":
    main()
