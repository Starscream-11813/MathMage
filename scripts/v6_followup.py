"""v6 follow-up (pre-registered 2026-10-24): fresh post-cutoff contests, a mechanism layer, MathMage-R.

Packages (rosters). OpenAI retired o3-mini before v6 (every request returns HTTP 404 model_not_found), so the
published SAT roster cannot be rerun. Its slot goes to a new SAT-style team, ``sat26`` (GPT-5-mini at low
effort, Claude Haiku 4.5, DeepSeek-V3; lanes/sat26), calibrated by the frozen offline phase on the published
16 training and 10 development problems (runs/live_20261018/program_v6/offline_sat26) and otherwise run with
the SAT roster's frozen policy. The open roster (``lane``, three gpt-oss-120b members) is unchanged.

Waves (each on its own ledger line; the staged plan files freeze jobs, units, code and data hashes before any call):

  F  Fresh confirmatory test on ``data/registry_v6`` (187 contest problems released November 2025 to April
     2026, transcribed from the organizers' PDFs; see scripts/v6_data.py). MathMage-16 is the published v3
     method (16-problem calibration, knee-rule price) run as a maximal run; self-consistency (sat26:
     DeepSeek-V3 SC@10 as for the SAT roster, and SC@5 of its strongest member, GPT-5-mini), universal
     self-consistency, one debate round and mixture-of-agents build on its saved stage-0 solutions with the
     v4/v5 templates and code paths unchanged. Majority voting and the open roster's pooled SC@k need no
     calls.                                                                   lines v6_fresh_sat / v6_fresh_open
  M  Mechanism layer on recorded stage-0 states: the open roster's frozen 2025-26 runs and AIME 2024 and 2025
     (states where a member erred, plus one in four all-correct states by hash), and every sat26 wave-F state;
     states run in one hash order. From each state, as separate branches: a response bank (one
     fresh attempt per member, a second per initially wrong member), one alternative program, a
     non-adaptive sweep (judge, diversified re-solve, the best-valued review), and a crossed hint trial.
     Every initially wrong member receives a hint from a right tutor (if one exists), a hint from a wrong
     tutor (if one exists) and a content-free placebo re-solve; every right member of a mixed problem
     receives a wrong tutor's hint (harm arm). Tutors and hint formats are drawn by hash; every hint also gets
     the residual-work audit as a semantic leak measurement that does not gate the trial.
                                                                              lines v6_mech_sat / v6_mech_open
  D  MathMage-R development runs (open roster): maximal runs on the ten development families from their saved
     stage-0 checkpoints with the MathMage-R artifact written by scripts/v6_fit.py (frozen with this plan),
     giving mu*_R by the frozen knee rule.                                     line v6_dev
  R  MathMage-R on the fresh test from the saved wave-F stage-0 checkpoints (open roster).  line v6_fresh_open
  Z  Dense ZPD sub-study (open roster; budget-gated, last): eight fresh attempts and eight hinted re-solves by
     one initially wrong member on each informative 2025-26 state at seed 0.  line v6_dense_open

Nothing under mathmage/ (either lane's copy), the frozen registries or their vaults changes. MathMage-R
differs from MathMage-16 only in its artifact (posterior parameters refitted on a larger calibration pool,
VOC factors per action type, its own knee-rule price), applied by this driver.

    python scripts/v6_followup.py prereg --package lane                   # stage 1: plan.json (no calls)
    python scripts/v6_followup.py prereg --package sat26                  # stage 2: plan_sat26.json (no calls)
    python scripts/v6_followup.py check --package sat26                   # verify plan, show progress (no calls)
    python scripts/v6_followup.py worker --package sat26 --wave F --index 0 --count 3
    python scripts/v6_followup.py mocktest --package lane                 # mock backend, temp folder (no calls)
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("lane", "sat26")
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "lane"
if PACKAGE not in PACKAGES:
    raise SystemExit("--package must be lane or sat26")
if PACKAGE == "sat26":
    # The sat26 code copy is imported first, so every later ``mathmage`` import (the v4/v5 helpers included)
    # binds to it; the v4/v5 runners are then loaded in their SAT-roster mode for the shared helpers only.
    sys.path.insert(0, str(REPO / "lanes" / "sat26"))
    import mathmage  # noqa: E402,F401
    import mathmage.paths  # noqa: E402,F401
    if Path(mathmage.__file__).resolve().parents[1] != (REPO / "lanes" / "sat26").resolve():
        raise SystemExit("the sat26 package did not load from lanes/sat26")

_argv = sys.argv
_helpers = "main" if PACKAGE == "sat26" else PACKAGE
sys.argv = [sys.argv[0], "--package", _helpers]            # the v4/v5 runners select their package at import
_spec = importlib.util.spec_from_file_location("v4_followup", REPO / "scripts" / "v4_followup.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
_spec5 = importlib.util.spec_from_file_location("v5_followup", REPO / "scripts" / "v5_followup.py")
F5 = importlib.util.module_from_spec(_spec5)
_spec5.loader.exec_module(F5)
sys.argv = _argv

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mathmage.team.engine as ENGINE  # noqa: E402
from mathmage import paths  # noqa: E402
from mathmage.data.registry import Registry  # noqa: E402
from mathmage.experiments.artifacts import digest, file_digest, read_frozen, read_record, seal_record  # noqa: E402
from mathmage.experiments.runner import _protocol_digest, grade_result, prompt_hashes, tools_runtime  # noqa: E402
from mathmage.experiments.runner import validate_artifact_runtime  # noqa: E402
from mathmage.grading.equiv import cluster  # noqa: E402
from mathmage.grading.vault import AnswerVault  # noqa: E402
from mathmage.grading.verify import grade_answer  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402
from mathmage.prompts.templates import HINT, HINT_KINDS, PLACEBO_NOTE, SOLVE, SOLVE_SCAFFOLD  # noqa: E402
from mathmage.scaffolds import HINT_LEVELS, residual_audit, structural_audit  # noqa: E402
from mathmage.team import TeamConfig, TeamEngine  # noqa: E402
from mathmage.team.engine import RESIDUAL_AUDIT, parse_json  # noqa: E402
from mathmage.team.policy import Action, Policy, PolicyConfig  # noqa: E402

V.atomic_json = F5.retrying(V.atomic_json)
atomic_json = V.atomic_json

V6_LIVE = V.LIVE / "program_v6"
OUT = V6_LIVE                        # replaced by a temporary folder in mocktest
REG6 = paths.DATA / "registry_v6"
SOFT_STOP = 149.5                    # global ledger guard under the $151 cap
FRESH_SEEDS = {"sat26": [0], "lane": list(range(8))}
LINES = {"fresh": {"sat26": "v6_fresh_sat", "lane": "v6_fresh_open"},
         "mech": {"sat26": "v6_mech_sat", "lane": "v6_mech_open"},
         "dev": {"lane": "v6_dev"},
         "dense": {"lane": "v6_dense_open"}}
WAVE_LINE = {"F": "fresh", "M": "mech", "D": "dev", "R": "fresh", "Z": "dense"}
# Rosters: the open roster's frozen v3 artifact and primary policy; sat26's v6 calibration with the SAT
# roster's frozen primary policy (the published method's settings; the price comes from its own artifact).
SPECS = {"lane": {"roster": "open", "artifact": V.OPEN / "offline_open" / "artifacts.json",
                  "policy_manifest": V.OPEN / "e2" / "manifest.json"},
         "sat26": {"roster": "sat26", "artifact": V6_LIVE / "offline_sat26" / "artifacts.json",
                   "policy_manifest": V.V3 / "e1_primary" / "manifest.json"}}
# Recorded states for the mechanism layer: (source run, benchmarks kept; None keeps every benchmark of a v4 job).
MECH_SOURCES = {"lane": [("primary", ("aime_2024", "aime_2025")), ("new_e2_mathmage_nov", None),
                         ("new_e2_mathmage_apr", None)],
                "sat26": [("fresh_mm", None)]}
ARTIFACT_LIVE = True                 # mocktest may run sat26 on a mock calibration (scripts only, no calls)
PLAN_FILES = {"lane": "plan.json", "sat26": "plan_sat26.json"}
DENSE_STATES, DENSE_DRAWS = 200, 8
RESTORE_MU = 1e12                    # restores a stage-0 state: the policy stops before any action
MECH_MAX_CALLS = 60
SC_BEST_K = 5                        # sat26 strongest-member SC: one MathMage-16 problem costs about four of its solves


def now():
    return datetime.now(timezone.utc).isoformat()


def pick(options, *key):
    """Deterministic draw from options, keyed by the unit and arm (a hash, fixed before any call)."""
    options = list(options)
    return options[int(digest(list(key))[:8], 16) % len(options)] if options else None


# ----------------------------------------------------------------------------- data
_CACHE = {}


def v6_registry():
    if "reg6" not in _CACHE:
        path = REG6 / "items.parquet"
        _CACHE["reg6"] = Registry.from_frame(pd.read_parquet(path)) if path.exists() else Registry([])
    return _CACHE["reg6"]


def v6_vault():
    if "vault6" not in _CACHE:
        _CACHE["vault6"] = AnswerVault(pd.read_parquet(REG6 / "answers.parquet"))
    return _CACHE["vault6"]


def item_of(item_id):
    reg = v6_registry()
    if item_id in reg:
        return reg[item_id]
    if item_id in base_registry():
        return base_registry()[item_id]
    if "ext" not in _CACHE:
        _CACHE["ext"] = V.ext_registry()
    return _CACHE["ext"][item_id]


def base_registry():
    if "base" not in _CACHE:
        _CACHE["base"] = V.load_registry()
    return _CACHE["base"]


def vault_for(item_id):
    """The vault holding item_id's answer: v6, the frozen base vault, or the v4 extension (each loaded once)."""
    if item_id in v6_registry():
        return v6_vault()
    if item_id in base_registry():
        if "base_vault" not in _CACHE:
            from mathmage.grading.vault import load_vault
            _CACHE["base_vault"] = load_vault()
        return _CACHE["base_vault"]
    if "ext_vault" not in _CACHE:
        _CACHE["ext_vault"] = V.ext_vault()
    return _CACHE["ext_vault"]


def correct_of(item_id):
    vault = vault_for(item_id)
    return lambda answer: bool(answer) and grade_answer(item_id, answer, vault).correct


def primary_manifest(package):
    """The manifest whose frozen policy the package's MathMage runs use (open: its e2 run; sat26: SAT e1)."""
    return json.loads(SPECS[package]["policy_manifest"].read_text(encoding="utf-8"))


def frozen_policy(package, **overrides):
    policy = {k: primary_manifest(package)["policy"][k] for k in V.POLICY_KEYS}
    policy.update(overrides)
    return policy


def roster_artifacts(package):
    spec = SPECS[package]
    artifacts = read_frozen(spec["artifact"], live=ARTIFACT_LIVE)
    validate_artifact_runtime(artifacts, spec["roster"], tools_runtime(["full"], "local"))
    return artifacts


def engine_for(client, package, seed, artifacts=None, **overrides):
    """A frozen MathMage engine of the package (v4's engine_for, with sat26's artifact and the SAT policy)."""
    artifacts = artifacts if artifacts is not None else roster_artifacts(package)
    config = TeamConfig(roster=SPECS[package]["roster"], condition="full", seed=seed, offline_artifacts=artifacts,
                        **frozen_policy(package, **overrides))
    return TeamEngine(client, config)


def v4_plan():
    plan = json.loads((V.V4 / "plan.json").read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v4 plan hash changed")
    return plan


def job_dir(package, job_id):
    """Each package's v6 jobs live in their own folder (the rosters share item ids and seed 0)."""
    return OUT / package / job_id


def recorded_folder(package, source):
    if source == "primary":
        return SPECS[package]["policy_manifest"].parent
    return job_dir(package, source) if source.startswith("fresh_") else V.V4 / source


def recorded_record(package, source, item_id, seed, plan=None):
    """The sealed agent record (with its stage-0 checkpoint) of a frozen MathMage run (v3, v4 or wave F)."""
    path = recorded_folder(package, source) / "agent" / V.name_of(item_id, seed, "full")
    identity = {"item_id": item_id, "seed": seed, "condition": "full"}
    if source == "primary":
        manifest = primary_manifest(package)
        return read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"], identity=identity)
    if source.startswith("fresh_"):
        return read_record(path, plan_id=job_plan_id(plan, package, source), mode=plan_mode(plan), identity=identity)
    return read_record(path, plan_id=V.job_plan_id(v4_plan(), package, source), mode="live", identity=identity)


ALL_CORRECT_RATE = {"sat26": 1, "lane": 4}  # every sat26 all-correct state; one in four open all-correct states


def all_correct_sampled(package, item_id, seed):
    return int(digest(["mech-all-correct", item_id, seed])[:8], 16) % ALL_CORRECT_RATE[package] == 0


def mech_weight(package, record):
    """Inverse sampling weight of a wave-M state (informative states and sat26 all-correct states weigh 1)."""
    return 1 if not all(record["stage0_member_correct"]) else ALL_CORRECT_RATE[package]


def mech_units(package):
    """States as (item_id, seed, source) in one hash order, so that any prefix a budget stop leaves is a uniform
    sample of every stratum: recorded states where a member erred (they carry the hint trial) and the sampled
    all-correct states (bank and sweep only). sat26 states are all its wave-F states, scheduled before any call."""
    if package == "sat26":
        units = [(i, s, "fresh_mm") for i, s in fresh_units(package)]
    else:
        units = []
        for source, benchmarks in MECH_SOURCES[package]:
            for path in sorted((recorded_folder(package, source) / "graded").glob("*.json")):
                row = json.loads(path.read_text(encoding="utf-8"))
                if benchmarks and row["benchmark"] not in benchmarks:
                    continue
                item_id, seed = row["item_id"], int(row.get("seed", 0))
                if (any(not ok for ok in (row.get("stage0_correct") or {}).values())
                        or all_correct_sampled(package, item_id, seed)):
                    units.append((item_id, seed, source))
    return sorted(units, key=lambda u: digest(["mech-order", u[0], u[1], u[2]]))


def dense_units(package):
    """Open roster: up to 200 informative 2025-26 states at seed 0 (hash order); the solver is the lowest wrong member."""
    if package != "lane":
        return []
    rows = []
    for source in ("new_e2_mathmage_nov", "new_e2_mathmage_apr"):
        for path in sorted((recorded_folder(package, source) / "graded").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if int(row.get("seed", 0)) == 0 and any(not ok for ok in (row.get("stage0_correct") or {}).values()):
                rows.append((row["item_id"], 0, source))
    rows.sort(key=lambda u: digest(["dense", u[0]]))
    return sorted(rows[:DENSE_STATES])


def fresh_units(package):
    return [(i, s) for i in sorted(it.item_id for it in v6_registry().all()) for s in FRESH_SEEDS[package]]


def offline_root(package):
    return SPECS[package]["artifact"].parent


def dev_checkpoints(package):
    """Stage-0 checkpoints of the ten development families, saved by the frozen offline phase."""
    root = offline_root(package)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dev_ids = set(manifest["dev_ids"])
    found = {}
    for path in sorted((root / "observations").glob("*.json")):
        try:
            data = read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"])["data"]
        except Exception:     # noqa: BLE001 - observations of other kinds lack the checkpoint schema
            continue
        if isinstance(data, dict) and data.get("item_id") in dev_ids and "stage0" in data and "sha256" in data:
            found.setdefault(data["item_id"], []).append(data)
    return found


def dev_units(package):
    manifest = json.loads((offline_root(package) / "manifest.json").read_text(encoding="utf-8"))
    return [(item_id, 0) for item_id in sorted(manifest["dev_ids"])]


# ----------------------------------------------------------------------------- jobs
def jobs(package):
    fresh = fresh_units(package)
    out = [{"id": "fresh_mm", "wave": "F", "method": "mathmage16", "units": fresh}]
    if package == "sat26":
        out.append({"id": "fresh_sc", "wave": "F", "method": "sc", "model": "deepseek-v3", "source": "fresh_mm",
                    "units": fresh})
        out.append({"id": "fresh_sc_best", "wave": "F", "method": "sc", "model": "gpt-5-mini-low", "k": SC_BEST_K,
                    "source": "fresh_mm", "units": fresh})
    out += [{"id": f"fresh_{m}", "wave": "F", "method": m, "source": "fresh_mm", "units": fresh}
            for m in ("debate", "usc", "moa")]
    out.append({"id": "mech", "wave": "M", "method": "mech", "units": mech_units(package)})
    if package == "lane":
        out.append({"id": "dev_r", "wave": "D", "method": "mathmage_r_dev", "units": dev_units(package)})
        out.append({"id": "fresh_r", "wave": "R", "method": "mathmage_r", "source": "fresh_mm", "units": fresh})
        out.append({"id": "dense", "wave": "Z", "method": "dense", "units": dense_units(package)})
    return out


def unit_key(unit):
    return unit[0], int(unit[1])


# ----------------------------------------------------------------------------- plan
def frozen_state(package):
    spec = SPECS[package]
    manifest = primary_manifest(package)
    calibration = json.loads((offline_root(package) / "manifest.json").read_text(encoding="utf-8"))
    return {"package_root": str(Path(paths.__file__).resolve().parents[1].relative_to(REPO)),
            "roster": spec["roster"],
            "protocol_sha256": _protocol_digest(), "prompt_hashes": prompt_hashes(),
            "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
            "registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
            "vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
            "ext_items_sha256": file_digest(V.EXT / "items.parquet"),
            "ext_answers_sha256": file_digest(V.EXT / "answers.parquet"),
            "v6_items_sha256": file_digest(REG6 / "items.parquet"),
            "v6_answers_sha256": file_digest(REG6 / "answers.parquet"),
            "v6_manifest_sha256": file_digest(REG6 / "manifest.json"),
            "artifact_sha256": file_digest(spec["artifact"]), "calibration_plan_id": calibration["plan_id"],
            "mu_star": float(read_frozen(spec["artifact"], live=ARTIFACT_LIVE)["mu_star"]),
            "policy_plan_id": manifest["plan_id"],
            "policy": frozen_policy(package),
            "baseline_templates": {t.id: t.text for t in (V.DEBATE, V.USC, F5.MOA, F5.MOA_QUERY)},
            "mechanism_templates": {"placebo_note": PLACEBO_NOTE, "hint_levels": list(HINT_LEVELS)},
            "v4_runner_sha256": file_digest(REPO / "scripts" / "v4_followup.py"),
            "v5_runner_sha256": file_digest(REPO / "scripts" / "v5_followup.py"),
            "runner_sha256": file_digest(Path(__file__)),
            "fit_sha256": file_digest(REPO / "scripts" / "v6_fit.py"),
            "tools_runtime": tools_runtime(["full"], "local")}


def job_plan_id(plan, package, job_id):
    return digest([plan["sha256"], package, job_id])


def read_plan_file(name):
    """A frozen v6 plan file, checked against its own sha256 (no frozen-state check)."""
    plan = json.loads((OUT / name).read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError(f"v6 plan {name} hash changed")
    return plan


def load_plan(package):
    """The package's frozen plan, verified against the current code, data and artifacts. sat26's stage-2 plan
    must name the stage-1 plan as its parent and reproduce the sat26 schedule that stage 1 froze."""
    plan = read_plan_file(PLAN_FILES[package])
    if package != "lane":
        parent = read_plan_file(PLAN_FILES["lane"])
        if plan.get("parent_plan_sha256") != parent["sha256"] or plan["analysis_plan"] != parent["analysis_plan"]:
            raise ValueError("stage-2 plan does not extend the frozen stage-1 plan")
        staged = {j["id"]: j["units_sha256"] for j in parent["jobs"][package]}
        if staged != {j["id"]: j["units_sha256"] for j in plan["jobs"][package]}:
            raise ValueError(f"{package} schedule differs from the one frozen in stage 1")
    expected, current = plan["frozen"][package], frozen_state(package)
    for key, value in expected.items():
        if current[key] != value:
            raise ValueError(f"{key} differs from the pre-registered v6 plan; the run must not proceed")
    scheduled = {j["id"]: j["units_sha256"] for j in plan["jobs"][package]}
    for job in jobs(package):
        if scheduled.get(job["id"]) != digest([list(u) for u in job["units"]]):
            raise ValueError(f"units of {job['id']} differ from the pre-registered v6 plan")
    return plan


# ----------------------------------------------------------------------------- guard
def install_diagnostics():
    """Record the redacted error metadata of failed provider responses (scripts/provider_diagnostics.py)."""
    sys.path.insert(0, str(REPO))
    from scripts.provider_diagnostics import ProviderDiagnostics
    from mathmage.llm.openrouter import OpenRouterSync
    diagnostics = ProviderDiagnostics(OUT, qwen_max_attempts=1)
    original = OpenRouterSync.complete

    def diagnosed(backend, settings, messages, response_format, spec):
        return diagnostics.complete(original, backend, settings, messages, response_format, spec)
    OpenRouterSync.complete = diagnosed
    return OpenRouterSync, original


class Guard:
    """Stop on STOP, any provider fault, unreconciled billing or the global soft stop (as in v4/v5)."""

    def __init__(self, package):
        self.package, self.ledger = package, BudgetLedger()
        self.attention = OUT / f"provider_attention_{package}.json"
        self.lock = threading.RLock()

    def reason(self):
        if (OUT / "STOP").exists():
            return "STOP requested"
        if self.attention.exists():
            return "provider attention recorded"
        balance = self.ledger.summary()
        if balance["unreconciled_calls"] or balance["spent_total"] + balance["pending_total"] >= SOFT_STOP:
            return "global spending or billing guard"
        return None

    def install(self):
        original, guard = LLMClient.complete, self

        def guarded(client, request):
            if client.live:
                why = guard.reason()
                if why:
                    raise BudgetExceeded(f"v6 stopped: {why}")
            response = original(client, request)
            if client.live and not response.cached and (response.error or response.billing_uncertain):
                if (response.error or "").startswith("BudgetExceeded"):
                    return response
                with guard.lock:
                    if not guard.attention.exists():
                        atomic_json(guard.attention, {
                            "time_utc": now(), "model": request.model_key, "sample_tag": request.sample_tag,
                            "item_id": request.meta.get("item_id"), "error": response.error,
                            "billing_uncertain": response.billing_uncertain, "attempts": response.attempts,
                            "cost_usd": response.cost_usd, "generation_id": response.generation_id})
                raise BudgetExceeded("v6 provider health gate; see provider attention file")
            return response
        LLMClient.complete = guarded
        return original


# ----------------------------------------------------------------------------- MathMage-16 and baselines
def source_record(package, job, item_id, seed, plan):
    """The sealed wave-F MathMage-16 record whose stage-0 solutions a baseline (or MathMage-R) builds on."""
    path = job_dir(package, job["source"]) / "agent" / V.name_of(item_id, seed, "full")
    return read_record(path, plan_id=job_plan_id(plan, package, job["source"]), mode=plan_mode(plan),
                       identity={"item_id": item_id, "seed": seed, "condition": "full"})


def plan_mode(plan):
    return plan.get("mode", "live")


def stage0_cost(record, member):
    return V.stage0_cost(record, member)


def run_mathmage16(client, package, item, seed):
    engine = engine_for(client, package, seed)
    payload = asdict(engine.run(item, checkpoint=None))
    return payload, grade_result(payload, item, mode="live" if client.live else "mock", vault=vault_for(item.item_id))


def run_baseline(client, package, job, item, seed, plan):
    """v4's SC / debate / USC (unchanged prompts and logic) on the wave-F stage-0 solutions."""
    engine = engine_for(client, package, seed)
    requests = V.stage0_requests(engine, item)
    record = source_record(package, job, item.item_id, seed, plan)
    texts = {c["id"]: c.get("text") or "" for c in record["candidates"]}
    members = list(engine.members)
    calls, extra = [], {}
    candidates = {c["id"]: c for c in record["candidates"]}
    stage0_answers = [candidates.get(f"s0-{i}", {}).get("answer") for i in range(len(members))]
    if job["method"] == "sc":
        member = members.index(job["model"])
        first = V.cached(client, requests[member])
        if first is None:
            raise RuntimeError(f"stage-0 request for {item.item_id} is not an exact cache hit; refusing to sample")
        calls.append(V.call_entry(requests[member], first, item, k=0, reused=True))
        calls[-1]["resource_cost_usd"] = stage0_cost(record, member)
        batch = [V.variant(requests[member], f"sc:k={k}", role=f"sample={k}") for k in range(1, job.get("k", V.K_SC))]
        for k, (request, response) in enumerate(zip(batch, V.complete_all(client, batch)), 1):
            calls.append(V.call_entry(request, response, item, k=k))
    elif job["method"] == "debate":
        batch = []
        for i, request in enumerate(requests):
            others = "\n\n".join(f"One agent solution:\n```\n{texts.get(f's0-{j}') or '(no solution)'}\n```"
                                 for j in range(len(members)) if j != i)
            messages = [request.messages[0], {"role": "assistant", "content": texts.get(f"s0-{i}") or "(no solution)"},
                        {"role": "user", "content": V.DEBATE.render(others=others, problem=engine.item.problem,
                                                                    format_instruction=engine.item.format_instruction)}]
            batch.append(V.variant(request, "debate:r2", messages=messages, template_id=V.DEBATE.id, role=f"member={i}"))
        for i, (request, response) in enumerate(zip(batch, V.complete_all(client, batch))):
            calls.append(V.call_entry(request, response, item, k=i))
        extra["stage0_answers"] = stage0_answers
        extra["stage0_cost_usd"] = sum(stage0_cost(record, i) for i in range(len(members)))
    elif job["method"] == "usc":
        judge = engine._judge_model()
        responses = "\n\n".join(f"Response {i}: {texts.get(f's0-{i}') or '(no solution)'}" for i in range(len(members)))
        prompt = V.USC.render(problem=engine.item.problem, responses=responses)
        request = V.variant(requests[0], "usc", messages=[{"role": "user", "content": prompt}], template_id=V.USC.id,
                            model_key=judge, role="selector")
        response = client.complete(request)
        calls.append(V.call_entry(request, response, item, k=0))
        import re as _re
        match = _re.search(r"most consistent response is Response\s*(\d+)", response.text or "", _re.I)
        choice = int(match.group(1)) if match and int(match.group(1)) < len(members) else None
        extra.update({"judge": judge, "choice": choice, "parsed": choice is not None,
                      "chosen_answer": candidates.get(f"s0-{choice}", {}).get("answer") if choice is not None else None,
                      "stage0_answers": stage0_answers,
                      "stage0_cost_usd": sum(stage0_cost(record, i) for i in range(len(members)))})
    elif job["method"] == "moa":
        judge = engine._judge_model()
        proposals = "\n".join(f"{i + 1}. {texts.get(f's0-{i}') or '(no solution)'}" for i in range(len(members)))
        messages = [{"role": "system", "content": F5.MOA.render(responses=proposals)},
                    {"role": "user", "content": F5.MOA_QUERY.render(problem=engine.item.problem,
                                                                     format_instruction=engine.item.format_instruction)}]
        request = V.variant(requests[0], "moa", messages=messages, template_id=F5.MOA.id, model_key=judge,
                            role="aggregator")
        response = client.complete(request)
        calls.append(V.call_entry(request, response, item, k=0))
        extra.update({"judge": judge, "stage0_answers": stage0_answers,
                      "stage0_cost_usd": sum(stage0_cost(record, i) for i in range(len(members)))})
    return {"job": job["id"], "method": job["method"], "item_id": item.item_id, "seed": seed, "condition": "full",
            "members": members, "calls": calls, **extra}


def majority(answers, answer_format):
    usable = [a for a in answers if a]
    if not usable:
        return None
    labels = cluster(usable, answer_format)
    return usable[max(range(len(usable)), key=lambda j: (labels.count(labels[j]), -j))]


def grade_baseline(payload, item):
    """v4/v5 grading (USC and MoA fall back to the stage-0 vote), with the v6 vault for fresh items."""
    correct = correct_of(item.item_id)
    out = {"job": payload["job"], "method": payload["method"], "item_id": item.item_id, "benchmark": item.benchmark,
           "family_id": item.family_id, "split": item.split, "item_date": item.item_date, "seed": payload["seed"],
           "samples": [{"k": c["k"], "model_key": c["model_key"], "answer": c["answer"], "correct": correct(c["answer"]),
                        "resource_cost_usd": c["resource_cost_usd"], "reused_stage0": c["reused_stage0"]}
                       for c in payload["calls"]],
           "new_cost_usd": sum(c["cost_usd"] for c in payload["calls"]),
           "resource_cost_usd": sum(c["resource_cost_usd"] for c in payload["calls"]) + payload.get("stage0_cost_usd", 0.0)}
    answers = payload.get("stage0_answers")
    if payload["method"] == "usc":
        chosen = payload["chosen_answer"] if payload["parsed"] else majority(answers, item.answer_format)
        out.update({"usc_parsed": payload["parsed"], "usc_choice": payload["choice"], "usc_answer": chosen,
                    "usc_correct": correct(chosen), "stage0_correct": [correct(a) for a in answers]})
    elif payload["method"] == "moa":
        call = payload["calls"][0]
        chosen = call["answer"] or majority(answers, item.answer_format)
        out.update({"moa_valid": bool(call["answer"]), "moa_answer": chosen, "moa_correct": correct(chosen),
                    "moa_fallback": not call["answer"], "judge": payload["judge"],
                    "stage0_correct": [correct(a) for a in answers]})
    elif payload["method"] == "debate":
        out["stage0_correct"] = [correct(a) for a in answers]
    return out


# ----------------------------------------------------------------------------- MathMage-R
def r_artifact_path(package):
    return OUT / f"artifacts_r_{package}.json"


def r_artifacts(package, plan):
    """MathMage-R's frozen artifact: written by scripts/v6_fit.py, whose hash the plan fixes."""
    artifacts = read_frozen(r_artifact_path(package), live=True)
    fit = artifacts.get("v6_fit", {})
    if fit.get("fit_sha256") != plan["frozen"][package]["fit_sha256"]:
        raise ValueError("MathMage-R artifact was not written by the pre-registered scripts/v6_fit.py")
    return artifacts


@contextlib.contextmanager
def voc_factors(factors):
    """Scale each action type's value of computation by its pre-registered factor (MathMage-R only)."""
    original = ENGINE.Policy
    factors = {str(k): float(v) for k, v in (factors or {}).items()}

    class CalibratedPolicy(original):
        def score(self, actions, *args, **kwargs):
            scored = super().score(actions, *args, **kwargs)
            for action in scored:
                action.voc *= factors.get(action.kind, 1.0)
            return scored

    ENGINE.Policy = CalibratedPolicy
    try:
        yield
    finally:
        ENGINE.Policy = original


def engine_r(client, package, seed, artifacts, mu=0.0):
    validate_artifact_runtime(artifacts, SPECS[package]["roster"], tools_runtime(["full"], "local"))
    return engine_for(client, package, seed, artifacts=artifacts, mu=mu)


def run_mathmage_r(client, package, item, seed, checkpoint, artifacts):
    engine = engine_r(client, package, seed, artifacts)
    with voc_factors(artifacts.get("voc_factors")):
        payload = asdict(engine.run(item, checkpoint=checkpoint))
    payload["meta"]["voc_factors"] = artifacts.get("voc_factors")
    return payload, grade_result(payload, item, mode="live" if client.live else "mock", vault=vault_for(item.item_id))


# ----------------------------------------------------------------------------- mechanism layer
def restored(client, package, item, seed, checkpoint):
    """A frozen MathMage-16 engine restored to a recorded stage-0 state (no call is made)."""
    engine = engine_for(client, package, seed, mu=RESTORE_MU, max_calls=MECH_MAX_CALLS)
    engine.run(item, checkpoint=checkpoint)
    return engine


def counters():
    return {"samples": Counter(), "hints": Counter(), "reviews": Counter(), "tool_alt": 0, "diversify": 0, "judges": 0}


def scored_state(engine):
    """Frozen-parameter posterior and the valued feasible actions at the engine's current state."""
    params = engine._posterior_params()
    cfg = PolicyConfig(mu=0.0, epsilon=engine.config.voc_epsilon, max_actions=engine.config.max_policy_actions,
                       max_item_cost_usd=engine.config.max_item_cost_usd)
    policy = Policy(params, cfg)
    label_of, obs, evidence, C = engine._posterior_state()
    selectable = {label_of[c.id] for c in engine._included() if c.id in label_of}
    result = policy.posterior_of(obs, evidence, C, selectable=selectable)
    actions = policy.score(engine._feasible_actions(label_of, C, result, counters(), cfg, params), obs, evidence, C,
                           lambda m: engine.members[m], lambda s, t: engine._coaching_spec(s, t, params, label_of),
                           result)
    return params, label_of, obs, evidence, C, result, actions


def state_summary(engine, label_of, obs, evidence, C, result):
    """Everything needed to recompute this state's posterior under any parameters (lifts by kind, not value)."""
    lift_kind = {}
    for c in engine.candidates:
        if c.phase in ("coached_solve",):
            lift_kind[c.id] = "coached"
        elif c.phase == "method_diversification":
            lift_kind[c.id] = "diversify"
        elif c.phase == "rebuttal_revision":
            lift_kind[c.id] = "revision"
        else:
            lift_kind[c.id] = "none"
    included = engine._included()
    observations = []
    for c in included:
        lab = label_of.get(c.id, -1)
        if c.phase in ENGINE.TOOL_PHASES and lab < 0:
            continue
        source = -1
        if c.id in engine._coach_source:
            source = label_of.get(engine._coach_source[c.id][0], -1)
        observations.append({"candidate": c.id, "agent": engine._agent_key(c), "cluster": lab,
                             "lift_kind": lift_kind[c.id], "source": source})
    answers = {}
    for c in included:
        if c.id in label_of:
            answers.setdefault(label_of[c.id], c.answer)
    best_rep = engine._representative(result.best, label_of) if C else None
    return {"observations": observations,
            "evidence": [{"kind": e.kind, "cluster": e.cluster, "options": list(e.options)} for e in evidence],
            "n_clusters": C, "cluster_answers": {str(k): v for k, v in answers.items()},
            "selectable": sorted({label_of[c.id] for c in included if c.id in label_of}),
            "best_cluster": int(result.best) if C else None, "best_answer": best_rep.answer if best_rep else None,
            "confidence": float(result.confidence), "q": [float(x) for x in result.q], "q_none": float(result.q_none)}


def hint_branch(engine, solver, tutor, level, arm, params):
    """One hint from ``tutor`` to ``solver`` at the stage-0 state, then the scaffolded re-solve (as _act_coach)."""
    current = engine._latest_candidate(solver)
    tutor_answer = engine._latest_candidate(tutor)
    role = f"{current.id}:{level}:{arm}:t{tutor}"
    response = engine._call(engine._request(engine.members[tutor], HINT, "mech_hint", role, problem=engine.item.problem,
                                            attempt=current.certificate, hint_kind=HINT_KINDS[level]))
    hint = response.text if response.ok and not response.truncated else ""
    audit = structural_audit(hint)
    leak = bool(hint) and engine._answer_leak(hint)
    if leak:
        audit.structural_pass = False
        audit.reasons.append("hint contains a candidate final answer")
    audited = None
    if hint:
        auditor = min((i for i in range(len(engine.members)) if i != tutor),
                      key=lambda i: engine.models.model(engine.members[i]).price.out)
        check = engine._call(engine._request(engine.members[auditor], RESIDUAL_AUDIT, "mech_hint_audit", role,
                                             problem=engine.item.problem, hint=hint))
        verdict = parse_json(check.text) if check.ok and not check.truncated else {}
        audited = {"auditor": engine.member_ids[auditor], "verdict": verdict,
                   "residual_pass": residual_audit(hint, verdict).residual_work_pass}
    delivered = audit.structural_pass           # the deployed screen: structural audit plus the answer filter
    engine.trace.append({"event": "mech_hint", "arm": arm, "solver": engine.member_ids[solver],
                         "tutor": engine.member_ids[tutor], "level": level, "hint": hint, "answer_leak_filtered": leak,
                         "structural": audit.to_dict(), "residual_audit": audited, "delivered": delivered})
    if not delivered:
        return None
    response = engine._call(engine._request(engine.members[solver], SOLVE_SCAFFOLD, "mech_coached_solve", role,
                                            problem=engine.item.problem,
                                            format_instruction=engine.item.format_instruction, note=hint))
    candidate = engine._candidate(response, solver, "coached_solve", current.id)
    if engine._tutor_dependent(params):
        engine._lifts[candidate.id] = params.lift["coached_wrong"]
        if tutor_answer is not None:
            engine._coach_source[candidate.id] = (tutor_answer.id, params.lift["coached_right"])
    else:
        engine._lifts[candidate.id] = engine._pair_lift(solver, tutor, params)
    return candidate


def placebo_branch(engine, solver):
    current = engine._latest_candidate(solver)
    response = engine._call(engine._request(engine.members[solver], SOLVE_SCAFFOLD, "mech_placebo",
                                            f"{current.id}:placebo", problem=engine.item.problem,
                                            format_instruction=engine.item.format_instruction, note=PLACEBO_NOTE))
    candidate = engine._candidate(response, solver, "placebo_solve", current.id)
    engine._lifts[candidate.id] = 0.0
    return candidate


def mech_arms(item_id, seed, correct, actions, engine):
    """The pre-registered arms of one state (tutors and formats drawn by hash, never by outcome)."""
    n = len(correct)
    wrong = [i for i in range(n) if not correct[i]]
    right = [i for i in range(n) if correct[i]]
    certified = {i for i in range(n) if (engine._latest_candidate(i) is not None
                                         and engine._latest_candidate(i).certificate)}
    arms = [{"kind": "sample", "member": i, "k": 1} for i in range(n)]
    arms += [{"kind": "sample", "member": i, "k": 2} for i in wrong]
    arms.append({"kind": "tool_alt"})
    arms.append({"kind": "diversify"})
    reviews = sorted((a for a in actions if a.kind == "review"), key=lambda a: (-a.voc, a.cluster))
    if reviews:
        arms.append({"kind": "review", "member": reviews[0].member, "cluster": reviews[0].cluster})
    judge = next((a for a in actions if a.kind == "judge"), None)
    if judge is not None:
        arms.append({"kind": "judge", "options": list(judge.options)})
    for s in wrong:
        if s in certified:
            for arm, pool in (("right", [t for t in right if t != s]), ("wrong", [t for t in wrong if t != s])):
                if pool:
                    arms.append({"kind": "hint", "arm": arm, "solver": s,
                                 "tutor": pick(pool, item_id, seed, s, arm, "tutor"),
                                 "level": pick(HINT_LEVELS, item_id, seed, s, arm, "level")})
        arms.append({"kind": "placebo", "solver": s})
    if wrong:
        for r in right:
            pool = [t for t in wrong if t != r]
            if pool and r in certified:
                arms.append({"kind": "hint", "arm": "harm", "solver": r, "tutor": pick(pool, item_id, seed, r, "harm"),
                             "level": pick(HINT_LEVELS, item_id, seed, r, "harm", "level")})
    return arms


def run_arm(client, package, item, seed, checkpoint, arm, actions):
    """One branch from the stage-0 state; ``actions`` are the stage-0 actions valued once by run_mech."""
    engine = restored(client, package, item, seed, checkpoint)
    params = engine._posterior_params()
    label_of = engine._posterior_state()[0]
    before = {c.id for c in engine.candidates}
    trace_start, evidence_start = len(engine.trace), len(engine._evidence_records)
    chosen = None
    if arm["kind"] == "sample":
        tally = counters()
        tally["samples"][arm["member"]] = arm["k"] - 1          # role sample=<k>:member=<i>, as the controller's
        chosen = next((a for a in actions if a.kind == "sample" and a.member == arm["member"]), None)
        engine._act_sample(arm["member"], tally)
    elif arm["kind"] == "tool_alt":
        chosen = next((a for a in actions if a.kind == "tool_alt"), None)
        engine._tool_forward(alt=True)
    elif arm["kind"] == "diversify":
        chosen = next((a for a in actions if a.kind == "diversify"), None)
        index = chosen.member if chosen else next((i for i, m in enumerate(engine.members) if m == engine.best_member), 0)
        engine._execute_action(Action("diversify", member=index), label_of, counters(), params)
    elif arm["kind"] == "review":
        chosen = next((a for a in actions if a.kind == "review" and a.member == arm["member"]
                       and a.cluster == arm["cluster"]), None)
        engine._execute_action(Action("review", member=arm["member"], cluster=arm["cluster"]), label_of, counters(), params)
    elif arm["kind"] == "judge":
        chosen = next((a for a in actions if a.kind == "judge"), None)
        engine._execute_action(Action("judge", options=tuple(arm["options"])), label_of, counters(), params)
    elif arm["kind"] == "hint":
        chosen = next((a for a in actions if a.kind == "coach" and a.member == arm["solver"]
                       and a.tutor == arm["tutor"]), None)
        hint_branch(engine, arm["solver"], arm["tutor"], arm["level"], arm["arm"], params)
    elif arm["kind"] == "placebo":
        placebo_branch(engine, arm["solver"])
    label_after, obs_after, ev_after, C_after = engine._posterior_state()
    policy = Policy(params, PolicyConfig(mu=0.0, epsilon=engine.config.voc_epsilon))
    selectable = {label_after[c.id] for c in engine._included() if c.id in label_after}
    result_after = policy.posterior_of(obs_after, ev_after, C_after, selectable=selectable)
    new = [c for c in engine.candidates if c.id not in before]
    return {"arm": arm, "predicted": chosen.to_dict() if chosen else None,
            "new_candidates": [asdict(c) for c in new], "trace": engine.trace[trace_start:],
            "new_evidence": [list(e[:2]) + [list(e[2])] for e in engine._evidence_records[evidence_start:]],
            "calls": engine.calls, "post": state_summary(engine, label_after, obs_after, ev_after, C_after, result_after),
            "new_cost_usd": sum(float(c.get("cost_usd") or 0) for c in engine.calls),
            "resource_cost_usd": sum(float(c.get("resource_cost_usd") or 0) for c in engine.calls)}


def run_mech(client, package, item, seed, source, checkpoint=None, plan=None):
    if checkpoint is None:
        checkpoint = recorded_record(package, source, item.item_id, seed, plan)["checkpoint"]
    correct = correct_of(item.item_id)
    stage0 = [c for c in checkpoint["stage0"] if c["id"] != "s0-tool"]
    member_correct = [bool(c.get("valid", True) and not c.get("error") and correct(c.get("answer"))) for c in stage0]
    engine = restored(client, package, item, seed, checkpoint)
    params, label_of, obs, evidence, C, result, actions = scored_state(engine)
    arms = mech_arms(item.item_id, seed, member_correct, actions, engine)
    with ThreadPoolExecutor(max_workers=4) as pool:
        branches = list(pool.map(lambda arm: run_arm(client, package, item, seed, checkpoint, arm, actions), arms))
    stopped = [c.get("error") for b in branches for c in b["calls"] if (c.get("error") or "").startswith("BudgetExceeded")]
    if stopped:
        raise BudgetExceeded(stopped[0])
    return {"job": "mech", "method": "mech", "item_id": item.item_id, "seed": seed, "source": source,
            "members": engine.member_ids, "stage0_member_correct": member_correct,
            "stage0_candidates": [{k: c.get(k) for k in ("id", "member", "model_key", "phase", "answer", "valid",
                                                          "error")} for c in checkpoint["stage0"]],
            "stage0": state_summary(engine, label_of, obs, evidence, C, result),
            "stage0_actions": [a.to_dict() for a in actions], "arms": branches}


def grade_mech(payload, item):
    correct = correct_of(item.item_id)
    rows = []
    for branch in payload["arms"]:
        new = branch["new_candidates"]
        rows.append({"arm": branch["arm"], "new_answers": [c.get("answer") for c in new],
                     "new_correct": [bool(c.get("valid", True) and not c.get("error") and correct(c.get("answer")))
                                     for c in new],
                     "post_correct": correct(branch["post"]["best_answer"]),
                     "predicted_voc": (branch["predicted"] or {}).get("voc"),
                     "new_cost_usd": branch["new_cost_usd"], "resource_cost_usd": branch["resource_cost_usd"]})
    return {"job": "mech", "method": "mech", "item_id": item.item_id, "benchmark": item.benchmark,
            "family_id": item.family_id, "seed": payload["seed"], "source": payload["source"],
            "stage0_member_correct": payload["stage0_member_correct"],
            "stage0_correct": correct(payload["stage0"]["best_answer"]), "arms": rows,
            "new_cost_usd": sum(r["new_cost_usd"] for r in rows)}


def run_dense(client, package, item, seed, source, checkpoint=None):
    """Eight fresh attempts and eight hinted re-solves (tutors alternate; formats by hash) by one wrong member."""
    checkpoint = checkpoint or recorded_record(package, source, item.item_id, seed)["checkpoint"]
    correct = correct_of(item.item_id)
    stage0 = [c for c in checkpoint["stage0"] if c["id"] != "s0-tool"]
    member_correct = [bool(c.get("valid", True) and not c.get("error") and correct(c.get("answer"))) for c in stage0]
    solver = member_correct.index(False)
    engine = restored(client, package, item, seed, checkpoint)
    params = engine._posterior_params()
    tutors = [t for t in range(len(engine.members)) if t != solver]
    retries = [engine._request(engine.members[solver], SOLVE, "dense_retry", f"k={k}:member={solver}",
                               problem=engine.item.problem, format_instruction=engine.item.format_instruction,
                               cards=engine._cards()) for k in range(DENSE_DRAWS)]
    responses = engine._independent_calls(retries)
    retry_answers = [ENGINE.normalize(ENGINE.extract_answer(r.text), engine.answer_format)
                     if r.ok and not r.truncated else None for r in responses]
    hints = []
    certified = bool(engine._latest_candidate(solver) and engine._latest_candidate(solver).certificate)
    for k in range(DENSE_DRAWS if certified else 0):
        tutor = tutors[k % len(tutors)]
        level = pick(HINT_LEVELS, item.item_id, seed, solver, "dense", k)
        candidate = hint_branch(engine, solver, tutor, level, f"dense{k}", params)
        hints.append({"k": k, "tutor": tutor, "level": level, "answer": candidate.answer if candidate else None,
                      "delivered": candidate is not None})
    return {"job": "dense", "method": "dense", "item_id": item.item_id, "seed": seed, "source": source,
            "solver": solver, "stage0_member_correct": member_correct, "retry_answers": retry_answers,
            "hints": hints, "trace": engine.trace, "calls": engine.calls,
            "new_cost_usd": sum(float(c.get("cost_usd") or 0) for c in engine.calls)}


def grade_dense(payload, item):
    correct = correct_of(item.item_id)
    return {"job": "dense", "item_id": item.item_id, "benchmark": item.benchmark, "family_id": item.family_id,
            "seed": payload["seed"], "solver": payload["solver"],
            "retry_correct": [correct(a) for a in payload["retry_answers"]],
            "hint_correct": [correct(h["answer"]) if h["delivered"] else None for h in payload["hints"]],
            "new_cost_usd": payload["new_cost_usd"]}


# ----------------------------------------------------------------------------- workers
def process(clients, package, job, unit, plan, r_arts=None, dev_cps=None):
    item_id, seed = unit_key(unit)
    target = job_dir(package, job["id"])
    out = target / "agent" / V.name_of(item_id, seed, "full")
    if out.exists():
        return False
    client = clients(WAVE_LINE[job["wave"]])
    item = item_of(item_id)
    mode = "live" if client.live else "mock"
    if job["method"] == "mathmage16":
        payload, graded = run_mathmage16(client, package, item, seed)
    elif job["method"] in ("sc", "debate", "usc", "moa"):
        if not (job_dir(package, job["source"]) / "agent" / out.name).exists():
            return False                          # its stage-0 record is not written yet
        payload = run_baseline(client, package, job, item, seed, plan)
        graded = grade_baseline(payload, item)
    elif job["method"] == "mech":
        if unit[2].startswith("fresh_") and not (job_dir(package, unit[2]) / "agent" / out.name).exists():
            return False                          # its wave-F state is not written yet
        payload = run_mech(client, package, item, seed, unit[2], plan=plan)
        graded = grade_mech(payload, item)
    elif job["method"] == "dense":
        payload = run_dense(client, package, item, seed, unit[2])
        graded = grade_dense(payload, item)
    elif job["method"] == "mathmage_r":
        if not (job_dir(package, job["source"]) / "agent" / out.name).exists():
            return False
        checkpoint = source_record(package, job, item_id, seed, plan)["checkpoint"]
        payload, graded = run_mathmage_r(client, package, item, seed, checkpoint, r_arts)
    elif job["method"] == "mathmage_r_dev":
        checkpoints = dev_cps.get(item_id) or []
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one saved development checkpoint for {item_id}, found {len(checkpoints)}")
        payload, graded = run_mathmage_r(client, package, item, seed, checkpoints[0], r_arts)
    else:
        raise ValueError(job["method"])
    stopped = [c.get("error") for c in (payload.get("calls") or payload.get("meta", {}).get("calls") or [])
               if (c.get("error") or "").startswith("BudgetExceeded")]
    if stopped:                                     # never keep a unit with a call the guard refused
        raise BudgetExceeded(stopped[0])
    atomic_json(out, seal_record(payload, plan_id=job_plan_id(plan, package, job["id"]), mode=mode))
    atomic_json(target / "graded" / out.name, graded)
    return True


def worker(package, wave, index, count, client_factory=None):
    plan = load_plan(package)
    guard = Guard(package)
    state = {"package": package, "wave": wave, "index": index, "of": count, "pid": os.getpid(), "done": 0,
             "waiting": 0, "status": "running", "started_at_utc": now()}
    status = OUT / "workers" / f"{package}_{wave}_{index}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    made = {}

    def clients(kind):
        if kind not in made:
            line = LINES[kind][package]
            made[kind] = client_factory(line) if client_factory else LLMClient(experiment=line, live=True)
        return made[kind]
    original = guard.install()
    transport, original_transport = install_diagnostics()
    r_arts = r_artifacts(package, plan) if wave in ("D", "R") else None
    dev_cps = dev_checkpoints(package) if wave == "D" else None
    try:
        for job in jobs(package):
            if job["wave"] != wave:
                continue
            for unit in job["units"]:
                item_id, seed = unit_key(unit)
                if V.partition(item_id, seed, count) != index:
                    continue
                why = guard.reason()
                if why:
                    state["status"] = f"stopped: {why}"
                    return
                try:
                    done = process(clients, package, job, unit, plan, r_arts, dev_cps)
                except BudgetExceeded as exc:
                    state["status"] = f"stopped: {exc}"
                    return
                if done:
                    state["done"] += 1
                state["current"] = f"{job['id']} {item_id} s{seed}"
                state["updated_at_utc"] = now()
                atomic_json(status, state)
        state["status"] = "complete"
    finally:
        state["updated_at_utc"] = now()
        atomic_json(status, state)
        LLMClient.complete = original
        transport.complete = original_transport
        for client in made.values():
            client.close()


# ----------------------------------------------------------------------------- estimates
def per_unit_costs(package):
    """Measured mean new spend per unit from finished v4/v5 jobs on 2025-26 contests (planning only)."""
    if package == "sat26":
        return sat26_costs()
    rate = lambda folder: float(np.mean([json.loads(p.read_text(encoding="utf-8")).get(  # noqa: E731
        "new_cost_usd", json.loads(p.read_text(encoding="utf-8")).get("cost_usd", 0.0))
        for p in (folder / "graded").glob("*.json")]))
    out = {"mathmage16": rate(V.V4 / "new_e2_mathmage_nov"), "debate": rate(V.V4 / "new_e2_debate_nov"),
           "usc": rate(V.V4 / "new_e2_usc_nov"), "moa": rate(V.LIVE / "program_v5" / "nov_e2_moa")}
    stage0 = []
    for path in (V.V4 / "new_e2_mathmage_nov" / "agent").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        stage0.append(sum(float(c.get("resource_cost_usd") or 0) for c in record["meta"]["calls"] if c["phase"] == "stage0"))
    out["controller"] = max(0.0, out["mathmage16"] - float(np.mean(stage0)))
    out["mech"] = 0.0075                                             # mechanism layer per state (trace-based plan)
    out["dense"] = 0.012
    return out


def sat26_costs():
    """sat26 planning rates from its own offline-phase calls (the call log rows after its first own call):
    MathMage-16 = mean spend per development problem (stage 0, program and the maximal policy run); baselines
    from the members' mean stage-0 solve cost; the mechanism layer from the mean call cost and the arm count."""
    import yaml
    members = yaml.safe_load((paths.CONFIGS / "models.yaml").read_text(encoding="utf-8"))["rosters"]["sat26"]["members"]
    manifest = json.loads((offline_root("sat26") / "manifest.json").read_text(encoding="utf-8"))
    rows, start = [], None
    with open(paths.LOGS / "calls" / "offline.jsonl", encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if start is None and row.get("model_key") == "gpt-5-mini-low":
                start = row["timestamp"]
            if start is not None and row.get("model_key") in members:
                rows.append(row)
    # Resource cost: a cache hit (an identical earlier SAT-roster request) is priced as the call it reuses,
    # since fresh problems have no cached requests.
    cost = lambda r: float(r.get("resource_cost_usd") or r.get("cost_usd") or 0)  # noqa: E731
    stage0 = {m: float(np.mean([cost(r) for r in rows if r["model_key"] == m and "stage0" in r["sample_tag"]]))
              for m in members}
    dev = {}
    for row in rows:
        if row.get("item_id") in manifest["dev_ids"]:
            dev[row["item_id"]] = dev.get(row["item_id"], 0.0) + cost(row)
    per_call = float(np.mean([cost(r) for r in rows]))
    out = {"mathmage16": float(np.mean(list(dev.values()))) if dev else 3.5 * sum(stage0.values()),
           "sc:deepseek-v3": (V.K_SC - 1) * stage0["deepseek-v3"],
           "sc:gpt-5-mini-low": (SC_BEST_K - 1) * stage0["gpt-5-mini-low"],
           "debate": 1.25 * sum(stage0.values()), "usc": 0.8 * stage0["gpt-5-mini-low"],
           "moa": 1.1 * stage0["gpt-5-mini-low"],
           "mech": 16 * per_call,             # about 16 calls per state: bank, sweep and the crossed hint trial
           "dense": 0.0, "controller": 0.0}
    out["dev_problems_measured"] = len(dev)
    return out


def estimate(package):
    rate = per_unit_costs(package)
    out = {}
    for job in jobs(package):
        method = job["method"]
        per = (rate["controller"] if method in ("mathmage_r", "mathmage_r_dev") else
               rate["mech"] if method == "mech" else rate["dense"] if method == "dense" else
               rate.get(f"{method}:{job.get('model')}", rate.get(method)))
        out[job["id"]] = {"wave": job["wave"], "units": len(job["units"]), "per_unit_usd": round(per, 5),
                          "expected_usd": round(per * len(job["units"]), 3)}
    return out


# ----------------------------------------------------------------------------- commands
ANALYSIS_PLAN = {
    "primary": [
        "H9 (sat26 team: GPT-5-mini low, Claude Haiku 4.5, DeepSeek-V3; fresh set, one seed): MathMage-16 at its "
        "own knee-rule mu* vs majority vote, DeepSeek-V3 SC@10, USC, one-round debate and MoA; Holm across the "
        "five contrasts. The team replaces the published SAT roster because o3-mini was retired.",
        "H10 (open roster, fresh set, eight seeds): MathMage-16 at mu* vs majority vote, SC@5 from the 24 pooled "
        "stage-0 samples, USC, debate and MoA; Holm across the five contrasts.",
        "Each contrast: paired macro difference over the fresh contest cells (benchmarks weighted equally), "
        "family-clustered bootstrap 95% CI (2,000 draws; crossed with seeds for the open roster), two-sided "
        "bootstrap p."],
    "secondary": [
        "S6 (MathMage-R vs MathMage-16 on the fresh set, open roster): paired difference at each variant's own "
        "knee-rule price and at the maximal run; two-sided.",
        "S7 (sat26 calibration transfer, descriptive): the sat26 team's offline parameters (abilities, collision "
        "rates, lifts, judge accuracy, mu*) beside the published SAT roster's, and its fresh-set coverage, vote "
        "and decision accuracy; no test.",
        "S8 (hint trial, both rosters; unit = initially wrong member of a recorded state): rescue rate after a "
        "right tutor's hint vs one fresh attempt, vs the better of two fresh attempts, vs a wrong tutor's hint "
        "and vs the placebo re-solve; harm rate of a wrong tutor's hint on right members. Paired differences "
        "with problem-clustered bootstrap CIs; also restricted to hints passing the residual-work audit. Lifts "
        "are refitted with the frozen IRT-consistent estimator and the proximal-zone shares recomputed.",
        "S9 (VOC calibration): for each swept action type at stage-0 states, the mean predicted value of "
        "computation (frozen MathMage-16 parameters and MathMage-R parameters) vs the realized change in decision "
        "correctness under the same parameters; non-adaptive, so rows are comparable treatment effects.",
        "S10 (calibration scale): cross-fitted learning curves of stage-0 accuracy, ECE and Brier score as the "
        "calibration pool grows from 16 to all recorded problems (no calls).",
        "S11 (dense ZPD, open roster, if run): per-problem retry and hint success rates over eight draws each; "
        "zones observed problem by problem at K = 1 and 2.",
        "S12 (strongest-member self-consistency, sat26): MathMage-16 at mu* vs GPT-5-mini SC@k at matched mean "
        "resource cost (k = the largest k <= 5 whose mean SC@k cost does not exceed MathMage-16's; SC@k uses "
        "samples 0..k-1, sample 0 being the reused stage-0 solution) and at k = 5; paired macro difference, "
        "family-clustered bootstrap CI, two-sided, reported whatever the outcome."],
    "mathmage_r": [
        "Open roster only. Fit by scripts/v6_fit.py (hash frozen here) on data available before wave R: the "
        "recorded non-fresh runs (stage-0 panels, certificate checks, judge picks, reviews), the original training "
        "evidence and public anchors, and the wave-M bank, sweep and hint trial. Same estimators and priors as "
        "the published calibration. VOC factor per action type = (sum of realized decision changes + 5 x mean "
        "predicted) / (sum of predicted values + 5 x mean predicted), clipped to [0.1, 1], from the non-adaptive "
        "wave-M sweep under the refitted parameters. mu*_R from wave D by the frozen knee rule (cheapest price "
        "within one point of the best development accuracy). No fresh outcome is read before wave R."],
    "rules": [
        "Sample sizes are fixed by this plan; no optional stopping or added data after results.",
        "Every job is reported whatever its outcome; jobs stopped by budget are reported as incomplete.",
        "MathMage-16 uses the v3 code and policy unchanged (open roster: its v3 artifact; sat26: the artifact of "
        "the frozen offline phase rerun for the new team, before any fresh call); baselines use the v4/v5 "
        "templates and logic.",
        "Mechanism-layer arms are assigned by hash before any call; gold labels define arms (wrong/right members) "
        "but never enter any method's decision. sat26 wave-M states are every wave-F state. Wave-M states run in "
        "one hash order, so a budget stop leaves a uniform sample of every stratum.",
        "Matched cost uses mean resource USD per problem, counting reused stage-0 samples."]}


ANALYSIS_PLAN["rules"].append(
    "Staged pre-registration: plan.json (stage 1) freezes this analysis plan, the open roster's frozen state and "
    "jobs, and the sat26 jobs and units; plan_sat26.json (stage 2) freezes sat26's frozen state (its calibration "
    "artifact and mu*, code and data hashes) after its offline phase and before any sat26 fresh-set call, names "
    "plan.json as its parent and must reproduce the stage-1 sat26 schedule.")


def schedule_of(job_list):
    return [{**{k: v for k, v in j.items() if k != "units"}, "n_units": len(j["units"]),
             "units": [list(u) for u in j["units"]], "units_sha256": digest([list(u) for u in j["units"]])}
            for j in job_list]


def state_command(package, jobs_only=False):
    if jobs_only:
        print(json.dumps({"jobs": jobs(package)}))
    else:
        print(json.dumps({"frozen": frozen_state(package), "jobs": jobs(package), "estimate": estimate(package)}))


def package_state(package, jobs_only=False):
    """The package's state, computed in its own process (each package binds its own ``mathmage`` copy)."""
    command = [sys.executable, str(Path(__file__)), "state", "--package", package] + (["--jobs-only"] if jobs_only else [])
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return json.loads(result.stdout.strip().splitlines()[-1])


APPROVAL = ("User approved on 2026-10-24 (AskUserQuestion): 'Fresh test + mechanism layer' and MathMage-R alongside "
            "the frozen method, pre-registered; ledger cap raised to $151 with v6 lines. After o3-mini's retirement "
            "(HTTP 404 model_not_found), the user chose 'New team with GPT-5-mini': the sat26 team calibrated exactly "
            "as published, then the fresh test and mechanism layer on its fresh states.")


def prereg(package):
    """Stage 1 (--package lane): plan.json. Stage 2 (--package sat26): plan_sat26.json, after its calibration."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / PLAN_FILES[package]
    if path.exists():
        raise SystemExit(f"{path.name} exists; the v6 plan for {package} is frozen")
    data = package_state(package)
    plan = {"created_at_utc": now(), "mode": "live", "stage": 1 if package == "lane" else 2,
            "approval": APPROVAL,
            "method_frozen": ("MathMage-16 = v3 code and policy per roster (open: its v3 artifact; sat26: its own run "
                              "of the frozen offline phase); MathMage-R per analysis_plan (open roster)."),
            "frozen": {package: data["frozen"]}, "jobs": {package: schedule_of(data["jobs"])},
            "estimates": {package: data["estimate"]},
            "budget": {"soft_stop_usd": SOFT_STOP, "lines": LINES,
                       "spent_before_usd": BudgetLedger().summary()["spent_total"]},
            "analysis_plan": ANALYSIS_PLAN}
    if package == "lane":
        for other in PACKAGES:
            if other != "lane":                    # later stages: schedules fixed now, frozen state deferred
                plan["jobs"][other] = schedule_of(package_state(other, jobs_only=True)["jobs"])
    else:
        parent = read_plan_file(PLAN_FILES["lane"])
        if parent["analysis_plan"] != ANALYSIS_PLAN:
            raise SystemExit("the analysis plan differs from the frozen stage-1 plan")
        staged = {j["id"]: j["units_sha256"] for j in parent["jobs"][package]}
        if staged != {j["id"]: j["units_sha256"] for j in plan["jobs"][package]}:
            raise SystemExit(f"{package} schedule differs from the one frozen in stage 1")
        plan["parent_plan_sha256"] = parent["sha256"]
    plan["sha256"] = digest(plan)
    atomic_json(path, plan)
    write_prereg_markdown(plan, package)
    estimates = plan["estimates"][package]
    print(json.dumps({"file": path.name, "sha256": plan["sha256"], "estimates": estimates,
                      "total_usd": round(sum(j["expected_usd"] for j in estimates.values()), 2)}, indent=1))


def write_prereg_markdown(plan, package):
    """A readable copy of the plan file (the JSON file and its sha256 are authoritative)."""
    name = "PREREGISTRATION.md" if package == "lane" else f"PREREGISTRATION_{package}.md"
    lines = [f"# v6 pre-registration, stage {plan['stage']} (plan sha256 `{plan['sha256']}`)", "",
             f"Frozen {plan['created_at_utc']} before any v6 call of this stage's package ({package}). "
             f"{plan['approval']}", ""]
    if plan.get("parent_plan_sha256"):
        lines += [f"Extends stage 1 (plan.json, sha256 `{plan['parent_plan_sha256']}`), whose analysis plan and "
                  f"{package} schedule it reproduces.", ""]
    lines += [f"Method: {plan['method_frozen']}", "", "## Jobs, units and planning estimates", "",
              "| Package | Job | Wave | Method | Units | Expected USD |", "|---|---|---|---|---|---|"]
    for pkg, schedule in plan["jobs"].items():
        for job in schedule:
            est = plan["estimates"].get(pkg, {}).get(job["id"])
            cost = f"{est['expected_usd']:.2f}" if est else "stage 2"
            lines.append(f"| {pkg} | {job['id']} | {job['wave']} | {job['method']} | {job['n_units']} | {cost} |")
    lines += ["", "Ledger lines: " + ", ".join(f"{k}: {v}" for k, v in plan["budget"]["lines"].items()),
              f"Global soft stop: ${plan['budget']['soft_stop_usd']}; ledger spend before this stage: "
              f"${plan['budget']['spent_before_usd']:.2f}.", ""]
    for section, entries in plan["analysis_plan"].items():
        lines += [f"## {section.replace('_', ' ').capitalize()}", ""] + [f"- {e}" for e in entries] + [""]
    lines += ["## Frozen hashes", ""]
    for pkg, frozen in plan["frozen"].items():
        lines += [f"- {pkg}: " + ", ".join(f"{k} `{str(v)[:12]}`" for k, v in frozen.items()
                                           if k.endswith("sha256") or k.endswith("plan_id") or k == "mu_star")]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(package):
    plan = load_plan(package)
    balance = BudgetLedger().summary()
    done = {j["id"]: len(list((job_dir(package, j["id"]) / "graded").glob("*.json")))
            if (job_dir(package, j["id"]) / "graded").exists() else 0 for j in jobs(package)}
    lines = {kind: {"line": LINES[kind][package], "remaining": BudgetLedger().remaining(LINES[kind][package])}
             for kind in LINES if package in LINES[kind]}
    print(json.dumps({"plan": plan["sha256"], "package": package, "lines": lines, "spent_total": balance["spent_total"],
                      "pending": balance["pending_calls"], "unreconciled": balance["unreconciled_calls"],
                      "done": done, "units": {j["id"]: len(j["units"]) for j in jobs(package)}}, indent=1))


def mocktest(package, artifact=None):
    """One unit of every method on the mock backend in a temporary folder (no paid call, global files untouched).
    ``artifact`` runs the package on a mock calibration (sat26 before its live offline phase has finished)."""
    global OUT, ARTIFACT_LIVE
    tmp = Path(tempfile.mkdtemp(prefix="v6_mock_"))
    OUT = tmp
    if artifact:
        SPECS[package]["artifact"], ARTIFACT_LIVE = Path(artifact), False
    plan = {"mode": "mock", "frozen": {package: frozen_state(package)}}
    plan["jobs"] = {package: [{"id": j["id"], "units_sha256": digest([list(u) for u in j["units"]])} for j in jobs(package)]}
    plan["sha256"] = digest({k: v for k, v in plan.items()})
    atomic_json(tmp / PLAN_FILES[package], plan)

    def factory(line):
        return LLMClient(experiment=line, live=False, cache_path=tmp / "cache.sqlite", ledger_path=tmp / "ledger.jsonl",
                         log_directory=tmp / "calls")
    made = {}

    def clients(kind):
        return made.setdefault(kind, factory(LINES.get(kind, {}).get(package, "smoke")))
    report = {}
    for job in jobs(package):
        if not job["units"] or job["wave"] in ("D", "R"):
            continue
        unit = job["units"][0]
        if job["method"] == "mech" and str(unit[2]).startswith("fresh_"):   # a state this mock run has written
            unit = next((u for u in job["units"] if (job_dir(package, u[2]) / "agent"
                                                     / V.name_of(u[0], int(u[1]), "full")).exists()), unit)
        try:
            if job["method"] in ("mech", "dense"):
                record = recorded_record(package, unit[2], unit[0], int(unit[1]), plan)
                checkpoint = mock_checkpoint(record["checkpoint"])
                item = item_of(unit[0])
                if job["method"] == "mech":
                    payload = run_mech(clients("mech"), package, item, int(unit[1]), unit[2], checkpoint=checkpoint)
                    report[job["id"]] = {"arms": len(payload["arms"]),
                                         "kinds": Counter(a["arm"]["kind"] for a in payload["arms"]),
                                         "graded_arms": len(grade_mech(payload, item)["arms"])}
                else:
                    report[job["id"]] = "dense uses recorded live checkpoints; exercised through run_mech"
                continue
            process(clients, package, job, unit, plan)
            graded = sorted((job_dir(package, job["id"]) / "graded").glob("*.json"))
            row = json.loads(graded[0].read_text(encoding="utf-8")) if graded else {}
            report[job["id"]] = {"graded": [p.name for p in graded][:1], "fields": sorted(row)[:12],
                                 "new_cost_usd": row.get("new_cost_usd", row.get("cost_usd"))}
        except Exception as exc:  # noqa: BLE001 - report every failing path
            report[job["id"]] = f"FAILED: {type(exc).__name__}: {exc}"
    for client in made.values():
        client.close()
    print(json.dumps({"tmp": str(tmp), "report": report}, indent=1, default=str))


def mock_checkpoint(checkpoint):
    cp = copy.deepcopy(checkpoint)
    cp["mode"] = "mock"
    cp["sha256"] = TeamEngine._checkpoint_digest(cp)
    return cp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prereg", "state", "check", "worker", "mocktest"])
    parser.add_argument("--package", default="lane", choices=list(PACKAGES))
    parser.add_argument("--wave", default="F", choices=["F", "M", "D", "R", "Z"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--jobs-only", action="store_true", help="state: the schedule only (no frozen state)")
    parser.add_argument("--artifact", help="mocktest: a mock calibration artifact for the package")
    args = parser.parse_args()
    if args.command == "prereg":
        prereg(args.package)
    elif args.command == "state":
        state_command(args.package, jobs_only=args.jobs_only)
    elif args.command == "check":
        check(args.package)
    elif args.command == "mocktest":
        mocktest(args.package, artifact=args.artifact)
    else:
        if args.wave not in {j["wave"] for j in jobs(args.package)}:
            raise SystemExit(f"{args.package} has no wave {args.wave} in the v6 plan")
        with file_lock(OUT / f".worker_{args.package}_{args.wave}_{args.index}.lock"):
            worker(args.package, args.wave, args.index, args.count)


if __name__ == "__main__":
    main()
