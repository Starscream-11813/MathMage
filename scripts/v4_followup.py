"""v4 follow-up (pre-registered): same-roster baselines and post-cutoff contests.

The MathMage method is frozen: every MathMage run uses the v3 code, offline artifacts
and policy of its roster, validated exactly as ``run_study`` validates them. Nothing
under ``mathmage/``, ``configs/``, or the frozen registry and vault changes. New contests
live in ``data/registry_ext/`` (see scripts/v4_data.py). Baseline prompts live here.

Jobs (pre-registered priority order; ``plan.json`` freezes them before any call):
  P1  e2_debate, e2_usc       open roster: one debate round / universal self-consistency
  P2  e1_sc                   SAT roster: DeepSeek-V3 self-consistency, K = 10 (matched cost)
  P3  e1_debate               SAT roster: one debate round over the saved stage-0 solutions
  P4  e1_usc                  SAT roster: universal self-consistency by the frozen judge
  P5  HMMT Nov 2025           both rosters: MathMage, then SC / debate / USC
  P6  April 2025 contests     both rosters: MathMage (open roster adds AIME 2026, HMMT Feb 2026)
  P7  e1_sat_sc               SAT roster: SAT's o3-mini SC (K = 10) on AIME 2026 + HMMT Feb 2026,
                              only if the e1_sat line can absorb 1.35 x its estimate after P1-P6

Self-consistency reuses each problem's saved stage-0 sample as its first sample: the
stage-0 request is rebuilt with the frozen engine and must be an exact cache hit
(identical prompt, sample tag and seed), otherwise the unit stops without a call.

    python scripts/v4_followup.py prereg                 # freeze plan.json (no calls)
    python scripts/v4_followup.py check --package main   # verify and estimate (no calls)
    python scripts/v4_followup.py run --package main --workers 4
    python scripts/v4_followup.py run --package lane --workers 6
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "lanes" / "open_weight"
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "main"
if PACKAGE not in ("main", "lane"):
    raise SystemExit("--package must be main or lane")
sys.path.insert(0, str(LANE if PACKAGE == "lane" else REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mathmage import paths  # noqa: E402
from mathmage.data.registry import Registry, load_registry  # noqa: E402
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, read_frozen, read_record, seal_record  # noqa: E402
from mathmage.experiments.runner import (_protocol_digest, grade_result, prompt_hashes, tools_runtime,  # noqa: E402
                                         validate_artifact_runtime)
from mathmage.grading.equiv import cluster, normalize  # noqa: E402
from mathmage.grading.vault import AnswerVault  # noqa: E402
from mathmage.grading.verify import grade_answer  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402
from mathmage.llm.types import LLMRequest, cache_key, resolve  # noqa: E402
from mathmage.prompts.templates import SOLVE, Template  # noqa: E402
from mathmage.team import TeamConfig, TeamEngine  # noqa: E402
from mathmage.team.engine import extract_answer  # noqa: E402

LIVE = paths.RUNS / "live_20261018"
V3, OPEN, V4 = LIVE / "program_v3", LIVE / "program_open", LIVE / "program_v4"
EXT = paths.DATA / "registry_ext"
SOFT_STOP = 73.0
POLICY_KEYS = ("sandbox", "router", "mu", "max_policy_actions", "max_item_cost_usd", "voc_epsilon", "hint_audit",
               "max_calls")
K_SC = 10
GATE_FACTOR = 1.35
APRIL = ("brumo_2025", "cmimc_2025", "smt_2025")

# Baseline prompts. Debate follows Du et al. (2023): each agent sees its own round-1
# solution in the conversation and the other agents' solutions in a new turn. USC
# follows Chen et al. (2023). Both keep MathMage's answer convention (\boxed{...}).
DEBATE = Template("debate_round2.v1", (
    "These are the solutions to the problem from other agents:\n\n{others}\n\n"
    "Using the reasoning from other agents as additional information, can you provide your answer to the problem? "
    "Examine your solution and the other agents' solutions step by step. Reason step by step, then give the final "
    "answer as \\boxed{{...}}.\n{format_instruction}\n\nProblem:\n{problem}\n"))
USC = Template("usc.v1", (
    "I have generated the following responses to the question: {problem}\n\n{responses}\n\n"
    "Evaluate these responses.\nSelect the most consistent response based on majority consensus.\n"
    "Start your answer with \"The most consistent response is Response X\" (without quotes)."))

ROSTERS = {"main": {"roster": "sat", "root": V3, "phase": "e1_primary", "artifact": V3 / "offline_sat" / "artifacts.json",
                    "line": "v4_sat"},
           "lane": {"roster": "open", "root": OPEN, "phase": "e2", "artifact": OPEN / "offline_open" / "artifacts.json",
                    "line": "v4_open"}}


def now():
    return datetime.now(timezone.utc).isoformat()


def name_of(item_id, seed, condition):
    return digest([item_id, seed, condition])[:24] + ".json"


# ----------------------------------------------------------------------------- data
def ext_registry():
    return Registry.from_frame(pd.read_parquet(EXT / "items.parquet"))


def ext_vault():
    return AnswerVault(pd.read_parquet(EXT / "answers.parquet"))


def item_of(item_id):
    base = load_registry()
    return base[item_id] if item_id in base else ext_registry()[item_id]


def vault_for(item_id):
    return None if item_id in load_registry() else ext_vault()


def saved_units(package):
    """(item_id, seed) of the frozen primary study, from its graded rows."""
    spec = ROSTERS[package]
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in (spec["root"] / spec["phase"] / "graded").glob("*.json")]
    return sorted({(r["item_id"], int(r.get("seed", 0))) for r in rows})


def jobs(package):
    """The pre-registered job list for one package, in priority order."""
    ext = ext_registry()
    nov = sorted(i.item_id for i in ext.all() if i.benchmark == "hmmt_nov_2025")
    april = sorted(i.item_id for i in ext.all() if i.benchmark in APRIL)
    if package == "main":
        primary = saved_units("main")
        fresh_63 = [(i, s) for i, s in primary if i.split("/")[0] in ("aime_2026", "hmmt_feb_2026")]
        return [
            {"id": "e1_sc", "priority": 2, "method": "sc", "model": "deepseek-v3", "source": "primary", "units": primary},
            {"id": "e1_debate", "priority": 3, "method": "debate", "source": "primary", "units": primary},
            {"id": "e1_usc", "priority": 4, "method": "usc", "source": "primary", "units": primary},
            {"id": "new_e1_mathmage_nov", "priority": 5, "method": "mathmage", "units": [(i, 0) for i in nov]},
            {"id": "new_e1_sc_nov", "priority": 5, "method": "sc", "model": "deepseek-v3", "source": "new_e1_mathmage_nov",
             "units": [(i, 0) for i in nov]},
            {"id": "new_e1_debate_nov", "priority": 5, "method": "debate", "source": "new_e1_mathmage_nov",
             "units": [(i, 0) for i in nov]},
            {"id": "new_e1_usc_nov", "priority": 5, "method": "usc", "source": "new_e1_mathmage_nov",
             "units": [(i, 0) for i in nov]},
            {"id": "new_e1_mathmage_apr", "priority": 6, "method": "mathmage", "units": [(i, 0) for i in april]},
            {"id": "e1_sat_sc", "priority": 7, "method": "sc", "model": "o3-mini-low", "source": "primary",
             "units": fresh_63, "gated": True},
        ]
    seeds = list(range(8))
    base = load_registry()
    fresh = sorted(i.item_id for i in base.all() if i.benchmark in ("aime_2026", "hmmt_feb_2026"))
    primary = saved_units("lane")
    return [
        {"id": "e2_debate", "priority": 1, "method": "debate", "source": "primary", "units": primary},
        {"id": "e2_usc", "priority": 1, "method": "usc", "source": "primary", "units": primary},
        {"id": "new_e2_mathmage_nov", "priority": 5, "method": "mathmage", "units": [(i, s) for i in nov for s in seeds]},
        {"id": "new_e2_debate_nov", "priority": 5, "method": "debate", "source": "new_e2_mathmage_nov",
         "units": [(i, s) for i in nov for s in seeds]},
        {"id": "new_e2_usc_nov", "priority": 5, "method": "usc", "source": "new_e2_mathmage_nov",
         "units": [(i, s) for i in nov for s in seeds]},
        {"id": "new_e2_mathmage_apr", "priority": 6, "method": "mathmage",
         "units": [(i, s) for i in april + fresh for s in seeds]},
    ]


# ----------------------------------------------------------------------------- plan
def frozen_state(package):
    spec = ROSTERS[package]
    manifest = json.loads((spec["root"] / spec["phase"] / "manifest.json").read_text(encoding="utf-8"))
    return {"protocol_sha256": _protocol_digest(), "prompt_hashes": prompt_hashes(),
            "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
            "budget_sha256": file_digest(paths.CONFIGS / "budget.yaml"),
            "registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
            "vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
            "ext_items_sha256": file_digest(EXT / "items.parquet"), "ext_answers_sha256": file_digest(EXT / "answers.parquet"),
            "artifact_sha256": file_digest(spec["artifact"]), "primary_plan_id": manifest["plan_id"],
            "policy": {k: manifest["policy"][k] for k in POLICY_KEYS},
            "baseline_templates": {t.id: t.text for t in (DEBATE, USC)},
            "runner_sha256": file_digest(Path(__file__))}


def job_plan_id(plan, package, job_id):
    return digest([plan["sha256"], package, job_id])


def load_plan(package):
    plan = json.loads((V4 / "plan.json").read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v4 plan hash changed")
    expected, current = plan["frozen"][package], frozen_state(package)
    for key, value in expected.items():
        if current[key] != value:
            raise ValueError(f"{key} differs from the pre-registered v4 plan; the run must not proceed")
    return plan


# ----------------------------------------------------------------------------- guard
class Guard:
    def __init__(self, package):
        self.package, self.ledger = package, BudgetLedger()
        self.attention = V4 / f"provider_attention_{package}.json"
        self.lock = threading.RLock()

    def reason(self):
        if (V4 / "STOP").exists() or (V3 / "STOP").exists():
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
                    raise BudgetExceeded(f"v4 stopped: {why}")
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
                raise BudgetExceeded("v4 provider health gate; see provider attention file")
            return response
        LLMClient.complete = guarded
        return original


# ----------------------------------------------------------------------------- methods
def engine_for(client, package, seed):
    spec = ROSTERS[package]
    artifacts = read_frozen(spec["artifact"], live=True)
    validate_artifact_runtime(artifacts, spec["roster"], tools_runtime(["full"], "local"))
    manifest = json.loads((spec["root"] / spec["phase"] / "manifest.json").read_text(encoding="utf-8"))
    config = TeamConfig(roster=spec["roster"], condition="full", seed=seed, offline_artifacts=artifacts,
                        **{k: manifest["policy"][k] for k in POLICY_KEYS})
    return TeamEngine(client, config)


def stage0_requests(engine, item):
    """The frozen engine's stage-0 solver requests, rebuilt exactly (same prompt, tag and seed)."""
    engine._start(item)
    cards = engine._cards()
    return [engine._request(model, SOLVE, "stage0", f"member={i}", problem=engine.item.problem,
                            format_instruction=engine.item.format_instruction, cards=cards)
            for i, model in enumerate(engine.members)]


def cached(client, request):
    return client.cache.get(cache_key(request, resolve(request, client.models.model(request.model_key))))


def complete_all(client, batch):
    """Concurrent completion that never keeps a partial unit.

    ``LLMClient.complete_many`` turns a budget or guard stop inside the batch into an error
    response; re-raise it so the unit is not written with missing samples.
    """
    responses = client.complete_many(batch)
    stopped = [r.error for r in responses if (r.error or "").startswith("BudgetExceeded")]
    if stopped:
        raise BudgetExceeded(stopped[0])
    return responses


def variant(request, tag, *, messages=None, template_id=None, model_key=None, role=None):
    seed = int(digest([request.sample_tag, tag])[:8], 16) % (2 ** 31)
    meta = {**request.meta, "phase": tag.split(":")[0], "role": role or tag, "baseline": tag}
    return replace(request, sample_tag=f"{request.sample_tag}|{tag}", seed=seed, meta=meta,
                   messages=messages if messages is not None else request.messages,
                   template_id=template_id or request.template_id, model_key=model_key or request.model_key)


def call_entry(request, response, item, k=None, reused=False):
    valid = response.ok and not response.truncated
    return {"k": k, "model_key": request.model_key, "sample_tag": request.sample_tag, "template_id": request.template_id,
            "text": response.text, "answer": normalize(extract_answer(response.text), item.answer_format) if valid else None,
            "valid": valid, "error": response.error or ("truncated" if response.truncated else None),
            "cached": bool(response.cached), "reused_stage0": reused,
            "cost_usd": 0.0 if response.cached else float(response.cost_usd or 0),
            "resource_cost_usd": float(response.resource_cost_usd or 0), "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out, "tokens_reasoning": response.tokens_reasoning,
            "latency_s": response.latency_s}


def stage0_record(job, package, item_id, seed, plan):
    """Saved stage-0 member solutions of the frozen MathMage run this baseline builds on."""
    if job["source"] == "primary":
        spec = ROSTERS[package]
        manifest = json.loads((spec["root"] / spec["phase"] / "manifest.json").read_text(encoding="utf-8"))
        path = spec["root"] / spec["phase"] / "agent" / name_of(item_id, seed, "full")
        return read_record(path, plan_id=manifest["plan_id"], mode=manifest["mode"],
                           identity={"item_id": item_id, "seed": seed, "condition": "full"})
    path = V4 / job["source"] / "agent" / name_of(item_id, seed, "full")
    return read_record(path, plan_id=job_plan_id(plan, package, job["source"]), mode="live",
                       identity={"item_id": item_id, "seed": seed, "condition": "full"})


def stage0_cost(record, member):
    return sum(float(c.get("resource_cost_usd") or 0) for c in record["meta"]["calls"]
               if c["phase"] == "stage0" and c["role"] == f"member={member}")


def run_baseline(client, package, job, item, seed, plan):
    engine = engine_for(client, package, seed)
    requests = stage0_requests(engine, item)
    record = stage0_record(job, package, item.item_id, seed, plan)
    texts = {c["id"]: c.get("text") or "" for c in record["candidates"]}
    members = list(engine.members)
    calls, extra = [], {}
    if job["method"] == "sc":
        member = members.index(job["model"])
        first = cached(client, requests[member])
        if first is None:
            raise RuntimeError(f"stage-0 request for {item.item_id} is not an exact cache hit; refusing to sample")
        calls.append(call_entry(requests[member], first, item, k=0, reused=True))
        calls[-1]["resource_cost_usd"] = stage0_cost(record, member)
        batch = [variant(requests[member], f"sc:k={k}", role=f"sample={k}") for k in range(1, K_SC)]
        for k, (request, response) in enumerate(zip(batch, complete_all(client, batch)), 1):
            calls.append(call_entry(request, response, item, k=k))
    elif job["method"] == "debate":
        batch = []
        for i, request in enumerate(requests):
            others = "\n\n".join(f"One agent solution:\n```\n{texts.get(f's0-{j}') or '(no solution)'}\n```"
                                 for j in range(len(members)) if j != i)
            messages = [request.messages[0], {"role": "assistant", "content": texts.get(f"s0-{i}") or "(no solution)"},
                        {"role": "user", "content": DEBATE.render(others=others, problem=engine.item.problem,
                                                                  format_instruction=engine.item.format_instruction)}]
            batch.append(variant(request, "debate:r2", messages=messages, template_id=DEBATE.id, role=f"member={i}"))
        for i, (request, response) in enumerate(zip(batch, complete_all(client, batch))):
            calls.append(call_entry(request, response, item, k=i))
        extra["stage0_answers"] = [c.get("answer") for c in record["candidates"] if c["id"] in
                                   [f"s0-{i}" for i in range(len(members))]]
        extra["stage0_cost_usd"] = sum(stage0_cost(record, i) for i in range(len(members)))
    elif job["method"] == "usc":
        judge = engine._judge_model()
        responses = "\n\n".join(f"Response {i}: {texts.get(f's0-{i}') or '(no solution)'}" for i in range(len(members)))
        prompt = USC.render(problem=engine.item.problem, responses=responses)
        request = variant(requests[0], "usc", messages=[{"role": "user", "content": prompt}], template_id=USC.id,
                          model_key=judge, role="selector")
        response = client.complete(request)
        calls.append(call_entry(request, response, item, k=0))
        match = re.search(r"most consistent response is Response\s*(\d+)", response.text or "", re.I)
        choice = int(match.group(1)) if match and int(match.group(1)) < len(members) else None
        candidates = {c["id"]: c for c in record["candidates"]}
        extra.update({"judge": judge, "choice": choice, "parsed": choice is not None,
                      "chosen_answer": candidates.get(f"s0-{choice}", {}).get("answer") if choice is not None else None,
                      "stage0_answers": [candidates.get(f"s0-{i}", {}).get("answer") for i in range(len(members))],
                      "stage0_cost_usd": sum(stage0_cost(record, i) for i in range(len(members)))})
    return {"job": job["id"], "method": job["method"], "item_id": item.item_id, "seed": seed, "condition": "full",
            "members": members, "calls": calls, **extra}


def grade_baseline(payload, item):
    vault = vault_for(item.item_id)
    correct = lambda a: bool(a) and grade_answer(item.item_id, a, vault).correct  # noqa: E731
    out = {"job": payload["job"], "method": payload["method"], "item_id": item.item_id, "benchmark": item.benchmark,
           "family_id": item.family_id, "split": item.split, "item_date": item.item_date, "seed": payload["seed"],
           "samples": [{"k": c["k"], "model_key": c["model_key"], "answer": c["answer"], "correct": correct(c["answer"]),
                        "resource_cost_usd": c["resource_cost_usd"], "reused_stage0": c["reused_stage0"]}
                       for c in payload["calls"]],
           "new_cost_usd": sum(c["cost_usd"] for c in payload["calls"]),
           "resource_cost_usd": sum(c["resource_cost_usd"] for c in payload["calls"]) + payload.get("stage0_cost_usd", 0.0)}
    if payload["method"] == "usc":
        answers = payload["stage0_answers"]
        fallback = None
        if not payload["parsed"]:
            usable = [a for a in answers if a]
            if usable:
                labels = cluster(usable, item.answer_format)
                fallback = usable[max(range(len(usable)), key=lambda j: (labels.count(labels[j]), -j))]
        chosen = payload["chosen_answer"] if payload["parsed"] else fallback
        out.update({"usc_parsed": payload["parsed"], "usc_choice": payload["choice"], "usc_answer": chosen,
                    "usc_correct": correct(chosen), "stage0_correct": [correct(a) for a in answers]})
    if payload["method"] == "debate":
        out["stage0_correct"] = [correct(a) for a in payload["stage0_answers"]]
    return out


def run_mathmage(client, package, item, seed, plan, job):
    engine = engine_for(client, package, seed)
    payload = asdict(engine.run(item, checkpoint=None))
    sealed = seal_record(payload, plan_id=job_plan_id(plan, package, job["id"]), mode="live")
    return sealed, grade_result(payload, item, mode="live", vault=vault_for(item.item_id))


# ----------------------------------------------------------------------------- workers
def partition(item_id, seed, count):
    return int(digest([item_id, seed])[:8], 16) % count


def process(client, package, job, item_id, seed, plan):
    target = V4 / job["id"]
    out = target / "agent" / name_of(item_id, seed, "full")
    if out.exists():
        return False
    item = item_of(item_id)
    if job["method"] == "mathmage":
        sealed, graded = run_mathmage(client, package, item, seed, plan, job)
        atomic_json(out, sealed)
        atomic_json(target / "graded" / out.name, graded)
    else:
        payload = run_baseline(client, package, job, item, seed, plan)
        atomic_json(out, seal_record(payload, plan_id=job_plan_id(plan, package, job["id"]), mode="live"))
        atomic_json(target / "graded" / out.name, grade_baseline(payload, item))
    return True


def worker(package, index, count, wave):
    plan = load_plan(package)
    guard = Guard(package)
    state = {"package": package, "index": index, "of": count, "wave": wave, "pid": os.getpid(), "done": 0,
             "status": "running", "started_at_utc": now()}
    status = V4 / "workers" / f"{package}_{index}.json"
    original = guard.install()
    client = LLMClient(experiment=ROSTERS[package]["line"], live=True)
    try:
        for job in jobs(package):
            if bool(job.get("gated")) != (wave == 2):
                continue
            for item_id, seed in job["units"]:
                if partition(item_id, seed, count) != index:
                    continue
                why = guard.reason()
                if why:
                    state["status"] = f"stopped: {why}"
                    return
                try:
                    process(client, package, job, item_id, seed, plan)
                except BudgetExceeded as exc:
                    state["status"] = f"stopped: {exc}"
                    return
                state["done"] += 1
                state["current"] = f"{job['id']} {item_id} s{seed}"
                state["updated_at_utc"] = now()
                atomic_json(status, state)
        state["status"] = "complete"
    finally:
        state["updated_at_utc"] = now()
        atomic_json(status, state)
        LLMClient.complete = original
        client.close()


# ----------------------------------------------------------------------------- estimates
def unit_costs(package):
    """Measured mean resource cost per call and per MathMage evaluation, by model and problem class."""
    spec = ROSTERS[package]
    calls, totals = {}, {}
    for path in (spec["root"] / spec["phase"] / "graded").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        hard = row["benchmark"] not in ("tqa_phys", "math500")
        totals.setdefault(hard, []).append(float(row["cost_usd"]))
        record = json.loads((spec["root"] / spec["phase"] / "agent" / path.name).read_text(encoding="utf-8"))
        for c in record["meta"]["calls"]:
            if c["phase"] == "stage0":
                calls.setdefault((c["model_key"], hard), []).append(float(c.get("resource_cost_usd") or 0))
    return {k: float(np.mean(v)) for k, v in calls.items()}, {k: float(np.mean(v)) for k, v in totals.items()}


def estimate(package):
    per_call, per_eval = unit_costs(package)
    members = {"main": ["o3-mini-low", "haiku-4.5", "deepseek-v3"], "lane": ["gpt-oss-120b-low"] * 3}[package]
    judge = {"main": "haiku-4.5", "lane": "gpt-oss-120b-low"}[package]
    out = {}
    for job in jobs(package):
        total = 0.0
        for item_id, _ in job["units"]:
            hard = not item_id.startswith(("tqa_phys", "math500"))
            if job["method"] == "mathmage":
                total += per_eval[hard]
            elif job["method"] == "sc":
                total += (K_SC - 1) * per_call[(job["model"], hard)]
            elif job["method"] == "debate":
                total += 1.6 * sum(per_call[(m, hard)] for m in members)        # longer inputs than stage 0
            elif job["method"] == "usc":
                total += 1.2 * per_call[(judge, hard)]
        out[job["id"]] = {"units": len(job["units"]), "expected_usd": round(total, 3), "priority": job["priority"],
                          "gated": bool(job.get("gated"))}
    return out


# ----------------------------------------------------------------------------- commands
def prereg():
    V4.mkdir(parents=True, exist_ok=True)
    if (V4 / "plan.json").exists():
        raise SystemExit("plan.json exists; the v4 plan is frozen")
    frozen, schedule, estimates = {}, {}, {}
    for package in ("main", "lane"):
        code = ("import json,sys;sys.argv=['x','--package','%s'];"
                "import importlib.util as u;s=u.spec_from_file_location('v4',r'%s');m=u.module_from_spec(s);"
                "s.loader.exec_module(m);print(json.dumps({'frozen':m.frozen_state('%s'),'jobs':[{k:v for k,v in j.items()} "
                "for j in m.jobs('%s')],'estimate':m.estimate('%s')}))" % (package, __file__, package, package, package))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        data = json.loads(result.stdout.strip().splitlines()[-1])
        frozen[package], estimates[package] = data["frozen"], data["estimate"]
        schedule[package] = [{**j, "units": [list(u) for u in j["units"]], "units_sha256": digest(j["units"])}
                             for j in data["jobs"]]
    plan = {
        "created_at_utc": now(), "approval": "User approved Tiers 1 + 2 on 2026-10-19 and kept the $75 cap.",
        "method_frozen": "MathMage v3 code, offline artifacts and policy per roster; no test-set tuning.",
        "frozen": frozen, "jobs": schedule, "estimates": estimates,
        "budget": {"soft_stop_usd": SOFT_STOP, "lines": {"main": "v4_sat", "lane": "v4_open"},
                   "gate_P7": f"run only if v4_sat remaining >= {GATE_FACTOR} x estimate after P1-P6"},
        "analysis_plan": {
            "primary": [
                "H1 (SAT roster, 237 problems): MathMage at dev mu* vs DeepSeek-V3 SC@10, vs debate, vs USC "
                "(paired macro difference, family-clustered bootstrap 95% CI, Holm across H1-H3).",
                "H2 (open roster, 580 evaluations): MathMage at dev mu* and maximal vs debate and USC; vs SC@k at matched "
                "cost from pooled seeds.",
                "H3 (contamination-free): SAT roster on AIME 2026 + HMMT Feb 2026 + HMMT Nov 2025; open roster on AIME "
                "2025 + AIME 2026 + HMMT Feb 2026 + the four new contests: MathMage vs majority vote and vs the "
                "baselines available on that set."],
            "secondary": [
                "H4: MathMage vs vote, weighted vote and single members on the 153 new problems, both rosters.",
                "H5 (if P7 runs): our o3-mini SC@10 vs SAT's reported SC (AIME 2026 53.4, HMMT Feb 2026 31.9)."],
            "rules": ["Sample sizes are fixed by this plan; no optional stopping or added data after results.",
                      "All jobs are reported, whatever their outcome; stopped jobs are reported as incomplete.",
                      "Matched cost uses mean resource USD per problem, counting reused stage-0 samples."]},
    }
    plan["sha256"] = digest(plan)
    atomic_json(V4 / "plan.json", plan)
    print(json.dumps({"sha256": plan["sha256"], "estimates": estimates}, indent=1))


def check(package):
    plan = load_plan(package)
    balance = BudgetLedger().summary()
    line = ROSTERS[package]["line"]
    est = estimate(package)
    done = {j["id"]: len(list((V4 / j["id"] / "graded").glob("*.json"))) if (V4 / j["id"] / "graded").exists() else 0
            for j in jobs(package)}
    print(json.dumps({"plan": plan["sha256"], "package": package, "line": line,
                      "line_remaining": BudgetLedger().remaining(line), "spent_total": balance["spent_total"],
                      "pending": balance["pending_calls"], "estimates": est, "done": done}, indent=1))


def run(package, workers):
    plan = load_plan(package)
    balance = BudgetLedger().summary()
    if balance["pending_calls"] or balance["unreconciled_calls"]:
        raise SystemExit("reconcile outstanding reservations or billing first")
    (V4 / "workers").mkdir(parents=True, exist_ok=True)
    for wave in (1, 2):
        if wave == 2:
            gated = [j for j in jobs(package) if j.get("gated")]
            if not gated:
                break
            need = sum(estimate(package)[j["id"]]["expected_usd"] for j in gated) * GATE_FACTOR
            remaining = BudgetLedger().remaining(ROSTERS[package]["line"])
            atomic_json(V4 / f"gate_{package}.json", {"time_utc": now(), "need_usd": need, "line_remaining_usd": remaining,
                                                      "run": remaining >= need})
            if remaining < need or Guard(package).reason():
                break
        processes = []
        for index in range(workers):
            log = open(V4 / "workers" / f"{package}_{index}.log", "a", encoding="utf-8")
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "worker", "--package", package,
                                               "--index", str(index), "--count", str(workers), "--wave", str(wave)],
                                              cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            time.sleep(10)
        if Guard(package).reason():
            break
    atomic_json(V4 / f"done_{package}.json", {"time_utc": now(), "plan": plan["sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prereg", "check", "run", "worker"])
    parser.add_argument("--package", default="main", choices=["main", "lane"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--wave", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prereg":
        prereg()
    elif args.command == "check":
        check(args.package)
    elif args.command == "worker":
        worker(args.package, args.index, args.count, args.wave)
    else:
        with file_lock(V4 / f".run_{args.package}.lock"):
            run(args.package, args.workers)


if __name__ == "__main__":
    main()
