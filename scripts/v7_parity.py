"""v7 parity follow-up (user-approved 2026-10-25): program-augmented baselines on the fresh set.

In the fourth wave only MathMage sees the coder's executed program, so a reader may ask how much of its lead is that
program. Here universal self-consistency, mixture-of-agents and one-round debate get the same program: the coder's
stage-0 program and its printed output, recorded in each unit's fresh-set MathMage-16 run
(runs/live_20261018/program_v6/<package>/fresh_mm), shown beside the members' solutions and labelled as a program.
When the program returned no answer, a baseline's inputs equal its v6 original's, and the original's sealed output is
reused (no call). Certificate checks, MathMage's other stage-0 evidence, are not shown.

Jobs per package (units: the v6 fresh-set units; ledger lines v7_parity_sat / v7_parity_open):
  fresh_usc_p     the frozen judge selects among the members' responses and the program; unparsed -> plurality
  fresh_moa_p     the frozen judge aggregates the members' proposals and the program; invalid -> plurality
  fresh_debate_p  one revision round in which every member also sees the program; the program then joins the final
                  plurality as a fourth participant that does not revise
(plurality: over the members' and the program's answers.) The v4/v5 templates are unchanged; only the solution lists
they render gain the labelled program.

The plan (program_v7/plan_v7.json) freezes this driver's hash, the v6 plan hashes, the units, the ledger lines and an
analysis plan that also fixes two simple stage-0 rules replayed on recorded answers (paper/v7_results.py), before any
call. Nothing under mathmage/, the lane code copies, the registries, the v6 plans or the v6 scripts changes; the v6
plan's frozen-state check runs before every worker starts.

    python scripts/v7_parity.py prereg                                # plan_v7.json (no calls)
    python scripts/v7_parity.py check --package sat26                 # verify plan, show progress (no calls)
    python scripts/v7_parity.py mocktest --package lane               # mock backend, temp folder (no calls)
    python scripts/v7_parity.py run --package lane --workers 12       # launch and wait for the workers
    python scripts/v7_parity.py worker --package sat26 --index 0 --count 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("lane", "sat26")
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "lane"
if PACKAGE not in PACKAGES:
    raise SystemExit("--package must be lane or sat26")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The grading guard binds the package's code copy first and runs the frozen exact_correct in a killable child (as
# scripts/v6_worker_guarded.py does for the v6 workers); the frozen v6 driver is then loaded as a library, never run.
GUARD = _load("v6_worker_guarded", REPO / "scripts" / "v6_worker_guarded.py")
GUARD.install()
V6 = _load("v6_followup", REPO / "scripts" / "v6_followup.py")
V, F5 = V6.V, V6.F5

import numpy as np  # noqa: E402

from mathmage.experiments.artifacts import digest, file_digest, read_record, seal_record  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402

atomic_json = V6.atomic_json
V7_LIVE = V.LIVE / "program_v7"
OUT = V7_LIVE                         # replaced by a temporary folder in mocktest
PLAN_FILE = "plan_v7.json"
SOFT_STOP = 149.5                     # global ledger guard under the $151 cap (as in v6)
LINES = {"sat26": "v7_parity_sat", "lane": "v7_parity_open"}
METHODS = ("usc_p", "moa_p", "debate_p")
BASE = {"usc_p": "fresh_usc", "moa_p": "fresh_moa", "debate_p": "fresh_debate"}
PROGRAM_LABEL = "a program written for this problem, with its printed output"
PROMPT_GROWTH = 1.15                  # planning only: the program's text lengthens each prompt


def now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- jobs
def jobs(package):
    units = V6.fresh_units(package)
    return [{"id": f"fresh_{m}", "method": m, "base": BASE[m], "source": "fresh_mm", "units": units} for m in METHODS]


def job_dir(package, job_id):
    return OUT / package / job_id


def job_plan_id(plan, package, job_id):
    return digest([plan["sha256"], package, job_id])


def program_of(record):
    """The coder's stage-0 program candidate when it returned an answer (the pool of the vote + program row)."""
    c = next((c for c in record["candidates"] if c["id"] == "s0-tool"), None)
    return c if c is not None and c.get("valid", True) and c.get("answer") is not None else None


def program_cost(record):
    return sum(float(c.get("resource_cost_usd") or 0) for c in record["meta"]["calls"]
               if c.get("phase") == "verify_forward")


# ----------------------------------------------------------------------------- one unit
def original_record(package, job, item_id, seed, plan6):
    """The sealed v6 record of the baseline this job augments."""
    path = V6.job_dir(package, job["base"]) / "agent" / V.name_of(item_id, seed, "full")
    return read_record(path, plan_id=V6.job_plan_id(plan6, package, job["base"]), mode=V6.plan_mode(plan6),
                       identity={"item_id": item_id, "seed": seed, "condition": "full"})


def run_unit(client, package, job, item, seed, plan6):
    engine = V6.engine_for(client, package, seed)
    requests = V.stage0_requests(engine, item)
    record = V6.source_record(package, {"source": "fresh_mm"}, item.item_id, seed, plan6)
    candidates = {c["id"]: c for c in record["candidates"]}
    members = list(engine.members)
    n = len(members)
    sols = [candidates.get(f"s0-{i}", {}).get("text") or "(no solution)" for i in range(n)]
    program = program_of(record)
    out = {"job": job["id"], "method": job["method"], "item_id": item.item_id, "seed": seed, "condition": "full",
           "members": members, "stage0_answers": [candidates.get(f"s0-{i}", {}).get("answer") for i in range(n)],
           "stage0_cost_usd": sum(V6.stage0_cost(record, i) for i in range(n)),
           "program_answer": program["answer"] if program else None, "program_cost_usd": program_cost(record)}
    if program is None:                       # inputs equal the v6 original's: reuse its sealed output
        original = original_record(package, job, item.item_id, seed, plan6)
        keep = {k: original[k] for k in ("judge", "choice", "parsed", "chosen_answer") if k in original}
        return {**out, **keep, "calls": [{**c, "cost_usd": 0.0, "reused_v6": True} for c in original["calls"]],
                "reused_from": job["base"]}
    text = program.get("text") or ""
    calls = []
    if job["method"] == "usc_p":
        judge = engine._judge_model()
        listed = "\n\n".join(f"Response {i}: {s}" for i, s in enumerate(sols)) + f"\n\nResponse {n} ({PROGRAM_LABEL}): {text}"
        prompt = V.USC.render(problem=engine.item.problem, responses=listed)
        request = V.variant(requests[0], "usc+program", messages=[{"role": "user", "content": prompt}],
                            template_id=V.USC.id, model_key=judge, role="selector")
        response = client.complete(request)
        calls.append(V.call_entry(request, response, item, k=0))
        match = re.search(r"most consistent response is Response\s*(\d+)", response.text or "", re.I)
        choice = int(match.group(1)) if match and int(match.group(1)) <= n else None
        chosen = None if choice is None else (program["answer"] if choice == n
                                              else candidates.get(f"s0-{choice}", {}).get("answer"))
        out.update({"judge": judge, "choice": choice, "parsed": choice is not None, "chosen_answer": chosen})
    elif job["method"] == "moa_p":
        judge = engine._judge_model()
        listed = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sols)) + f"\n{n + 1}. ({PROGRAM_LABEL}) {text}"
        messages = [{"role": "system", "content": F5.MOA.render(responses=listed)},
                    {"role": "user", "content": F5.MOA_QUERY.render(problem=engine.item.problem,
                                                                     format_instruction=engine.item.format_instruction)}]
        request = V.variant(requests[0], "moa+program", messages=messages, template_id=F5.MOA.id, model_key=judge,
                            role="aggregator")
        response = client.complete(request)
        calls.append(V.call_entry(request, response, item, k=0))
        out["judge"] = judge
    else:
        batch = []
        for i, request in enumerate(requests):
            others = "\n\n".join(f"One agent solution:\n```\n{sols[j]}\n```" for j in range(n) if j != i)
            others += f"\n\nOne more solution, {PROGRAM_LABEL}:\n{text}"
            messages = [request.messages[0], {"role": "assistant", "content": sols[i]},
                        {"role": "user", "content": V.DEBATE.render(others=others, problem=engine.item.problem,
                                                                    format_instruction=engine.item.format_instruction)}]
            batch.append(V.variant(request, "debate+program:r2", messages=messages, template_id=V.DEBATE.id,
                                   role=f"member={i}"))
        for i, (request, response) in enumerate(zip(batch, V.complete_all(client, batch))):
            calls.append(V.call_entry(request, response, item, k=i))
    return {**out, "calls": calls, "program_text_sha256": digest(text)}


def grade(payload, item):
    """v6 grading with the program in the fallback pool and, for debate, in the final plurality."""
    correct = V6.correct_of(item.item_id)
    members, program = payload["stage0_answers"], payload.get("program_answer")
    pool = members + ([program] if program else [])
    out = {"job": payload["job"], "method": payload["method"], "item_id": item.item_id, "benchmark": item.benchmark,
           "family_id": item.family_id, "split": item.split, "item_date": item.item_date, "seed": payload["seed"],
           "samples": [{"k": c["k"], "model_key": c["model_key"], "answer": c["answer"], "correct": correct(c["answer"]),
                        "resource_cost_usd": c["resource_cost_usd"], "reused_stage0": c.get("reused_stage0", False)}
                       for c in payload["calls"]],
           "new_cost_usd": sum(float(c["cost_usd"]) for c in payload["calls"]),
           "resource_cost_usd": (sum(float(c["resource_cost_usd"]) for c in payload["calls"])
                                 + payload["stage0_cost_usd"] + payload["program_cost_usd"]),
           "program_answer": program, "program_correct": bool(program) and correct(program),
           "reused_from": payload.get("reused_from"), "stage0_correct": [correct(a) for a in members]}
    if payload["method"] == "usc_p":
        chosen = payload.get("chosen_answer") if payload.get("parsed") else V6.majority(pool, item.answer_format)
        out.update({"usc_parsed": bool(payload.get("parsed")), "usc_choice": payload.get("choice"),
                    "usc_answer": chosen, "usc_correct": correct(chosen)})
    elif payload["method"] == "moa_p":
        call = payload["calls"][0]
        chosen = call["answer"] or V6.majority(pool, item.answer_format)
        out.update({"moa_valid": bool(call["answer"]), "moa_answer": chosen, "moa_correct": correct(chosen),
                    "moa_fallback": not call["answer"], "judge": payload.get("judge")})
    elif program:
        out["samples"].append({"k": len(members), "model_key": "program", "answer": program, "correct": correct(program),
                               "resource_cost_usd": 0.0, "reused_stage0": True, "program_voter": True})
    return out


# ----------------------------------------------------------------------------- plan
def frozen_state(package):
    return {"driver_sha256": file_digest(Path(__file__)),
            "v6_runner_sha256": file_digest(REPO / "scripts" / "v6_followup.py"),
            "guard_sha256": file_digest(REPO / "scripts" / "v6_worker_guarded.py"),
            "grading_child_sha256": file_digest(REPO / "scripts" / "v6_grading_child.py"),
            "v6_plan_sha256": {name: V6.read_plan_file(name)["sha256"] for name in sorted(set(V6.PLAN_FILES.values()))},
            "templates": {t.id: t.text for t in (V.DEBATE, V.USC, F5.MOA, F5.MOA_QUERY)},
            "program_label": PROGRAM_LABEL, "line": LINES[package]}


def estimate(package):
    """Planning only: the v6 originals' mean new spend per unit, times the share of units whose program answered."""
    plan6 = V6.load_plan(package)
    present = 0
    units = V6.fresh_units(package)
    for item_id, seed in units:
        present += program_of(V6.source_record(package, {"source": "fresh_mm"}, item_id, seed, plan6)) is not None
    out = {}
    for job in jobs(package):
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in (V6.job_dir(package, job["base"]) / "graded").glob("*.json")]
        per = float(np.mean([r["new_cost_usd"] for r in rows])) * PROMPT_GROWTH
        out[job["id"]] = {"units": len(units), "units_with_program": present, "per_call_unit_usd": round(per, 5),
                          "expected_usd": round(per * present, 3)}
    return out


def schedule_of(job_list):
    return [{**{k: v for k, v in j.items() if k != "units"}, "n_units": len(j["units"]),
             "units": [list(u) for u in j["units"]], "units_sha256": digest([list(u) for u in j["units"]])}
            for j in job_list]


ANALYSIS_PLAN = {
    "primary": [
        "P1 (sat26 team: GPT-5-mini low, Claude Haiku 4.5, DeepSeek-V3; 187 fresh problems, one seed) and P2 (open "
        "roster: three gpt-oss-120b; 187 fresh problems x 8 seeds): MathMage-16 at its frozen mu* vs USC + program, "
        "MoA + program and debate + program; Holm across each team's three contrasts.",
        "Each contrast: paired macro difference over the five fresh contest cells (benchmarks weighted equally), "
        "family-clustered bootstrap 95% CI (2,000 draws; crossed with seeds for the open roster), two-sided bootstrap "
        "p (the v6 estimator, paper/v4_results.contrast)."],
    "secondary": [
        "Q1: MathMage-16's maximal run vs each program-augmented baseline (same estimator; descriptive).",
        "Q2: each program-augmented baseline vs its v6 original on the same units, i.e. the value of showing the "
        "program (same estimator; descriptive).",
        "Q3 (no calls): two simple stage-0 rules replayed on the recorded stage-0 answers of every evaluation set of "
        "the paper's main table (E1, the SAT-style team's 2025-26 contests, the sat26 fresh set; E2, the open "
        "roster's 2025-26 contests, the open fresh set), each compared with MathMage's stage-0 posterior pick and "
        "with MathMage-16 at mu* (paired, same estimator; descriptive): (a) program first: the program's answer "
        "whenever it returned one, otherwise the members' majority vote (the frozen vote, ties in roster order); "
        "(b) accuracy-weighted vote: plurality over the members' and the program's answers, each weighted by its "
        "author's Laplace-smoothed accuracy (k + 1) / (n + 2) over the answers it gave on its team's 16 training "
        "problems (the stage-0 panels of the frozen offline phase; the open roster's three members share one pooled "
        "weight), ties broken in pool order (members in roster order, then the program). Both rules cost the "
        "members' stage-0 calls plus the program's."],
    "rules": [
        "Units are the v6 fresh-set units (sat26: 187 problems x seed 0; open: 187 x seeds 0-7). Sample sizes are "
        "fixed by this plan; no optional stopping.",
        "The program shown to a baseline is the coder's stage-0 program and its printed output from the unit's "
        "fresh-set MathMage-16 record, labelled as a program; certificate checks are not shown.",
        "When the program returned no answer, the program-augmented baseline's inputs equal the v6 original's and "
        "its sealed output is reused (no call).",
        "USC + program: the frozen judge selects among the members' responses and the program; an unparsed selection "
        "falls back to the plurality of the members' and the program's answers (ties in pool order). MoA + program: "
        "the frozen judge aggregates the members' proposals and the program; an invalid aggregate falls back "
        "likewise. Debate + program: every member sees the other members' solutions and the program and revises "
        "once; the program joins the final plurality (ties uniform at random, the v4 rule) as a fourth participant "
        "that does not revise.",
        "Cost per unit = the members' stage-0 calls + the program's call + the baseline's new calls (resource USD).",
        "Every job and contrast of this plan is reported whatever its outcome."]}

APPROVAL = ("User request on 2026-10-25, after a costed proposal (universal self-consistency, mixture-of-agents and "
            "debate with the program's answer on both fresh sets, about $10-15, a new budget line and a frozen plan): "
            "'work on addressing the \"Baselines don't get the program.\" concern' and 'replay two simple rules on "
            "the recorded answers'.")


def read_plan():
    plan = json.loads((OUT / PLAN_FILE).read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v7 plan hash changed")
    return plan


def load_plan(package):
    """The frozen v7 plan, verified against this package's current state and schedule (and the v6 plan's)."""
    plan = read_plan()
    V6.load_plan(package)
    expected, current = plan["frozen"][package], frozen_state(package)
    for key, value in expected.items():
        if current[key] != value:
            raise ValueError(f"{key} differs from the frozen v7 plan; the run must not proceed")
    scheduled = {j["id"]: j["units_sha256"] for j in plan["jobs"][package]}
    for job in jobs(package):
        if scheduled.get(job["id"]) != digest([list(u) for u in job["units"]]):
            raise ValueError(f"units of {job['id']} differ from the frozen v7 plan")
    return plan


def package_state(package):
    """The package's frozen state, schedule and estimate, computed in its own process (each binds its code copy)."""
    command = [sys.executable, str(Path(__file__)), "state", "--package", package]
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return json.loads(result.stdout.strip().splitlines()[-1])


def prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / PLAN_FILE
    if path.exists():
        raise SystemExit(f"{path.name} exists; the v7 plan is frozen")
    states = {p: package_state(p) for p in PACKAGES}
    plan = {"created_at_utc": now(), "mode": "live", "approval": APPROVAL,
            "frozen": {p: s["frozen"] for p, s in states.items()}, "jobs": {p: schedule_of(s["jobs"]) for p, s in states.items()},
            "estimates": {p: s["estimate"] for p, s in states.items()},
            "budget": {"soft_stop_usd": SOFT_STOP, "lines": LINES,
                       "spent_before_usd": BudgetLedger().summary()["spent_total"]},
            "analysis_plan": ANALYSIS_PLAN}
    plan["sha256"] = digest(plan)
    atomic_json(path, plan)
    write_prereg_markdown(plan)
    total = sum(j["expected_usd"] for s in plan["estimates"].values() for j in s.values())
    print(json.dumps({"file": path.name, "sha256": plan["sha256"], "estimates": plan["estimates"],
                      "total_usd": round(total, 2)}, indent=1))


def write_prereg_markdown(plan):
    lines = [f"# v7 parity plan (sha256 `{plan['sha256']}`)", "",
             f"Frozen {plan['created_at_utc']} before any v7 call. {plan['approval']}", "",
             "## Jobs, units and planning estimates", "",
             "| Package | Job | Units | Units with a program answer | Expected USD |", "|---|---|---|---|---|"]
    for package, schedule in plan["jobs"].items():
        for job in schedule:
            est = plan["estimates"][package][job["id"]]
            lines.append(f"| {package} | {job['id']} | {job['n_units']} | {est['units_with_program']} | "
                         f"{est['expected_usd']:.2f} |")
    lines += ["", "Ledger lines: " + ", ".join(f"{k}: {v}" for k, v in plan["budget"]["lines"].items()),
              f"Global soft stop: ${plan['budget']['soft_stop_usd']}; ledger spend before: "
              f"${plan['budget']['spent_before_usd']:.2f}.", ""]
    for section, entries in plan["analysis_plan"].items():
        lines += [f"## {section.capitalize()}", ""] + [f"- {e}" for e in entries] + [""]
    lines += ["## Frozen hashes", ""]
    for package, frozen in plan["frozen"].items():
        lines += [f"- {package}: " + ", ".join(f"{k} `{str(v)[:12]}`" for k, v in frozen.items() if k.endswith("sha256")
                                               and not isinstance(v, dict))]
        lines += [f"  v6 plans: " + ", ".join(f"{k} `{v[:12]}`" for k, v in frozen["v6_plan_sha256"].items())]
    (OUT / "PLAN_v7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- guard and workers
class Guard:
    """Stop on STOP, any provider fault, unreconciled billing or the global soft stop (as in v6)."""

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
                    raise BudgetExceeded(f"v7 stopped: {why}")
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
                raise BudgetExceeded("v7 provider health gate; see provider attention file")
            return response
        LLMClient.complete = guarded
        return original


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


def process(client, package, job, unit, plan6, plan7):
    item_id, seed = unit[0], int(unit[1])
    target = job_dir(package, job["id"])
    out = target / "agent" / V.name_of(item_id, seed, "full")
    if out.exists():
        return False
    item = V6.item_of(item_id)
    payload = run_unit(client, package, job, item, seed, plan6)
    stopped = [c.get("error") for c in payload["calls"] if (c.get("error") or "").startswith("BudgetExceeded")]
    if stopped:                                     # never keep a unit with a call the guard refused
        raise BudgetExceeded(stopped[0])
    graded = grade(payload, item)
    mode = "live" if client.live else "mock"
    atomic_json(out, seal_record(payload, plan_id=job_plan_id(plan7, package, job["id"]), mode=mode))
    atomic_json(target / "graded" / out.name, graded)
    return True


def worker(package, index, count):
    plan7 = load_plan(package)
    plan6 = V6.load_plan(package)
    guard = Guard(package)
    state = {"package": package, "index": index, "of": count, "pid": os.getpid(), "done": 0, "status": "running",
             "started_at_utc": now()}
    status = OUT / "workers" / f"{package}_{index}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    client = LLMClient(experiment=LINES[package], live=True)
    original = guard.install()
    transport, original_transport = install_diagnostics()
    try:
        for job in jobs(package):
            for unit in job["units"]:
                item_id, seed = unit[0], int(unit[1])
                if V.partition(item_id, seed, count) != index:
                    continue
                why = guard.reason()
                if why:
                    state["status"] = f"stopped: {why}"
                    return
                try:
                    done = process(client, package, job, unit, plan6, plan7)
                except BudgetExceeded as exc:
                    state["status"] = f"stopped: {exc}"
                    return
                state["done"] += bool(done)
                state["current"] = f"{job['id']} {item_id} s{seed}"
                state["updated_at_utc"] = now()
                atomic_json(status, state)
        state["status"] = "complete"
    finally:
        state["updated_at_utc"] = now()
        atomic_json(status, state)
        LLMClient.complete = original
        transport.complete = original_transport
        client.close()


def remaining(package, plan):
    left = 0
    for job in plan["jobs"][package]:
        directory = OUT / package / job["id"] / "graded"
        left += job["n_units"] - (len(list(directory.glob("*.json"))) if directory.exists() else 0)
    return left


def run(package, count):
    """Launch the package's workers (no console windows), wait for them and write a done file when complete."""
    plan = load_plan(package)
    (OUT / "workers").mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = []
    for index in range(count):
        log = open(OUT / "workers" / f"{package}_{index}.log", "a", encoding="utf-8")
        processes.append(subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "worker", "--package", package, "--index", str(index),
             "--count", str(count)], cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=flags))
    while any(p.poll() is None for p in processes):
        time.sleep(15)
    left = remaining(package, plan)
    result = {"time_utc": now(), "package": package, "plan": plan["sha256"], "remaining_units": left,
              "exit_codes": [p.returncode for p in processes]}
    if left == 0:
        atomic_json(OUT / f"done_{package}.json", result)
    print(json.dumps(result))


def check(package):
    plan = load_plan(package)
    balance = BudgetLedger().summary()
    done = {j["id"]: len(list((job_dir(package, j["id"]) / "graded").glob("*.json")))
            if (job_dir(package, j["id"]) / "graded").exists() else 0 for j in jobs(package)}
    print(json.dumps({"plan": plan["sha256"], "package": package, "line": LINES[package],
                      "remaining_usd": BudgetLedger().remaining(LINES[package]), "spent_total": balance["spent_total"],
                      "pending": balance["pending_calls"], "unreconciled": balance["unreconciled_calls"],
                      "done": done, "units": {j["id"]: len(j["units"]) for j in jobs(package)}}, indent=1))


def mocktest(package):
    """Every job on the mock backend, for a unit whose program answered and one whose did not (temp folder)."""
    global OUT
    tmp = Path(tempfile.mkdtemp(prefix="v7_mock_"))
    OUT = tmp
    plan6 = V6.load_plan(package)
    plan7 = {"mode": "mock", "sha256": digest(["v7-mock", package])}
    client = LLMClient(experiment="smoke", live=False, cache_path=tmp / "cache.sqlite", ledger_path=tmp / "ledger.jsonl",
                       log_directory=tmp / "calls")
    with_program = without = None
    for item_id, seed in V6.fresh_units(package):
        has = program_of(V6.source_record(package, {"source": "fresh_mm"}, item_id, seed, plan6)) is not None
        if has and with_program is None:
            with_program = (item_id, seed)
        if not has and without is None:
            without = (item_id, seed)
        if with_program and without:
            break
    report = {}
    for job in jobs(package):
        for label, unit in (("program", with_program), ("no program", without)):
            try:
                process(client, package, job, unit, plan6, plan7)
                row = json.loads((job_dir(package, job["id"]) / "graded" / V.name_of(unit[0], unit[1], "full"))
                                 .read_text(encoding="utf-8"))
                report[f"{job['id']} ({label})"] = {k: row.get(k) for k in (
                    "program_answer", "reused_from", "new_cost_usd", "resource_cost_usd", "usc_parsed", "usc_correct",
                    "moa_valid", "moa_correct")} | {"samples": [(s["model_key"], s["answer"]) for s in row["samples"]]}
            except Exception as exc:  # noqa: BLE001 - report every failing path
                report[f"{job['id']} ({label})"] = f"FAILED: {type(exc).__name__}: {exc}"
    client.close()
    print(json.dumps({"tmp": str(tmp), "units": {"program": with_program, "no program": without}, "report": report},
                     indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prereg", "state", "check", "worker", "run", "mocktest"])
    parser.add_argument("--package", default="lane", choices=list(PACKAGES))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "prereg":
            prereg()
        elif args.command == "state":
            print(json.dumps({"frozen": frozen_state(args.package), "jobs": jobs(args.package),
                              "estimate": estimate(args.package)}))
        elif args.command == "check":
            check(args.package)
        elif args.command == "mocktest":
            mocktest(args.package)
        elif args.command == "run":
            run(args.package, args.workers)
        else:
            with file_lock(OUT / f".worker_{args.package}_{args.index}.lock"):
                worker(args.package, args.index, args.count)
    finally:
        GUARD.CHILD.shutdown()
        GUARD.PARSE_CHILD.shutdown()


if __name__ == "__main__":
    main()
