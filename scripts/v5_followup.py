"""v5 follow-up (pre-registered 2026-10-20): complete the same-roster baseline matrix.

v4 ran debate, USC and SC on only one of the four new contests (HMMT Nov 2025) and had no
mixture-of-agents baseline. v5 runs debate and USC on every remaining post-cutoff problem,
SAT-roster SC on the April contests, and a same-roster mixture-of-agents (MoA) baseline on
every evaluation set. v5 was decided after the v4 results were known. Its confirmatory tests
therefore use only v5 data; pooled v4 + v5 estimates are secondary (see plan.json).

MathMage is not re-run. Every baseline builds on the saved stage-0 solutions of a frozen
MathMage run (the v3 primary studies or the v4 new-contest jobs), exactly as in v4. The
frozen v4 runner is imported unchanged (its pinned hash is verified against the v4 plan) and
supplies the SC, debate and USC code. v5 adds only the MoA aggregator, its job list, its
guard and its budget lines.

Jobs (pre-registered priority order):
  P1  open roster  debate, USC     BRUMO, CMIMC, SMT 2025, AIME 2026, HMMT Feb 2026 (186 problems x 8 seeds)
  P2  SAT roster   debate, USC     BRUMO, CMIMC, SMT 2025 (123 problems)
  P3  open roster  MoA             E2 (580), HMMT Nov 2025 (240), the P1 units (1,488)
  P4  SAT roster   MoA             April contests (123), HMMT Nov 2025 (30), E1 (237)
  P5  SAT roster   SC@10           DeepSeek-V3 on the April contests (123), matched cost

    python scripts/v5_followup.py prereg                  # freeze plan.json (no calls)
    python scripts/v5_followup.py check --package main    # verify and estimate (no calls)
    python scripts/v5_followup.py worker --package lane --index 0 --count 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "main"
if PACKAGE not in ("main", "lane"):
    raise SystemExit("--package must be main or lane")

_argv = sys.argv
sys.argv = [sys.argv[0], "--package", PACKAGE]             # the v4 runner selects its package at import
_spec = importlib.util.spec_from_file_location("v4_followup", REPO / "scripts" / "v4_followup.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
sys.argv = _argv

import numpy as np  # noqa: E402

from mathmage.experiments.artifacts import digest, file_digest, seal_record  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger  # noqa: E402
from mathmage.llm.locking import file_lock  # noqa: E402
from mathmage.prompts.templates import Template  # noqa: E402


def retrying(write, attempts=40, pause_s=0.25):
    """Windows-safe atomic write: retry while a reader holds the target (v4 amendment 3)."""
    def wrapped(path, value):
        for attempt in range(attempts):
            try:
                return write(path, value)
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(pause_s)
    return wrapped


V.atomic_json = retrying(V.atomic_json)
atomic_json = V.atomic_json

V5 = V.LIVE / "program_v5"
SOFT_STOP = 73.0
LINES = {"main": "v5_sat", "lane": "v5_open"}

# Mixture-of-agents (Wang et al., 2024): the aggregate-and-synthesize system prompt over the
# proposals, followed by the user query. Two layers: the saved stage-0 solutions are the
# proposer layer and the frozen judge is the aggregator. MathMage's answer convention is kept.
MOA = Template("moa_aggregate.v1", (
    "You have been provided with a set of responses from various models to the latest user query. Your task is to "
    "synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the "
    "information provided in these responses, recognizing that some of it may be biased or incorrect. Your response "
    "should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to "
    "the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of "
    "accuracy and reliability.\n\nResponses from models:\n{responses}"))
MOA_QUERY = Template("moa_query.v1", (
    "{problem}\n\nReason step by step, then give the final answer as \\boxed{{...}}.\n{format_instruction}"))


def now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- jobs
def jobs(package):
    """The pre-registered v5 job list for one package, in priority order."""
    ext = V.ext_registry()
    nov = sorted(i.item_id for i in ext.all() if i.benchmark == "hmmt_nov_2025")
    april = sorted(i.item_id for i in ext.all() if i.benchmark in V.APRIL)
    if package == "main":
        apr = [(i, 0) for i in april]
        return [
            {"id": "apr_e1_debate", "priority": 2, "method": "debate", "source": "new_e1_mathmage_apr", "units": apr},
            {"id": "apr_e1_usc", "priority": 2, "method": "usc", "source": "new_e1_mathmage_apr", "units": apr},
            {"id": "apr_e1_moa", "priority": 4, "method": "moa", "source": "new_e1_mathmage_apr", "units": apr},
            {"id": "nov_e1_moa", "priority": 4, "method": "moa", "source": "new_e1_mathmage_nov",
             "units": [(i, 0) for i in nov]},
            {"id": "e1_moa", "priority": 4, "method": "moa", "source": "primary", "units": V.saved_units("main")},
            {"id": "apr_e1_sc", "priority": 5, "method": "sc", "model": "deepseek-v3", "source": "new_e1_mathmage_apr",
             "units": apr},
        ]
    seeds = list(range(8))
    fresh = sorted(i.item_id for i in V.load_registry().all() if i.benchmark in ("aime_2026", "hmmt_feb_2026"))
    later = [(i, s) for i in april + fresh for s in seeds]          # the units of v4's new_e2_mathmage_apr
    return [
        {"id": "apr_e2_debate", "priority": 1, "method": "debate", "source": "new_e2_mathmage_apr", "units": later},
        {"id": "apr_e2_usc", "priority": 1, "method": "usc", "source": "new_e2_mathmage_apr", "units": later},
        {"id": "e2_moa", "priority": 3, "method": "moa", "source": "primary", "units": V.saved_units("lane")},
        {"id": "nov_e2_moa", "priority": 3, "method": "moa", "source": "new_e2_mathmage_nov",
         "units": [(i, s) for i in nov for s in seeds]},
        {"id": "apr_e2_moa", "priority": 3, "method": "moa", "source": "new_e2_mathmage_apr", "units": later},
    ]


# ----------------------------------------------------------------------------- plan
def v4_plan():
    plan = json.loads((V.V4 / "plan.json").read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v4 plan hash changed")
    return plan


def frozen_state(package):
    state = V.frozen_state(package)
    state["v4_runner_sha256"] = state.pop("runner_sha256")
    state.update({"v4_plan_sha256": v4_plan()["sha256"], "runner_sha256": file_digest(Path(__file__)),
                  "moa_templates": {t.id: t.text for t in (MOA, MOA_QUERY)}})
    return state


def job_plan_id(plan, package, job_id):
    return digest([plan["sha256"], package, job_id])


def load_plan(package):
    plan = json.loads((V5 / "plan.json").read_text(encoding="utf-8"))
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v5 plan hash changed")
    four = v4_plan()
    if four["frozen"][package]["runner_sha256"] != file_digest(REPO / "scripts" / "v4_followup.py"):
        raise ValueError("the v4 runner differs from its pre-registered hash; the run must not proceed")
    expected, current = plan["frozen"][package], frozen_state(package)
    for key, value in expected.items():
        if current[key] != value:
            raise ValueError(f"{key} differs from the pre-registered v5 plan; the run must not proceed")
    return plan, four


# ----------------------------------------------------------------------------- guard
class Guard:
    """The v4 guard with v5 paths: stop on STOP, any provider fault, unreconciled billing or the soft stop."""

    def __init__(self, package):
        self.package, self.ledger = package, BudgetLedger()
        self.attention = V5 / f"provider_attention_{package}.json"
        self.lock = threading.RLock()

    def reason(self):
        if (V5 / "STOP").exists():
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
                    raise BudgetExceeded(f"v5 stopped: {why}")
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
                raise BudgetExceeded("v5 provider health gate; see provider attention file")
            return response
        LLMClient.complete = guarded
        return original


# ----------------------------------------------------------------------------- MoA
def run_moa(client, package, job, item, seed, four):
    engine = V.engine_for(client, package, seed)
    requests = V.stage0_requests(engine, item)
    record = V.stage0_record(job, package, item.item_id, seed, four)
    texts = {c["id"]: c.get("text") or "" for c in record["candidates"]}
    members = list(engine.members)
    judge = engine._judge_model()
    proposals = "\n".join(f"{i + 1}. {texts.get(f's0-{i}') or '(no solution)'}" for i in range(len(members)))
    messages = [{"role": "system", "content": MOA.render(responses=proposals)},
                {"role": "user", "content": MOA_QUERY.render(problem=engine.item.problem,
                                                              format_instruction=engine.item.format_instruction)}]
    request = V.variant(requests[0], "moa", messages=messages, template_id=MOA.id, model_key=judge, role="aggregator")
    response = client.complete(request)
    candidates = {c["id"]: c for c in record["candidates"]}
    return {"job": job["id"], "method": "moa", "item_id": item.item_id, "seed": seed, "condition": "full",
            "members": members, "judge": judge, "calls": [V.call_entry(request, response, item, k=0)],
            "stage0_answers": [candidates.get(f"s0-{i}", {}).get("answer") for i in range(len(members))],
            "stage0_cost_usd": sum(V.stage0_cost(record, i) for i in range(len(members)))}


def grade_moa(payload, item):
    """MoA's answer is the aggregator's; with no extractable answer it falls back to the stage-0 vote (as USC does)."""
    out = V.grade_baseline({**payload, "method": "moa_samples"}, item)
    out["method"] = "moa"
    vault = V.vault_for(item.item_id)
    correct = lambda a: bool(a) and V.grade_answer(item.item_id, a, vault).correct  # noqa: E731
    call = payload["calls"][0]
    answers = payload["stage0_answers"]
    fallback = None
    if not call["answer"]:
        usable = [a for a in answers if a]
        if usable:
            labels = V.cluster(usable, item.answer_format)
            fallback = usable[max(range(len(usable)), key=lambda j: (labels.count(labels[j]), -j))]
    chosen = call["answer"] or fallback
    out.update({"moa_valid": bool(call["answer"]), "moa_answer": chosen, "moa_correct": correct(chosen),
                "moa_fallback": not call["answer"], "judge": payload["judge"],
                "stage0_correct": [correct(a) for a in answers]})
    return out


# ----------------------------------------------------------------------------- workers
def process(client, package, job, item_id, seed, plan, four):
    target = V5 / job["id"]
    out = target / "agent" / V.name_of(item_id, seed, "full")
    if out.exists():
        return False
    item = V.item_of(item_id)
    if job["method"] == "moa":
        payload = run_moa(client, package, job, item, seed, four)
        graded = grade_moa(payload, item)
    else:
        payload = V.run_baseline(client, package, job, item, seed, four)
        graded = V.grade_baseline(payload, item)
    stopped = [c["error"] for c in payload["calls"] if (c.get("error") or "").startswith("BudgetExceeded")]
    if stopped:                                     # never keep a unit with a call the guard refused
        raise BudgetExceeded(stopped[0])
    atomic_json(out, seal_record(payload, plan_id=job_plan_id(plan, package, job["id"]), mode="live"))
    atomic_json(target / "graded" / out.name, graded)
    return True


def worker(package, index, count):
    plan, four = load_plan(package)
    guard = Guard(package)
    state = {"package": package, "index": index, "of": count, "pid": os.getpid(), "done": 0,
             "status": "running", "started_at_utc": now()}
    status = V5 / "workers" / f"{package}_{index}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    original = guard.install()
    client = LLMClient(experiment=LINES[package], live=True)
    try:
        for job in jobs(package):
            for item_id, seed in job["units"]:
                if V.partition(item_id, seed, count) != index:
                    continue
                why = guard.reason()
                if why:
                    state["status"] = f"stopped: {why}"
                    return
                try:
                    process(client, package, job, item_id, seed, plan, four)
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
def measured_rate(job_id):
    """Mean new spend per unit of a finished v4 job."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in (V.V4 / job_id / "graded").glob("*.json")]
    return float(np.mean([r.get("new_cost_usd", r.get("cost_usd", 0.0)) for r in rows]))


def judge_output_usd(package, source):
    """Expected aggregator output cost: the judge's own mean stage-0 output on the source problems."""
    spec = V.ROSTERS[package]
    folder = (spec["root"] / spec["phase"] if source == "primary" else V.V4 / source) / "agent"
    judge = {"main": "haiku-4.5", "lane": "gpt-oss-120b-low"}[package]
    price = {"main": 5.00, "lane": 0.17}[package]
    tokens = []
    for path in list(folder.glob("*.json"))[:300]:
        record = json.loads(path.read_text(encoding="utf-8"))
        tokens += [c.get("tokens_out") or 0 for c in record["meta"]["calls"]
                   if c["phase"] == "stage0" and c["model_key"] == judge]
    return float(np.mean(tokens)) * price / 1e6


def estimate(package):
    """Pre-call cost estimates from measured v4 rates (hard contests use the HMMT Nov 2025 jobs)."""
    e = "e1" if package == "main" else "e2"
    rate = {("debate", "hard"): measured_rate(f"new_{e}_debate_nov"), ("usc", "hard"): measured_rate(f"new_{e}_usc_nov"),
            ("usc", "primary"): measured_rate(f"{e}_usc")}
    if package == "main":
        rate[("sc", "hard")] = measured_rate("new_e1_sc_nov")
    out = {}
    for job in jobs(package):
        kind = "primary" if job["source"] == "primary" else "hard"
        if job["method"] == "moa":
            per = rate[("usc", kind)] + judge_output_usd(package, job["source"])
        else:
            per = rate[(job["method"], "hard")]
        out[job["id"]] = {"units": len(job["units"]), "expected_usd": round(per * len(job["units"]), 3),
                          "per_unit_usd": round(per, 5), "priority": job["priority"]}
    return out


# ----------------------------------------------------------------------------- commands
ANALYSIS_PLAN = {
    "primary": [
        "H6 (SAT roster; 123 April-2025 contest problems: BRUMO, CMIMC, SMT): MathMage at dev mu* (its sealed v4 "
        "run) vs debate, vs USC, vs MoA, vs DeepSeek-V3 SC@10 (matched cost, as in v4).",
        "H7 (open roster; 1,488 evaluations: BRUMO, CMIMC, SMT 2025, AIME 2026, HMMT Feb 2026 x 8 seeds): "
        "mu* vs debate, vs USC, vs MoA.",
        "H8 (MoA on the original suites): SAT roster E1 (237 problems) and open roster E2 (580 evaluations): mu* vs MoA.",
        "Each contrast: paired macro difference (benchmarks weighted equally), family-clustered bootstrap 95% CI "
        "(crossed with seeds for the open roster), two-sided bootstrap p; Holm across the nine primary contrasts."],
    "secondary": [
        "S1 (complete contamination-free sets, v4 + v5 pooled): SAT roster 216 problems (AIME 2026, HMMT Feb 2026, "
        "HMMT Nov 2025, BRUMO, CMIMC, SMT 2025); open roster 1,968 evaluations (those plus AIME 2025): mu* vs debate, "
        "USC, MoA and SC. v5 was decided after v4's debate contrasts were seen, so the pooled debate contrasts are "
        "judged against a two-look Pocock boundary (nominal two-sided alpha 0.0294); the others are descriptive.",
        "S2: the anytime frontier's maximal point vs each baseline on the H6-H8 sets.",
        "S3: cost per correct answer and Pareto dominance on every set.",
        "S4: same-roster MoA on E1 vs the MoA accuracy SAT reports on its suite (57.3); descriptive only.",
        "S5 (non-inferiority to debate): mu* minus debate pooled over every roster-benchmark cell where debate ran "
        "(v4 + v5; families resampled jointly across rosters); MathMage is declared non-inferior if the lower bound "
        "of the two-sided 95% interval exceeds -3 pp (one-sided alpha 0.025), reported with the cost ratio."],
    "disclosed_before_freeze": [
        "Exploratory offline analyses already computed from v3/v4 data on 2026-10-20 and therefore not "
        "pre-registered here: adaptive-compute baselines (Adaptive-Consistency, ESC), a mixture-of-thought cascade, "
        "stopping rules on identical trajectories, and a pooled v4 debate contrast (+2.0 [-1.9, +5.9] pp). The "
        "pooled v4 estimate motivated S5's margin."],
    "rules": [
        "Sample sizes are fixed by this plan; no optional stopping or added data after results.",
        "All jobs are reported, whatever their outcome; stopped jobs are reported as incomplete.",
        "MathMage is not re-run: its sealed v3/v4 results on the same units are the comparator.",
        "Debate, USC and SC use the v4 templates and code unchanged. MoA uses the aggregate-and-synthesize prompt of "
        "Wang et al. (2024) with the frozen judge as aggregator; if the aggregator gives no extractable answer, MoA "
        "falls back to the stage-0 majority answer, as USC does on a parse failure.",
        "Matched cost uses mean resource USD per problem, counting reused stage-0 samples.",
        "Timing disclosure: this plan was frozen on 2026-10-20, after the v4 analysis. MathMage's accuracy on these "
        "units was known; no baseline outcome on them was."]}


def state_command(package):
    print(json.dumps({"frozen": frozen_state(package), "jobs": jobs(package), "estimate": estimate(package)}))


def prereg():
    V5.mkdir(parents=True, exist_ok=True)
    if (V5 / "plan.json").exists():
        raise SystemExit("plan.json exists; the v5 plan is frozen")
    frozen, schedule, estimates = {}, {}, {}
    for package in ("main", "lane"):
        result = subprocess.run([sys.executable, str(Path(__file__)), "state", "--package", package], cwd=str(REPO),
                                capture_output=True, text=True, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        data = json.loads(result.stdout.strip().splitlines()[-1])
        frozen[package], estimates[package] = data["frozen"], data["estimate"]
        schedule[package] = [{**j, "units": [list(u) for u in j["units"]], "units_sha256": digest(j["units"])}
                             for j in data["jobs"]]
    plan = {
        "created_at_utc": now(),
        "approval": "User request of 2026-10-20 to raise the paper to ARR 4 with background experiments; $75 cap kept.",
        "method_frozen": "MathMage v3 code, offline artifacts and policy per roster; MathMage itself is not re-run.",
        "frozen": frozen, "jobs": schedule, "estimates": estimates,
        "budget": {"soft_stop_usd": SOFT_STOP, "lines": LINES, "spent_before_usd": BudgetLedger().summary()["spent_total"]},
        "analysis_plan": ANALYSIS_PLAN,
    }
    plan["sha256"] = digest(plan)
    atomic_json(V5 / "plan.json", plan)
    print(json.dumps({"sha256": plan["sha256"], "estimates": estimates,
                      "total_usd": round(sum(j["expected_usd"] for p in estimates.values() for j in p.values()), 2)},
                     indent=1))


def check(package):
    plan, _ = load_plan(package)
    balance = BudgetLedger().summary()
    done = {j["id"]: len(list((V5 / j["id"] / "graded").glob("*.json"))) if (V5 / j["id"] / "graded").exists() else 0
            for j in jobs(package)}
    print(json.dumps({"plan": plan["sha256"], "package": package, "line": LINES[package],
                      "line_remaining": BudgetLedger().remaining(LINES[package]), "spent_total": balance["spent_total"],
                      "pending": balance["pending_calls"], "estimates": plan["estimates"][package], "done": done}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prereg", "state", "check", "worker"])
    parser.add_argument("--package", default="main", choices=["main", "lane"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prereg":
        prereg()
    elif args.command == "state":
        state_command(args.package)
    elif args.command == "check":
        check(args.package)
    else:
        with file_lock(V5 / f".worker_{args.package}_{args.index}.lock"):
            worker(args.package, args.index, args.count)


if __name__ == "__main__":
    main()
