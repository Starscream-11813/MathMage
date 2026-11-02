"""Training-only learning: posterior calibration, tutoring, discernment, cards, operating point.

Everything learned here uses training items (AIME 2023, HMMT February 2025); the
development split is a held-out check of calibration and selects the pre-registered
operating point. Run with no flags for deterministic mock validation. All partial
observations are checkpointed; an interrupted run never produces a frozen artifact.
Synthetic observations may validate code but cannot initialize a live evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mathmage import paths
from mathmage.config import load_models
from mathmage.data.registry import load_registry
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, freeze, seal_record, read_record
from mathmage.data.splits import stratified_sample
from mathmage.llm.locking import file_lock
from mathmage.llm.types import LLMRequest
from mathmage.prompts.templates import CARD, HINT, HINT_KINDS, JUDGE, REVIEW, SOLVE, SOLVE_REVISE, SOLVE_SCAFFOLD, TEMPLATES

# Public MathArena configurations used only as linked anchor agents on training items.
MATHARENA_ALIASES = {"sat": {"o3-mini (low)": "o3-mini-low@matharena", "DeepSeek-V3": "deepseek-v3@matharena"},
                     "open": {"GPT OSS 120B (high)": "gpt-oss-120b@matharena"},
                     "sat26": {"GPT-5-mini (high)": "gpt-5-mini@matharena", "DeepSeek-V3": "deepseek-v3@matharena"}}


def _delta_interval(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "lift": None, "low": None, "high": None}
    # Each input is already a per-problem mean. The offline sampling split is
    # family-disjoint; aggregate any repeated same-family observations upstream.
    arr = np.asarray(values)
    rng = np.random.default_rng(0)
    boot = rng.choice(arr, (2000, len(arr))).mean(axis=1)
    return {"n": len(arr), "lift": float(arr.mean()),
            "low": float(np.quantile(boot, .025)) if len(arr) > 1 else None,
            "high": float(np.quantile(boot, .975)) if len(arr) > 1 else None}


def _sample_split(registry, split: str, limit: int):
    """Fixed benchmark-stratified sampling before observing any correctness labels."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("positive integer training and development limits required")
    items = registry.select(split=split)
    ids = stratified_sample(items, min(limit, len(items)), key=lambda item: item.benchmark,
                            salt=f"offline-{split}")
    return [registry[iid] for iid in ids]


def _family_means(rows: list[dict], value) -> list[float]:
    groups = {}
    for row in rows:
        groups.setdefault(row["family_id"], []).append(float(value(row)))
    return [float(np.mean(groups[family])) for family in sorted(groups)]


def _merge_item_records(records):
    """Keep all annotators of one item on a single shared difficulty draw.

    Public anchors and newly collected responses are distinct draws, but they
    describe the same item. Treating them as separate items would sever that
    link and count the item twice in calibration and scoring.
    """
    from mathmage.psychometrics.calibrate import ItemRecord

    merged = {}
    for record in records:
        previous = merged.get(record.item_id)
        if previous is None:
            merged[record.item_id] = ItemRecord(record.item_id, record.answer_format,
                                                list(record.responses), record.family_id)
        else:
            if (previous.answer_format, previous.family_id) != (record.answer_format, record.family_id):
                raise ValueError("conflicting metadata for calibration item " + record.item_id)
            previous.responses.extend(record.responses)
    return [merged[key] for key in sorted(merged)]


def _strategy_scores(validation: list[dict], zone: str, strategies) -> list[dict]:
    scores = []
    for strategy in strategies:
        rows = [row for row in validation if row["zone"] == zone and row["strategy"] == strategy]
        if not rows:
            continue
        utility = _family_means(rows, lambda row: int(row["final_correct"]) - int(row["initial_correct"])
                                 - int(row["initial_correct"] and not row["final_correct"]))
        costs = _family_means(rows, lambda row: row["cost_usd"])
        scores.append({"strategy": strategy, "n": len(rows), "n_families": len(utility),
                       "utility": float(np.mean(utility)), "cost_usd": float(np.mean(costs))})
    return sorted(scores, key=lambda row: (-row["utility"], row["cost_usd"], strategies.index(row["strategy"])))


def _public_log_records(roster: str, train_ids: set[str]) -> list:
    """Free MathArena attempts on training items, as linked anchor agents.

    They sharpen the difficulty scale and wrong-answer coincidence estimates; they never
    set our agents' own abilities, which come only from our prompts and settings.
    """
    from mathmage.data.loaders import matharena_outputs
    from mathmage.grading.equiv import normalize
    from mathmage.grading.extract import extract_final
    from mathmage.grading.verify import grade_answer
    from mathmage.psychometrics.calibrate import ItemRecord, Response

    aliases = MATHARENA_ALIASES.get(roster, {})
    if not aliases:
        return []
    registry = load_registry()
    records: dict[str, list] = {}
    for competition in ("hmmt_feb_2025",):
        try:
            df = matharena_outputs(competition, columns=["problem_idx", "model_name", "answer"])
        except Exception:                                   # offline or unavailable: no anchors
            return []
        for row in df[df.model_name.isin(list(aliases))].to_dict("records"):
            item_id = f"{competition}/{int(row['problem_idx']):02d}"
            if item_id not in train_ids:
                continue
            raw = extract_final(row["answer"] or "", lenient=False)
            answer = normalize(raw, registry[item_id].answer_format) if raw else None
            correct = bool(answer is not None and grade_answer(item_id, answer).correct)
            records.setdefault(item_id, []).append(Response(aliases[row["model_name"]], answer, correct))
    return [ItemRecord(i, registry[i].answer_format, r, registry[i].family_id) for i, r in sorted(records.items())]


def _public_kappa_panels(roster: str, train_ids: set[str], team_size: int = 4) -> list:
    """Disjoint, team-sized panels of public first attempts on training items (collision odds only).

    Each public configuration's first attempt on a training item lands in exactly one
    panel (a seeded per-item shuffle), so no response is counted twice and every panel
    has the size of a deployed proposal pool. Configurations already linked as ability
    anchors are excluded. Development and test items are never read.
    """
    from mathmage.data.loaders import matharena_outputs
    from mathmage.grading.equiv import normalize
    from mathmage.grading.extract import extract_final
    from mathmage.grading.verify import grade_answer
    from mathmage.psychometrics.calibrate import ItemRecord, Response

    excluded = set(MATHARENA_ALIASES.get(roster, {}))
    registry = load_registry()
    panels = []
    for competition in ("hmmt_feb_2025",):
        try:
            df = matharena_outputs(competition, columns=["problem_idx", "model_name", "answer"])
        except Exception:                                   # offline or unavailable: no panels
            return []
        firsts: dict[str, dict] = {}
        for row in df.to_dict("records"):
            item_id = f"{competition}/{int(row['problem_idx']):02d}"
            if item_id not in train_ids or row["model_name"] in excluded:
                continue
            firsts.setdefault(item_id, {}).setdefault(row["model_name"], row["answer"])
        for item_id in sorted(firsts):
            answer_format = registry[item_id].answer_format
            responses = []
            for name in sorted(firsts[item_id]):
                raw = extract_final(firsts[item_id][name] or "", lenient=False)
                answer = normalize(raw, answer_format) if raw else None
                responses.append(Response(name, answer, bool(answer is not None and grade_answer(item_id, answer).correct)))
            order = np.random.default_rng(int(digest(item_id)[:8], 16)).permutation(len(responses))
            shuffled = [responses[i] for i in order]
            for start in range(0, len(shuffled) - team_size + 1, team_size):
                panels.append(ItemRecord(item_id, answer_format, shuffled[start:start + team_size],
                                         registry[item_id].family_id))
    return panels


def offline_phase(*, roster="sat", live=False, train_limit=40, dev_limit=20, output=None, estimate=False,
                  approved_plan=None, sandbox="auto", extra_samples=1, review_evidence=True,
                  strategy_validation=False, policy_dev=True, use_public_logs=True, policy_actions=14,
                  max_calls=40, max_item_cost_usd=0.75, voc_epsilon=0.002, hint_audit="structural"):
    if type(extra_samples) is not int or extra_samples < 0:
        raise ValueError("extra_samples must be a nonnegative integer")
    if type(policy_actions) is not int or policy_actions < 0:
        raise ValueError("policy_actions must be a nonnegative integer")
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    if (isinstance(max_item_cost_usd, bool) or not isinstance(max_item_cost_usd, (int, float))
            or not math.isfinite(max_item_cost_usd) or max_item_cost_usd <= 0):
        raise ValueError("max_item_cost_usd must be finite and positive")
    if (isinstance(voc_epsilon, bool) or not isinstance(voc_epsilon, (int, float))
            or not math.isfinite(voc_epsilon) or voc_epsilon < 0):
        raise ValueError("voc_epsilon must be finite and nonnegative")
    if not isinstance(hint_audit, str) or hint_audit not in {"structural", "residual"}:
        raise ValueError("hint_audit must be structural or residual")
    from mathmage.analysis.frontier import frontier, operating_point
    from mathmage.experiments.runner import (grade_result, _protocol_digest, tools_runtime, require_live_tools,
                                             budget_review)
    from mathmage.grading.equiv import normalize
    from mathmage.grading.verify import grade_answer
    from mathmage.llm.client import LLMClient
    from mathmage.psychometrics.calibrate import (CalibrationData, CoachingRecord, EvidenceRecord, ItemRecord,
                                                  Response, calibrate, reliability)
    from mathmage.scaffolds import structural_audit
    from mathmage.team import TeamConfig, TeamEngine
    from mathmage.team.artifacts import DEFAULT_BANK, STRATEGIES
    from mathmage.team.engine import TEAM_TEMPLATES, default_posterior_params, extract_answer, parse_json

    registry = load_registry()
    train = _sample_split(registry, "train", train_limit)
    dev = _sample_split(registry, "dev", dev_limit)
    if not train or not dev:
        raise ValueError("positive nonempty training and development samples required")
    if {x.family_id for x in train} & {x.family_id for x in dev}:
        raise ValueError("train/dev family overlap")
    learning_families = {x.family_id for x in train + dev}
    if learning_families & {x.family_id for x in registry.select(split="test")}:
        raise ValueError("learning/test family overlap")
    mode = "live" if live else "mock"
    models = load_models()
    rs = models.rosters[roster]
    n = len(rs.members)
    # These bounds define the maximal traces used to choose mu. Deployment must
    # inherit them unless an explicitly different policy is declared in its plan.
    policy_config = {"max_policy_actions": policy_actions, "max_item_cost_usd": max_item_cost_usd,
                     "voc_epsilon": voc_epsilon, "hint_audit": hint_audit, "max_calls": max_calls}
    # Freeze free anchor inputs before constructing the plan or a paid client.
    # A changed cache must not silently change a resumed calibration and produce
    # additional development calls under the same approved plan.
    public = _public_log_records(roster, {i.item_id for i in train}) if use_public_logs else []
    public_snapshot = [asdict(record) for record in public]
    kappa_panels = _public_kappa_panels(roster, {i.item_id for i in train}) if use_public_logs else []
    kappa_snapshot = [asdict(record) for record in kappa_panels]
    stage0_allowance = min(max_calls, n + 1 + n)
    # These two development allocations jointly bound one capped trace. A short
    # actual stage0 can leave more room for continuation than this allocation.
    dev_continuation_allowance = min(max_calls, stage0_allowance + policy_actions * 3) - stage0_allowance
    estimate_lines = {"train_stage0_and_program": len(train) * stage0_allowance,
                      "train_extra_member_samples": len(train) * n * extra_samples,
                      "train_self_revision": len(train) * n,
                      # Each tutor authors one hint, replayed to at most n-1 solvers.
                      "train_hint_replay_upper": len(train) * n * n,
                      "train_review_evidence": len(train) * n if review_evidence else 0,
                      "train_discernment_upper": len(train) * n,
                      "train_cards_upper": len(train),
                      "dev_stage0_and_program": len(dev) * stage0_allowance,
                      # Up to hint + residual audit + solve, or review + revision + check.
                      "dev_policy_maximal_run_upper": len(dev) * dev_continuation_allowance if policy_dev else 0,
                      "dev_rule_strategy_validation": len(dev) * len(STRATEGIES) * policy_config["max_calls"]
                                                        if strategy_validation else 0}
    unit = max(m.price.cost(2500, m.expected_out("olympiad")) for m in [models.model(k) for k in rs.members])
    manifest = {"schema_version": 2, "experiment": "offline", "mode": mode, "roster": roster,
                "train_ids": [x.item_id for x in train], "dev_ids": [x.item_id for x in dev],
                "registry_sha256": file_digest(paths.REGISTRY / "items.parquet"),
                "answer_vault_sha256": file_digest(paths.REGISTRY / "answers.parquet"),
                "models_sha256": file_digest(paths.CONFIGS / "models.yaml"),
                "budget_sha256": file_digest(paths.CONFIGS / "budget.yaml"),
                "protocol_sha256": _protocol_digest(),
                "tools_runtime": tools_runtime(["full"], sandbox),
                "prompt_hashes": {k: t.id for k, t in {**TEMPLATES, **TEAM_TEMPLATES}.items()},
                "options": {"sandbox": sandbox, "extra_samples": extra_samples, "review_evidence": review_evidence,
                            "strategy_validation": strategy_validation, "policy_dev": policy_dev,
                            "use_public_logs": use_public_logs, "policy_actions": policy_actions},
                "policy_config": policy_config,
                "public_anchor_snapshot": {"sha256": digest(public_snapshot), "n_items": len(public),
                                           "n_responses": sum(len(record.responses) for record in public)},
                "public_kappa_snapshot": {"sha256": digest(kappa_snapshot), "n_panels": len(kappa_panels),
                                          "n_items": len({record.item_id for record in kappa_panels}),
                                          "team_size": 4},
                "planning_calls": estimate_lines,
                "planning_expected_usd": sum(estimate_lines.values()) * unit,
                "total_usd_ceiling": None,
                "note": ("Upper call allowances priced at the costliest member's expected olympiad solve with "
                         "2500 input tokens; this is a planning allowance, not a total-dollar upper bound. Actual "
                         "contexts, output lengths and retries vary. Every live call reserves its full output ceiling. "
                         "Development stage0 and continuation call allocations jointly bound each capped trace; "
                         "the item-cost limit is an action-planning stop, not a hard dollar guarantee. "
                         "Both rosters share the offline line plus 20% tolerance and the unchanged $75 global cap. "
                         "No batch discount assumed.")}
    manifest["plan_id"] = digest(manifest)
    if estimate:
        review = {**manifest, "budget_review": budget_review("offline", manifest["planning_expected_usd"])}
        print(json.dumps(review, indent=2))
        return review
    if live and approved_plan != manifest["plan_id"]:
        raise ValueError("live offline phase requires reviewed --approved-plan from --estimate --live")
    runtime = require_live_tools(live, ["full"], sandbox)
    if live and runtime != manifest["tools_runtime"]:
        raise ValueError("sandbox runtime changed after planning; review a fresh offline plan")
    target = Path(output) if output else paths.RUNS / f"offline_{roster}_{mode}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        if json.loads((target / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("offline manifest changed; choose a new output directory")
    else:
        atomic_json(target / "manifest.json", manifest)
    kwargs = {} if live else {"cache_path": target / "cache.sqlite", "ledger_path": target / "ledger.jsonl",
                              "log_directory": target / "calls"}
    client = LLMClient(experiment="offline", live=live, **kwargs)

    def saved(name, operation):
        observation_id = digest(name)
        path = target / "observations" / (observation_id[:24] + ".json")
        with file_lock(path.with_suffix(".lock")):
            if path.exists():
                return read_record(path, plan_id=manifest["plan_id"], mode=mode,
                                   identity={"observation_id": observation_id})["data"]
            try:
                value = operation()
            except BaseException:
                atomic_json(target / "execution.json", {"status": "interrupted", "mode": mode,
                            "final_artifact_created": False, "client": client.summary()})
                client.close()
                raise
            atomic_json(path, seal_record({"observation_id": observation_id, "data": value},
                                         plan_id=manifest["plan_id"], mode=mode))
            return value

    def call(item, model, template, tag, **kw):
        request = LLMRequest(model_key=model, messages=[{"role": "user", "content": template.render(**kw)}],
            template_id=template.id, sample_tag=f"offline:{tag}", kind=item.kind,
            meta={"item_id": item.item_id, "family_id": item.family_id, "source": item.source,
                  "band": item.band, "item_date": item.item_date, "condition": tag, "role": tag})
        return saved([item.item_id, model, template.id, tag, kw], lambda: asdict(client.complete(request)))

    def usable(c):
        return bool(c.get("valid", True) and not c.get("error") and c.get("answer"))

    def agent_of(c):
        return "tool" if c.get("phase") in ("tool_forward", "verify_forward", "tool_alt") else c["model_key"]

    def responses_of(item, candidates, grades):
        # Record attempted programs without answers as abstentions too. They
        # estimate tool availability without pretending a failed execution is a
        # mathematical wrong answer.
        return [Response(agent_of(c), c["answer"] if usable(c) else None, grades[c["id"]])
                for c in candidates]

    def lineage(key):
        return models.model(key).lineage

    trials, discernment_trials, cards, review_trials = [], [], [], []
    calibration_items, revision_pairs = [], []
    member_scores = {k: [] for k in rs.members}
    for item in train:
        base = saved(["base", item.item_id], lambda: TeamEngine(client, TeamConfig(roster=roster, sandbox=sandbox,
                                                                                  **policy_config))
                     .create_checkpoint(item))
        stage0 = base["stage0"]
        initial = [c for c in stage0 if c["id"] != "s0-tool"]
        grades = {c["id"]: bool(usable(c) and grade_answer(item.item_id, c["answer"]).correct) for c in stage0}
        for candidate in initial:
            member_scores[candidate["model_key"]].append({"family_id": item.family_id,
                                                        "correct": float(grades[candidate["id"]])})
        responses = responses_of(item, stage0, grades)
        # Extra independent samples estimate within-model error correlation (kappa_same).
        for k in range(extra_samples):
            for i, member in enumerate(rs.members):
                r = call(item, member, SOLVE, f"extra:{k}:member={i}", problem=item.problem,
                         format_instruction=item.view().format_instruction, cards="")
                raw = extract_answer(r["text"]) if not r.get("error") and not r.get("truncated") else None
                ans = normalize(raw, item.answer_format) if raw else None
                responses.append(Response(member, ans, bool(ans and grade_answer(item.item_id, ans).correct)))
        calibration_items.append(ItemRecord(item.item_id, item.answer_format, responses, item.family_id))
        self_revision = {}
        for i, candidate in enumerate(initial):
            response = call(item, candidate["model_key"], SOLVE_REVISE, f"self-revision:{i}",
                problem=item.problem, format_instruction=item.view().format_instruction,
                solution=candidate["text"], concerns="Check your own reasoning independently and correct any local defect you find.")
            self_revision[i] = bool(not response.get("error") and not response.get("truncated") and
                                   grade_answer(item.item_id, extract_answer(response["text"])).correct)
            revision_pairs.append((self_revision[i], grades[candidate["id"]]))
        # Identical-hint replay, restricted to solvers whose initial answer was wrong and to other tutors:
        # the same authored hint goes verbatim to every such solver, separating content from uptake.
        for ti, tutor in enumerate(rs.members):
            wrong_solvers = [si for si, c in enumerate(initial) if not grades[c["id"]] and si != ti]
            if not wrong_solvers:
                continue
            source = initial[wrong_solvers[0]]
            # Whether the tutor's own answer was right: coaching lifts are fitted separately for each case.
            tutor_correct = bool(ti < len(initial) and usable(initial[ti]) and grades[initial[ti]["id"]])
            hint_response = call(item, tutor, HINT, f"replay-hint:{ti}", problem=item.problem,
                                 attempt=source["certificate"], hint_kind=HINT_KINDS["strategic"])
            hint = hint_response["text"]
            structural = structural_audit(hint)
            leak = any(len(str(c.get("answer") or "")) >= 2 and
                       re.search(r"(?<![\w.])" + re.escape(str(c["answer"])) + r"(?![\w.])", hint) for c in initial)
            accepted = bool(not hint_response.get("error") and not hint_response.get("truncated")
                            and structural.structural_pass and not leak)
            for si in wrong_solvers:
                candidate = initial[si]
                row = {"item_id": item.item_id, "family_id": item.family_id, "solver": candidate["model_key"],
                       "solver_instance": si, "tutor": tutor, "tutor_instance": ti, "tutor_correct": tutor_correct,
                       "hint_id": digest(hint), "panel": "identical_hint_replay", "self_revision_correct": self_revision[si],
                       "audit_accepted": accepted, "answer_leak_filtered": bool(leak), "empirical_leakage": None,
                       "hint_validity": None, "structural_audit": asdict(structural), "correct": None}
                if accepted:
                    response = call(item, candidate["model_key"], SOLVE_SCAFFOLD, f"replay:{ti}:{si}",
                        problem=item.problem, format_instruction=item.view().format_instruction, note=hint)
                    row["correct"] = bool(not response.get("error") and not response.get("truncated") and
                                          grade_answer(item.item_id, extract_answer(response["text"])).correct)
                trials.append(row)
        # One review of every initial candidate by another lineage (evidence likelihoods, training only).
        if review_evidence:
            checker = TeamEngine(client, TeamConfig(roster=roster, sandbox=sandbox))
            checker._start(item)
            for i, candidate in enumerate(initial):
                if not usable(candidate):
                    continue
                others = [j for j in range(n) if j != i]
                different = [j for j in others if lineage(rs.members[j]) != lineage(candidate["model_key"])]
                reviewer = (different or others or [i])[0]
                r = call(item, rs.members[reviewer], REVIEW, f"review-evidence:{i}", problem=item.problem,
                         solution=candidate["text"])
                report = parse_json(r["text"]) if not r.get("error") and not r.get("truncated") else {}
                located = type(report.get("first_error_line")) is int and report["first_error_line"] > 0
                objection = report.get("verdict") == "invalid" and located and bool(report.get("diagnosis"))
                verified = False
                code = report.get("check_code", "")
                if objection and isinstance(code, str) and code.strip() and checker._can_execute():
                    res = saved(["review-check", item.item_id, i, digest(code)],
                                lambda: asdict(checker._execute(code, "review_objection", candidate["id"])))
                    verified = bool(res["executed"] and res["exit_code"] == 0 and not res["error"]
                                    and "CLAIM_FALSE" in (res["stdout"] or "").splitlines())
                kind = ("review_objection_verified" if verified else "review_objection_unverified" if objection
                        else "review_valid" if report.get("verdict") == "valid" else "review_unsure")
                review_trials.append({"item_id": item.item_id, "family_id": item.family_id, "kind": kind,
                                      "target_correct": grades[candidate["id"]], "reviewer": rs.members[reviewer]})
        correct = [c for c in initial if grades[c["id"]]]
        wrong = [c for c in initial if not grades[c["id"]]]
        if correct and wrong:
            pair = [correct[0], wrong[0]]
            if int(digest(item.item_id)[:2], 16) % 2:
                pair.reverse()
            pool = json.dumps([{"number": k + 1, "solution": c["text"]} for k, c in enumerate(pair)])
            for judge in dict.fromkeys(rs.members):
                response = call(item, judge, JUDGE, "discernment", problem=item.problem, candidates=pool)
                choice = parse_json(response["text"]).get("choice")
                valid = (type(choice) is int and 1 <= choice <= 2 and not response.get("error")
                         and not response.get("truncated"))
                discernment_trials.append({"item_id": item.item_id, "family_id": item.family_id, "judge": judge,
                                           "correct_choice": bool(valid and grades[pair[choice - 1]["id"]])})
        if correct:
            response = call(item, rs.card_writer, CARD, "card", problem=item.problem, solution=correct[0]["text"])
            content = parse_json(response["text"])
            allowed = {"method", "applies_when", "pitfall", "checklist"}
            # This method-only syntax screen does not establish empirical leakage
            # safety or full proof correctness of the answer-correct source candidate.
            if (not response.get("error") and not response.get("truncated") and set(content) == allowed
                    and all(isinstance(content[k], str) for k in allowed - {"checklist"})
                    and isinstance(content["checklist"], list)
                    and all(isinstance(entry, str) for entry in content["checklist"])):
                if not re.search(r"\d|\\boxed", json.dumps(content)):
                    cards.append({"id": digest(content)[:16], "source_item_id": item.item_id,
                                  "family_id": item.family_id, **content})

    tutoring, tutoring_details = {}, {}
    for solver in dict.fromkeys(rs.members):
        tutoring[solver], tutoring_details[solver] = {}, {}
        for tutor in dict.fromkeys(rs.members):
            values = {}
            for r in trials:
                if r["solver"] == solver and r["tutor"] == tutor and r["correct"] is not None:
                    values.setdefault(r["family_id"], []).append(int(r["correct"]) - int(r["self_revision_correct"]))
            summary = _delta_interval([float(np.mean(v)) for v in values.values()])
            tutoring_details[solver][tutor] = summary
            if summary["n"]:
                tutoring[solver][tutor] = summary["lift"]
    discernment = {}
    for member in dict.fromkeys(rs.members):
        values = _family_means([r for r in discernment_trials if r["judge"] == member], lambda r: r["correct_choice"])
        if values:
            discernment[member] = sum(values) / len(values)
    best = max(dict.fromkeys(rs.members), key=lambda k: np.mean(_family_means(member_scores[k], lambda r: r["correct"]))
               if member_scores[k] else -1)

    # ---- supervised training calibration for reference-free test inference ----
    authored_hints = {(row["item_id"], row["tutor_instance"], row["hint_id"]): row["audit_accepted"]
                      for row in trials}
    data = CalibrationData(items=_merge_item_records(calibration_items + public),
                           evidence=[EvidenceRecord(r["kind"], r["target_correct"]) for r in review_trials],
                           judge_trials=[t["correct_choice"] for t in discernment_trials],
                           coaching=[(bool(r["correct"]), bool(r["self_revision_correct"])) for r in trials
                                     if r["correct"] is not None],
                           revision=revision_pairs, hint_acceptance_trials=list(authored_hints.values()),
                           coaching_records=[CoachingRecord(r["item_id"], r["solver"], bool(r["correct"]),
                                                            bool(r["tutor_correct"]))
                                             for r in trials if r["correct"] is not None],
                           kappa_items=kappa_panels)
    params, calibration_diagnostics = calibrate(
        data, provenance={"train_ids": [i.item_id for i in train], "public_log_items": [r.item_id for r in public],
                          "public_log_agents": sorted(MATHARENA_ALIASES.get(roster, {}).values()) if public else []},
        base=default_posterior_params(rs.members, rs.checker))
    provisional = {"frozen": True, "mode": mode, "roster": roster,
                   "provenance": {"train_ids": [i.item_id for i in train], "dev_ids": [i.item_id for i in dev]},
                   "best_member": best, "tutoring_matrix": tutoring, "discernment": discernment,
                   "cards": cards, "strategy_bank": DEFAULT_BANK, "posterior_params": params.to_dict(),
                   "policy_config": policy_config}
    provisional["sha256"] = digest(provisional)

    # ---- development: held-out calibration and the pre-registered operating point ----
    dev_records, dev_policy, validation = [], [], []
    for item in dev:
        base = saved(["dev-base", item.item_id, provisional["sha256"]], lambda:
                     TeamEngine(client, TeamConfig(roster=roster, sandbox=sandbox, offline_artifacts=provisional))
                     .create_checkpoint(item))
        stage0 = base["stage0"]
        grades = {c["id"]: bool(usable(c) and grade_answer(item.item_id, c["answer"]).correct) for c in stage0}
        dev_records.append(ItemRecord(item.item_id, item.answer_format, responses_of(item, stage0, grades),
                                      item.family_id))
        if policy_dev:
            result = saved(["dev-policy", item.item_id, provisional["sha256"]], lambda:
                asdict(TeamEngine(client, TeamConfig(roster=roster, sandbox=sandbox, offline_artifacts=provisional,
                                                     mu=0.0, max_policy_actions=policy_actions)).run(item, checkpoint=base)))
            dev_policy.append(grade_result(result, item, mode=mode))
        if strategy_validation:
            for strategy in STRATEGIES:
                result = saved(["dev-strategy", item.item_id, strategy, provisional["sha256"]], lambda s=strategy:
                    asdict(TeamEngine(client, TeamConfig(roster=roster, sandbox=sandbox, offline_artifacts=provisional,
                        strategy_override=[s])).run(item, checkpoint=base)))
                graded = grade_result(result, item, mode=mode)
                validation.append({**graded, "strategy": strategy})
    bank, bank_scores = {}, {}
    for zone in "ALPU":
        ranked = _strategy_scores(validation, zone, STRATEGIES) if validation else []
        bank[zone] = [s["strategy"] for s in ranked] if ranked else DEFAULT_BANK[zone]
        bank_scores[zone] = ranked
    dev_reliability = reliability(dev_records, params) if dev_records else {}
    dev_frontier = frontier(dev_policy) if dev_policy else []
    mu_star, mu_rule = 0.0, "maximal run (no development policy runs)"
    if dev_frontier:
        top = max(p["accuracy"] for p in dev_frontier)
        # Pre-registered knee rule: the cheapest threshold within one point of the best development accuracy.
        knee = min((p for p in dev_frontier if p["accuracy"] >= top - 0.01), key=lambda p: p["cost_usd"])
        mu_star, mu_rule = float(knee["mu"]), "cheapest threshold within 1 point of the best development accuracy"
    artifact = {**provisional, "schema_version": 2,
        "provenance": {**provisional["provenance"], "registry_sha256": manifest["registry_sha256"],
                       "models_sha256": manifest["models_sha256"], "prompt_hashes": manifest["prompt_hashes"],
                       "answer_vault_sha256": manifest["answer_vault_sha256"],
                       "budget_sha256": manifest["budget_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                       "tools_runtime": manifest["tools_runtime"],
                       "public_anchor_snapshot": manifest["public_anchor_snapshot"],
                       "public_kappa_snapshot": manifest["public_kappa_snapshot"],
                       "manifest_plan_id": manifest["plan_id"]},
        "zone_thresholds": {}, "strategy_bank": bank, "mu_star": mu_star,
        "measurements": {"tutoring": tutoring_details, "strategy_validation": bank_scores,
            "posterior_calibration": calibration_diagnostics, "dev_reliability": dev_reliability,
            "dev_operating_point": operating_point(dev_policy, mu_star) if dev_policy else None,
            "dev_frontier": dev_frontier, "mu_rule": mu_rule,
            "zone_calibration": ("posterior parameters fitted on synthetic training fixtures; empirical zone calibration unmeasured"
                                 if not live else "posterior calibrated on training items; development reliability reported; empirical zones unvalidated"),
            "empirical_hint_leakage": "unmeasured; structural screen and candidate-answer filter only",
            "live_contingent_tutoring": "unmeasured; identical-hint replay panel only",
            "reflection_tuning": "not run",
            "card_screen": "method-only structure and digit/boxed-marker screen; empirical leakage unmeasured",
            "matrix_missingness": "missing tutoring or discernment entries remain absent; no success labels imputed",
            "status": "synthetic plumbing validation" if not live else "observed offline measurements"}}
    atomic_json(target / "training_evidence.json", seal_record(
        {"tutoring_trials": trials, "discernment_trials": discernment_trials, "review_trials": review_trials,
         "public_anchor_snapshot": public_snapshot, "public_kappa_snapshot": kappa_snapshot},
        plan_id=manifest["plan_id"], mode=mode))
    atomic_json(target / "development_evidence.json", seal_record(
        {"policy_rows": dev_policy, "strategy_rows": validation}, plan_id=manifest["plan_id"], mode=mode))
    frozen = freeze(artifact, target / "artifacts.json")
    atomic_json(target / "execution.json", {"status": "complete", "client": client.summary(), "artifact_sha256": frozen["sha256"]})
    client.close()
    print(f"Frozen {mode} artifacts: {target / 'artifacts.json'}")
    return target / "artifacts.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", choices=["sat", "xolver", "open", "sat26"], default="sat")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--approved-plan")
    ap.add_argument("--train-limit", type=int, default=40)
    ap.add_argument("--dev-limit", type=int, default=20)
    ap.add_argument("--sandbox", choices=["auto", "docker", "local"], default="auto")
    ap.add_argument("--extra-samples", type=int, default=1)
    ap.add_argument("--policy-actions", type=int, default=14)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--max-item-cost-usd", type=float, default=0.75)
    ap.add_argument("--voc-epsilon", type=float, default=0.002)
    ap.add_argument("--hint-audit", choices=["structural", "residual"], default="structural")
    ap.add_argument("--no-review-evidence", action="store_true")
    ap.add_argument("--strategy-validation", action="store_true")
    ap.add_argument("--no-policy-dev", action="store_true")
    ap.add_argument("--no-public-logs", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    offline_phase(roster=args.roster, live=args.live, train_limit=args.train_limit, dev_limit=args.dev_limit,
                  output=args.output, estimate=args.estimate, approved_plan=args.approved_plan, sandbox=args.sandbox,
                  extra_samples=args.extra_samples, review_evidence=not args.no_review_evidence,
                  strategy_validation=args.strategy_validation, policy_dev=not args.no_policy_dev,
                  use_public_logs=not args.no_public_logs, policy_actions=args.policy_actions,
                  max_calls=args.max_calls, max_item_cost_usd=args.max_item_cost_usd,
                  voc_epsilon=args.voc_epsilon, hint_audit=args.hint_audit)


if __name__ == "__main__":
    main()
