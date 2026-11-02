"""The posterior router (main MathMage system): routing, actions, ablations, anytime property."""

from __future__ import annotations

import json

import pytest

from mathmage.analysis.frontier import budget_matched, frontier, operating_point, pareto, stop_index
from mathmage.data.registry import ItemView
from mathmage.experiments.artifacts import digest
from mathmage.llm.types import LLMResponse
from mathmage.team import TeamConfig, TeamEngine
from mathmage.team.posterior import AgentProfile, PosteriorParams

ITEM = ItemView("synthetic/01", "synthetic", "Find the sum of the integers from 1 through 4.",
                "Give one integer.", "olympiad")


def solution(answer):
    return f"Working.\nCERTIFICATE\n1. Sum the four consecutive integers.\n\\boxed{{{answer}}}"


class Script:
    """Deterministic fake client. Stage-0 answers are fixed; later solves follow ``later``."""

    live = False

    def __init__(self, stage0=(10, 10, 10), later=(10,), review="valid", choice=1, hint="Pair the terms.",
                 program="print('ANSWER:', sum(range(1, 5)))"):
        self.stage0, self.later, self.review, self.choice = stage0, list(later), review, choice
        self.hint, self.program = hint, program
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        phase = request.meta["phase"]
        if phase == "stage0":
            text = solution(self.stage0[int(request.meta["role"].split("=")[-1])])
        elif phase in ("verify_forward", "tool_alt"):
            text = f"```python\n{self.program}\n```"
        elif phase == "review":
            verdict = {"verdict": self.review, "first_error_line": 1 if self.review == "invalid" else None,
                       "diagnosis": "unsupported step" if self.review == "invalid" else "", "check_code": ""}
            text = json.dumps(verdict)
        elif phase == "selection":
            text = json.dumps({"choice": self.choice})
        elif phase == "hint":
            text = self.hint
        else:
            answer = self.later.pop(0) if len(self.later) > 1 else self.later[0]
            text = solution(answer)
        return LLMResponse(text=text, cost_usd=0.0, tokens_in=200, tokens_out=400, latency_s=0.01)

    def complete_many(self, requests):
        return [self.complete(r) for r in requests]


def artifact(params: PosteriorParams, **extra):
    data = {"frozen": True, "mode": "mock", "roster": "sat",
            "provenance": {"train_ids": ["aime_2023/01"], "dev_ids": ["aime_2023/02"]},
            "posterior_params": params.to_dict(), **extra}
    data["sha256"] = digest(data)
    return data


def strong_params(**kw):
    agents = {"o3-mini-low": AgentProfile(2.2), "haiku-4.5": AgentProfile(1.8), "deepseek-v3": AgentProfile(1.2),
              "tool": AgentProfile(1.0, abstain=0.3)}
    return PosteriorParams(agents=agents, calibrated=True, provenance={"train_ids": ["aime_2023/01"]}, **kw)


def run(script, **config):
    config.setdefault("sandbox", "docker")         # no tools unless a test asks for the local sandbox
    return TeamEngine(script, TeamConfig(**config)).run(ITEM)


def test_confident_unanimous_answer_is_accepted_without_extra_calls():
    script = Script(stage0=(10, 10, 10))
    result = run(script, offline_artifacts=artifact(strong_params()))
    assert result.zone.label == "A" and result.zone.calibrated and not result.zone.uncertain
    assert len(script.requests) == 3 and result.final_id == "s0-0"
    assert len(result.meta["anytime"]) == 1 and result.meta["anytime"][0]["stop_reason"]
    assert result.meta["router"] == "posterior" and result.meta["posterior_calibrated"]


def test_disagreement_triggers_actions_and_final_is_the_posterior_argmax():
    script = Script(stage0=(10, 11, 12), later=(10,))
    result = run(script, offline_artifacts=artifact(strong_params()), max_policy_actions=6)
    steps = result.meta["anytime"]
    assert result.zone.label in {"L", "U", "P"} and len(steps) > 1
    assert steps[-1]["best_candidate"] == result.final_id
    assert [s["calls"] for s in steps] == sorted(s["calls"] for s in steps)
    final = next(c for c in result.candidates if c.id == result.final_id)
    assert final.answer == "10"


def test_anytime_property_threshold_runs_are_prefixes_of_the_maximal_run():
    full = run(Script(stage0=(10, 11, 12), later=(10, 11, 10, 10)), offline_artifacts=artifact(strong_params()),
               max_policy_actions=6)
    steps = full.meta["anytime"]
    ratios = [s["chosen"]["ratio"] for s in steps if s["chosen"]]
    assert len(ratios) >= 2
    mu = (ratios[0] + ratios[1]) / 2 if ratios[1] < ratios[0] else ratios[0] * 0.999
    cut = stop_index(steps, mu)
    partial = run(Script(stage0=(10, 11, 12), later=(10, 11, 10, 10)), offline_artifacts=artifact(strong_params()),
                  max_policy_actions=6, mu=mu)
    labels_full = [s["chosen"]["label"] for s in steps if s["chosen"]]
    labels_partial = [s["chosen"]["label"] for s in partial.meta["anytime"] if s["chosen"]]
    assert labels_partial == labels_full[:len(labels_partial)] and len(labels_partial) == cut
    assert partial.final_id == steps[cut]["best_candidate"]


def test_uniform_budget_ablation_spends_the_full_action_budget():
    result = run(Script(stage0=(10, 10, 10)), offline_artifacts=artifact(strong_params()), condition="A1",
                 max_policy_actions=3)
    assert sum(1 for s in result.meta["anytime"] if s["chosen"]) == 3


def test_a2_uses_sat_certificate_judge_and_no_reviews():
    script = Script(stage0=(10, 11, 12), choice=2)
    result = run(script, offline_artifacts=artifact(strong_params()), condition="A2", max_policy_actions=2)
    phases = [r.meta["phase"] for r in script.requests]
    assert "review" not in phases
    judge = [r for r in script.requests if r.meta["phase"] == "selection"]
    assert judge and judge[-1].template_id.startswith("sat_judge")
    assert result.final_id is not None


def test_unverified_objection_never_revises_but_a3_overwrites():
    params = strong_params()
    full = run(Script(stage0=(10, 11, 11), review="invalid"), offline_artifacts=artifact(params),
               max_policy_actions=4)
    reviews = [t for t in full.trace if t["event"] == "review"]
    assert all(not t["revision_allowed"] for t in reviews)
    a3 = run(Script(stage0=(10, 11, 11), review="invalid"), offline_artifacts=artifact(params), condition="A3",
             max_policy_actions=4)
    if any(t["event"] == "review" for t in a3.trace):
        assert any(t["revision_allowed"] for t in a3.trace if t["event"] == "review")


def test_hints_that_contain_a_candidate_answer_are_rejected():
    script = Script(stage0=(17, 23, 31), hint="Try 23 first; then pair terms.", later=(23,))
    result = run(script, offline_artifacts=artifact(strong_params()), max_policy_actions=8)
    hints = [t for t in result.trace if t["event"] == "hint"]
    assert all(h["answer_leak_filtered"] and not h["audit"]["accepted"] for h in hints)
    assert not any(r.meta["phase"] == "coached_solve" for r in script.requests)


def test_program_answer_joins_the_pool_with_the_local_sandbox():
    script = Script(stage0=(10, 11, 12))
    result = run(script, sandbox="local", offline_artifacts=artifact(strong_params()), max_policy_actions=0)
    tool = next(c for c in result.candidates if c.id == "s0-tool")
    assert tool.answer == "10" and tool.valid
    assert result.meta["stage0_tool"] == ["s0-tool"] and result.meta["sandbox_backend"] == "local-process"
    assert result.stage0 == ["s0-0", "s0-1", "s0-2"]
    # With the independent program agreeing, answer 10 leads.
    assert next(c for c in result.candidates if c.id == result.final_id).answer == "10"
    a7 = run(Script(stage0=(10, 11, 12)), sandbox="local", offline_artifacts=artifact(strong_params()),
             condition="A7", max_policy_actions=0)
    assert not any(c.id == "s0-tool" for c in a7.candidates) and a7.meta["stage0_tool"] == []


def test_frontier_helpers_truncate_and_find_pareto_points():
    items = [{"benchmark": "x", "anytime": [
        {"calls": 3, "cost_usd": 0.03, "correct": False, "chosen": {"ratio": 5.0}},
        {"calls": 4, "cost_usd": 0.05, "correct": True, "chosen": {"ratio": 1.0}},
        {"calls": 5, "cost_usd": 0.07, "correct": True, "chosen": None}]},
        {"benchmark": "x", "anytime": [{"calls": 3, "cost_usd": 0.03, "correct": True, "chosen": None}]}]
    assert stop_index(items[0]["anytime"], 10.0) == 0
    assert stop_index(items[0]["anytime"], 2.0) == 1
    assert stop_index(items[0]["anytime"], 0.0) == 2
    assert operating_point(items, 10.0)["accuracy"] == pytest.approx(0.5)
    assert operating_point(items, 0.0)["accuracy"] == pytest.approx(1.0)
    pts = frontier(items, [0.0, 2.0, 10.0])
    assert [p["accuracy"] for p in pareto(pts)] == [0.5, 1.0]


def test_frontier_enumerates_narrow_threshold_intervals_and_rejects_nonreplayable_ablations():
    items = [{"item_id": "test/1", "benchmark": "x", "anytime": [
        {"calls": 1, "cost_usd": 0.01, "correct": False, "chosen": {"ratio": 1.00001}},
        {"calls": 2, "cost_usd": 0.02, "correct": True, "chosen": {"ratio": 1.0}},
        {"calls": 3, "cost_usd": 0.03, "correct": False, "chosen": None}]}]
    assert {p["cost_usd"] for p in frontier(items)} == {0.01, 0.02, 0.03}
    with pytest.raises(ValueError, match="oracle"):
        budget_matched(items, 0.02)
    oracle = budget_matched(items, 0.02, descriptive_oracle=True)
    assert oracle["accuracy"] == 1 and not oracle["confirmatory"]
    frozen = budget_matched(items, 0.02, mu=0.0)
    assert frozen["accuracy"] == 0 and not frozen["within_budget"]
    items[0]["frontier_replay_valid"] = False
    with pytest.raises(ValueError, match="not a replayable"):
        frontier(items)


def test_budget_matching_selects_on_development_without_using_test_labels():
    def item(name, outcomes):
        return {"item_id": name, "benchmark": "x", "anytime": [
            {"calls": 1, "cost_usd": 0.01, "correct": outcomes[0], "chosen": {"ratio": 5.0}},
            {"calls": 2, "cost_usd": 0.02, "correct": outcomes[1], "chosen": {"ratio": 1.0}},
            {"calls": 3, "cost_usd": 0.03, "correct": outcomes[2], "chosen": None}]}
    dev = [item("dev/1", [False, True, False])]
    test = [item("test/1", [True, False, True])]
    result = budget_matched(test, 0.025, development_items=dev)
    assert result["cost_usd"] == 0.02 and result["accuracy"] == 0
    assert result["selection"] == "development selected threshold" and result["confirmatory"]
    with pytest.raises(ValueError, match="overlap"):
        budget_matched(test, 0.025, development_items=test)


def test_tool_phase_is_independent_and_a7_replay_removes_all_tool_answers(monkeypatch):
    from mathmage.tools import ToolResult

    class Sandbox:
        backend, image, reason, available = "fixture", "fixture-image", "fixture", True

        def run(self, code):
            return ToolResult(True, True, "ANSWER: 10\n", "", 0)

    monkeypatch.setattr("mathmage.team.engine.make_sandbox", lambda _: Sandbox())
    client = Script(stage0=(11, 12, 13))
    source = TeamEngine(client, TeamConfig(max_policy_actions=0))
    checkpoint = source.create_checkpoint(ITEM)
    tool = next(row for row in checkpoint["stage0"] if row["id"] == "s0-tool")
    assert tool["phase"] == "tool_forward"
    assert source._agent_key(next(c for c in source.candidates if c.id == "s0-tool")) == "tool"
    assert sum(r.meta["phase"] == "verify_forward" for r in client.requests) == 1
    a7 = TeamEngine(Script(), TeamConfig(condition="A7", max_policy_actions=0)).run(ITEM, checkpoint)
    assert all(c.id != "s0-tool" and c.answer != "10" for c in a7.candidates)
    assert a7.meta["stage0_tool"] == [] and a7.costs["calls"] == 3
    assert all(not c.evidence.full_answer_verified for c in a7.candidates)
    independent_a7 = TeamEngine(Script(), TeamConfig(condition="A7")).create_checkpoint(ITEM)
    with pytest.raises(ValueError, match="tool/evidence context"):
        TeamEngine(Script(), TeamConfig(max_policy_actions=0)).run(ITEM, independent_a7)


def test_mu_uses_frozen_default_and_explicit_zero_requests_maximal():
    frozen = artifact(strong_params())
    frozen["mu_star"] = 100.0
    frozen["sha256"] = digest({k: v for k, v in frozen.items() if k != "sha256"})
    default = run(Script(stage0=(10, 11, 12)), offline_artifacts=frozen)
    maximal = run(Script(stage0=(10, 11, 12)), offline_artifacts=frozen, mu=0.0, max_policy_actions=2)
    assert default.meta["mu"] == 100.0 and default.meta["mu_source"] == "frozen development choice"
    assert maximal.meta["mu"] == 0 and maximal.meta["mu_source"] == "explicit"


def test_direct_engine_resolves_frozen_policy_limits_without_mutating_caller():
    frozen = artifact(strong_params(), policy_config={"max_policy_actions": 0, "max_item_cost_usd": 0.15,
                                                     "voc_epsilon": 0.01, "hint_audit": "residual", "max_calls": 9})
    requested = TeamConfig(offline_artifacts=frozen)
    engine = TeamEngine(Script(), requested)
    assert requested.max_policy_actions is None
    assert engine.config.max_policy_actions == 0 and engine.config.max_item_cost_usd == 0.15
    assert engine.config.voc_epsilon == 0.01 and engine.config.hint_audit == "residual"
    assert engine.config.max_calls == 9
    explicit = TeamEngine(Script(), TeamConfig(offline_artifacts=frozen, max_policy_actions=3, max_calls=4))
    assert explicit.config.max_policy_actions == 3 and explicit.config.max_calls == 4


@pytest.mark.parametrize("kw", [{"mu": float("nan")}, {"mu": float("inf")},
                                  {"max_item_cost_usd": float("inf")},
                                  {"voc_epsilon": float("nan")}, {"voc_epsilon": -1},
                                  {"max_calls": 3.5}, {"max_policy_actions": True}])
def test_policy_limits_reject_nonfinite_or_invalid_values(kw):
    with pytest.raises(ValueError):
        TeamConfig(**kw)


def test_live_posterior_requires_calibration_but_training_checkpoint_does_not():
    class LiveScript(Script):
        live = True

    client = LiveScript()
    engine = TeamEngine(client, TeamConfig(condition="A7", max_policy_actions=0))
    with pytest.raises(ValueError, match="calibrated training-derived"):
        engine.run(ITEM)
    assert client.requests == []
    assert engine.create_checkpoint(ITEM)["stage0"]


def test_a2_terminal_anytime_snapshot_matches_judged_output_and_resources():
    result = run(Script(stage0=(10, 11, 12), choice=2), condition="A2", max_policy_actions=0)
    last = result.meta["anytime"][-1]
    assert last["best_candidate"] == result.final_id == "s0-1"
    assert last["calls"] == result.costs["calls"]
    assert last["cost_usd"] == pytest.approx(result.costs["resource_cost_usd"])
    assert not result.meta["frontier_replay_valid"]


@pytest.mark.parametrize("encoded", [False, True])
def test_single_call_budget_denial_aborts_without_candidate_or_call(encoded):
    from mathmage.llm.ledger import BudgetExceeded

    class Denied(Script):
        def complete(self, request):
            if encoded:
                return LLMResponse(text="", error="BudgetExceeded: exhausted")
            raise BudgetExceeded("exhausted")

    engine = TeamEngine(Denied(), TeamConfig(condition="C1", sandbox="docker"))
    with pytest.raises(BudgetExceeded, match="exhausted"):
        engine.run(ITEM)
    assert engine.calls == [] and engine.candidates == []


def test_concurrent_budget_denial_preserves_successful_calls_but_aborts_observation():
    from mathmage.llm.ledger import BudgetExceeded

    class PartiallyDenied(Script):
        def complete_many(self, requests):
            return [LLMResponse(text=solution(10), tokens_in=5, tokens_out=6),
                    LLMResponse(text="", error="BudgetExceeded: exhausted"),
                    LLMResponse(text=solution(11), tokens_in=7, tokens_out=8)]

    engine = TeamEngine(PartiallyDenied(), TeamConfig(sandbox="docker"))
    with pytest.raises(BudgetExceeded, match="exhausted"):
        engine.run(ITEM)
    assert [call["role"] for call in engine.calls] == ["member=0", "member=2"]
    assert engine.candidates == []
    assert all(not call["error"] for call in engine.calls)


def test_raised_batch_budget_denial_is_not_replaced_with_synthetic_failures():
    from mathmage.llm.ledger import BudgetExceeded

    class DeniedBatch(Script):
        def complete_many(self, requests):
            raise BudgetExceeded("exhausted")

    engine = TeamEngine(DeniedBatch(), TeamConfig(sandbox="docker"))
    with pytest.raises(BudgetExceeded):
        engine.run(ITEM)
    assert engine.calls == [] and engine.candidates == []


def test_a5_discussion_bypasses_hint_screen_propensity(monkeypatch):
    from mathmage.team.policy import Policy

    configs = []

    def capture(params, cfg):
        configs.append(cfg)
        return Policy(params, cfg)

    monkeypatch.setattr("mathmage.team.engine.Policy", capture)
    run(Script(), condition="A5", max_policy_actions=0)
    assert not configs[-1].screen_coaching
    run(Script(), max_policy_actions=0)
    assert configs[-1].screen_coaching


def _tutor_dependent_params():
    return strong_params(lift={"coached": 1.0, "coached_right": 2.5, "coached_wrong": 0.2, "diversify": 0.2,
                               "revision": 0.3, "discussion": 0.3})


def test_tutor_dependent_coaching_considers_every_tutor_and_links_the_tutor_answer():
    from collections import Counter
    from mathmage.team.policy import Policy, PolicyConfig

    params = _tutor_dependent_params()
    script = Script(stage0=(10, 11, 12), later=(11,))
    engine = TeamEngine(script, TeamConfig(sandbox="docker", offline_artifacts=artifact(params), max_policy_actions=0))
    engine.run(ITEM)
    counters = {"samples": Counter(), "hints": Counter(), "reviews": Counter(), "tool_alt": 0, "diversify": 0,
                "judges": 0}
    label_of, obs, evidence, C = engine._posterior_state()
    result = Policy(params).posterior_of(obs, evidence, C)
    coach = [a for a in engine._feasible_actions(label_of, C, result, counters, PolicyConfig(), params)
             if a.kind == "coach"]
    assert {(a.member, a.tutor) for a in coach} == {(s, t) for s in range(3) for t in range(3) if s != t}
    lift, source, source_lift = engine._coaching_spec(2, 1, params, label_of)
    assert (lift, source_lift) == (0.2, 2.5) and source == label_of[engine._latest_candidate(1).id]
    engine._act_coach(2, 1, counters, params)
    label_of, obs, _, _ = engine._posterior_state()
    coached = obs[-1]
    tutor_cluster = label_of[engine._latest_candidate(1).id]
    assert coached.cluster == tutor_cluster
    assert (coached.source, coached.source_lift, coached.lift) == (tutor_cluster, 2.5, 0.2)


def test_single_lift_artifacts_keep_one_matched_tutor_and_budget_zones_are_recorded():
    from collections import Counter
    from mathmage.team.policy import Policy, PolicyConfig

    params = strong_params()
    engine = TeamEngine(Script(stage0=(10, 11, 12)), TeamConfig(sandbox="docker", offline_artifacts=artifact(params),
                                                                max_policy_actions=3))
    result = engine.run(ITEM)
    assert result.zone.signals["member_zones_budget"]["K"] == 3
    assert set(result.zone.signals["member_zones_budget"]["zones"]) == set(result.zone.signals["member_zones"])
    counters = {"samples": Counter(), "hints": Counter(), "reviews": Counter(), "tool_alt": 0, "diversify": 0,
                "judges": 0}
    label_of, obs, evidence, C = engine._posterior_state()
    posterior = Policy(params).posterior_of(obs, evidence, C)
    coach = [a for a in engine._feasible_actions(label_of, C, posterior, counters, PolicyConfig(), params)
             if a.kind == "coach"]
    assert len({a.member for a in coach}) == len(coach)


def test_a3_retired_tutor_remains_historical_evidence_but_cannot_be_selected():
    from collections import Counter
    from mathmage.team.policy import Policy, PolicyConfig

    params = _tutor_dependent_params()
    script = Script(stage0=(10, 11, 12), later=(13, 14), review="invalid")
    engine = TeamEngine(script, TeamConfig(condition="A3", tools_enabled=False, sandbox="docker",
                                          offline_artifacts=artifact(params), max_policy_actions=0))
    engine.run(ITEM)
    counters = {"samples": Counter(), "hints": Counter(), "reviews": Counter(), "tool_alt": 0, "diversify": 0,
                "judges": 0}
    tutor = engine._latest_candidate(0)
    engine._act_coach(1, 0, counters, params)
    coached = engine._latest_candidate(1)
    engine._act_review(tutor, 2, counters, params)
    assert tutor.id in engine._overwritten and tutor not in engine._included()
    labels, obs, evidence, C = engine._posterior_state()
    assert tutor.id in labels                     # historical truth reference still exists
    assert obs[-2].source == labels[tutor.id]       # the coached response precedes the revision
    assert obs[-2].source_lift == params.lift["coached_right"]
    assert engine._coach_source[coached.id][0] == tutor.id
    eligible = {labels[c.id] for c in engine._included() if c.id in labels}
    assert labels[tutor.id] not in eligible
    posterior = Policy(params).posterior_of(obs, evidence, C, selectable=eligible)
    actions = engine._feasible_actions(labels, C, posterior, counters, PolicyConfig(), params)
    assert all(a.cluster != labels[tutor.id] for a in actions if a.kind == "review")
    assert all(labels[tutor.id] not in a.options for a in actions if a.kind == "judge")
    final, _, steps, _ = engine._run_policy()
    assert final in {c.id for c in engine._included()} and final != tutor.id
    assert steps[-1]["confidence"] == pytest.approx(posterior.confidence)
