from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from mathmage.data.registry import ItemView
from mathmage.llm.types import LLMResponse
from mathmage.scaffolds import residual_audit, structural_audit
from mathmage.team import CheckEvidence, TeamConfig, TeamEngine
from mathmage.team.artifacts import STRATEGIES, load_artifacts
from mathmage.team.engine import extract_answer
from mathmage.experiments.artifacts import digest
from mathmage.tools import DockerSandbox
from mathmage.tools import computational_evidence


ITEM = ItemView("synthetic/01", "synthetic", "Find the sum of the integers from 1 through 4.",
                "Give one integer.", "olympiad")


def solution(answer):
    return f"Working.\nCERTIFICATE\n1. Sum the four consecutive integers.\n\\boxed{{{answer}}}"


class MockClient:
    live = False

    def __init__(self, answers=(10, 10, 11), objection=False, selection=2):
        self.requests = []
        self.answers = answers
        self.objection = objection
        self.selection = selection

    def complete(self, request):
        self.requests.append(request)
        phase = request.meta["phase"]
        if phase == "stage0":
            index = int(request.meta["role"].split("=")[-1])
            text = solution(self.answers[index])
        elif phase == "review":
            text = json.dumps({"verdict": "invalid" if self.objection else "unsure", "first_error_line": 1,
                               "diagnosis": "The claimed arithmetic is unsupported.", "check_code": ""})
        elif phase == "selection":
            text = json.dumps({"choice": self.selection})
        elif phase == "hint":
            text = "Pair terms symmetrically and look for a repeated pair sum."
        elif phase == "hint_audit":
            text = json.dumps({"reveals_answer": False, "one_step_completion": False,
                               "remaining_nontrivial_steps": ["Establish the pairing", "Count the disjoint pairs"]})
        else:
            text = solution(12)
        return LLMResponse(text=text, cost_usd=0.01, tokens_in=10, tokens_out=20, latency_s=0.1)

    def complete_many(self, requests):
        return [self.complete(request) for request in requests]


class Verifier:
    def __init__(self, mode="none"):
        self.mode = mode

    def verify(self, item, candidate):
        assert isinstance(item, ItemView)
        if self.mode == "all":
            return CheckEvidence("passed", "injected_test", ["independent computation"], [], True)
        if self.mode == "first" and candidate.id == "s0-0":
            return CheckEvidence("passed", "injected_test", ["independent computation"], [], True)
        if self.mode == "partial":
            return CheckEvidence("partial", "injected_test", ["line 1 checked"])
        return CheckEvidence(details="no executable check")


def engine(client=None, *, condition="full", verifier=None, **kwargs):
    # These tests pin the operational-rule router (ablation R1). The posterior router (the
    # main system) is tested in tests/test_policy_engine.py.
    kwargs.setdefault("router", "rule")
    return TeamEngine(client or MockClient(), TeamConfig(condition=condition, **kwargs), verifier=verifier or Verifier())


def test_verified_consensus_accepts_without_judge_and_costs_serialize():
    client = MockClient(answers=(10, 10, 10))
    result = engine(client, verifier=Verifier("all")).run(ITEM)
    assert result.zone.label == "A"
    assert result.final_id == "s0-0"
    assert len(client.requests) == 3
    assert result.costs["cost_usd"] == pytest.approx(0.03)
    assert result.costs["tokens_in"] == 30
    json.dumps(asdict(result))


def test_agreement_without_checks_is_uncertain_latent():
    result = engine().run(ITEM)
    assert result.zone.label == "L"
    assert result.zone.uncertain and not result.zone.calibrated
    assert result.zone.signals["verified_answers"] == 0
    assert result.zone.probabilities == {}


def test_independent_stage0_tags_reused_across_ablations():
    clients = [MockClient(), MockClient()]
    for client, condition in zip(clients, ("full", "A3")):
        engine(client, condition=condition).run(ITEM)
    initial = [[r for r in c.requests if r.meta["phase"] == "stage0"] for c in clients]
    assert [r.sample_tag for r in initial[0]] == [r.sample_tag for r in initial[1]]
    assert [r.messages for r in initial[0]] == [r.messages for r in initial[1]]
    assert all(len(r.messages) == 1 for r in initial[0])
    assert len({r.sample_tag for r in initial[0]}) == 3
    assert len({r.seed for r in initial[0]}) == 3


def test_unverified_objection_cannot_trigger_revision():
    client = MockClient(objection=True)
    result = engine(client).run(ITEM)
    assert len(result.candidates) == 3
    assert all(not event["revision_allowed"] for event in result.trace if event["event"] == "review")


def test_a3_allows_revision_and_retains_originals():
    client = MockClient(objection=True, selection=4)
    result = engine(client, condition="A3").run(ITEM)
    assert len(result.candidates) == 6
    assert result.final_id == "c3"
    assert result.stage0 == ["s0-0", "s0-1", "s0-2"]
    assert {c.id for c in result.candidates} >= set(result.stage0)
    assert all(c.parent_id for c in result.candidates[3:])


def test_autonomy_guard_blocks_unverified_judge_choice():
    result = engine(MockClient(selection=2), verifier=Verifier("first")).run(ITEM)
    assert result.final_id == "s0-0"
    selection = next(t for t in result.trace if t["event"] == "selection")
    assert selection["autonomy_guard_applied"]


def test_checkpoint_replay_is_immutable_and_does_not_repeat_stage0():
    first = engine().run(ITEM)
    saved = json.dumps(first.checkpoint, sort_keys=True)
    client = MockClient(objection=True)
    replay = engine(client, condition="A3").run(ITEM, checkpoint=first.checkpoint)
    assert not any(r.meta["phase"] == "stage0" for r in client.requests)
    assert json.dumps(first.checkpoint, sort_keys=True) == saved
    assert replay.stage0 == first.stage0
    assert replay.costs["calls"] == 10
    assert replay.costs["dispatch_calls"] == 7
    assert replay.costs["inherited_calls"] == 3
    assert replay.checkpoint["stage0_costs"]["calls"] == 3
    with pytest.raises(ValueError, match="problem text"):
        engine().run(ItemView(ITEM.item_id, ITEM.benchmark, "A changed problem.", ITEM.format_instruction, ITEM.kind),
                     checkpoint=first.checkpoint)


def test_no_tools_does_not_call_verifier_or_code_generator():
    class ForbiddenVerifier:
        def verify(self, *_):
            raise AssertionError("tools are disabled")
    client = MockClient()
    result = engine(client, condition="A7", verifier=ForbiddenVerifier()).run(ITEM)
    assert all(c.evidence.status == "unavailable" for c in result.candidates)
    assert not any(r.meta["phase"].startswith("verify") for r in client.requests)


def test_single_agent_and_homogeneous_controls_have_distinct_contexts():
    single, homogeneous = MockClient(), MockClient()
    engine(single, condition="C1").run(ITEM)
    engine(homogeneous, condition="C2").run(ITEM)
    s0 = [r for r in single.requests if r.meta["phase"] == "stage0"]
    h0 = [r for r in homogeneous.requests if r.meta["phase"] == "stage0"]
    assert [len(r.messages) for r in s0] == [1, 3, 5]
    assert [len(r.messages) for r in h0] == [1, 1, 1]
    assert len({r.model_key for r in single.requests}) == 1
    assert len({r.model_key for r in h0}) == 1


def test_graded_hint_sequence_and_unmeasured_empirical_leakage():
    client = MockClient(answers=(1, 2, 3))
    result = engine(client, verifier=Verifier("partial"), max_strategies=1, max_hint_levels=2).run(ITEM)
    assert result.zone.label == "P"
    hints = [t for t in result.trace if t["event"] == "hint"]
    assert [h["level"] for h in hints] == ["strategic", "decomposition"] * 3
    assert all(h["audit"]["accepted"] for h in hints)
    assert all(h["audit"]["empirical_leakage"] == "unmeasured" for h in hints)
    assert all("\\boxed" not in r.messages[-1]["content"] for r in client.requests if r.meta["phase"] == "hint")


def test_hint_audit_fails_closed_without_residual_work():
    assert not structural_audit("The final answer is 10.").structural_pass
    assert not residual_audit("Try pairing.", {}).accepted
    assert not residual_audit("Try pairing.", {"reveals_answer": False, "one_step_completion": False,
                                               "remaining_nontrivial_steps": ["Add up"]}).accepted


def test_a1_runs_all_six_templates_and_a2_uses_certificate_only_judge():
    a1 = engine(condition="A1", verifier=Verifier("all")).run(ITEM)
    assert [t["name"] for t in a1.trace if t["event"] == "strategy"] == list(STRATEGIES)
    client = MockClient()
    a2 = engine(client, condition="A2").run(ITEM)
    assert not any(r.meta["phase"] == "review" for r in client.requests)
    request = next(r for r in client.requests if r.meta["phase"] == "selection")
    assert request.template_id.startswith("sat_judge")
    assert '"evidence"' not in request.messages[-1]["content"]
    assert next(t for t in a2.trace if t["event"] == "selection")["certificate_only"]


def test_sandbox_never_falls_back_to_host_python(monkeypatch):
    monkeypatch.setattr("mathmage.tools.sandbox.shutil.which", lambda _: None)
    monkeypatch.setattr("mathmage.tools.sandbox.subprocess.Popen",
                        lambda *_args, **_kwargs: pytest.fail("must never launch host Python"))
    result = DockerSandbox().run("import os; print(os.environ)")
    assert not result.available and not result.executed


def test_artifacts_reject_unfrozen_nontraining_cards_and_mock_live():
    with pytest.raises(ValueError, match="frozen"):
        load_artifacts({})
    data = {"frozen": True, "mode": "mock", "provenance": {"train_ids": ["train/1"], "dev_ids": []},
            "cards": [{"source_item_id": "test/1", "method": "Pair terms"}]}
    data["sha256"] = digest(data)
    with pytest.raises(ValueError, match="training items"):
        load_artifacts(data)
    data["cards"] = []
    data.pop("sha256")
    data["sha256"] = digest(data)
    with pytest.raises(ValueError, match="live runs"):
        load_artifacts(data, live=True)


def test_boxed_parser_handles_nested_braces_and_integer_normalization():
    assert extract_answer(r"First \boxed{0010}") == "0010"
    assert extract_answer(r"Second \boxed{\frac{3}{7}}") == r"\frac{3}{7}"


def test_constant_output_pseudo_verifiers_are_rejected():
    assert not computational_evidence("print('ANSWER: 10')", "independent_forward")[0]
    assert not computational_evidence("print('CLAIM_FALSE')", "review_objection")[0]
    assert computational_evidence("answer = sum(range(1, 5))\nprint('ANSWER:', answer)", "independent_forward")[0]
    assert computational_evidence("actual = 2 + 2\nprint('CLAIM_FALSE' if actual != 5 else 'CLAIM_TRUE')",
                                  "review_objection")[0]
    assert not computational_evidence("print(sum(range(5)))\nprint('ANSWER: 10')", "independent_forward")[0]
    assert not computational_evidence("x = 2 < 3\nprint('CLAIM_FALSE')", "review_objection")[0]
    assert computational_evidence("actual = 2 + 2\nif actual != 5:\n    print('CLAIM_FALSE')", "review_objection")[0]


def test_checkpoint_and_artifact_tampering_are_detected():
    first = engine().run(ITEM)
    first.checkpoint["stage0"][0]["answer"] = "999"
    with pytest.raises(ValueError, match="content hash"):
        engine().run(ITEM, checkpoint=first.checkpoint)
    artifact = {"frozen": True, "mode": "mock", "provenance": {"train_ids": [], "dev_ids": []}}
    artifact["sha256"] = digest(artifact)
    artifact["mode"] = "live"
    with pytest.raises(ValueError, match="content hash"):
        load_artifacts(artifact)


def test_checkpoint_resource_cost_includes_shared_work_without_rebilling():
    first = engine().run(ITEM)
    replay = engine(condition="A2").run(ITEM, checkpoint=first.checkpoint)
    assert replay.costs["calls"] == 4
    assert replay.costs["dispatch_calls"] == 1
    assert replay.costs["marginal_cost_usd"] == pytest.approx(0.01)
    assert replay.costs["full_method_cost_usd"] > first.checkpoint["stage0_costs"]["resource_cost_usd"]
    assert replay.costs["tokens_in"] == 40
    assert replay.costs["wall_latency_s"] >= 0


def test_strategy_override_replays_one_specific_mechanism():
    initial = engine().run(ITEM)
    result = engine(strategy_override=["method_diversification"]).run(ITEM, checkpoint=initial.checkpoint)
    assert [t["name"] for t in result.trace if t["event"] == "strategy"] == ["method_diversification"]
    assert len(result.candidates) == 4


def test_create_checkpoint_has_no_judging_or_treatment_calls():
    client = MockClient()
    checkpoint = engine(client).create_checkpoint(ITEM)
    assert len(client.requests) == 3
    assert all(request.meta["phase"] == "stage0" for request in client.requests)
    assert checkpoint["stage0_costs"]["calls"] == 3
    assert checkpoint["mode"] == "mock"
    assert checkpoint["prompt_hashes"]
    assert checkpoint["models_sha256"]
    replay = engine(condition="A2").run(ITEM, checkpoint=checkpoint)
    assert replay.costs["calls"] == 4


def test_checkpoint_binds_format_and_verification_context():
    checkpoint = engine().create_checkpoint(ITEM)
    changed_format = ItemView(ITEM.item_id, ITEM.benchmark, ITEM.problem, "Give a fraction.", ITEM.kind, "symbolic")
    with pytest.raises(ValueError, match="evidence context"):
        engine().run(changed_format, checkpoint)
    with pytest.raises(ValueError, match="evidence context"):
        engine(checker="haiku-4.5").run(ITEM, checkpoint)
    with pytest.raises(ValueError, match="tool/evidence context"):
        engine(sandbox="local").run(ITEM, checkpoint)


def test_live_tools_protocol_fails_before_model_calls_without_isolation(monkeypatch):
    class LiveClient(MockClient):
        live = True
    client = LiveClient()
    monkeypatch.setattr(DockerSandbox, "available", property(lambda _: False))
    with pytest.raises(RuntimeError, match="requires isolated Docker"):
        TeamEngine(client, TeamConfig())
    assert client.requests == []
    # A7 is the explicit no-tools protocol and needs no Docker prerequisite.
    TeamEngine(client, TeamConfig(condition="A7"))


def test_call_metadata_keeps_provenance_out_of_prompt_and_binds_checkpoint():
    from mathmage.data.registry import Item
    item = Item("synthetic/02", "synthetic", "synthetic source", "Find a sum.", "exact", "algebra",
                "family-2", "train", "2026-10", "foundations", "olympiad")
    client = MockClient()
    result = engine(client).run(item)
    first = client.requests[0]
    assert first.meta["family_id"] == "family-2"
    assert first.meta["source"] == "synthetic source"
    assert first.meta["stage"] == "stage0" and first.meta["attempt_idx"] == 0
    assert "family-2" not in first.messages[0]["content"]
    continuation = next(r for r in client.requests if r.meta["phase"] == "selection")
    assert continuation.meta["checkpoint_id"] == result.checkpoint["sha256"]


def test_default_offline_client_runs_every_condition_without_network(tmp_path, monkeypatch):
    from mathmage.llm.client import LLMClient
    monkeypatch.setattr("mathmage.tools.sandbox.shutil.which", lambda _: None)
    with LLMClient(cache_path=tmp_path / "cache.sqlite", ledger_path=tmp_path / "ledger.jsonl",
                   log_directory=tmp_path / "calls") as client:
        for condition in ("full", *(f"A{i}" for i in range(1, 9)), "C1", "C2"):
            result = TeamEngine(client, TeamConfig(condition=condition, max_hint_levels=1)).run(ITEM)
            assert result.final_id in {candidate.id for candidate in result.candidates}
            assert result.costs["marginal_cost_usd"] == 0
            assert result.costs["resource_cost_usd"] > 0
            assert result.meta["tools_status"] in {"disabled", "unavailable"}
            assert all(call["mode"] == "mock" for call in result.meta["calls"])

