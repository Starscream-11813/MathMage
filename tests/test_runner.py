from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from mathmage.data.registry import Item
from mathmage.experiments import runner
from mathmage.experiments.artifacts import atomic_json, freeze, read_record, seal_record
from mathmage.grading.vault import AnswerVault
from mathmage.grading.verify import Grade
from mathmage.team.types import Candidate, TeamResult, ZoneEstimate


def toy_item():
    return Item("toy/1", "aime_2024", "fixture", "Compute 2+2.", "aime_integer", "algebra",
                "family/1", "train", "2024-02", "olympiad", "olympiad")


def vault():
    return AnswerVault(pd.DataFrame([{"item_id": "toy/1", "answer": "4", "answer_type": "integer",
                                      "answer_format": "aime_integer", "solution": "2+2=4"}]))


def payload(*, condition="full", seed=0):
    return asdict(TeamResult("toy/1", condition, seed,
        [Candidate("s0-0", "o3-mini-low#0", "o3-mini-low", "fixture", "4", "certificate", "stage0"),
         Candidate("s0-1", "haiku-4.5#1", "haiku-4.5", "fixture", "5", "certificate", "stage0")],
        ["s0-0", "s0-1"], "s0-0", ZoneEstimate("D", {}, True, False, "fixture"),
        {"fixture_checkpoint": True}, {}, [],
        {"cost_usd": 0.01, "marginal_cost_usd": 0.01, "resource_cost_usd": 0.03,
         "tokens_in": 10, "tokens_out": 20, "calls": 3, "latency_s": 0.2,
         "resource_latency_s": 1.5, "backend_calls": 1}))


class FakeClient:
    instances = []
    def __init__(self, experiment, live=False, **kwargs):
        self.live = live
        self.closed = False
        self.instances.append(self)
    def summary(self):
        return {"mode": "live" if self.live else "mock", "cost_usd": 0.0}
    def close(self):
        self.closed = True


class FakeEngine:
    calls = []
    configs = []
    failure = None
    def __init__(self, client, config):
        self.config = config
        self.configs.append(config)
    def run(self, item, checkpoint=None):
        if self.failure:
            raise self.failure
        self.calls.append((self.config.condition, checkpoint))
        p = payload(condition=self.config.condition, seed=self.config.seed)
        return TeamResult(**p)


@pytest.fixture
def isolated_runner(monkeypatch):
    FakeClient.instances = []
    FakeEngine.calls = []
    FakeEngine.configs = []
    FakeEngine.failure = None
    monkeypatch.setattr(runner, "study_items", lambda *args: [toy_item()])
    monkeypatch.setattr(runner, "file_digest", lambda path: Path(path).name + "-digest")
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "source-v1")
    monkeypatch.setattr(runner, "tools_runtime", lambda conditions, sandbox="auto": {"requested": True, "available": True,
                                                                   "image_id": "sha256:fixture", "reason": "fixture"})
    monkeypatch.setattr("mathmage.llm.client.LLMClient", FakeClient)
    monkeypatch.setattr("mathmage.team.TeamEngine", FakeEngine)
    monkeypatch.setattr("mathmage.grading.verify.grade_answer", lambda item_id, answer, vault=None:
                        Grade(item_id, answer, answer == "4", "fixture"))
    monkeypatch.setattr("mathmage.analysis.make_tables.build_report", lambda rows, output:
                        atomic_json(output / "report.json", {"observations": len(rows)}))


def test_grade_mapping_keeps_logical_resources_and_marginal_spending_separate():
    row = runner.grade_result(payload(), toy_item(), mode="mock", vault=vault())
    assert row["covered"] and row["initial_correct"] and row["final_correct"]
    assert row["strict_final_correct"] is True
    assert row["stage0_correct"] == {"o3-mini-low#0": True, "haiku-4.5#1": False}
    assert row["cost_usd"] == 0.03 and row["marginal_cost_usd"] == 0.01
    assert row["latency_s"] == 1.5 and row["marginal_latency_s"] == 0.2
    assert row["calls"] == 3 and row["tokens_in"] == 10 and row["backend_calls"] == 1


def test_invalid_or_truncated_candidate_cannot_receive_credit_or_vote():
    result = payload()
    result["candidates"][0]["valid"] = False
    result["candidates"][0]["error"] = "truncated"
    row = runner.grade_result(result, toy_item(), mode="mock", vault=vault())
    assert not row["covered"] and not row["final_correct"]
    assert row["initial_vote_id"] == "s0-1"
    assert row["strict_final_correct"] is False


@pytest.mark.parametrize("mutation", ["wrong_item", "missing_final", "duplicate_id", "missing_initial"])
def test_malformed_result_fails_closed(mutation):
    result = payload()
    if mutation == "wrong_item":
        result["item_id"] = "other"
    elif mutation == "missing_final":
        result["final_id"] = "absent"
    elif mutation == "duplicate_id":
        result["candidates"][1]["id"] = result["candidates"][0]["id"]
    else:
        result["stage0"].append("absent")
    with pytest.raises(ValueError):
        runner.grade_result(result, toy_item(), mode="mock", vault=vault())


def test_default_runner_saves_sealed_records_and_resumes_without_agent_calls(tmp_path, isolated_runner):
    target = runner.run_study("pilot", output=tmp_path)
    assert target == tmp_path and len(FakeEngine.calls) == 1
    record = next((target / "agent").glob("*.json"))
    stored = json.loads(record.read_text(encoding="utf-8"))
    assert stored["run_provenance"]["mode"] == "mock"
    assert "candidate_grades" not in stored
    runner.run_study("pilot", output=tmp_path)
    assert len(FakeEngine.calls) == 1
    assert all(client.closed for client in FakeClient.instances)
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "complete"


def test_resume_rejects_modified_agent_output(tmp_path, isolated_runner):
    runner.run_study("pilot", output=tmp_path)
    record = next((tmp_path / "agent").glob("*.json"))
    stored = json.loads(record.read_text())
    stored["candidates"][0]["answer"] = "changed"
    atomic_json(record, stored)
    with pytest.raises(ValueError, match="content hash"):
        runner.run_study("pilot", output=tmp_path)
    assert len(FakeEngine.calls) == 1


def test_resume_rejects_changed_manifest_or_code(tmp_path, isolated_runner, monkeypatch):
    runner.run_study("pilot", output=tmp_path)
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "source-v2")
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_study("pilot", output=tmp_path)


def test_mode_barrier_prevents_live_reusing_mock_output(tmp_path, isolated_runner):
    runner.run_study("pilot", output=tmp_path)
    manifest = runner.run_study("pilot", live=True, estimate=True, router="rule")
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_study("pilot", live=True, approved_plan=manifest["plan_id"], output=tmp_path, router="rule")
    assert all(not c.live for c in FakeClient.instances)


def test_live_artifact_barrier_and_plan_review_before_client_creation(tmp_path, isolated_runner):
    artifact_path = tmp_path / "mock-artifact.json"
    freeze({"mode": "mock", "provenance": {"train_ids": [], "dev_ids": []}}, artifact_path)
    with pytest.raises(ValueError, match="live models"):
        runner.run_study("e1", live=True, artifacts_path=artifact_path, output=tmp_path / "run")
    with pytest.raises(ValueError, match="approved-plan"):
        runner.run_study("pilot", live=True, output=tmp_path / "run", router="rule")
    assert FakeClient.instances == []


def test_shared_stage0_and_a8_proposal_independence(tmp_path, isolated_runner):
    runner.run_study("pilot", conditions=["full", "A7", "A8", "C1", "C2"], output=tmp_path)
    assert FakeEngine.calls[0] == ("full", None)
    assert FakeEngine.calls[1] == ("A7", {"fixture_checkpoint": True})
    assert FakeEngine.calls[2:] == [("A8", None), ("C1", None), ("C2", None)]


def test_failed_execution_is_marked_and_client_closes(tmp_path, isolated_runner):
    FakeEngine.failure = RuntimeError("fixture failure")
    with pytest.raises(RuntimeError):
        runner.run_study("pilot", output=tmp_path)
    record = json.loads((tmp_path / "execution.json").read_text())
    assert record["status"] == "failed" and record["error_type"] == "RuntimeError"
    assert FakeClient.instances[0].closed


def test_generic_records_reject_mode_plan_and_identity_mismatch(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, seal_record({"item_id": "toy/1", "phase": "train"}, plan_id="plan", mode="mock"))
    assert read_record(path, plan_id="plan", mode="mock", identity={"phase": "train"})["item_id"] == "toy/1"
    for kwargs in [{"plan_id": "other", "mode": "mock"}, {"plan_id": "plan", "mode": "live"},
                   {"plan_id": "plan", "mode": "mock", "identity": {"phase": "dev"}}]:
        with pytest.raises(ValueError):
            read_record(path, **kwargs)


def test_real_team_and_client_api_work_in_offline_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "study_items", lambda *args: [toy_item()])
    monkeypatch.setattr(runner, "file_digest", lambda path: Path(path).name + "-digest")
    monkeypatch.setattr("mathmage.grading.verify.grade_answer", lambda item_id, answer, vault=None:
                        Grade(item_id, answer, answer == "4", "fixture"))
    monkeypatch.setattr("mathmage.analysis.make_tables.build_report", lambda rows, output:
                        atomic_json(output / "report.json", {"observations": len(rows)}))
    runner.run_study("pilot", conditions=["A7"], output=tmp_path)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["status"] == "complete"
    assert execution["client"]["mode"] == "mock" and execution["client"]["cost_usd"] == 0
    assert execution["client"]["backend_calls"] >= 3


def test_e2_uses_eight_aime_seeds_and_one_math500_seed(tmp_path, isolated_runner, monkeypatch):
    from dataclasses import replace
    items = [toy_item(), replace(toy_item(), item_id="math500/1", benchmark="math500")]
    monkeypatch.setattr(runner, "study_items", lambda *args: items)
    original = FakeEngine.run
    def run(engine, item, checkpoint=None):
        result = original(engine, item, checkpoint)
        result.item_id = item.item_id
        return result
    monkeypatch.setattr(FakeEngine, "run", run)
    estimate = runner.run_study("e2", estimate=True)
    assert estimate["seeds"] == list(range(8))
    assert estimate["n_item_seed_pairs"] == 9
    assert estimate["benchmark_repetitions"] == {"aime_2024": 8, "math500": 1}
    assert estimate["cost_estimate"]["lines"][0]["calls"] == 9
    runner.run_study("e2", output=tmp_path)
    rows = [json.loads(path.read_text()) for path in (tmp_path / "graded").glob("*.json")]
    assert len(rows) == 9
    assert sorted(r["seed"] for r in rows if r["benchmark"] == "aime_2024") == list(range(8))
    assert [r["seed"] for r in rows if r["benchmark"] == "math500"] == [0]


@pytest.mark.parametrize("field", ["registry_sha256", "answer_vault_sha256", "models_sha256", "prompt_hashes", "protocol_sha256"])
def test_frozen_artifact_runtime_drift_is_rejected(field, tmp_path, isolated_runner):
    artifact = {"roster": "sat", "mode": "mock", "provenance": {
        "train_ids": [], "dev_ids": [], "registry_sha256": "items.parquet-digest",
        "answer_vault_sha256": "answers.parquet-digest",
        "models_sha256": "models.yaml-digest", "prompt_hashes": runner.prompt_hashes(),
        "protocol_sha256": "source-v1"}}
    runner.validate_artifact_runtime(artifact, "sat")
    artifact["provenance"][field] = "changed"
    path = tmp_path / "artifact.json"
    freeze(artifact, path)
    with pytest.raises(ValueError, match=field):
        runner.run_study("pilot", artifacts_path=path, output=tmp_path / "run")
    assert FakeClient.instances == []


def test_live_tool_requirement_fails_closed_except_a7(monkeypatch):
    class MissingSandbox:
        image = "fixture"
        available = False
        reason = "fixture unavailable"
    monkeypatch.setattr("mathmage.tools.DockerSandbox", MissingSandbox)
    with pytest.raises(ValueError, match="prepared Docker sandbox"):
        runner.require_live_tools(True, ["full"])
    assert not runner.require_live_tools(True, ["A7"])["requested"]
    assert not runner.require_live_tools(False, ["full"])["available"]


def test_policy_configuration_is_frozen_and_reaches_the_engine(tmp_path, isolated_runner):
    runner.run_study("pilot", output=tmp_path, sandbox="local", router="posterior", mu=3.0,
                     max_policy_actions=4, max_item_cost_usd=0.2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    policy = manifest["policy"]
    config = FakeEngine.configs[0]
    assert policy["mu"] == config.mu == 3.0 and policy["mu_source"] == "explicit"
    assert config.sandbox == "local" and config.max_policy_actions == 4 and config.max_item_cost_usd == 0.2
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_study("pilot", output=tmp_path, sandbox="local", mu=4.0,
                         max_policy_actions=4, max_item_cost_usd=0.2)


def test_frozen_development_threshold_is_used_unless_explicitly_overridden(tmp_path, isolated_runner):
    frozen_limits = {"max_policy_actions": 4, "max_item_cost_usd": 0.2,
                     "voc_epsilon": 0.007, "hint_audit": "residual", "max_calls": 12}
    artifact = {"roster": "sat", "mode": "mock", "mu_star": 7.0, "policy_config": frozen_limits, "provenance": {
        "train_ids": [], "dev_ids": [], "registry_sha256": "items.parquet-digest",
        "answer_vault_sha256": "answers.parquet-digest",
        "models_sha256": "models.yaml-digest", "prompt_hashes": runner.prompt_hashes(),
        "protocol_sha256": "source-v1"}}
    path = tmp_path / "artifact.json"
    freeze(artifact, path)
    selected = runner.run_study("pilot", artifacts_path=path, estimate=True)
    maximal = runner.run_study("pilot", artifacts_path=path, estimate=True, mu=0.0)
    assert selected["policy"]["mu"] == 7.0 and maximal["policy"]["mu"] == 0.0
    assert all(selected["policy"][key] == value for key, value in frozen_limits.items())
    override = runner.run_study("pilot", artifacts_path=path, estimate=True,
                                max_policy_actions=2, max_item_cost_usd=0.1,
                                voc_epsilon=0.01, hint_audit="structural", max_calls=9)
    assert override["policy"]["max_policy_actions"] == 2
    assert override["policy"]["max_item_cost_usd"] == 0.1
    assert override["policy"]["voc_epsilon"] == 0.01 and override["policy"]["hint_audit"] == "structural"
    assert override["policy"]["max_calls"] == 9
    assert selected["plan_id"] != maximal["plan_id"]
    from mathmage.team.artifacts import load_artifacts
    assert load_artifacts(path)["mu_star"] == 7.0
    assert load_artifacts(path)["policy_config"] == frozen_limits


def test_live_runtime_drift_requires_remeasurement(isolated_runner):
    runtime = {"available": True, "backend": "docker", "image_id": "sha256:one", "sandbox_image": "fixture"}
    artifact = {"roster": "sat", "mode": "live", "provenance": {
        "registry_sha256": "items.parquet-digest", "models_sha256": "models.yaml-digest",
        "answer_vault_sha256": "answers.parquet-digest",
        "prompt_hashes": runner.prompt_hashes(), "protocol_sha256": "source-v1", "tools_runtime": runtime}}
    runner.validate_artifact_runtime(artifact, "sat", {**runtime, "requested": True})
    with pytest.raises(ValueError, match="tools_runtime image_id"):
        runner.validate_artifact_runtime(artifact, "sat", {**runtime, "requested": True, "image_id": "sha256:two"})
    # The explicitly declared no-tools control removes evidence and does not require a different calibration run.
    runner.validate_artifact_runtime(artifact, "sat", {"requested": False})


def test_live_uncalibrated_pilot_requires_rule_router(tmp_path, isolated_runner):
    with pytest.raises(ValueError, match="--router rule"):
        runner.run_study("pilot", live=True, estimate=True)
    assert FakeClient.instances == []


def test_ablations_include_the_rule_router_control():
    assert "R1" in runner.STUDIES["ablations"][3]


def test_estimate_uses_policy_horizon_call_caps_and_actual_tool_availability(isolated_runner):
    small = runner.run_study("pilot", estimate=True, max_policy_actions=0)
    large = runner.run_study("pilot", estimate=True, max_policy_actions=4)
    capped = runner.run_study("pilot", estimate=True, max_policy_actions=4, max_calls=5)
    no_tools = runner.run_study("pilot", estimate=True, conditions=["A7"], max_policy_actions=0)
    rule = runner.run_study("pilot", estimate=True, router="rule", max_policy_actions=0)
    costs = lambda p: p["cost_estimate"]
    assert costs(small)["request_ceilings"][0]["stage0_call_ceiling"] == 7
    assert costs(no_tools)["request_ceilings"][0]["stage0_call_ceiling"] == 3
    assert costs(large)["additional_dispatch_call_ceiling"] > costs(small)["additional_dispatch_call_ceiling"]
    assert costs(capped)["additional_dispatch_call_ceiling"] == 5
    assert costs(rule)["request_ceilings"][0]["router"] == "rule"
    assert costs(rule)["additional_dispatch_call_ceiling"] == 40
    assert costs(large)["expected_usd"] > costs(small)["expected_usd"]
    assert costs(large)["total_usd_ceiling"] is None
    assert large["pricing"]["o3-mini-low"]["max_output_tokens"] == 25000
    assert FakeClient.instances == []


def test_remaining_budget_is_read_only_advisory_and_does_not_change_plan_id(tmp_path, isolated_runner, monkeypatch):
    from mathmage.config import BudgetConfig

    budget = BudgetConfig(global_cap_usd=75.0, experiments={"pilot": 1.0, "offline": 14.0})
    monkeypatch.setattr(runner, "load_budget", lambda: budget)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr("mathmage.llm.ledger.DEFAULT_LEDGER", ledger)
    first = runner.run_study("pilot", estimate=True)
    assert not ledger.exists() and not ledger.with_suffix(".jsonl.lock").exists()
    assert first["budget_review"]["global_remaining_usd"] == 75.0
    ledger.write_text(json.dumps({"experiment": "offline", "cost_usd": 3.0}) + "\n")
    second = runner.run_study("pilot", estimate=True)
    assert first["plan_id"] == second["plan_id"]
    assert second["budget_review"]["global_remaining_usd"] == 72.0
    assert second["budget_review"]["offline_shared_remaining_usd"] == pytest.approx(
        budget.experiments["offline"] * (1 + budget.overrun_tolerance) - 3.0)
    assert FakeClient.instances == []


def test_checkpoint_reuse_ceiling_matches_execution_order(isolated_runner):
    shared = runner.run_study("pilot", estimate=True, conditions=["full", "R1"], max_policy_actions=0)
    separate = runner.run_study("pilot", estimate=True, conditions=["R1", "full"], max_policy_actions=0)
    a, b = shared["cost_estimate"], separate["cost_estimate"]
    assert a["request_ceilings"][1]["reuses_full_checkpoint"]
    assert not any(row["reuses_full_checkpoint"] for row in b["request_ceilings"])
    assert a["additional_dispatch_call_ceiling"] < b["additional_dispatch_call_ceiling"]


def test_changed_answer_vault_prevents_resume_before_regrading(tmp_path, isolated_runner, monkeypatch):
    runner.run_study("pilot", output=tmp_path)
    assert len(FakeEngine.calls) == 1
    original = runner.file_digest
    monkeypatch.setattr(runner, "file_digest", lambda path: "changed-vault" if Path(path).name == "answers.parquet" else original(path))
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_study("pilot", output=tmp_path)
    assert len(FakeEngine.calls) == 1


def test_protocol_digest_ignores_presentation_modules_only():
    import ast
    from pathlib import Path
    from mathmage.experiments import runner

    package = Path(runner.__file__).resolve().parents[1]
    assert all((package / name).exists() for name in runner.PRESENTATION_ONLY)
    excluded = {"mathmage." + name[:-3].replace("/", ".") for name in runner.PRESENTATION_ONLY}
    for path in package.rglob("*.py"):
        relative = path.relative_to(package).as_posix()
        if relative in runner.PRESENTATION_ONLY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module} | {f"{node.module}.{alias.name}" for alias in node.names}
            elif isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            else:
                continue
            assert not names & excluded, f"{relative} imports presentation-only code; it must stay in the protocol"
    assert "analysis/frontier.py" not in runner.PRESENTATION_ONLY   # selects the operating point
