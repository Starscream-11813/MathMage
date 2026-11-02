"""Offline calibration must link anchors to the actual shared item."""

import json
import re

import pytest

from mathmage.experiments.offline import _merge_item_records, offline_phase
from mathmage.psychometrics.calibrate import ItemRecord, Response
from mathmage.data.registry import Item, Registry


def test_anchor_and_live_observations_share_one_item_without_mutating_inputs():
    actual = ItemRecord("train/1", "aime_integer", [Response("actual", "17", True)], "family1")
    anchor = ItemRecord("train/1", "aime_integer", [Response("anchor", "23", False)], "family1")
    other = ItemRecord("train/2", "aime_integer", [Response("actual", None, False)], "family2")
    merged = _merge_item_records([anchor, other, actual])
    assert len(merged) == 2
    assert {r.agent for r in merged[0].responses} == {"actual", "anchor"}
    assert len(actual.responses) == len(anchor.responses) == 1
    assert merged[1].responses[0].answer is None
    with pytest.raises(ValueError, match="conflicting metadata"):
        _merge_item_records([actual, ItemRecord("train/1", "exact", [], "family1")])


@pytest.mark.parametrize("kwargs", [{"extra_samples": -1}, {"extra_samples": True},
                                   {"policy_actions": -1}, {"policy_actions": 1.2}])
def test_invalid_collection_limits_fail_before_registry_or_client(kwargs):
    with pytest.raises(ValueError, match="nonnegative integer"):
        offline_phase(**kwargs)


def test_public_anchor_content_is_bound_to_plan_before_any_model_client(tmp_path, monkeypatch):
    from mathmage.experiments import offline, runner
    from mathmage.llm.client import LLMClient

    registry = Registry([
        Item(f"fixture/{split}", "aime_2023", "fixture", "Find two plus two.", "aime_integer",
             "algebra", f"family-{split}", split, "2023-02", "olympiad", "olympiad")
        for split in ("train", "dev")])
    for name in ("items.parquet", "answers.parquet"):
        (tmp_path/name).write_bytes(b"fixture")
    monkeypatch.setattr(offline, "load_registry", lambda: registry)
    monkeypatch.setattr(offline.paths, "REGISTRY", tmp_path)
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "fixture-protocol")
    monkeypatch.setattr(runner, "tools_runtime", lambda *args: {"requested": True, "available": False})
    monkeypatch.setattr(LLMClient, "__init__", lambda *args, **kw: pytest.fail("model client created"))
    anchor = ItemRecord("fixture/train", "aime_integer", [Response("anchor", "4", True)], "family-train")
    monkeypatch.setattr(offline, "_public_log_records", lambda *args: [anchor])
    monkeypatch.setattr(offline, "_public_kappa_panels", lambda *args: [])
    first = offline_phase(train_limit=1, dev_limit=1, estimate=True, policy_actions=3)
    assert first["public_anchor_snapshot"]["n_items"] == 1
    assert first["policy_config"]["max_policy_actions"] == 3
    assert first["planning_calls"]["train_hint_replay_upper"] == 9
    anchor.responses[0].answer = "5"
    second = offline_phase(train_limit=1, dev_limit=1, estimate=True, policy_actions=3)
    assert first["plan_id"] != second["plan_id"]
    assert first["public_anchor_snapshot"]["sha256"] != second["public_anchor_snapshot"]["sha256"]


@pytest.mark.parametrize("kwargs, message", [
    ({"max_calls": 0}, "positive integer"),
    ({"max_calls": True}, "positive integer"),
    ({"max_calls": 2.5}, "positive integer"),
    ({"max_item_cost_usd": 0}, "finite and positive"),
    ({"max_item_cost_usd": float("inf")}, "finite and positive"),
    ({"max_item_cost_usd": True}, "finite and positive"),
    ({"voc_epsilon": -0.1}, "finite and nonnegative"),
    ({"voc_epsilon": float("nan")}, "finite and nonnegative"),
    ({"voc_epsilon": False}, "finite and nonnegative"),
    ({"hint_audit": "unchecked"}, "structural or residual"),
    ({"hint_audit": None}, "structural or residual"),
])
def test_invalid_policy_caps_precede_registry_and_client(monkeypatch, kwargs, message):
    from mathmage.experiments import offline
    from mathmage.llm.client import LLMClient

    monkeypatch.setattr(offline, "load_registry", lambda: pytest.fail("registry opened"))
    monkeypatch.setattr(LLMClient, "__init__", lambda *args, **kw: pytest.fail("model client created"))
    with pytest.raises(ValueError, match=message):
        offline_phase(live=True, **kwargs)


def test_custom_policy_caps_are_frozen_and_used_for_development(tmp_path, monkeypatch):
    from mathmage.experiments import offline, runner
    from mathmage.experiments.artifacts import read_frozen
    from mathmage.grading import verify
    from mathmage.llm.mock import MockBackend
    from mathmage.team import TeamEngine
    from mathmage.tools import DockerSandbox

    registry = Registry([
        Item(f"fixture/{split}", "aime_2023", "fixture", "Find two plus two.", "aime_integer",
             "algebra", f"family-{split}", split, "2023-02", "olympiad", "olympiad")
        for split in ("train", "dev")])
    for name in ("items.parquet", "answers.parquet"):
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(offline, "load_registry", lambda: registry)
    monkeypatch.setattr(offline.paths, "REGISTRY", tmp_path)
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "fixture-protocol")
    monkeypatch.setattr(DockerSandbox, "available", property(lambda self: False))
    monkeypatch.setattr(verify, "grade_answer", lambda iid, answer, *a, **kw:
                        verify.Grade(iid, answer, bool(answer == "4"), "test_fixture"))
    original_complete = MockBackend.complete

    def fixture_complete(self, settings, messages, response_format, spec):
        response, count = original_complete(self, settings, messages, response_format, spec)
        if "CERTIFICATE" in response.text:
            response.text = re.sub(r"\\boxed\{\d+\}", lambda _: r"\boxed{4}", response.text)
        return response, count

    monkeypatch.setattr(MockBackend, "complete", fixture_complete)
    original_run = TeamEngine.run
    dev_configs = []

    def capture_run(self, item, *args, **kwargs):
        dev_configs.append(self.config)
        return original_run(self, item, *args, **kwargs)

    monkeypatch.setattr(TeamEngine, "run", capture_run)
    options = dict(train_limit=1, dev_limit=1, use_public_logs=False, policy_actions=1,
                   max_calls=5, max_item_cost_usd=0.20, voc_epsilon=0.004, hint_audit="residual")
    plan = offline_phase(estimate=True, **options)
    assert plan["planning_calls"]["dev_stage0_and_program"] == 5
    assert plan["planning_calls"]["dev_policy_maximal_run_upper"] == 0
    expected = {"max_policy_actions": 1, "max_calls": 5, "max_item_cost_usd": 0.20,
                "voc_epsilon": 0.004, "hint_audit": "residual"}
    assert plan["policy_config"] == expected
    changed = offline_phase(estimate=True, **{**options, "max_item_cost_usd": 0.30})
    assert changed["plan_id"] != plan["plan_id"]
    artifact_path = offline_phase(output=tmp_path / "run", **options)
    artifact = read_frozen(artifact_path)
    assert artifact["policy_config"] == expected
    manifest = json.loads((artifact_path.parent / "manifest.json").read_text())
    assert manifest["policy_config"] == expected
    assert dev_configs and all({key: getattr(config, key) for key in expected} == expected
                               for config in dev_configs)
    assert all(config.mu == 0.0 for config in dev_configs)
    development = json.loads((artifact_path.parent / "development_evidence.json").read_text())
    assert all(row["calls"] <= 5 for row in development["policy_rows"])


def test_offline_cli_passes_custom_policy_caps(monkeypatch):
    from mathmage.experiments import offline

    captured = []
    monkeypatch.setattr(offline, "offline_phase", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr("sys.argv", ["offline_phase.py", "--estimate", "--max-calls", "20",
        "--max-item-cost-usd", "0.20", "--voc-epsilon", "0.003", "--hint-audit", "residual"])
    offline.main()
    assert captured[0]["max_calls"] == 20
    assert captured[0]["max_item_cost_usd"] == 0.20
    assert captured[0]["voc_epsilon"] == 0.003
    assert captured[0]["hint_audit"] == "residual"
