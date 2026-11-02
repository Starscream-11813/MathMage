"""Offline learning boundaries, family weighting, and resumable integrity."""

import json
import re
from types import SimpleNamespace

import pytest

from mathmage.data.registry import Item, Registry
from mathmage.experiments.artifacts import read_frozen
from mathmage.experiments.offline import _delta_interval, _sample_split, _strategy_scores, offline_phase


def _fixture_registry(tmp_path, monkeypatch):
    from mathmage.experiments import offline

    registry = Registry([
        Item(f"fixture/{split}-{index}", benchmark, "fixture", "Compute the sum of two and two.",
             "aime_integer", "algebra", f"family-{split}-{index}", split, "2023-02", "olympiad", "olympiad")
        for split in ("train", "dev")
        for index, benchmark in enumerate(("aime_2023", "hmmt_feb_2025"))
    ])
    registry_path = tmp_path / "registry"
    registry_path.mkdir()
    (registry_path / "items.parquet").write_bytes(b"fixture registry hash input")
    (registry_path / "answers.parquet").write_bytes(b"REFERENCE_SECRET_NOT_FOR_PROMPTS")
    monkeypatch.setattr(offline, "load_registry", lambda: registry)
    monkeypatch.setattr(offline, "_public_log_records", lambda *args: [])
    monkeypatch.setattr(offline, "_public_kappa_panels", lambda *args: [])
    monkeypatch.setattr(offline.paths, "REGISTRY", registry_path)


def test_training_sample_is_balanced_deterministic_and_excludes_test():
    items = [SimpleNamespace(item_id=f"{bench}/{i}", benchmark=bench, split=split)
             for bench, split in [("aime_2023", "train"), ("hmmt_feb_2025", "train"), ("aime_2026", "test")]
             for i in range(5)]
    first = _sample_split(Registry(items), "train", 2)
    second = _sample_split(Registry(reversed(items)), "train", 2)
    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert {item.benchmark for item in first} == {"aime_2023", "hmmt_feb_2025"}
    assert all(item.split == "train" for item in first)
    with pytest.raises(ValueError):
        _sample_split(Registry(items), "train", -1)


def test_strategy_selection_weights_families_and_penalizes_harm_twice():
    rows = []
    for family, initial, final, repeats in [("duplicate-family", False, True, 3), ("other-family", True, False, 1)]:
        rows.extend({"family_id": family, "zone": "P", "strategy": "risky", "initial_correct": initial,
                     "final_correct": final, "cost_usd": 1.0} for _ in range(repeats))
        rows.append({"family_id": family, "zone": "P", "strategy": "safe", "initial_correct": initial,
                     "final_correct": initial, "cost_usd": 1.0})
    ranked = _strategy_scores(rows, "P", ("risky", "safe"))
    assert ranked[0]["strategy"] == "safe"
    assert ranked[1]["utility"] == pytest.approx(-0.5)
    assert ranked[1]["n_families"] == 2
    assert _strategy_scores(rows, "A", ("risky", "safe")) == []
    assert _delta_interval([])["lift"] is None
    assert _delta_interval([1.0])["low"] is None


def test_mock_offline_cards_replay_missing_labels_and_resume_integrity(tmp_path, monkeypatch):
    from mathmage.experiments import runner
    from mathmage.grading import verify
    from mathmage.llm.mock import MockBackend
    from mathmage.tools import DockerSandbox

    # The regression suite needs no downloaded benchmark files or vault values.
    _fixture_registry(tmp_path, monkeypatch)

    # Force the card-generation branch with fixture correctness. This validates
    # orchestration only; the resulting artifact must remain explicitly mock.
    monkeypatch.setattr(verify, "grade_answer", lambda iid, answer, *a, **kw:
                        verify.Grade(iid, answer, True, "test_fixture"))
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "fixture-protocol")
    monkeypatch.setattr(DockerSandbox, "available", property(lambda self: False))
    prompts = []
    original = MockBackend.complete

    def capture(self, settings, messages, response_format, spec):
        prompts.extend(message["content"] for message in messages)
        response, count = original(self, settings, messages, response_format, spec)
        # The fixture grader above marks every usable answer correct. Keep one
        # answer-equivalence cluster so it represents a possible training item.
        if "CERTIFICATE" in response.text:
            response.text = re.sub(r"\\boxed\{\d+\}", lambda _: r"\boxed{4}", response.text)
        return response, count

    monkeypatch.setattr(MockBackend, "complete", capture)
    artifact_path = offline_phase(train_limit=2, dev_limit=2, output=tmp_path)
    artifact = read_frozen(artifact_path)
    assert artifact["mode"] == "mock"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert artifact["provenance"]["public_kappa_snapshot"] == manifest["public_kappa_snapshot"]
    assert artifact["cards"]
    assert all(card["source_item_id"] in artifact["provenance"]["train_ids"] for card in artifact["cards"])
    assert all(not values for values in artifact["tutoring_matrix"].values())
    assert artifact["discernment"] == {}
    assert "unmeasured" in artifact["measurements"]["zone_calibration"]
    assert "unmeasured" in artifact["measurements"]["empirical_hint_leakage"]
    assert "not run" in artifact["measurements"]["reflection_tuning"]
    assert "tutoring_trials" not in artifact["measurements"]
    assert "discernment_trials" not in artifact["measurements"]
    serialized = json.dumps(artifact)
    assert '"final_correct"' not in serialized and '"self_revision_correct"' not in serialized
    assert all('"final_correct"' not in prompt and '"self_revision_correct"' not in prompt for prompt in prompts)
    assert all("REFERENCE_SECRET_NOT_FOR_PROMPTS" not in prompt for prompt in prompts)
    assert (tmp_path / "training_evidence.json").exists()
    with pytest.raises(ValueError, match="live"):
        read_frozen(artifact_path, live=True)

    # Resuming uses sealed observations, including the same card-aware stage0,
    # and performs no additional backend calls.
    prompts.clear()
    assert offline_phase(train_limit=2, dev_limit=2, output=tmp_path) == artifact_path
    assert prompts == []
    assert json.loads((tmp_path / "execution.json").read_text())["client"]["requests"] == 0
    observation = next((tmp_path / "observations").glob("*.json"))
    tampered = json.loads(observation.read_text())
    tampered["data"]["tampered"] = True
    observation.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash changed"):
        offline_phase(train_limit=2, dev_limit=2, output=tmp_path)


def test_live_sandbox_guard_precedes_client_creation_even_with_reviewed_plan(tmp_path, monkeypatch):
    from mathmage.experiments import runner
    from mathmage.llm.client import LLMClient
    from mathmage.tools import DockerSandbox

    _fixture_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_protocol_digest", lambda: "fixture-protocol")
    monkeypatch.setattr(DockerSandbox, "available", property(lambda self: False))
    plan = offline_phase(train_limit=2, dev_limit=2, output=tmp_path, live=True, estimate=True)
    assert not plan["tools_runtime"]["available"]
    monkeypatch.setattr(LLMClient, "__init__", lambda *args, **kwargs: pytest.fail("paid client created"))
    with pytest.raises(ValueError, match="prepared Docker sandbox"):
        offline_phase(train_limit=2, dev_limit=2, output=tmp_path, live=True, approved_plan=plan["plan_id"])
