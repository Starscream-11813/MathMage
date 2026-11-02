from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from mathmage.data.registry import Item, Registry
from mathmage.experiments import mechanisms as m
from mathmage.experiments.artifacts import atomic_json, digest, freeze
from mathmage.grading.verify import Grade
from mathmage.team.types import Candidate, TeamResult, ZoneEstimate


def item(i=1, family="family/1", split="dev", benchmark="aime_2024"):
    return Item(f"toy/{i}", benchmark, "fixture", "Compute 2+2.", "aime_integer", "algebra",
                family, split, "2024-02", "olympiad", "olympiad")


def test_assignment_is_stable_and_family_seed_clustered():
    first = m.assigned_arm("z3", "family/1", 0, 17)
    assert first == m.assigned_arm("z3", "family/1", 0, 17)
    assert set(m.assigned_arm("z3", f"family/{i}", 0, 17) for i in range(50)) == set(m.ARMS["z3"])
    manifest = m.mechanism_plan("z3", [item(), item(2)], assignment_seed=17)
    assert {a["assigned_arm"] for a in manifest["assignments"]} == {first}
    assert all(len(a["executed_arms"]) == 1 for a in manifest["assignments"])
    assert "correct" not in json.dumps(manifest["assignments"])
    assert manifest["resource_limits"]["input_token_ceiling"] is None
    assert manifest["resource_limits"]["total_cost_ceiling_usd"] is None
    assert "do not match" in manifest["resource_limits"]["note"]


def test_selection_excludes_training_and_requires_named_test_subset(monkeypatch, tmp_path):
    registry = Registry([item(), item(2, split="train"), item(3, split="test")])
    monkeypatch.setattr(m, "load_registry", lambda: registry)
    assert [i.item_id for i in m.select_items("z3", 10)] == ["toy/1"]
    with pytest.raises(ValueError, match="only Z3"):
        m.select_items("z6", 1, "ablation_45")
    monkeypatch.setattr(m.paths, "REGISTRY", tmp_path)
    atomic_json(tmp_path / "subsets.json", {"ablation_45": ["toy/3"]})
    assert m.select_items("z3", 2, "ablation_45")[0].split == "test"
    atomic_json(tmp_path / "subsets.json", {"ablation_45": ["toy/2"]})
    with pytest.raises(ValueError, match="test items only"):
        m.select_items("z3", 2, "ablation_45")


def test_explicit_population_preserves_order_without_limit_truncation(monkeypatch, tmp_path):
    registry = Registry([item(i, family=f"family/{i}") for i in range(1, 5)]
                        + [item(5, split="train"), item(6, split="test"), item(7, split="test")])
    monkeypatch.setattr(m, "load_registry", lambda: registry)
    requested = ["toy/4", "toy/2", "toy/1"]
    assert [it.item_id for it in m.select_items("z6", 1, item_ids=requested)] == requested
    for invalid in ([], ["toy/1", "toy/1"], "toy/1", [None], [""]):
        with pytest.raises(ValueError, match="nonempty unique"):
            m.select_items("z6", item_ids=invalid)
    for invalid in (["toy/5"], ["toy/6"], ["missing"]):
        with pytest.raises(ValueError, match="declared development"):
            m.select_items("z6", item_ids=invalid)
    monkeypatch.setattr(m.paths, "REGISTRY", tmp_path)
    atomic_json(tmp_path / "subsets.json", {"ablation_45": ["toy/6"]})
    assert m.select_items("z3", subset="ablation_45", item_ids=["toy/6"])[0].split == "test"
    with pytest.raises(ValueError, match="declared development"):
        m.select_items("z3", subset="ablation_45", item_ids=["toy/7"])


def test_mechanism_plan_binds_explicit_population_and_flags_tuning_families(monkeypatch):
    shared = item(1, family="tuning-family")
    related = item(2, family="tuning-family")
    heldout = item(3, family="new-family")
    monkeypatch.setattr(m, "load_registry", lambda: Registry([shared, related, heldout]))
    artifacts = {"provenance": {"train_ids": [], "dev_ids": [shared.item_id]}}
    plan = m.mechanism_plan("z6", [related, heldout], artifacts=artifacts, explicit_item_ids=True)
    selection = plan["selection_provenance"]
    assert selection["item_ids"] == [related.item_id, heldout.item_id]
    assert selection["item_ids_sha256"] == digest(selection["item_ids"])
    assert selection["source"] == "explicit ordered item IDs" and not selection["limit_applies"]
    assert selection["offline_learning_overlap_item_ids"] == []
    assert selection["offline_learning_overlap_family_ids"] == ["tuning-family"]
    assert selection["interpretation"].startswith("exploratory:")
    independent = m.mechanism_plan("z6", [heldout], artifacts=artifacts, explicit_item_ids=True)
    assert independent["selection_provenance"]["offline_learning_overlap_family_ids"] == []
    assert "confirmatory status additionally requires" in independent["selection_provenance"]["interpretation"]
    reordered = m.mechanism_plan("z6", [heldout, related], artifacts=artifacts, explicit_item_ids=True)
    assert reordered["plan_id"] != plan["plan_id"]


def test_forced_strategy_and_review_gate_controls():
    common = {"roster": "sat", "seed": 3, "artifacts": None, "max_calls": 20}
    assert m.arm_config("z3", "graded_coaching", **common).strategy_override == ["graded_coaching"]
    assert m.arm_config("z3", "fresh_verified_search", **common).strategy_override == ["fresh_verified_search"]
    gated = m.arm_config("z6", "gated_review", **common)
    ungated = m.arm_config("z6", "ungated_review", **common)
    assert gated.condition == "full" and ungated.condition == "A3"
    assert gated.strategy_override == ungated.strategy_override == ["evidence_gated_review"]
    assert gated.max_calls == ungated.max_calls == 20
    assert m.arm_config("z3", "graded_coaching", sandbox="local", **common).sandbox == "local"
    assert gated.router == ungated.router == "rule"


class FakeClient:
    instances = []
    def __init__(self, experiment, live=False, **kwargs):
        self.live, self.closed = live, False
        self.instances.append(self)
    def summary(self):
        return {"mode": "live" if self.live else "mock", "cost_usd": 0}
    def close(self):
        self.closed = True


class FakeEngine:
    events = []
    target = None
    def __init__(self, client, config):
        self.config = config
        assert (self.target / "assignments.json").exists()
    def create_checkpoint(self, it):
        self.events.append("checkpoint")
        return {"item_id": it.item_id, "sha256": "checkpoint-hash", "stage0_costs":
                {"calls": 3, "tokens_in": 10, "tokens_out": 5, "resource_cost_usd": .01}}
    def run(self, it, checkpoint=None):
        arm = self.config.strategy_override[0]
        self.events.append(arm)
        assert checkpoint["sha256"] == "checkpoint-hash"
        return TeamResult(it.item_id, self.config.condition, self.config.seed,
            [Candidate("s0-0", "m#0", "o3-mini-low", "fixture", "4", "fixture", "stage0")],
            ["s0-0"], "s0-0", ZoneEstimate("D", {}, True, False, "fixture"), checkpoint, {}, [],
            {"calls": 5, "dispatch_calls": 2, "tokens_in": 15, "tokens_out": 9,
             "resource_cost_usd": .02, "marginal_cost_usd": .005})


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    FakeClient.instances, FakeEngine.events, FakeEngine.target = [], [], tmp_path
    monkeypatch.setattr(m, "select_items", lambda *args: [item()])
    monkeypatch.setattr(m, "_protocol_digest", lambda: "source-v1")
    monkeypatch.setattr("mathmage.llm.LLMClient", FakeClient)
    monkeypatch.setattr("mathmage.team.TeamEngine", FakeEngine)
    original = m.grade_result
    def grade(*args, **kwargs):
        FakeEngine.events.append("grade")
        return original(*args, **kwargs)
    monkeypatch.setattr(m, "grade_result", grade)
    monkeypatch.setattr("mathmage.grading.verify.grade_answer", lambda iid, answer, vault=None:
                        Grade(iid, answer, answer == "4", "fixture"))


def test_paired_continuations_sealed_before_gold_and_resume(tmp_path, isolated):
    m.run_mechanism("z3", paired=True, output=tmp_path)
    assert FakeEngine.events == ["checkpoint", "graded_coaching", "fresh_verified_search", "grade", "grade"]
    raw = [json.loads(p.read_text()) for p in (tmp_path / "agent").glob("*.json")]
    assert all("candidate_grades" not in p["result"] for p in raw)
    assert all(p["run_provenance"]["mode"] == "mock" for p in raw)
    rows = [json.loads(p.read_text()) for p in (tmp_path / "graded").glob("*.json")]
    assert all(r["continuation_tokens_in"] == 5 and r["continuation_tokens_out"] == 4 for r in rows)
    assert all(r["continuation_calls"] == 2 and r["covered"] and not r["pre_blocked"] for r in rows)
    events = FakeEngine.events.copy()
    m.run_mechanism("z3", paired=True, output=tmp_path)
    assert FakeEngine.events == events + ["grade", "grade"]
    assert all(c.closed for c in FakeClient.instances)
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["status"] == "synthetic validation only"
    assert "not claimed" in report["matching_claim"]
    assert report["comparisons"]["blocked"]["risk_difference_first_minus_second"] is None


def test_parallel_default_executes_only_preassigned_arm(tmp_path, isolated):
    m.run_mechanism("z3", output=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assigned = manifest["assignments"][0]["assigned_arm"]
    assert FakeEngine.events == ["checkpoint", assigned, "grade"]
    assert len(list((tmp_path / "agent").glob("*.json"))) == 1


def test_tampered_checkpoint_and_changed_plan_cannot_resume(tmp_path, isolated):
    m.run_mechanism("z3", output=tmp_path)
    with pytest.raises(ValueError, match="manifest changed"):
        m.run_mechanism("z3", max_calls=21, output=tmp_path)
    path = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint = json.loads(path.read_text())
    checkpoint["checkpoint"]["sha256"] = "modified"
    atomic_json(path, checkpoint)
    with pytest.raises(ValueError, match="content hash"):
        m.run_mechanism("z3", output=tmp_path)


def test_changed_answer_vault_prevents_mechanism_resume(tmp_path, isolated, monkeypatch):
    m.run_mechanism("z3", output=tmp_path)
    events = FakeEngine.events.copy()
    original = m.file_digest
    monkeypatch.setattr(m, "file_digest", lambda path: "changed-vault" if Path(path).name == "answers.parquet" else original(path))
    with pytest.raises(ValueError, match="manifest changed"):
        m.run_mechanism("z3", output=tmp_path)
    assert FakeEngine.events == events


def test_live_requires_frozen_live_artifacts_before_client(tmp_path, isolated):
    with pytest.raises(ValueError, match="frozen live"):
        m.run_mechanism("z6", live=True, output=tmp_path)
    path = tmp_path / "artifacts.json"
    freeze({"mode": "mock", "provenance": {"train_ids": [], "dev_ids": []}}, path)
    with pytest.raises(ValueError, match="live models"):
        m.run_mechanism("z6", live=True, artifacts_path=path, output=tmp_path)
    assert FakeClient.instances == []


def test_real_paired_z6_reuses_identical_reviewer_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "select_items", lambda *args: [item()])
    monkeypatch.setattr("mathmage.grading.verify.grade_answer", lambda iid, answer, vault=None:
                        Grade(iid, answer, answer == "4", "fixture"))
    m.run_mechanism("z6", paired=True, output=tmp_path)
    records = [json.loads(line) for line in (tmp_path / "calls" / "e5_zpd.jsonl").read_text().splitlines()]
    reviews = [row for row in records if row.get("role", "").startswith("s0-") and row["prompt_template_id"].startswith("review.")]
    assert len(reviews) == 6
    assert sum(r["cached"] for r in reviews) == 3
    assert len({r["cache_key"] for r in reviews}) == 3
    assert all(r["mode"] == "mock" and r["cost_usd"] == 0 for r in records)


@pytest.mark.parametrize("artifact_args, expected", [([], None), (["--artifacts", "fixture-policy.json"], "fixture-policy.json")])
def test_cli_artifact_argument_matches_runner_signature(monkeypatch, artifact_args, expected):
    import inspect
    signature = inspect.signature(m.run_mechanism)
    captured = []
    def capture(**kwargs):
        signature.bind(**kwargs)  # Detect unsupported CLI keyword names.
        captured.append(kwargs)
    monkeypatch.setattr(m, "run_mechanism", capture)
    m.main(["z3", "--limit", "2", *artifact_args])
    assert len(captured) == 1
    assert captured[0]["artifacts_path"] == expected
    assert "artifacts" not in captured[0]
    assert captured[0]["study"] == "z3" and captured[0]["limit"] == 2
    assert captured[0]["live"] is False


def test_cli_reads_explicit_population_without_sampling(monkeypatch, tmp_path):
    path = tmp_path / "mechanism_ids.json"
    path.write_text(json.dumps(["toy/2", "toy/1"]), encoding="utf-8")
    captured = []
    monkeypatch.setattr(m, "run_mechanism", lambda **kwargs: captured.append(kwargs))
    m.main(["z6", "--item-ids-json", str(path), "--limit", "1", "--estimate"])
    assert captured[0]["item_ids"] == ["toy/2", "toy/1"]
    assert "item_ids_json" not in captured[0]
    assert captured[0]["estimate"] is True
