"""Research-relevant atlas checks: finite-sample limits, horizons, and provenance."""

import pandas as pd
import pytest

from mathmage.analysis.atlas import analyze_attempts, posterior_zones


def test_posterior_uses_observed_n_and_distinct_sampling_horizon():
    result = posterior_zones(0, 4, tau=0.5, horizon_k=8)
    assert result["tau_k"] == pytest.approx(1 - 0.5 ** (1 / 8))
    assert result["posterior_mean"] == pytest.approx(1 / 6)
    assert result["zero_success_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 4))
    assert result["zero_success_upper_bound"] > 0.5
    assert sum(result[f"p_{z}"] for z in ("autonomous", "latent", "unassisted_below_sampling_threshold")) == pytest.approx(1)
    changed = posterior_zones(0, 4, horizon_k=16)
    assert changed["posterior_mean"] == result["posterior_mean"]
    assert changed["zero_success_upper_bound"] == result["zero_success_upper_bound"]
    assert changed["p_unassisted_below_sampling_threshold"] < result["p_unassisted_below_sampling_threshold"]


def test_strong_autonomous_evidence_and_sparse_evidence():
    assert posterior_zones(4, 4)["inferred_zone"] == "autonomous"
    assert posterior_zones(0, 4)["inferred_zone"] == "inconclusive"
    assert posterior_zones(0, 100)["inferred_zone"] == "unassisted_below_sampling_threshold"
    for successes, trials in [(-1, 4), (5, 4), (0, 0)]:
        with pytest.raises(ValueError):
            posterior_zones(successes, trials)


def _attempts():
    return pd.DataFrame([
        {"problem_idx": item, "model_name": "public model", "model_config": config,
         "idx_answer": attempt, "correct": correct}
        for item, config, outcomes in [(1, "config-a", [True, False]), (2, "config-a", [True] * 4),
                                        (1, "config-b", [False, False])]
        for attempt, correct in enumerate(outcomes)
    ])


def test_atlas_preserves_configurations_and_reports_real_counts():
    frame = _attempts()
    frame.attrs["source_revision"] = "test-revision"
    tables, report = analyze_attempts({"aime_2025": frame}, bootstrap_samples=30)
    assert len(tables["item_posteriors"]) == 3
    assert len(tables["model_summary"]) == 2
    assert report["n_analyzed_attempts"] == 8
    assert report["n_public_configurations"] == 2
    assert report["sources"]["aime_2025"]["revision"] == "test-revision"
    a = tables["model_summary"].query("model_config == 'config-a'").iloc[0]
    assert a.mean_item_pass1 == pytest.approx(0.75)
    assert a.attempt_accuracy == pytest.approx(5 / 6)
    sensitivity = tables["threshold_sensitivity"]
    assert set(sensitivity.horizon_k) == {4, 8, 16}
    assert set(sensitivity.tau) == {0.4, 0.5, 0.6}
    assert not any("proximal" in c or "unreachable" in c for c in tables["item_posteriors"].columns)


def test_duplicate_attempts_and_nonboolean_labels_are_rejected():
    frame = _attempts()
    with pytest.raises(ValueError, match="duplicate"):
        analyze_attempts({"aime_2025": pd.concat([frame, frame.iloc[:1]])})
    frame["correct"] = frame.correct.astype(str)
    with pytest.raises(ValueError, match="booleans"):
        analyze_attempts({"aime_2025": frame})


def test_known_erratum_is_excluded_and_disclosed():
    frame = _attempts()
    frame["problem_idx"] = frame.problem_idx.replace({1: 10})
    tables, report = analyze_attempts({"aime_2026": frame}, bootstrap_samples=0)
    assert report["source_counts"]["aime_2026"]["excluded_errata_attempts"] == 4
    assert tables["item_posteriors"].item_id.tolist() == ["aime_2026/02"]
