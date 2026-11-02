import pytest

from mathmage.analysis.metrics import decomposition, macro_accuracy, macro_decomposition
from mathmage.analysis.stats import cluster_interval, paired_contrast, holm
from mathmage.experiments.artifacts import freeze, read_frozen


def toy():
    return [dict(item_id=str(i), family_id=str(i), benchmark="a", seed=0,
                 covered=c, initial_correct=a, final_correct=b)
            for i, (c, a, b) in enumerate([(True, True, True), (True, True, False),
                                           (True, False, True), (False, False, True),
                                           (False, False, False)])]


def test_hand_computed_decomposition_and_rescue_harm():
    m = decomposition(toy())
    assert m["coverage"] == 3 / 5
    assert m["selection"] == 2 / 3
    assert m["generation"] == 1 / 2
    assert m["harm"] == 1 / 2
    assert m["rescue"] == 2 / 3
    assert m["accuracy"] == pytest.approx(m["decomposed_accuracy"])
    assert m["net_gain"] == pytest.approx(m["rescue_harm_gain"])


def test_undefined_denominators_are_null_not_zero():
    m = decomposition([toy()[0]])
    assert m["generation"] is None and m["rescue"] is None
    assert decomposition([])["accuracy"] is None


def test_macro_not_micro_and_deterministic_cluster_bootstrap():
    rows = toy() + [{**toy()[0], "item_id": "extra", "family_id": "extra", "benchmark": "b"}]
    assert macro_accuracy(rows) == .8
    assert cluster_interval(rows, samples=100) == cluster_interval(rows, samples=100)


def test_paired_requires_same_units_and_omits_invalid_mcnemar():
    a = toy()
    b = [{**r, "final_correct": False} for r in a]
    assert paired_contrast(a, b, samples=100)["wins"] == 3
    with pytest.raises(ValueError):
        paired_contrast(a, b[:-1])
    repeated_a = a + [{**r, "seed": 1} for r in a]
    assert paired_contrast(repeated_a, repeated_a, samples=100)["mcnemar_exact_p"] is None
    assert holm([.01, .04, .03]) == [.03, .06, .06]


def test_artifact_integrity_and_mock_live_separation(tmp_path):
    p = tmp_path / "artifact.json"
    freeze({"mode": "mock", "provenance": {"train_ids": ["a"], "dev_ids": ["b"]}, "cards": []}, p)
    assert read_frozen(p)["mode"] == "mock"
    with pytest.raises(ValueError):
        read_frozen(p, live=True)
    p.write_text(p.read_text().replace('"mock"', '"live"'))
    with pytest.raises(ValueError):
        read_frozen(p)


def test_macro_decomposition_preserves_both_accounting_identities():
    rows = toy() + [{**toy()[0], "item_id": "b0", "family_id": "b0", "benchmark": "b"}]
    macro, micro = macro_decomposition(rows), decomposition(rows)
    assert macro["accuracy"] == .8
    assert micro["accuracy"] == 4 / 6
    assert macro["accuracy"] == pytest.approx(macro["decomposed_accuracy"])
    assert macro["net_gain"] == pytest.approx(macro["rescue_harm_gain"])
    assert macro["selection"] == pytest.approx(.875)
    assert macro["n"] == 6 and macro["aggregation"] == "benchmark_macro"
    assert micro["cost_usd_per_problem"] is None


def test_bootstrap_rejects_partially_missing_or_declared_absent_seeds():
    rows = toy() + [{**toy()[0], "seed": 1}]
    with pytest.raises(ValueError, match="incomplete seed panel"):
        cluster_interval(rows, samples=20)
    with pytest.raises(ValueError, match="incomplete seed panel"):
        cluster_interval(toy(), samples=20, expected_seeds={"a": [0, 1]})
    with pytest.raises(ValueError, match="schedule must match"):
        cluster_interval(toy(), samples=20, expected_seeds={"other": [0]})


def test_bootstrap_supports_different_benchmark_seed_schedules_and_shared_family():
    base = toy()[0]
    rows = [{**base, "benchmark": "a", "seed": s} for s in range(3)] + [{**base, "benchmark": "b", "seed": 0}]
    def check_draw(draw):
        assert len([r for r in draw if r["benchmark"] == "a"]) == 3
        assert len([r for r in draw if r["benchmark"] == "b"]) == 1
        return macro_accuracy(draw)
    interval = cluster_interval(rows, statistic=check_draw, samples=25,
                                expected_seeds={"a": [0, 1, 2], "b": [0]})
    assert interval["samples"] == 25
    assert interval["seed_panels"] == {"a": [0, 1, 2], "b": [0]}


def test_report_separates_macro_micro_and_service_from_wall_latency(tmp_path):
    from mathmage.analysis.make_tables import build_report
    rows = [{**r, "condition": "full", "mode": "mock", "resource_latency_s": 100.,
             "wall_latency_s": 3., "tools_status": "unavailable"} for r in toy()]
    rows.append({**rows[0], "item_id": "b0", "family_id": "b0", "benchmark": "b"})
    report = build_report(rows, tmp_path, samples=20)
    table = {r["benchmark"]: r for r in report["accuracy"]}
    assert table["macro_average"]["accuracy"] == .8
    assert table["micro_average"]["accuracy"] == 4 / 6
    assert table["a"]["resource_service_latency_median_s"] == 100
    assert table["a"]["wall_latency_median_s"] == 3
    assert report["tools_status"] == {"full": {"unavailable": 6}}
