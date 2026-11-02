"""Offline calibration recovers generating parameters and yields calibrated posteriors."""

import numpy as np
import pytest

from mathmage.psychometrics.calibrate import (CalibrationData, CoachingRecord, EvidenceRecord, ItemRecord, Response, calibrate,
                                              fit_coaching_lifts, fit_evidence, fit_irt, fit_kappa, fit_kappa_mixture,
                                              reliability)


def simulate(n_items, theta, kappa=0.15, b_sd=1.6, seed=0, abstain=0.0):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        b = rng.normal(0.0, b_sd)
        truth = int(rng.integers(0, 1000))
        responses, wrong_seen = [], []
        for agent, th in theta.items():
            if rng.random() < abstain:
                responses.append(Response(agent, None, False))
                continue
            if rng.random() < 1 / (1 + np.exp(-(th - b))):
                responses.append(Response(agent, str(truth), True))
            else:
                if wrong_seen and rng.random() < kappa * len(set(wrong_seen)):
                    ans = int(rng.choice(sorted(set(wrong_seen))))
                else:
                    ans = (truth + int(rng.integers(1, 999))) % 1000
                wrong_seen.append(ans)
                responses.append(Response(agent, str(ans), False))
        items.append(ItemRecord(f"sim/{i}", "aime_integer", responses))
    return items


THETA = {"o3-mini-low": 0.9, "haiku-4.5": 0.1, "deepseek-v3": -0.7, "tool": 0.3}


def test_irt_recovers_abilities_and_spread():
    items = simulate(500, THETA, seed=1)
    fit = fit_irt(items)
    assert fit["a"] == 1 and not fit["correlation_estimated"]
    est = fit["theta"]
    # Ordering is exact and values are close at this sample size.
    order = sorted(THETA, key=THETA.get)
    assert sorted(est, key=est.get) == order
    for k, v in THETA.items():
        assert abs(est[k] - v) < 0.35, (k, est[k], v)
    assert abs(fit["b_sd"] - 1.6) < 0.45 and fit["converged"]


def test_small_training_set_keeps_the_ability_ordering_of_well_separated_agents():
    items = simulate(60, {"strong": 1.2, "weak": -1.0}, seed=3)
    est = fit_irt(items)["theta"]
    assert est["strong"] > est["weak"]


def test_kappa_estimate_is_reasonable():
    items = simulate(600, THETA, kappa=0.2, seed=2)
    kappa, kappa_same = fit_kappa(items)
    assert 0.08 < kappa < 0.35
    assert kappa_same >= kappa


def test_calibrated_posterior_is_well_calibrated_out_of_sample():
    train = simulate(300, THETA, seed=4, abstain=0.03)
    test = simulate(400, THETA, seed=5, abstain=0.03)
    data = CalibrationData(items=train, judge_trials=[True] * 14 + [False] * 6,
                           hint_acceptance_trials=[True] * 2 + [False] * 6)
    params, diag = calibrate(data, provenance={"train_ids": [i.item_id for i in train]})
    rel = reliability(test, params)
    assert params.calibrated and diag["n_items"] == 300
    assert rel["ece"] < 0.06, rel
    assert abs(params.judge_accuracy - 0.7) < 0.05
    assert params.hint_acceptance == pytest.approx(6 / 13)
    assert diag["n_hint_acceptance_trials"] == 8


def test_reliability_includes_items_with_no_usable_answer():
    from mathmage.team.posterior import AgentProfile, PosteriorParams

    item = ItemRecord("missing/1", "aime_integer", [Response("a", None, False)])
    result = reliability([item], PosteriorParams({"a": AgentProfile(0, abstain=1)}))
    assert result["n"] == 1 and result["accuracy"] == 0 and result["mean_confidence"] == 0


def test_evidence_likelihoods_shrink_toward_priors_and_learn_direction():
    records = ([EvidenceRecord("review_objection_verified", False)] * 30
               + [EvidenceRecord("review_valid", True)] * 30
               + [EvidenceRecord("review_objection_unverified", True)] * 3)
    ev = fit_evidence(records)
    pc, pw = ev["review_objection_verified"]
    assert pw > pc                      # verified objections are evidence against a candidate
    vc, vw = ev["review_valid"]
    assert vc > vw
    for kind in ev:
        assert 0 < ev[kind][0] < 1 and 0 < ev[kind][1] < 1


def test_abstentions_do_not_lower_conditional_ability_estimates():
    original = simulate(70, {"strong": 1.1, "weak": -0.8}, seed=52)
    with_abstentions = [ItemRecord(it.item_id, it.answer_format,
                                  it.responses + [Response("strong", None, False)] * 4)
                       for it in original]
    first, second = fit_irt(original), fit_irt(with_abstentions)
    assert first["theta"] == pytest.approx(second["theta"], abs=1e-10)
    assert first["b_sd"] == pytest.approx(second["b_sd"], abs=1e-10)


def test_repeated_responses_can_identify_residual_model_correlation():
    rng = np.random.default_rng(874)
    theta, rho = {"a": 0.8, "b": 0.0, "c": -0.7}, 0.65
    concentration = (1 - rho) / rho
    items = []
    for i in range(180):
        b = rng.normal(0, 1.6)
        responses = []
        for name, th in theta.items():
            p = 1 / (1 + np.exp(-(th - b)))
            latent = rng.beta(concentration * p, concentration * (1 - p))
            for attempt in range(3):
                correct = rng.random() < latent
                responses.append(Response(name, "0" if correct else str(1 + attempt), bool(correct)))
        items.append(ItemRecord(f"correlated/{i}", "aime_integer", responses))
    fit = fit_irt(items)
    assert fit["converged"] and fit["correlation_estimated"]
    assert 0.35 < fit["same_agent_correlation"] < 0.85


def test_calibration_reports_unmeasured_evidence_and_action_channels_as_priors():
    params, diag = calibrate(CalibrationData(items=simulate(45, {"a": 0.8, "b": -0.8}, seed=66)))
    scope = diag["calibration_scope"]
    assert params.none_weight == 1 and scope["same_agent_correlation"] == "prior only"
    assert scope["judge"] == "prior only" and scope["diversify_lift"] == "prior only"
    assert scope["evidence"]["check_passed"] == 0
    assert params.provenance["calibration_scope"] == scope


def test_calibration_rejects_split_records_for_the_same_latent_item():
    records = simulate(2, {"a": 0.5}, seed=12)
    records[1].item_id = records[0].item_id
    with pytest.raises(ValueError, match="merge response draws"):
        calibrate(CalibrationData(items=records))


def test_calibration_rejects_inconsistent_answer_cluster_labels():
    records = [ItemRecord("contradiction/1", "aime_integer", [Response("a", "1", True), Response("b", "1", False)])]
    with pytest.raises(ValueError, match="inconsistent"):
        calibrate(CalibrationData(items=records))


# ---- model v3: trap mixture and tutor-dependent coaching lifts -------------------------

def simulate_collisions(n_items, n_agents, weight, kappa, kappa_trap, p_correct=0.4, seed=0, panels=1):
    """Sequential collision process with a per-item type shared by all of the item's panels."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i in range(n_items):
        k = kappa_trap if rng.random() < weight else kappa
        records = []
        for panel in range(panels):
            wrong_clusters, responses = [], []
            for a in range(n_agents):
                if rng.random() < p_correct:
                    responses.append(Response(f"m{a}", "0", True))
                    continue
                odds = np.full(len(wrong_clusters), k / (1 - k))
                z = 1 + odds.sum()
                u = rng.random() * z
                if u < odds.sum():
                    answer = wrong_clusters[int(np.searchsorted(np.cumsum(odds), u, side="right"))]
                else:
                    answer = str(1 + len(wrong_clusters) + 100 * panel)
                    wrong_clusters.append(answer)
                responses.append(Response(f"m{a}", answer, False))
            records.append(ItemRecord(f"sim/{i}", "aime_integer", responses))
        groups[f"sim/{i}"] = records
    return groups


def test_trap_mixture_recovers_heterogeneous_collisions():
    groups = simulate_collisions(160, 6, weight=0.25, kappa=0.03, kappa_trap=0.45, seed=7, panels=4)
    fit = fit_kappa_mixture(groups)
    assert fit["converged"] and fit["n_items"] > 100
    assert 0.12 < fit["trap_weight"] < 0.4, fit
    assert fit["kappa"] < 0.08 and fit["kappa_trap"] > 0.3, fit
    assert fit["loglik_gain"] > 5


def test_trap_mixture_stays_harmless_without_heterogeneity():
    groups = simulate_collisions(160, 6, weight=0.0, kappa=0.05, kappa_trap=0.05, seed=8, panels=4)
    fit = fit_kappa_mixture(groups)
    # Either the trap type is rare or it behaves like the ordinary type: unanimity is not discounted much.
    assert fit["trap_weight"] * (fit["kappa_trap"] - fit["kappa"]) < 0.05, fit
    assert fit["loglik_gain"] < 3


def test_coaching_lifts_recover_tutor_dependence():
    rng = np.random.default_rng(11)
    theta = {"solver": -0.3, "peer": 0.4}
    items, records = {}, []
    for i in range(700):
        b = rng.normal(0, 1.6)
        peer_right = rng.random() < 1 / (1 + np.exp(-(theta["peer"] - b)))
        solver_right = rng.random() < 1 / (1 + np.exp(-(theta["solver"] - b)))
        items[f"c/{i}"] = ItemRecord(f"c/{i}", "aime_integer", [Response("peer", "0" if peer_right else "1", peer_right),
                                                               Response("solver", "0" if solver_right else "2", solver_right)])
        if solver_right:
            continue                 # replays only coach initially wrong solvers
        lift = 2.5 if peer_right else 0.3
        coached = rng.random() < 1 / (1 + np.exp(-(theta["solver"] + lift - b)))
        records.append(CoachingRecord(f"c/{i}", "solver", bool(coached), bool(peer_right)))
    fit = fit_coaching_lifts(records, items, theta, 1.6, rho=0.0)
    assert fit["n_right"] > 60 and fit["n_wrong"] > 200
    assert fit["coached_right"] - fit["coached_wrong"] > 1.2, fit
    assert abs(fit["coached_wrong"] - 0.3) < 0.6 and abs(fit["coached_right"] - 2.5) < 1.0, fit


def test_calibrate_freezes_trap_and_tutor_dependent_parameters():
    train = simulate(80, THETA, seed=21)
    records = [CoachingRecord(item.item_id, "haiku-4.5", bool(i % 3 == 0), bool(i % 2 == 0))
               for i, item in enumerate(train[:30])]
    extra = [panel for panels in simulate_collisions(20, 4, .3, .03, .5, seed=3, panels=2).values() for panel in panels]
    params, diag = calibrate(CalibrationData(items=train, coaching_records=records, kappa_items=extra))
    assert {"coached", "coached_right", "coached_wrong"} <= set(params.lift)
    assert 0 <= params.trap_weight < 1 and params.kappa <= params.kappa_trap < 1
    assert diag["n_kappa_panels"] == len(extra) and diag["coaching_lifts"]["n"] == 30
    assert "trap" in diag["calibration_scope"]["collision_types"]
    from mathmage.team.posterior import TeamPosterior
    TeamPosterior(params)            # frozen parameters are valid for the v3 posterior


def test_coaching_fit_uses_beta_binomial_difficulty_likelihood_for_repeats():
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    # Three solver failures and two peer successes: repeated outcomes provide
    # less difficulty evidence than five independent trials. Use the closed
    # beta-binomial predictive product as an independent numerical reference.
    theta, rho, b_sd = {"solver": 0.2, "peer": 0.7}, 0.6, 2.0
    item = ItemRecord("repeat/01", "aime_integer", [Response("solver", "1", False)] * 3 +
                      [Response("peer", "0", True)] * 2)
    records = [CoachingRecord(item.item_id, "solver", i == 0, True) for i in range(8)]
    actual = fit_coaching_lifts(records, {item.item_id: item}, theta, b_sd, rho)
    nodes, weights = np.polynomial.hermite_e.hermegauss(61)
    b, concentration = b_sd * nodes, (1 - rho) / rho
    solver_p, peer_p = expit(theta["solver"] - b), expit(theta["peer"] - b)
    failures = np.prod([(concentration * (1 - solver_p) + i) / (concentration + i) for i in range(3)], axis=0)
    successes = peer_p * (concentration * peer_p + 1) / (concentration + 1)
    posterior = weights * failures * successes
    posterior /= posterior.sum()

    def reference_loss(lift):
        p = np.dot(posterior, concentration * expit(theta["solver"] + lift - b) / (concentration + 3))
        return -np.log(p) - 7 * np.log1p(-p) + 0.5 * ((lift - 0.5) / 1.5) ** 2

    expected = minimize_scalar(reference_loss, bounds=(-4, 6), method="bounded").x
    assert actual["coached"] == pytest.approx(expected, abs=1e-6)
    assert actual["coached_right"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("failure", ["not_converged", "nonfinite"])
def test_calibration_refuses_unreliable_collision_fit(monkeypatch, failure):
    import importlib
    module = importlib.import_module("mathmage.psychometrics.calibrate")
    result = {"converged": failure != "not_converged", "kappa": 0.1, "kappa_same": 0.2,
              "trap_weight": 0.1, "kappa_trap": 0.3, "loglik_gain": 0.0}
    if failure == "nonfinite":
        result["kappa"] = np.nan
    monkeypatch.setattr(module, "fit_irt", lambda _: {"converged": True})
    monkeypatch.setattr(module, "fit_kappa", lambda _: (0.1, 0.2))
    monkeypatch.setattr(module, "fit_kappa_mixture", lambda _: result)
    with pytest.raises(ValueError, match="collision mixture"):
        calibrate(CalibrationData(items=simulate(4, THETA, seed=31)))
