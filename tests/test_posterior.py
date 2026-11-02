"""The label-free team posterior: behavior, normalization, VOC, and calibration."""

import numpy as np
import pytest

from mathmage.team.posterior import AgentProfile, Evidence, Obs, PosteriorParams, TeamPosterior


def params(**kw):
    agents = {"strong": AgentProfile(1.2), "mid": AgentProfile(0.2), "weak": AgentProfile(-0.8),
              "tool": AgentProfile(0.4, abstain=0.4)}
    return PosteriorParams(agents=agents, **kw)


def test_agreement_raises_confidence_and_disagreement_lowers_it():
    tp = TeamPosterior(params())
    unanimous = tp.compute([Obs("strong", 0), Obs("mid", 0), Obs("weak", 0)])
    split = tp.compute([Obs("strong", 0), Obs("mid", 1), Obs("weak", 1)])
    scattered = tp.compute([Obs("strong", 0), Obs("mid", 1), Obs("weak", 2)])
    assert unanimous.confidence > 0.95
    assert 0.3 < split.confidence < 0.85
    assert scattered.q_none > unanimous.q_none
    assert abs(unanimous.q.sum() + unanimous.q_none - 1) < 1e-9
    # A strong lone dissenter outweighs nothing when the majority is weak: abilities matter.
    assert split.q[0] > 0.2


def test_independent_tool_agreement_and_verified_objection():
    tp = TeamPosterior(params())
    base = [Obs("strong", 0), Obs("mid", 1)]
    without = tp.compute(base)
    with_tool = tp.compute(base + [Obs("tool:checker", 1)])
    assert with_tool.q[1] > without.q[1]
    objection = tp.compute(base, [Evidence("review_objection_verified", 0)])
    assert objection.q[0] < without.q[0]


def test_same_agent_repetition_counts_less_than_independent_agreement():
    tp = TeamPosterior(params())
    repeat = tp.compute([Obs("mid", 0), Obs("mid", 0), Obs("strong", 1)])
    independent = tp.compute([Obs("mid", 0), Obs("weak", 0), Obs("strong", 1)])
    # Two agreeing samples from one model share errors (kappa_same > kappa) and are weaker evidence
    # than agreement between different models, even when the second model is weaker.
    assert repeat.q[0] < independent.q[0] + 0.15


def test_predictive_outcomes_are_a_distribution():
    tp = TeamPosterior(params())
    obs = [Obs("strong", 0), Obs("mid", 1), Obs("weak", 2)]
    result = tp.compute(obs)
    members = {0: ["strong"], 1: ["mid"], 2: ["weak"]}
    outcomes = tp.sample_outcomes(result, "strong", 0.0, 3, members)
    assert abs(sum(p for _, p in outcomes) - 1) < 1e-9
    assert all(p >= 0 for _, p in outcomes)


def test_voc_is_small_when_confident_and_positive_when_split():
    tp = TeamPosterior(params())
    confident = [Obs("strong", 0), Obs("mid", 0), Obs("weak", 0), Obs("tool:x", 0)]
    split = [Obs("strong", 0), Obs("mid", 1)]
    assert tp.voc_sample(confident, [], 1, "strong") < 0.01
    assert tp.voc_sample(split, [], 2, "strong") > 0.03
    # Equal-ability disagreement: a judge pick can change the decision, so it has value.
    even = [Obs("mid", 0), Obs("mid", 1)]
    assert tp.voc_judge(even, [], 2, [0, 1]) > 0.02
    assert tp.voc_evidence(even, [], 2, ["review_objection_verified", "review_objection_unverified",
                                          "review_valid", "review_unsure"], 0) > 0.0
    # Myopic VOC is exactly zero when no single outcome can flip the decision (here a 0.7-accurate
    # judge cannot overturn a strong member's lead), so the policy prefers actions that add answers.
    assert tp.voc_judge(split, [], 2, [0, 1]) == 0.0


def test_member_zones_sum_to_one_and_track_difficulty():
    tp = TeamPosterior(params())
    easy = tp.compute([Obs("strong", 0), Obs("mid", 0), Obs("weak", 0)])
    hard = tp.compute([Obs("strong", -1), Obs("mid", 1), Obs("weak", 2)])
    ze, zh = tp.member_zones(easy, "mid"), tp.member_zones(hard, "mid")
    assert abs(sum(ze.values()) - 1) < 1e-9 and abs(sum(zh.values()) - 1) < 1e-9
    assert ze["A"] > zh["A"]
    assert zh["U"] + zh["P"] > ze["U"] + ze["P"]


def _simulate(rng, theta, b_mean, b_sd, kappa, n_items, n_wrong_pool=50):
    """Draw items from the generative model the posterior assumes."""
    data = []
    for _ in range(n_items):
        b = rng.normal(b_mean, b_sd)
        answers = []
        for th in theta:
            if rng.random() < 1 / (1 + np.exp(-(th - b))):
                answers.append(0)                      # the true answer has label 0
            else:
                previous_wrong = [x for x in answers if x != 0]
                if previous_wrong and rng.random() < kappa * len(set(previous_wrong)):
                    answers.append(rng.choice(sorted(set(previous_wrong))))
                else:
                    answers.append(int(rng.integers(1, n_wrong_pool + 1)) + 100 * len(answers))
        data.append(answers)
    return data


def test_posterior_is_calibrated_on_its_own_generative_model():
    rng = np.random.default_rng(7)
    theta = [1.0, 0.0, -0.6]
    p = PosteriorParams(agents={"a0": AgentProfile(theta[0], abstain=0), "a1": AgentProfile(theta[1], abstain=0),
                                "a2": AgentProfile(theta[2], abstain=0)}, b_mean=0.0, b_sd=1.6, kappa=0.12,
                        kappa_same=0.3, none_weight=1.0)
    tp = TeamPosterior(p)
    conf, correct = [], []
    for answers in _simulate(rng, theta, 0.0, 1.6, 0.12, 2500):
        labels, keys = {}, []
        for x in answers:
            labels.setdefault(x, len(labels))
            keys.append(labels[x])
        res = tp.compute([Obs(f"a{i}", c) for i, c in enumerate(keys)], [], len(labels))
        inv = {v: k for k, v in labels.items()}
        conf.append(res.confidence)
        correct.append(inv[res.best] == 0)
    conf, correct = np.array(conf), np.array(correct, dtype=float)
    # Mean confidence matches realized accuracy, and the reliability curve is close to the diagonal.
    assert abs(conf.mean() - correct.mean()) < 0.03
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01)):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum() > 80:
            assert abs(conf[sel].mean() - correct[sel].mean()) < 0.06, (lo, hi, conf[sel].mean(), correct[sel].mean())


def test_params_round_trip():
    p = params(calibrated=True, provenance={"train_ids": ["x"]})
    q = PosteriorParams.from_dict(p.to_dict())
    assert q.agents["strong"].theta == pytest.approx(1.2) and q.evidence == p.evidence and q.calibrated
    old = p.to_dict()
    old.pop("model_version")
    with pytest.raises(ValueError, match="recalibrate"):
        PosteriorParams.from_dict(old)


@pytest.mark.parametrize("rho", [0.0, 0.4])
@pytest.mark.parametrize("clusters", [0, 1, 3, 12])
def test_predictive_update_is_bayesian_martingale_even_with_many_collisions(rho, clusters):
    """A hypothetical observation cannot systematically move an existing truth probability.

    Tests the full density over difficulty, not just normalization after a silent
    renormalization. It catches inconsistent fresh-answer mass, growing-hypothesis
    priors, independent-abstention updates, and duplicate-agent reinforcement.
    """
    tp = TeamPosterior(params(kappa=0.7, kappa_same=0.9, same_agent_correlation=rho, none_weight=1.7))
    obs = [Obs("mid" if i % 2 else "strong", i) for i in range(clusters)]
    ev = [Evidence("review_valid", 0)] if clusters else []
    result = tp.compute(obs, ev, clusters)
    members = {i: [o.agent] for i, o in enumerate(obs)}
    outcomes = tp.sample_outcomes(result, "strong", 0.4, clusters, members)
    assert sum(pr for _, pr in outcomes) == pytest.approx(1.0, abs=1e-12)
    expected = np.zeros_like(result.joint)
    for cluster, pr in outcomes:
        after = tp.compute(obs + [Obs("strong", cluster, 0.4)], ev,
                           clusters + 1 if cluster == clusters else clusters)
        # Predictive probabilities must be the model's marginal likelihood ratio.
        assert np.exp(after.log_marginal - result.log_marginal) == pytest.approx(pr, abs=1e-12)
        expected[:clusters] += pr * after.joint[:clusters]
        expected[clusters] += pr * (after.joint[clusters:].sum(axis=0))
    np.testing.assert_allclose(expected, result.joint, atol=1e-12)


def test_abstention_carries_no_difficulty_or_correctness_information():
    tp = TeamPosterior(params())
    obs = [Obs("strong", 0), Obs("weak", 1)]
    before = tp.compute(obs)
    after = tp.compute(obs + [Obs("tool", -1)] * 20)
    np.testing.assert_allclose(before.joint, after.joint, atol=1e-13)


def test_same_model_correlation_prevents_false_independent_replication():
    agents = {f"a{i}": AgentProfile(1, abstain=0) for i in range(12)}
    # A model can keep repeating its own systematic wrong answer. Both error
    # identity and correctness dependence must be represented in this example.
    correlated = TeamPosterior(PosteriorParams(agents, kappa_same=0.95, same_agent_correlation=0.8))
    independent = TeamPosterior(PosteriorParams(agents, kappa_same=0.95, same_agent_correlation=0))
    repeated = [Obs("a0", 0)] * 8
    diverse = [Obs(f"a{i}", 0) for i in range(8)]
    assert correlated.compute(repeated).confidence < 0.9
    assert correlated.compute(repeated).confidence < independent.compute(repeated).confidence - 0.1
    assert correlated.compute(diverse).confidence > 0.99


def test_zone_sampling_horizon_respects_response_correlation_and_abstention():
    # These parameters concentrate b near 1.5: one-shot p=.18, iid K=8 reach=.80,
    # but near-perfect response dependence cannot turn repeated failures into reach.
    p = PosteriorParams({"a": AgentProfile(0, abstain=0)}, b_mean=1.5, b_sd=0.02, same_agent_correlation=0.95)
    correlated = TeamPosterior(p)
    r = correlated.compute([])
    assert correlated.member_zones(r, "a")["U"] > 0.99
    p.same_agent_correlation = 0
    iid = TeamPosterior(p)
    assert iid.member_zones(iid.compute([]), "a")["L"] > 0.99
    p.agents["a"].abstain = 1
    unusable = TeamPosterior(p)
    assert unusable.member_zones(unusable.compute([]), "a")["U"] == pytest.approx(1)


def test_incomplete_evidence_outcomes_cannot_be_valued_as_a_partition():
    tp = TeamPosterior(params())
    with pytest.raises(ValueError, match="exhaustive"):
        tp.voc_evidence([Obs("mid", 0)], [], 1, ["check_passed", "check_failed"], 0)


def test_coaching_value_includes_screen_acceptance_but_discussion_bypasses_it():
    from mathmage.team.policy import Action, Policy, PolicyConfig

    p = params(hint_acceptance=0.0)
    obs = [Obs("mid", 0), Obs("mid", 1)]
    def scored(screen):
        policy = Policy(p, PolicyConfig(screen_coaching=screen))
        return policy.score([Action("coach", member=0, tutor=1)], obs, [], 2,
                            lambda _: "mid", lambda *_: 0.6)[0].voc
    assert scored(True) == 0
    full = scored(False)
    assert full > 0.01
    p.hint_acceptance = 0.25
    assert scored(True) == pytest.approx(0.25 * full)


def test_repeated_model_posterior_calibrates_on_normalized_generative_process():
    rng = np.random.default_rng(192)
    agents = {"a": AgentProfile(0.5, abstain=0.1), "b": AgentProfile(-0.3, abstain=0.1)}
    rho, k, ks = 0.4, 0.3, 0.65
    tp = TeamPosterior(PosteriorParams(agents, same_agent_correlation=rho, kappa=k, kappa_same=ks))
    confidence, correct = [], []
    for _ in range(1400):
        b = rng.normal(0, 1.6)
        concentration = (1 - rho) / rho
        latent = {}
        for name, agent in agents.items():
            p = 1 / (1 + np.exp(-(agent.theta - b)))
            latent[name] = rng.beta(concentration * p, concentration * (1 - p))
        labels, wrong_members, obs = {}, {}, []
        for name in ["a", "b", "a", "b", "a", "b"]:
            if rng.random() < agents[name].abstain:
                obs.append(Obs(name, -1))
                continue
            if rng.random() < latent[name]:
                answer = "true"
            else:
                keys = list(wrong_members)
                odds = [(ks if name in wrong_members[c] else k) / (1 - (ks if name in wrong_members[c] else k))
                        for c in keys] + [1.0]
                choice = rng.choice(len(odds), p=np.asarray(odds) / sum(odds))
                answer = keys[choice] if choice < len(keys) else f"wrong-{len(keys)}"
                wrong_members.setdefault(answer, []).append(name)
            labels.setdefault(answer, len(labels))
            obs.append(Obs(name, labels[answer]))
        if not labels:
            continue
        result = tp.compute(obs)
        confidence.append(result.confidence)
        correct.append(result.best == labels.get("true", -1))
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    assert abs(confidence.mean() - correct.mean()) < 0.035
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 100:
            assert abs(confidence[mask].mean() - correct[mask].mean()) < 0.06


# ---- model v3: trap items and tutor-dependent coaching --------------------------------

def _predictive_matches_evidence_ratio(tp, obs, evidence, C, new_agent, lift=0.0, source=-1, source_lift=None):
    """P(next response = c | data) must equal Z(data + response) / Z(data) for every outcome."""
    before = tp.compute(obs, evidence, C)
    members = {}
    for o in obs:
        if o.cluster >= 0:
            members.setdefault(o.cluster, []).append(o.agent)
    outcomes = tp.sample_outcomes(before, new_agent, lift, C, members, source, source_lift)
    assert sum(p for _, p in outcomes) == pytest.approx(1.0, abs=1e-9)
    for cluster, prob in outcomes:
        after = tp.compute(obs + [Obs(new_agent, cluster, lift, source, source_lift)], evidence,
                           C + 1 if cluster == C else C)
        assert prob == pytest.approx(np.exp(after.log_marginal - before.log_marginal), rel=1e-7, abs=1e-12)


def test_single_type_limit_and_trap_mixture_are_normalized():
    obs = [Obs("strong", 0), Obs("mid", 0), Obs("weak", 1)]
    plain = TeamPosterior(params()).compute(obs)
    degenerate = TeamPosterior(params(trap_weight=1e-12, kappa_trap=0.12)).compute(obs)
    assert np.allclose(plain.q, degenerate.q, atol=1e-9) and plain.trap_probability == 0
    mixed = TeamPosterior(params(trap_weight=0.2, kappa_trap=0.5)).compute(obs)
    assert mixed.parts.shape == (2, 3, 121)
    assert mixed.q.sum() + mixed.q_none == pytest.approx(1.0)
    assert np.allclose(mixed.parts.sum(axis=0), mixed.joint)


def test_trap_items_temper_confidence_in_unanimous_answers():
    unanimous = [Obs("strong", 0), Obs("mid", 0), Obs("weak", 0)]
    plain = TeamPosterior(params(kappa=0.04)).compute(unanimous)
    trap = TeamPosterior(params(kappa=0.04, trap_weight=0.2, kappa_trap=0.45)).compute(unanimous)
    assert trap.confidence < plain.confidence
    # Agreement itself is evidence for the trap type relative to its prior.
    assert trap.trap_probability > 0.2
    # A lone correct-looking dissent is more plausible when the majority may be trapped.
    dissent = unanimous + [Obs("tool:checker", 1)]
    assert (TeamPosterior(params(kappa=0.04, trap_weight=0.2, kappa_trap=0.45)).compute(dissent).q[1]
            > TeamPosterior(params(kappa=0.04)).compute(dissent).q[1])


@pytest.mark.parametrize("trap", [0.0, 0.25])
def test_predictive_is_exact_under_traps_and_tutor_dependent_lifts(trap):
    tp = TeamPosterior(params(trap_weight=trap, kappa_trap=0.4 if trap else 0.0))
    obs = [Obs("strong", 0), Obs("mid", 1), Obs("weak", 1), Obs("mid", 2, 0.3, source=0, source_lift=2.0)]
    evidence = [Evidence("review_valid", 0)]
    _predictive_matches_evidence_ratio(tp, obs, evidence, 3, "weak")
    _predictive_matches_evidence_ratio(tp, obs, evidence, 3, "weak", lift=0.4, source=1, source_lift=2.5)
    _predictive_matches_evidence_ratio(tp, obs, evidence, 3, "tool:checker")


def test_coached_success_rises_with_the_posterior_that_the_tutor_is_right():
    tp = TeamPosterior(params())
    obs = [Obs("strong", 0), Obs("mid", 0), Obs("weak", 1)]
    result = tp.compute(obs, [], 2)
    assert result.q[0] > result.q[1]
    members = {0: ["strong", "mid"], 1: ["weak"]}
    from_likely = tp.success_probability(result, "weak", 0.3, members, source=0, source_lift=2.5)
    from_unlikely = tp.success_probability(result, "weak", 0.3, members, source=1, source_lift=2.5)
    uncoached = tp.success_probability(result, "weak", 0.0, members)
    assert from_likely > from_unlikely > uncoached
    # The coached response's predicted chance of joining each cluster sums to the success probability.
    outcomes = dict(tp.sample_outcomes(result, "weak", 0.3, 2, members, 0, 2.5))
    assert outcomes[0] + outcomes[1] + outcomes[2] == pytest.approx(1 - tp.params.agent("weak").abstain)
    # Without a source the single-lift model is recovered exactly.
    assert tp.voc_sample(obs, [], 2, "weak", 0.7, result) == pytest.approx(
        tp.voc_sample(obs, [], 2, "weak", 0.7, result, source=-1, source_lift=2.5))


def test_coaching_source_must_be_an_observed_cluster():
    with pytest.raises(ValueError, match="coaching source"):
        TeamPosterior(params()).compute([Obs("strong", 0), Obs("mid", 1, 0.3, source=2, source_lift=1.0)])


def test_trap_parameters_are_validated():
    with pytest.raises(ValueError, match="trap"):
        TeamPosterior(params(kappa=0.2, trap_weight=0.1, kappa_trap=0.1))


def test_retired_cluster_changes_utility_without_changing_beliefs_or_predictive():
    tp = TeamPosterior(params(trap_weight=0.2, kappa_trap=0.45))
    obs = [Obs("strong", 0), Obs("mid", 0), Obs("weak", 1)]
    evidence = [Evidence("review_valid", 0)]
    full = tp.compute(obs, evidence, 2)
    current = tp.compute(obs, evidence, 2, selectable={1})
    assert full.best == 0 and current.best == 1
    assert current.confidence == pytest.approx(full.q[1])
    assert current.coverage == pytest.approx(full.q[1])
    assert np.array_equal(current.joint, full.joint)
    assert np.array_equal(current.parts, full.parts)
    assert current.log_marginal == full.log_marginal
    members = {0: ["strong", "mid"], 1: ["weak"]}
    outcomes = tp.sample_outcomes(current, "weak", 0.2, 2, members, source=0, source_lift=2.0)
    assert outcomes == tp.sample_outcomes(full, "weak", 0.2, 2, members, source=0, source_lift=2.0)
    expected, averaged_joint = 0.0, np.zeros_like(current.joint)
    for answer, probability in outcomes:
        after = tp.compute(obs + [Obs("weak", answer, 0.2, source=0, source_lift=2.0)], evidence,
                           3 if answer == 2 else 2)
        assert probability == pytest.approx(np.exp(after.log_marginal - current.log_marginal), abs=1e-10)
        eligible = {1} | ({answer} if answer >= 0 else set())
        expected += probability * max(after.q[c] for c in eligible)
        projected = after.joint if answer != 2 else np.vstack([after.joint[:2], after.joint[2:].sum(axis=0)])
        averaged_joint += probability * projected
    assert np.allclose(averaged_joint, current.joint, atol=1e-10)
    assert tp.voc_sample(obs, evidence, 2, "weak", 0.2, current, source=0, source_lift=2.0) == pytest.approx(
        max(0.0, expected - current.confidence))
    # Updating evidence cannot make the retired answer selectable.
    assert tp.voc_evidence(obs, evidence, 2,
                           ["review_objection_verified", "review_objection_unverified", "review_valid", "review_unsure"],
                           0, current) == pytest.approx(0.0, abs=1e-12)
    assert tp.voc_judge(obs, evidence, 2, [0, 1], current) == pytest.approx(0.0, abs=1e-12)


def test_empty_or_invalid_selectable_cluster_pool():
    tp = TeamPosterior(params())
    result = tp.compute([Obs("strong", 0)], selectable=set())
    assert result.best == -1 and result.confidence == 0 and result.coverage == 0
    with pytest.raises(ValueError, match="selectable"):
        tp.compute([Obs("strong", 0)], selectable={1})
