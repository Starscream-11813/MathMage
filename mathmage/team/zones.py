"""Conservative operational routing, never ground-truth zone classification."""
from __future__ import annotations

from collections import Counter
from mathmage.grading.equiv import cluster
from mathmage.team.types import Candidate, ZoneEstimate


def estimate_zone(candidates: list[Candidate], thresholds: dict | None = None, answer_format: str = "exact") -> ZoneEstimate:
    thresholds = thresholds or {}
    valid = [c for c in candidates if c.valid and c.answer is not None]
    votes = Counter(cluster([c.answer for c in valid], answer_format))
    agreeing = max(votes.values(), default=0)
    verified = [c for c in valid if c.evidence.full_answer_verified and c.evidence.status == "passed"]
    failed = any(c.evidence.status == "failed" or c.evidence.failed_claims for c in candidates)
    partial = any(c.evidence.verified_claims for c in candidates)
    verified_votes = Counter(cluster([c.answer for c in verified], answer_format))
    consensus = max(verified_votes.values(), default=0)
    # Agreement alone is never evidence of correctness. A requires independent
    # computational support for two agreeing candidates and no failed checks.
    if consensus >= max(2, int(thresholds.get("verified_consensus", 2))) and not failed:
        label, reason = "A", "verified agreement with no failed executable check"
    elif verified or agreeing >= 2:
        label, reason = "L", "candidate evidence or unverified agreement requires adjudication"
    elif partial:
        label, reason = "P", "verified partial progress without a verified complete answer"
    else:
        label, reason = "U", "no verified progress observed; finite samples cannot establish impossibility"
    # These are explicit routing scores, not posterior probabilities of latent
    # empirical zones. Only separately calibrated artifacts can supply confidence.
    calibration = thresholds.get("zone_probabilities", {}).get(label)
    calibrated = (isinstance(calibration, dict) and set(calibration) == set("ALPU")
                  and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in calibration.values())
                  and abs(sum(calibration.values()) - 1) < 1e-6)
    probabilities = dict(calibration) if calibrated else {}
    uncertainty_limit = float(thresholds.get("confidence_threshold", 0.8))
    uncertain = not calibrated or probabilities[label] < uncertainty_limit
    return ZoneEstimate(label, probabilities, uncertain, calibrated, reason, {
        "valid_samples": len(valid), "agreement_count": agreeing,
        "verified_answers": len(verified), "verified_consensus": consensus,
        "any_failed_check": failed, "verified_partial_progress": partial,
        "probability_semantics": "calibrated routing confidence" if calibrated else "unmeasured; operational routing only",
    })
