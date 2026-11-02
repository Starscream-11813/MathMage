"""Validate frozen offline components without accessing grading data."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


STRATEGIES = ("verify_and_accept", "evidence_gated_review", "graded_coaching",
              "method_diversification", "divergence_reconciliation", "fresh_verified_search")
DEFAULT_BANK = {
    "A": ["verify_and_accept"],
    "L": ["evidence_gated_review"],
    "P": ["graded_coaching", "method_diversification"],
    "U": ["fresh_verified_search", "method_diversification"],
}


def load_artifacts(value, *, live: bool = False) -> dict:
    if value is None:
        return {}
    data = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    if data.get("frozen") is not True:
        raise ValueError("offline artifacts must be frozen")
    expected = data.get("sha256")
    unhashed = {k: v for k, v in data.items() if k != "sha256"}
    actual = hashlib.sha256(json.dumps(unhashed, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if expected != actual:
        raise ValueError("offline artifact content hash is missing or changed")
    if live and data.get("mode") != "live":
        raise ValueError("live runs cannot use mock or unmeasured offline artifacts")
    provenance = data.get("provenance", {})
    train = set(provenance.get("train_ids", []))
    dev = set(provenance.get("dev_ids", []))
    if train & dev:
        raise ValueError("offline training and development item IDs overlap")
    if not train and any(data.get(k) for k in ("cards", "tutoring_matrix", "discernment")):
        raise ValueError("learned components require training provenance")
    for card in data.get("cards", []):
        if card.get("source_item_id") not in train:
            raise ValueError("strategy cards must come from training items only")
        allowed = {"id", "source_item_id", "family_id", "method", "applies_when", "pitfall", "checklist"}
        if set(card) - allowed:
            raise ValueError("strategy cards may contain methods and checks only")
    for order in data.get("strategy_bank", {}).values():
        if any(s not in STRATEGIES for s in order):
            raise ValueError("offline strategy bank contains an unknown strategy")
    posterior = data.get("posterior_params")
    if posterior is not None:
        if posterior.get("calibrated") and not train:
            raise ValueError("calibrated posterior parameters require training provenance")
        if posterior.get("calibrated") and set(posterior.get("provenance", {}).get("train_ids", [])) - train:
            raise ValueError("posterior calibration used items outside the training split")
    # Deployment receives aggregate learned components only. Trial-level grading
    # records belong to the offline evidence files and never enter agent state.
    fields = {"schema_version", "mode", "frozen", "provenance", "zone_thresholds", "tutoring_matrix",
              "discernment", "best_member", "strategy_bank", "cards", "sha256", "roster", "posterior_params", "mu_star", "policy_config"}
    return {key: value for key, value in data.items() if key in fields}
