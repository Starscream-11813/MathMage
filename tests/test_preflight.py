"""GET-only preflight checks with no network, completions, or credential output."""

import json
from types import SimpleNamespace

import pytest
import requests

from mathmage.config import ModelSpec
from scripts.preflight import run_preflight


class Ledger:
    def summary(self):
        return {"spent_total": 0, "pending_total": 0, "global_cap": 75, "unreconciled_calls": 0, "pending_calls": 0}


class Session:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.routes[url.split("/api/v1/")[1]]
        if isinstance(value, Exception):
            raise value
        code, data = value
        return SimpleNamespace(status_code=code, json=lambda: data)


def fixture():
    spec = ModelSpec(key="fixture", slug="fixture/model", canonical_slug="fixture/canonical", provider="Pinned",
                     lineage="fixture", max_tokens=8000, temperature=0.7, reasoning={"effort": "low"},
                     supports_seed=True, price={"in": 1, "out": 2})
    roster = SimpleNamespace(members=["fixture", "fixture"], checker="fixture", card_writer="fixture", default_judge="fixture")
    models = SimpleNamespace(rosters={"sat": roster}, frozen_at="2026-10-18", model=lambda name: spec)
    endpoint = {"provider_name": "Pinned", "tag": "pinned", "max_completion_tokens": 12000,
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "supported_parameters": ["max_tokens", "temperature", "reasoning", "seed"]}
    routes = {"key": (200, {"data": {"label": "SECRET-DO-NOT-PRINT", "limit_remaining": 20}}),
              "models": (200, {"data": [{"id": spec.slug, "canonical_slug": spec.canonical_slug}]}),
              "models/fixture/model/endpoints": (200, {"data": {"endpoints": [endpoint]}})}
    return models, endpoint, routes


def test_success_checks_pinned_prices_and_parameters_without_credential_disclosure():
    models, _, routes = fixture()
    session = Session(routes)
    report = run_preflight(models=models, credential="SECRET-DO-NOT-PRINT", session=session, ledger=Ledger())
    assert report["status"] == "ok"
    assert len(report["models"]) == 1
    assert len(session.calls) == 3
    assert all("chat/completions" not in url for url, _ in session.calls)
    assert "SECRET-DO-NOT-PRINT" not in json.dumps(report)
    assert "Authorization" not in json.dumps(report)
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.parametrize("defect", ["price", "provider", "token_limit", "parameter", "canonical"])
def test_explicit_mismatches_fail(defect):
    models, endpoint, routes = fixture()
    if defect == "price":
        endpoint["pricing"]["completion"] = "0.000003"
    elif defect == "provider":
        endpoint["provider_name"], endpoint["tag"] = "Other", "other"
    elif defect == "token_limit":
        endpoint["max_completion_tokens"] = 1000
    elif defect == "parameter":
        endpoint["supported_parameters"].remove("reasoning")
    else:
        routes["models"][1]["data"][0]["canonical_slug"] = "new/version"
    report = run_preflight(models=models, credential="secret", session=Session(routes), ledger=Ledger())
    assert report["status"] == "mismatch"


def test_missing_fields_stay_unknown_and_http401_is_explicit_sanitized():
    models, endpoint, routes = fixture()
    del endpoint["max_completion_tokens"]
    del endpoint["supported_parameters"]
    report = run_preflight(models=models, credential="secret", session=Session(routes), ledger=Ledger())
    assert report["status"] == "unknown"
    routes["key"] = (401, {"error": {"message": "secret echoed; do not print"}})
    report = run_preflight(models=models, credential="secret", session=Session(routes), ledger=Ledger())
    assert report["status"] == "mismatch"
    assert report["credential"]["http_status"] == 401
    assert "HTTP 401" in report["credential"]["detail"]
    assert "secret" not in json.dumps(report)


def test_offline_never_loads_credentials_or_uses_network(monkeypatch):
    from scripts import preflight

    models, _, _ = fixture()
    monkeypatch.setattr(preflight, "load_api_key", lambda: pytest.fail("offline credential access"))
    report = run_preflight(models=models, offline=True, session=Session({}), ledger=Ledger())
    assert report["status"] == "unknown"
    assert report["completion_calls"] == 0


def test_exception_details_are_suppressed():
    models, _, routes = fixture()
    routes["key"] = requests.ConnectionError("secret Authorization header")
    report = run_preflight(models=models, credential="secret", session=Session(routes), ledger=Ledger())
    assert report["status"] == "unknown"
    assert "secret" not in json.dumps(report)


def test_provider_pin_with_multiple_variants_checks_every_published_price():
    from copy import deepcopy

    models, endpoint, routes = fixture()
    other = deepcopy(endpoint)
    other["pricing"]["completion"] = "0.000004"
    routes["models/fixture/model/endpoints"][1]["data"]["endpoints"].append(other)
    report = run_preflight(models=models, credential="secret", session=Session(routes), ledger=Ledger())
    assert report["models"][0]["status"] == "mismatch"


def test_malformed_metadata_and_credential_loading_errors_are_sanitized(monkeypatch):
    from scripts import preflight

    models, endpoint, routes = fixture()
    endpoint["pricing"] = "unrecognized metadata shape"
    monkeypatch.setattr(preflight, "load_api_key", lambda: (_ for _ in ()).throw(ValueError("SECRET-DO-NOT-PRINT")))
    report = run_preflight(models=models, session=Session(routes), ledger=Ledger())
    assert report["status"] == "unknown"
    assert "SECRET-DO-NOT-PRINT" not in json.dumps(report)


def test_cli_exits_nonzero_only_for_mismatches(monkeypatch, capsys):
    from scripts import preflight

    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: {"status": "unknown"})
    assert preflight.main(["--offline"]) == 0
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: {"status": "mismatch"})
    assert preflight.main([]) == 1
