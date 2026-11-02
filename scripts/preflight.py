"""Read-only OpenRouter metadata checks before paid experiments (GET requests only).

Never send a completion, mutate configuration, or print credential labels/headers.
Missing metadata is reported as unknown; explicit configuration mismatches fail.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from mathmage.config import load_api_key, load_models  # noqa: E402
from mathmage.experiments.artifacts import atomic_json  # noqa: E402
from mathmage.llm.ledger import BudgetLedger  # noqa: E402
from mathmage.llm.openrouter import API  # noqa: E402


def _status(checks):
    statuses = [check["status"] for check in checks]
    return "mismatch" if "mismatch" in statuses else ("unknown" if "unknown" in statuses else "ok")


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _get(session, route: str, *, credential=None):
    """Return only a parsed object and sanitized status; never forward error text."""
    headers = {"Authorization": f"Bearer {credential}"} if credential else {}
    try:
        response = session.get(f"{API}/{route}", headers=headers, timeout=20, allow_redirects=False)
        code = response.status_code
        if code != 200:
            return None, {"status": "mismatch" if code == 401 else "unknown", "http_status": code,
                          "detail": "HTTP 401: credential rejected" if code == 401 else "metadata request unsuccessful"}
        data = response.json()
        if not isinstance(data, dict):
            return None, {"status": "unknown", "http_status": code, "detail": "metadata schema unrecognized"}
        return data, {"status": "ok", "http_status": code}
    except (requests.RequestException, ValueError, TypeError):
        return None, {"status": "unknown", "detail": "metadata request failed; response details suppressed"}


def _field(field, expected, observed, *, upper_bound=False):
    """Compare finite numeric metadata across every endpoint allowed by the pin."""
    known = [value for value in observed if value is not None]
    mismatch = any(expected > value if upper_bound else not math.isclose(expected, value, rel_tol=1e-6, abs_tol=1e-9)
                   for value in known)
    status = "mismatch" if mismatch else ("ok" if len(known) == len(observed) and observed else "unknown")
    return {"field": field, "status": status, "configured": expected, "observed": sorted(set(known)),
            "unreported_endpoints": len(observed) - len(known)}


def inspect_model(spec, catalog, endpoint_payload, endpoint_status):
    """Validate pinned-provider data, avoiding cheapest-provider catalog pricing."""
    checks = []
    entry = next((item for item in catalog if isinstance(item, dict) and item.get("id") == spec.slug), None)
    endpoints_data = endpoint_payload.get("data", {}) if endpoint_payload else {}
    endpoints = endpoints_data.get("endpoints") if isinstance(endpoints_data, dict) else None
    if entry is not None:
        checks.append({"field": "model_slug", "status": "ok"})
    elif endpoint_status.get("http_status") == 404:
        checks.append({"field": "model_slug", "status": "mismatch", "detail": "configured slug unavailable"})
    else:
        checks.append({"field": "model_slug", "status": "unknown", "detail": "slug not verified in model catalog"})
    canonical = entry.get("canonical_slug") if entry else None
    canonical = canonical if isinstance(canonical, str) else None
    if spec.canonical_slug:
        checks.append({"field": "canonical_slug", "status": ("ok" if canonical == spec.canonical_slug else
                       "mismatch" if canonical else "unknown"), "configured": spec.canonical_slug,
                       "observed": canonical})
    if not isinstance(endpoints, list):
        checks.append({"field": "pinned_provider", "status": "unknown", "detail": "provider metadata unavailable"})
        return {"model": spec.key, "slug": spec.slug, "provider": spec.provider,
                "status": _status(checks), "checks": checks, "endpoint_request": endpoint_status}
    matched = [endpoint for endpoint in endpoints if isinstance(endpoint, dict) and
               spec.provider.casefold() in {str(endpoint.get("provider_name", "")).casefold(),
                                           str(endpoint.get("tag", "")).casefold()}]
    checks.append({"field": "pinned_provider", "status": "ok" if matched else "mismatch",
                   "matching_endpoints": len(matched)})
    if matched:
        for field, remote, expected in [("input_usd_per_million", "prompt", spec.price.in_),
                                         ("output_usd_per_million", "completion", spec.price.out),
                                         ("cache_read_usd_per_million", "input_cache_read", spec.price.cache_read)]:
            if expected is not None:
                prices = [_number(endpoint["pricing"].get(remote))
                          if isinstance(endpoint.get("pricing"), dict) else None for endpoint in matched]
                checks.append(_field(field, expected, [price * 1e6 if price is not None else None for price in prices]))
        limits = [_number(endpoint.get("max_completion_tokens")) for endpoint in matched]
        checks.append(_field("max_output_tokens", spec.max_tokens, limits, upper_bound=True))
        required = ["max_tokens"]
        required += [name for name in ("temperature", "top_p", "reasoning") if getattr(spec, name) is not None]
        if spec.supports_seed:
            required.append("seed")
        for parameter in required:
            support = [endpoint.get("supported_parameters") for endpoint in matched]
            known = [parameters for parameters in support if isinstance(parameters, (list, dict))]
            checks.append({"field": f"parameter:{parameter}",
                           "status": "mismatch" if any(parameter not in parameters for parameters in known) else
                                     "ok" if len(known) == len(matched) else "unknown"})
    return {"model": spec.key, "slug": spec.slug, "provider": spec.provider,
            "status": _status(checks), "checks": checks, "endpoint_request": endpoint_status}


def _ledger_summary(ledger):
    try:
        data = ledger.summary()
        fields = ("spent_total", "pending_total", "global_cap", "unreconciled_calls", "pending_calls")
        result = {name: _number(data.get(name)) for name in fields}
        status = "mismatch" if result["unreconciled_calls"] else "ok"
        if (result["global_cap"] is not None and result["spent_total"] is not None and result["pending_total"] is not None
                and result["spent_total"] + result["pending_total"] >= result["global_cap"]):
            status = "mismatch"
        return {"status": status, **result}
    except (RuntimeError, ValueError, OSError):
        return {"status": "mismatch", "detail": "local ledger is unreadable or invalid"}


def run_preflight(*, roster="sat", offline=False, models=None, credential=None, session=None, ledger=None):
    models = models or load_models()
    if roster not in models.rosters:
        raise ValueError("unknown roster")
    roles = models.rosters[roster]
    selected = list(dict.fromkeys([*roles.members, roles.checker, roles.card_writer, roles.default_judge]))
    try:
        ledger = ledger if ledger is not None else BudgetLedger()
        local = _ledger_summary(ledger)
    except (RuntimeError, ValueError, OSError):
        local = {"status": "mismatch", "detail": "local ledger is unreadable or invalid"}
    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "roster": roster,
              "mode": "offline" if offline else "free_metadata_get", "completion_calls": 0,
              "frozen_at": models.frozen_at, "ledger": local,
              "notes": ["Configuration is never modified.",
                        "Missing metadata remains unknown; public availability does not verify account routing restrictions.",
                        "Parameter names are checked where published; exact accepted values require a separately authorized pilot.",
                        "Provider prices are published per-token prices converted to USD per million; no completion is sent."]}
    if offline:
        report.update({"status": "mismatch" if local["status"] == "mismatch" else "unknown",
                       "credential": {"status": "unknown", "detail": "not inspected in offline mode"},
                       "models": [{"model": name, "slug": models.model(name).slug,
                                   "provider": models.model(name).provider, "status": "unknown",
                                   "detail": "remote metadata not checked"} for name in selected]})
        return report
    client = session if session is not None else requests.Session()
    try:
        credential_error = False
        if credential is None:
            try:
                credential = load_api_key()
            except (OSError, ValueError, TypeError):
                credential_error = True
        if credential_error:
            account = {"status": "unknown", "detail": "credential configuration unreadable; details suppressed"}
        elif credential:
            account_data, account = _get(client, "key", credential=credential)
            if account_data is not None:
                # Account labels can contain credential fragments: whitelist only
                # numeric account limits, never labels, IDs, headers, or raw bodies.
                data = account_data.get("data", {})
                if isinstance(data, dict):
                    for name in ("limit", "limit_remaining", "usage"):
                        account[name] = _number(data.get(name))
                else:
                    account = {"status": "unknown", "detail": "account metadata schema unrecognized"}
        else:
            account = {"status": "mismatch", "detail": "credential is not configured"}
        catalog_data, catalog_status = _get(client, "models")
        catalog = catalog_data.get("data", []) if catalog_data else []
        if not isinstance(catalog, list):
            catalog, catalog_status = [], {"status": "unknown", "detail": "model catalog schema unrecognized"}
        cached = {}
        checked = []
        for name in selected:
            spec = models.model(name)
            if spec.slug not in cached:
                cached[spec.slug] = _get(client, f"models/{quote(spec.slug, safe='/')}/endpoints")
            checked.append(inspect_model(spec, catalog, *cached[spec.slug]))
        report.update({"credential": account, "catalog": catalog_status, "models": checked,
                       "status": _status([local, account, catalog_status, *checked])})
        return report
    finally:
        if session is None:
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", choices=["sat", "xolver"], default="sat")
    parser.add_argument("--offline", action="store_true", help="Inspect local configuration/ledger without network or credentials")
    parser.add_argument("--output", type=Path, help="Write the same sanitized JSON report to this path")
    args = parser.parse_args(argv)
    report = run_preflight(roster=args.roster, offline=args.offline)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["status"] == "mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
