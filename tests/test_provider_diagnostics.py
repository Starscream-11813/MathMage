"""Network-free tests for operational response diagnostics."""
from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


source = Path(__file__).resolve().parents[1] / "scripts" / "provider_diagnostics.py"
module_spec = importlib.util.spec_from_file_location("provider_diagnostics_under_test", source)
diagnostics = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(diagnostics)


def settings(qwen=True):
    return SimpleNamespace(provider="SiliconFlow" if qwen else "OpenAI",
                           slug="qwen/qwen3-32b" if qwen else "openai/o3-mini")


def backend():
    session = requests.Session()
    return SimpleNamespace(max_attempts=6, _session=lambda: session), session


def response(status, body, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://openrouter.ai/api/v1/chat/completions"
    result._content = json.dumps(body).encode()
    result.headers.update(headers or {})
    return result


def dispatch(session, result):
    for hook in session.hooks["response"]:
        replacement = hook(result)
        if replacement is not None:
            result = replacement
    return result


def test_failure_metadata_redaction_headers_and_attempt_restore(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    old_hook = lambda result: None
    old_hooks = [old_hook]
    session.hooks["response"] = old_hooks
    result = response(429, {"error": {
        "code": 429, "message": "bad sk-or-v1-SUPERSECRET Bearer OTHERSECRET",
        "metadata": {"error_type": "rate_limit_exceeded", "provider_name": "SiliconFlow",
                     "provider_code": "quota", "raw": 'api_key="HIDDEN" ' + "x" * 5000,
                     "private": "omit me"}}},
        {"Retry-After": "90", "X-RateLimit-Remaining": "0", "Authorization": "SECRET"})
    returned = (SimpleNamespace(error="failed after 1 attempts: HTTP 429 rate limited"), 1)

    def complete(current, *args):
        assert current.max_attempts == 1
        assert dispatch(session, result) is result
        return returned

    assert adapter.complete(complete, target, settings(), [{"content": "PRIVATE PROMPT"}], None, None) is returned
    assert target.max_attempts == 6
    assert session.hooks["response"] is old_hooks
    text = (tmp_path / "provider_diagnostics.jsonl").read_text()
    record = json.loads(text)
    assert record["http_status"] == 429
    assert record["retry_after_seconds"] == 90
    assert 89 < adapter.last_provider_backoff(settings()) <= 90
    assert set(record["metadata"]) == {"error_type", "provider_name", "provider_code", "raw"}
    assert len(record["metadata"]["raw"]) <= 4000
    for secret in ("SUPERSECRET", "OTHERSECRET", "HIDDEN", "PRIVATE PROMPT", "Authorization", "omit me"):
        assert secret not in text


def test_success_does_not_inspect_or_log_body(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    result = response(200, {"content": "private successful output"})
    result.json = lambda: pytest.fail("success body must never be read")
    returned = object()

    def complete(*args):
        dispatch(session, result)
        return returned

    assert adapter.complete(complete, target, settings(), [], None, None) is returned
    assert not (tmp_path / "provider_diagnostics.jsonl").exists()


@pytest.mark.parametrize("value,expected", [("120", 120), ("-1", None), ("junk", None), (None, None),
    ("Wed, 21 Oct 2015 07:28:00 GMT", 60), ("Wed, 21 Oct 2015 07:26:00 GMT", 0)])
def test_retry_after_parsing(value, expected):
    assert diagnostics.retry_after_seconds(value, now=1445412420) == expected


def test_capture_failure_never_changes_transport_response(tmp_path, monkeypatch):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    result = response(429, {"error": {"message": "rate limited"}})
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    def complete(*args):
        assert dispatch(session, result) is result
        return "unchanged", 1

    assert adapter.complete(complete, target, settings(), [], None, None) == ("unchanged", 1)
    assert adapter.diagnostic_errors == 1
    assert target.max_attempts == 6


def test_unexpected_callback_failure_cannot_change_response(tmp_path, monkeypatch):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    result = response(429, {})
    monkeypatch.setattr(adapter, "_capture", lambda *args: (_ for _ in ()).throw(RuntimeError("broken diagnostic")))

    def complete(*args):
        assert dispatch(session, result) is result
        return "unchanged", 1

    assert adapter.complete(complete, target, settings(), [], None, None) == ("unchanged", 1)
    assert adapter.diagnostic_errors == 1


def test_escaped_nested_credentials_are_redacted():
    text = diagnostics.redact(r'{\"api_key\":\"PRIVATE\",\"other\":\"SK-or-v1-HIDDEN\"}')
    assert "PRIVATE" not in text
    assert "HIDDEN" not in text


def test_restart_restores_longest_provider_hint_while_ignoring_partial_records(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.time, "time", lambda: 1445412420)
    records = [
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:27:00+00:00", "retry_after_seconds": 3600},
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:28:00+00:00", "retry_after_seconds": 60},
        {"provider": "OpenAI", "time_utc": "2015-10-21T07:27:00+00:00", "retry_after_seconds": 120},
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:27:00+00:00", "retry_after_seconds": None},
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:27:00", "retry_after_seconds": 9000},
        {"provider": "SiliconFlow", "time_utc": "bad date", "retry_after_seconds": 9000},
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:27:00+00:00", "retry_after_seconds": float("inf")},
        {"provider": "SiliconFlow", "time_utc": "2015-10-21T07:27:00+00:00", "retry_after_seconds": True},
        ["not an object"],
    ]
    (tmp_path / "provider_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + '\n{"provider":', encoding="utf-8")
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    assert adapter.last_provider_backoff(settings()) == 3600
    assert adapter.last_provider_backoff("OpenAI") == 120
    assert adapter.last_provider_backoff("Unknown") == 0


def test_expired_persisted_hint_never_produces_negative_delay(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.time, "time", lambda: 1445412420)
    (tmp_path / "provider_diagnostics.jsonl").write_text(json.dumps({
        "provider": "SiliconFlow", "time_utc": "2015-10-21T07:20:00+00:00", "retry_after_seconds": 60}),
        encoding="utf-8")
    assert diagnostics.ProviderDiagnostics(tmp_path).last_provider_backoff("SiliconFlow") == 0


def test_backend_exception_restores_attempts_and_hook(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    old_hooks = session.hooks["response"]

    def complete(*args):
        raise RuntimeError("original transport failure")

    with pytest.raises(RuntimeError, match="original transport failure"):
        adapter.complete(complete, target, settings(), [], None, None)
    assert target.max_attempts == 6
    assert session.hooks["response"] is old_hooks


def test_nonqwen_attempt_policy_unchanged_and_nonjson_body_omitted(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    target, session = backend()
    result = response(503, {})
    result._content = b"PRIVATE NONJSON BODY"

    def complete(current, *args):
        assert current.max_attempts == 6
        dispatch(session, result)
        return "unchanged", 6

    assert adapter.complete(complete, target, settings(False), [], None, None) == ("unchanged", 6)
    text = (tmp_path / "provider_diagnostics.jsonl").read_text()
    assert "PRIVATE NONJSON BODY" not in text
    assert "not JSON" in text


def test_shared_qwen_backend_mutation_is_serialized(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path, qwen_max_attempts=2)
    target, session = backend()
    old_hooks = session.hooks["response"]

    def complete(current, *args):
        assert current.max_attempts == 2
        assert len(session.hooks["response"]) == 1
        return "unchanged", 2

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: adapter.complete(complete, target, settings(), [], None, None), range(20)))
    assert results == [("unchanged", 2)] * 20
    assert target.max_attempts == 6
    assert session.hooks["response"] is old_hooks


def test_unrelated_endpoint_is_never_logged(tmp_path):
    adapter = diagnostics.ProviderDiagnostics(tmp_path)
    result = response(429, {"error": {"message": "private elsewhere"}})
    result.url = "https://example.com/api/v1/chat/completions"
    adapter._capture(result, settings())
    assert not (tmp_path / "provider_diagnostics.jsonl").exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_invalid_attempt_configuration(value, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        diagnostics.ProviderDiagnostics(tmp_path, qwen_max_attempts=value)
