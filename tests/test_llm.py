from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from mathmage.config import BudgetConfig, load_models
from mathmage.llm import BudgetExceeded, BudgetLedger, LLMClient, LLMRequest, LLMResponse
from mathmage.llm.cache import ResponseCache
from mathmage.llm.client import max_cost_usd
from mathmage.llm.openrouter import (NonRetryableError, OpenRouterBatch, OpenRouterSync,
                                    build_body, parse_completion)
from mathmage.llm.types import cache_key, resolve


def request(**kwargs):
    return LLMRequest(model_key="o3-mini-low", messages=[{"role": "user", "content": "Compute 2+2."}],
                      template_id="solve.v1#test", **kwargs)


def make_client(directory, **kwargs):
    return LLMClient(cache_path=directory / "cache.sqlite", ledger_path=directory / "ledger.jsonl",
                     log_directory=directory / "calls", **kwargs)


class FakeBackend:
    def __init__(self, *, uncertain=False, error=None, raises=False):
        self.calls = 0
        self.uncertain = uncertain
        self.error = error
        self.raises = raises

    def complete(self, *_):
        self.calls += 1
        time.sleep(0.01)
        if self.raises:
            raise requests.ReadTimeout("sensitive transport detail")
        return LLMResponse(text="fixture", tokens_in=8, tokens_out=3, cost_usd=0.001,
                           error=self.error, billing_uncertain=self.uncertain), 1


def _reserve_worker(path):
    ledger = BudgetLedger(BudgetConfig(global_cap_usd=1, experiments={"smoke": 1}, overrun_tolerance=0), Path(path))
    try:
        ledger.reserve("smoke", 0.6)
        return "reserved"
    except BudgetExceeded:
        return "capped"


def test_reservations_survive_restart_and_other_instances(tmp_path):
    budget = BudgetConfig(global_cap_usd=1, experiments={"smoke": 1}, overrun_tolerance=0)
    first = BudgetLedger(budget, tmp_path / "ledger.jsonl")
    reserved = first.reserve("smoke", 0.6, "request-key")
    second = BudgetLedger(budget, first.path)
    assert second.pending() == pytest.approx(0.6)
    assert second.unreconciled("request-key")
    with pytest.raises(BudgetExceeded):
        second.reserve("smoke", 0.5)
    second.commit(reserved, 0.25, {})
    assert first.remaining("smoke") == pytest.approx(0.75)
    assert first.summary()["spent_total"] == 0.25
    with pytest.raises(ValueError):
        first.commit(reserved, 0.25, {})


def test_cross_process_budget_cap(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_reserve_worker, [path, path]))
    assert sorted(results) == ["capped", "reserved"]
    assert BudgetLedger(path=Path(path)).pending() == pytest.approx(0.6)


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_invalid_reservation_fails_closed(tmp_path, amount):
    ledger = BudgetLedger(path=tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        ledger.reserve("smoke", amount)


def test_corrupt_ledger_fails_closed_and_legacy_entries_are_readable(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"experiment":"smoke","cost_usd":0.1}\n', encoding="utf-8")
    assert BudgetLedger(path=path).summary()["spent_total"] == 0.1
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":')
    with pytest.raises(RuntimeError, match="reconcile"):
        BudgetLedger(path=path)


def test_offline_default_never_reads_key_or_uses_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline mode attempted a credential or network access")
    monkeypatch.setattr("mathmage.llm.openrouter.load_api_key", forbidden)
    monkeypatch.setattr(requests.Session, "post", forbidden)
    with make_client(tmp_path) as client:
        first = client.complete(request())
        second = client.complete(request())
        assert first.ok and first.mode == "mock" and "MOCK" in first.text
        assert second.cached and second.text == first.text
        assert first.cost_usd == second.cost_usd == 0
        assert second.attempts == 0
        assert client.stats()["backend_calls"] == 1
        assert client.stats()["attempts"] == 1
        assert client.summary()["ledger"]["spent_total"] == 0
    assert not (tmp_path / "ledger.jsonl").exists()


def test_concurrent_identical_requests_single_flight_across_clients(tmp_path):
    backend = FakeBackend()
    first = make_client(tmp_path, backend=backend)
    second = make_client(tmp_path, backend=backend)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: (first if n % 2 else second).complete(request()), range(16)))
    assert backend.calls == 1
    assert sum(r.cached for r in results) == 15
    assert all(r.text == "fixture" for r in results)
    first.close()
    second.close()


def test_live_and_mock_namespaces_never_mix(tmp_path):
    mock = make_client(tmp_path)
    mock_response = mock.complete(request())
    backend = FakeBackend()
    live = make_client(tmp_path, live=True, backend=backend, cache=mock.cache)
    assert live.complete(request()).text == "fixture"
    assert backend.calls == 1
    assert mock.complete(request()).text == mock_response.text
    assert mock.cache.namespace == "mock" and live.cache.namespace == "live"
    mock.close()
    live.close()


def test_budget_reserves_maximum_output_not_expected_output(tmp_path):
    req = request()
    spec = load_models().model(req.model_key)
    bound = max_cost_usd(req, resolve(req), spec)
    assert bound >= spec.price.cost(0, spec.max_tokens)
    budget = BudgetConfig(global_cap_usd=0.05, experiments={"smoke": 0.05}, overrun_tolerance=0)
    ledger = BudgetLedger(budget, tmp_path / "ledger.jsonl")
    backend = FakeBackend()
    client = make_client(tmp_path, live=True, backend=backend, ledger=ledger)
    with pytest.raises(BudgetExceeded):
        client.complete(req)
    assert backend.calls == 0
    assert client.stats()["errors"] == 1
    client.close()


def test_uncertain_timeout_is_charged_conservatively_and_never_replayed(tmp_path):
    backend = FakeBackend(raises=True)
    first = make_client(tmp_path, live=True, backend=backend)
    response = first.complete(request())
    assert response.billing_uncertain and not response.ok
    assert response.cost_usd >= load_models().model("o3-mini-low").price.cost(0, 25000)
    assert "sensitive" not in response.error
    first.close()
    second = make_client(tmp_path, live=True, backend=backend)
    replay = second.complete(request())
    assert not replay.ok and "replay blocked" in replay.error
    assert backend.calls == 1
    assert second.ledger.summary()["unreconciled_calls"] == 1
    assert second.ledger.summary()["spent_total"] == pytest.approx(response.cost_usd)
    second.close()


def test_definite_failure_releases_reservation_and_can_retry(tmp_path):
    backend = FakeBackend(error="HTTP 401")
    # Definite rejection costs zero; providers must report a zero cost for it.
    original = backend.complete
    def reject(*args):
        response, attempts = original(*args)
        response.cost_usd = 0
        return response, attempts
    backend.complete = reject
    client = make_client(tmp_path, live=True, backend=backend)
    assert not client.complete(request()).ok
    assert client.ledger.pending() == 0
    assert client.ledger.summary()["spent_total"] == 0
    assert not client.complete(request()).ok
    assert backend.calls == 2
    client.close()


def test_log_has_mode_attempts_and_seed_support(tmp_path):
    client = make_client(tmp_path)
    client.complete(request(seed=7))
    client.complete(LLMRequest(model_key="haiku-4.5", messages=[{"role": "user", "content": "hello"}],
                               template_id="test", seed=7))
    records = [json.loads(line) for line in client.log.path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["seed_requested"] == records[0]["seed_sent"] == 7
    assert records[1]["seed_requested"] == 7 and records[1]["seed_sent"] is None
    assert all(r["correct"] is None and r["mode"] == "mock" and r["attempts"] == 1 for r in records)
    client.close()


def test_seed_and_sample_tag_are_in_cache_key():
    req = request(seed=1)
    other = request(seed=2)
    assert cache_key(req, resolve(req)) != cache_key(other, resolve(other))
    sample = request(seed=1, sample_tag="s1")
    assert cache_key(req, resolve(req)) != cache_key(sample, resolve(sample))
    assert build_body(resolve(req), req.messages, None)["seed"] == 1


def test_complete_many_preserves_order_and_rejects_unsafe_batch(tmp_path):
    client = make_client(tmp_path)
    reqs = [request(sample_tag=f"s{i}") for i in range(4)]
    responses = client.complete_many(reqs)
    assert len(responses) == 4
    assert [r.text for r in responses] == [client.complete(r).text for r in reqs]
    with pytest.raises(NotImplementedError, match="reconciliation"):
        client.complete_many(reqs, batch=True)
    client.close()


class FakeHTTPResponse:
    def __init__(self, status, data=None):
        self.status_code = status
        self.data = data
    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
    def post(self, *args, **kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def transport(outcomes, monkeypatch):
    backend = OpenRouterSync(api_key="test-only-no-network", max_attempts=3, live=True)
    session = FakeSession(outcomes)
    monkeypatch.setattr(backend, "_session", lambda: session)
    monkeypatch.setattr("mathmage.llm.openrouter.time.sleep", lambda *_: None)
    return backend, session


@pytest.mark.parametrize("outcome", [requests.ReadTimeout(), requests.ConnectionError(),
                                     FakeHTTPResponse(503), FakeHTTPResponse(200, ValueError("bad json"))])
def test_transport_does_not_replay_ambiguous_failures(monkeypatch, outcome):
    backend, session = transport([outcome], monkeypatch)
    req = request()
    response, attempts = backend.complete(resolve(req), req.messages, None, load_models().model(req.model_key))
    assert attempts == 1 and session.calls == 1
    assert response.billing_uncertain and not response.ok


def test_transport_retries_only_safe_rejections_and_connect_timeout(monkeypatch):
    success = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.001}}
    backend, session = transport([FakeHTTPResponse(429), requests.ConnectTimeout(), FakeHTTPResponse(200, success)], monkeypatch)
    req = request()
    response, attempts = backend.complete(resolve(req), req.messages, None, load_models().model(req.model_key))
    assert response.ok and attempts == 3 and session.calls == 3
    assert response.attempts == 3 and response.cost_usd == 0.001


def test_parse_missing_usage_never_claims_free_completion():
    spec = load_models().model("o3-mini-low")
    response = parse_completion({"choices": [{"message": {"content": "answer"}}]}, resolve(request()), spec.price)
    assert response.billing_uncertain and response.cost_estimated


def test_live_transport_requires_explicit_opt_in():
    with pytest.raises(NonRetryableError, match="live=True"):
        OpenRouterSync(api_key="test")
    with pytest.raises(NonRetryableError, match="live=True"):
        OpenRouterBatch(api_key="test")

class FileBackend:
    def __init__(self, path):
        self.path = Path(path)
    def complete(self, *_):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("dispatch\n")
        time.sleep(0.05)
        return LLMResponse(text="fixture", tokens_in=5, tokens_out=3), 1


def _cache_worker(directory):
    directory = Path(directory)
    client = make_client(directory, backend=FileBackend(directory / "dispatches.txt"))
    response = client.complete(request())
    client.close()
    return response.cached


def test_cross_process_identical_requests_single_flight(tmp_path):
    with ProcessPoolExecutor(max_workers=3) as pool:
        cached = list(pool.map(_cache_worker, [str(tmp_path)] * 3))
    assert sum(cached) == 2
    assert (tmp_path / "dispatches.txt").read_text(encoding="utf-8").splitlines() == ["dispatch"]


def test_process_interruption_retains_reservation_and_blocks_replay(tmp_path):
    class InterruptedBackend:
        def complete(self, *_):
            raise SystemExit("simulate interruption after dispatch")
    first = make_client(tmp_path, live=True, backend=InterruptedBackend())
    with pytest.raises(SystemExit):
        first.complete(request())
    assert first.ledger.pending() > 0
    first.close()
    backend = FakeBackend()
    resumed = make_client(tmp_path, live=True, backend=backend)
    response = resumed.complete(request())
    assert "replay blocked" in response.error
    assert backend.calls == 0
    assert resumed.ledger.pending() > 0
    resumed.close()


def test_cache_keeps_logical_resources_but_zeros_marginal_cost(tmp_path):
    client = make_client(tmp_path, live=True, backend=FakeBackend())
    original = client.complete(request())
    cached = client.complete(request())
    assert cached.cost_usd == cached.latency_s == 0
    assert cached.resource_cost_usd == original.resource_cost_usd > 0
    assert cached.resource_latency_s == original.resource_latency_s > 0
    assert cached.tokens_in == original.tokens_in and cached.tokens_out == original.tokens_out
    client.close()


def test_complete_many_preserves_successful_results_when_another_is_capped(tmp_path):
    # One call fits, but a simultaneous second full-output reservation does not.
    budget = BudgetConfig(global_cap_usd=0.15, experiments={"smoke": 0.15}, overrun_tolerance=0)
    capped = threading.Event()
    class GateLedger(BudgetLedger):
        def reserve(self, *args, **kwargs):
            try:
                return super().reserve(*args, **kwargs)
            except BudgetExceeded:
                capped.set()
                raise
    class GateBackend(FakeBackend):
        def complete(self, *args):
            assert capped.wait(5), "second request did not attempt its reservation"
            return super().complete(*args)
    ledger = GateLedger(budget, tmp_path / "ledger.jsonl")
    backend = GateBackend()
    client = make_client(tmp_path, live=True, ledger=ledger, backend=backend)
    responses = client.complete_many([request(sample_tag="one"), request(sample_tag="two")])
    assert len(responses) == 2
    assert sum(r.ok for r in responses) == 1
    assert sum("BudgetExceeded" in (r.error or "") for r in responses) == 1
    assert backend.calls == 1
    assert ledger.summary()["spent_total"] == pytest.approx(0.001)
    client.close()
