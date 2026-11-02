"""Safe cached LLM orchestration. The default backend is entirely offline."""
from __future__ import annotations
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional
from mathmage import paths
from mathmage.config import ModelsConfig, load_models
from mathmage.llm.cache import DEFAULT_CACHE, ResponseCache
from mathmage.llm.calllog import CallLog
from mathmage.llm.ledger import DEFAULT_LEDGER, BudgetExceeded, BudgetLedger
from mathmage.llm.locking import file_lock
from mathmage.llm.mock import MockBackend
from mathmage.llm.openrouter import OpenRouterSync
from mathmage.llm.types import LLMRequest, LLMResponse, ResolvedSettings, cache_key, resolve


def max_cost_usd(req: LLMRequest, settings: ResolvedSettings, spec) -> float:
    """Reserve all possible output tokens, including hidden reasoning.

    UTF-8 byte length bounds ordinary text tokenization conservatively. Additional
    message/template/JSON overhead is deliberately generous. Expected output lengths
    are planning estimates and must never be used to enforce a hard spending cap.
    """
    input_bound = 256 + sum(32 + len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in req.messages)
    if req.response_format:
        input_bound += len(json.dumps(req.response_format, ensure_ascii=False).encode("utf-8")) + 64
    return spec.price.cost(input_bound, settings.max_tokens)


class LLMClient:
    def __init__(self, experiment: str = "smoke", live: bool = False, *, backend=None,
                 cache: Optional[ResponseCache] = None, ledger: Optional[BudgetLedger] = None,
                 cache_path: Path = DEFAULT_CACHE, ledger_path: Path = DEFAULT_LEDGER,
                 log_directory: Path = paths.LOGS / "calls", call_log: Optional[CallLog] = None,
                 models: Optional[ModelsConfig] = None, max_workers: int = 4):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.experiment = experiment
        self.live = live
        self.mode = "live" if live else "mock"
        self.models = models or load_models()
        if backend is not None and getattr(backend, "is_live", False) and not live:
            raise ValueError("a live backend requires explicit live=True")
        self.backend = backend if backend is not None else (OpenRouterSync(live=True) if live else MockBackend())
        if live and isinstance(self.backend, MockBackend):
            raise ValueError("a mock backend cannot be recorded as live research data")
        self.cache = cache if cache is not None and cache.namespace == self.mode else ResponseCache(
            cache.path if cache is not None else Path(cache_path), namespace=self.mode)
        self.ledger = ledger if ledger is not None else BudgetLedger(path=Path(ledger_path))
        if live and not self.ledger.enforce:
            raise ValueError("live calls require budget enforcement")
        if live:
            self.ledger.line(experiment)
        self.log = call_log or CallLog(experiment, Path(log_directory))
        self.max_workers = max_workers
        self._stats_lock = threading.Lock()
        self._stats = {"requests": 0, "backend_calls": 0, "cache_hits": 0, "errors": 0,
                       "attempts": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    def _record(self, req, settings, response, key):
        from mathmage.grading.extract import extract_final
        parsed = extract_final(response.text, lenient=False) if response.ok and not response.truncated else None
        self.log.write(req, settings, response, key, parsed_answer=parsed)
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["cache_hits"] += int(response.cached)
            self._stats["errors"] += int(not response.ok)
            self._stats["attempts"] += response.attempts
            if not response.cached:
                self._stats["tokens_in"] += response.tokens_in
                self._stats["tokens_out"] += response.tokens_out
                self._stats["cost_usd"] += response.cost_usd
        return response

    def complete(self, req: LLMRequest) -> LLMResponse:
        spec = self.models.model(req.model_key)
        settings = resolve(req, spec)
        key = cache_key(req, settings)
        # A per-key OS lock covers check, dispatch, cache write, and budget commit.
        # Waiting callers recheck the shared cache after acquiring the lock.
        lock_path = self.cache.path.parent / (self.cache.path.name + ".locks") / f"{self.mode}-{key}.lock"
        with file_lock(lock_path):
            cached = self.cache.get(key)
            if cached is not None:
                return self._record(req, settings, cached, key)
            if self.live and self.ledger.unreconciled(key):
                response = LLMResponse(text="", error="previous call has unreconciled billing; automatic replay blocked",
                                       billing_uncertain=True, mode=self.mode, model_slug=settings.slug,
                                       provider=settings.provider)
                return self._record(req, settings, response, key)
            reservation = None
            if self.live:
                try:
                    reservation = self.ledger.reserve(self.experiment, max_cost_usd(req, settings, spec), key)
                except BudgetExceeded as exc:
                    response = LLMResponse(text="", error=f"BudgetExceeded: {exc}", model_slug=settings.slug,
                                           provider=settings.provider, mode=self.mode)
                    self._record(req, settings, response, key)
                    raise
            started = time.monotonic()
            with self._stats_lock:
                self._stats["backend_calls"] += 1
            try:
                result = self.backend.complete(settings, req.messages, req.response_format, spec)
                response, attempts = result if isinstance(result, tuple) else (result, 1)
                if not isinstance(response, LLMResponse):
                    raise TypeError("backend returned an invalid response")
                response.attempts = attempts
                if not math.isfinite(response.cost_usd) or response.cost_usd < 0:
                    raise ValueError("backend returned invalid cost")
            except Exception as exc:
                # Dispatch may have succeeded before an arbitrary backend failed.
                # Conservatively retain its full reservation and block replay.
                response = LLMResponse(text="", error=f"backend {type(exc).__name__}; billing unreconciled" if self.live
                                       else f"mock backend {type(exc).__name__}", billing_uncertain=self.live,
                                       attempts=1, model_slug=settings.slug, provider=settings.provider)
            response.mode = self.mode
            response.latency_s = response.latency_s or time.monotonic() - started
            response.resource_latency_s = response.latency_s
            response.resource_cost_usd = spec.price.cost(response.tokens_in, response.tokens_out)
            if not self.live:
                response.cost_usd = 0.0
                response.billing_uncertain = False
            elif reservation and response.billing_uncertain:
                response.cost_usd = max(response.cost_usd, reservation.amount)
                response.cost_estimated = True
            # Cache before finalizing the reservation. If we crash in between, the
            # response remains reusable and the conservative reservation stays held.
            if response.ok:
                self.cache.put(key, response, settings.slug, settings.provider, req.template_id,
                               json.dumps(asdict(req), ensure_ascii=False))
            if reservation:
                if response.cost_usd or response.ok or response.billing_uncertain:
                    self.ledger.commit(reservation, response.cost_usd,
                                       {"cache_key": key, "model_key": req.model_key, "attempts": response.attempts,
                                        "unreconciled": response.billing_uncertain and not response.ok,
                                        "cost_estimated": response.cost_estimated,
                                        "generation_id": response.generation_id})
                else:
                    self.ledger.release(reservation)
            return self._record(req, settings, response, key)

    def complete_many(self, requests: Iterable[LLMRequest], *, batch: bool = False) -> list[LLMResponse]:
        """Complete concurrently in input order. This never silently changes pricing."""
        if batch:
            raise NotImplementedError("durable batch budgeting/reconciliation is not implemented; use concurrent sync calls")
        def one(req):
            try:
                return self.complete(req)
            except BudgetExceeded as exc:
                # Other concurrent requests may already have succeeded. Preserve
                # every result rather than replacing that whole group with errors.
                settings = resolve(req, self.models.model(req.model_key))
                return LLMResponse(text="", error=f"BudgetExceeded: {exc}", model_slug=settings.slug,
                                   provider=settings.provider, mode=self.mode)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(one, requests))

    def stats(self) -> dict:
        with self._stats_lock:
            return {"mode": self.mode, **self._stats}

    def summary(self) -> dict:
        return {**self.stats(), "ledger": self.ledger.summary()}

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
