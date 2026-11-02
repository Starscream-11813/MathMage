"""OpenRouter backends: synchronous chat completions and the asynchronous Batch API.

Provider pinning: every request names one provider, disables fallbacks, and requires the
provider to support every sent parameter, so the serving stack is fixed and logged.
Only explicit rate limits and connect timeouts are retried. Read timeouts, disconnects,
malformed successful responses, and gateway failures may already have been billed;
they are marked uncertain and never automatically replayed.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from mathmage import paths
from mathmage.config import ModelSpec, load_api_key
from mathmage.llm.types import LLMResponse, ResolvedSettings

API = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
HEADERS_EXTRA = {"HTTP-Referer": "https://github.com/MathMage/MathMage", "X-Title": "MathMage"}


class NonRetryableError(RuntimeError):
    pass


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **HEADERS_EXTRA}


def build_body(settings: ResolvedSettings, messages: list[dict], response_format: Optional[dict],
               include_model: bool = True, include_provider: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if include_model:
        body["model"] = settings.slug
    body["messages"] = messages
    body["max_tokens"] = settings.max_tokens
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.top_p is not None:
        body["top_p"] = settings.top_p
    if settings.reasoning is not None:
        body["reasoning"] = settings.reasoning
    if settings.seed is not None:
        body["seed"] = settings.seed
    if response_format is not None:
        body["response_format"] = response_format
    if include_provider:
        # The lane lists a short vetted set of hosts serving the same weights; OpenRouter
        # tries them in order and never routes outside the list (allow_fallbacks=False).
        body["provider"] = {"order": [settings.provider, *getattr(settings, "provider_order", ())],
                            "allow_fallbacks": False, "require_parameters": True}
    body["usage"] = {"include": True}
    return body


def parse_completion(data: dict, settings: ResolvedSettings, price_fallback=None) -> LLMResponse:
    if "error" in data and not data.get("choices"):
        err = data["error"]
        raise RuntimeError(f"provider error: {json.dumps(err)[:500]}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response: {json.dumps(data)[:300]}")
    ch = choices[0]
    msg = ch.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    tin = int(usage.get("prompt_tokens") or 0)
    tout = int(usage.get("completion_tokens") or 0)
    cost = usage.get("cost")
    estimated = cost is None
    if cost is None and price_fallback is not None:
        cost = price_fallback.cost(tin, tout)
    finish = ch.get("finish_reason") or ch.get("native_finish_reason") or ""
    return LLMResponse(
        text=msg.get("content") or "",
        reasoning=msg.get("reasoning") or "",
        tokens_in=tin,
        tokens_out=tout,
        tokens_reasoning=int(details.get("reasoning_tokens") or 0),
        cost_usd=float(cost or 0.0),
        provider=data.get("provider") or settings.provider,
        model_slug=data.get("model") or settings.slug,
        finish_reason=str(finish),
        truncated=str(finish) == "length",
        generation_id=data.get("id"),
        cost_estimated=estimated,
        billing_uncertain=estimated and not (tin or tout),
    )


class OpenRouterSync:
    is_live = True
    def __init__(self, api_key: Optional[str] = None, read_timeout: float = 900.0, max_attempts: int = 6,
                 *, live: bool = False):
        if not live:
            raise NonRetryableError("paid completion transport requires explicit live=True")
        self.key = api_key or load_api_key()
        if not self.key:
            raise NonRetryableError("OPENROUTER_API_KEY not configured (api/config.json or environment)")
        self.read_timeout = read_timeout
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def complete(self, settings: ResolvedSettings, messages: list[dict], response_format: Optional[dict],
                 spec: ModelSpec) -> tuple[LLMResponse, int]:
        """Returns (response, number_of_attempts)."""
        body = build_body(settings, messages, response_format)
        last_err = ""
        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                r = self._session().post(f"{API}/chat/completions", headers=_headers(self.key), json=body,
                                         timeout=(30, self.read_timeout))
                if r.status_code == 429:
                    last_err = "HTTP 429 rate limited"
                elif r.status_code != 200:
                    uncertain = r.status_code not in {400, 401, 402, 403, 404, 405, 413, 422}
                    return LLMResponse(text="", error=f"HTTP {r.status_code}", billing_uncertain=uncertain,
                                       attempts=attempt, latency_s=time.monotonic() - started,
                                       model_slug=settings.slug, provider=settings.provider), attempt
                else:
                    try:
                        data = r.json()
                        resp = parse_completion(data, settings, spec.price)
                        resp.latency_s = time.monotonic() - started
                        resp.attempts = attempt
                        return resp, attempt
                    except (RuntimeError, ValueError, TypeError, KeyError):
                        return LLMResponse(text="", error="invalid or failed provider response; billing unreconciled",
                                           billing_uncertain=True, attempts=attempt,
                                           latency_s=time.monotonic() - started,
                                           model_slug=settings.slug, provider=settings.provider), attempt
            except requests.ConnectTimeout:
                last_err = "ConnectTimeout"
            except requests.RequestException as exc:
                return LLMResponse(text="", error=f"{type(exc).__name__}; billing unreconciled",
                                   billing_uncertain=True, attempts=attempt,
                                   latency_s=time.monotonic() - started,
                                   model_slug=settings.slug, provider=settings.provider), attempt
            if attempt < self.max_attempts:
                time.sleep(min(90.0, 2.0 ** attempt) + random.uniform(0, 1.5))
        return LLMResponse(text="", error=f"failed after {self.max_attempts} attempts: {last_err}",
                           attempts=self.max_attempts, latency_s=time.monotonic() - started,
                           model_slug=settings.slug, provider=settings.provider), self.max_attempts

    # -------------------------------------------------------------- account checks (free)
    def key_info(self) -> dict:
        r = requests.get(f"{API}/key", headers=_headers(self.key), timeout=30)
        if r.status_code != 200:
            raise NonRetryableError(f"key check failed: HTTP {r.status_code}: {r.text[:300]}")
        return r.json().get("data", {})

    def credits(self) -> dict:
        r = requests.get(f"{API}/credits", headers=_headers(self.key), timeout=30)
        if r.status_code != 200:
            raise NonRetryableError(f"credits check failed: HTTP {r.status_code}: {r.text[:300]}")
        return r.json().get("data", {})


@dataclass
class BatchJob:
    batch_id: str
    model_key: str
    custom_ids: list[str]
    submitted_at: float


class OpenRouterBatch:
    """Verified API shape: https://openrouter.ai/docs/batch-quickstart (2026-10-18).

    Low-level transport only. LLMClient does not submit batches until durable job and
    budget reconciliation is implemented; use its concurrent synchronous calls.
    """

    STATE = paths.LOGS / "batches.jsonl"
    TERMINAL = {"completed", "failed", "expired", "cancelled"}

    def __init__(self, api_key: Optional[str] = None, poll_s: float = 30.0, max_wait_s: float = 26 * 3600,
                 *, live: bool = False):
        if not live:
            raise NonRetryableError("batch submission requires explicit live=True")
        self.key = api_key or load_api_key()
        if not self.key:
            raise NonRetryableError("OPENROUTER_API_KEY not configured")
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s

    def submit(self, spec: ModelSpec, settings_list: list[ResolvedSettings], messages_list: list[list[dict]],
               formats: list[Optional[dict]], custom_ids: list[str]) -> BatchJob:
        assert spec.batch is not None, f"{spec.key} has no batch variant"
        reqs = []
        for s, m, f, cid in zip(settings_list, messages_list, formats, custom_ids):
            reqs.append({"custom_id": cid, "body": build_body(s, m, f, include_model=False, include_provider=False)})
        # Field order matters: endpoint, model, provider, completion_window before requests.
        payload = {"endpoint": "/v1/chat/completions", "model": spec.slug,
                   "provider": {"only": [spec.provider], "allow_fallbacks": False},
                   "completion_window": "24h", "requests": reqs}
        r = requests.post(f"{API}/batches", headers=_headers(self.key), data=json.dumps(payload), timeout=300)
        if r.status_code not in (200, 201, 202):
            raise NonRetryableError(f"batch submit failed: HTTP {r.status_code}: {r.text[:800]}")
        bid = r.json().get("id") or r.json().get("data", {}).get("id")
        job = BatchJob(bid, spec.key, list(custom_ids), time.time())
        self.STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"batch_id": bid, "model_key": spec.key, "n": len(custom_ids),
                                 "submitted_at": job.submitted_at, "custom_ids": custom_ids}) + "\n")
        return job

    def status(self, batch_id: str) -> dict:
        r = requests.get(f"{API}/batches/{batch_id}", headers=_headers(self.key), timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"batch poll failed: HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        return d.get("data", d)

    def wait(self, job: BatchJob) -> dict:
        t_end = time.time() + self.max_wait_s
        while True:
            try:
                st = self.status(job.batch_id)
                if st.get("status") in self.TERMINAL:
                    return st
            except RuntimeError:
                pass
            if time.time() > t_end:
                raise TimeoutError(f"batch {job.batch_id} did not finish within {self.max_wait_s/3600:.1f} h")
            time.sleep(self.poll_s)

    @staticmethod
    def results(state: dict, spec: ModelSpec, settings_by_id: dict[str, ResolvedSettings]) -> dict[str, LLMResponse]:
        out: dict[str, LLMResponse] = {}
        assert spec.batch is not None
        for res in state.get("results") or []:
            cid = res.get("custom_id")
            s = settings_by_id.get(cid)
            if s is None:
                continue
            if res.get("error"):
                out[cid] = LLMResponse(text="", error=f"batch item error: {json.dumps(res['error'])[:300]}", batch=True)
                continue
            resp = res.get("response") or {}
            body = resp.get("body", resp)
            try:
                lr = parse_completion(body, s, spec.batch.price)
                # Bill at batch prices when the body does not carry a cost.
                if not (body.get("usage") or {}).get("cost"):
                    lr.cost_usd = spec.batch.price.cost(lr.tokens_in, lr.tokens_out)
                lr.batch = True
                out[cid] = lr
            except RuntimeError as e:
                out[cid] = LLMResponse(text="", error=str(e), batch=True)
        return out
