"""Request/response types and settings resolution for model calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from mathmage.config import ModelSpec, load_models


@dataclass
class LLMRequest:
    """One model call.

    ``sample_tag`` distinguishes repeated samples of an identical prompt (seeds, extra
    samples) and is part of the cache key; ``meta`` (experiment, item, role, condition)
    is logged but never part of the key, so identical calls are never paid for twice.
    """

    model_key: str
    messages: list[dict[str, str]]
    template_id: str
    sample_tag: str = "s0"
    kind: str = "olympiad"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    response_format: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@dataclass
class ResolvedSettings:
    model_key: str
    slug: str
    provider: str
    max_tokens: int
    temperature: Optional[float]
    top_p: Optional[float]
    reasoning: Optional[dict[str, Any]]
    seed: Optional[int] = None
    # Ordered same-weight hosts tried after ``provider``; no other provider is ever used.
    provider_order: tuple[str, ...] = ()

    @property
    def reasoning_effort(self) -> Optional[str]:
        if not self.reasoning:
            return None
        if "effort" in self.reasoning:
            return str(self.reasoning["effort"])
        return "enabled" if self.reasoning.get("enabled") else "disabled"


@dataclass
class LLMResponse:
    text: str
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    provider: str = ""
    model_slug: str = ""
    finish_reason: str = ""
    truncated: bool = False
    cached: bool = False
    batch: bool = False
    error: Optional[str] = None
    generation_id: Optional[str] = None
    attempts: int = 0
    billing_uncertain: bool = False
    cost_estimated: bool = False
    mode: str = "live"
    resource_cost_usd: float = 0.0
    resource_latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "LLMResponse":
        return cls(**json.loads(s))


def resolve(req: LLMRequest, spec: Optional[ModelSpec] = None) -> ResolvedSettings:
    spec = spec or load_models().model(req.model_key)
    if req.max_tokens is not None and (isinstance(req.max_tokens, bool) or not isinstance(req.max_tokens, int)
                                       or req.max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer")
    if not req.messages or any(not isinstance(m.get("content"), str) for m in req.messages):
        raise ValueError("messages must contain text content")
    return ResolvedSettings(
        model_key=req.model_key,
        slug=spec.slug,
        provider=spec.provider,
        max_tokens=int(req.max_tokens or spec.max_tokens),
        temperature=req.temperature if req.temperature is not None else spec.temperature,
        top_p=req.top_p if req.top_p is not None else spec.top_p,
        reasoning=spec.reasoning,
        seed=req.seed if spec.supports_seed else None,
        provider_order=tuple(spec.provider_fallback_candidates),
    )


def cache_key(req: LLMRequest, settings: ResolvedSettings) -> str:
    payload = {
        "slug": settings.slug,
        "provider": settings.provider,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "reasoning": settings.reasoning,
        "response_format": req.response_format,
        "messages": req.messages,
        "sample_tag": req.sample_tag,
        "seed": settings.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def approx_tokens(text: str) -> int:
    """Rough token count (about 3.6 characters per token for English and LaTeX)."""
    return max(1, int(len(text) / 3.6) + 4)


def request_tokens_in(req: LLMRequest) -> int:
    return sum(approx_tokens(m.get("content", "")) + 4 for m in req.messages)
