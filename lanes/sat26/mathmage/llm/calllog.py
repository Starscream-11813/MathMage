"""Append-only call log: one JSON line per model call (research plan, Appendix B schema).

``correct`` is always null at call time: correctness is attached later by the grader in
separate files, so nothing an agent can read ever contains a correctness label.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mathmage import paths
from mathmage.llm.types import LLMRequest, LLMResponse, ResolvedSettings
from mathmage.llm.locking import file_lock

RUN_ID = uuid.uuid4().hex[:12]

SCHEMA_META_FIELDS = (
    "item_id", "family_id", "band", "source", "item_date", "condition", "scaffold_type", "level", "scaffold_id",
    "leakage", "valid", "tutor_slug", "blinded", "framing", "checkpoint_id", "attempt_idx", "role", "stage",
    "variant", "seed",
)


class CallLog:
    def __init__(self, experiment: str, directory: Path = paths.LOGS / "calls"):
        directory.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.path = directory / f"{experiment}.jsonl"
        self._lock = threading.Lock()

    def write(self, req: LLMRequest, settings: ResolvedSettings, resp: LLMResponse, key: str,
              parsed_answer: Optional[str] = None, tool_calls: Optional[int] = None) -> None:
        rec: dict[str, Any] = {
            "run_id": RUN_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": self.experiment,
            "model_key": settings.model_key,
            "model_slug": resp.model_slug or settings.slug,
            "provider": resp.provider or settings.provider,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "reasoning_effort": settings.reasoning_effort,
            "prompt_template_id": req.template_id,
            "sample_tag": req.sample_tag,
        }
        for f in SCHEMA_META_FIELDS:
            rec[f] = req.meta.get(f)
        rec["seed_requested"] = req.seed
        rec["seed_sent"] = settings.seed
        rec.update({
            "response": resp.text,
            "parsed_answer": parsed_answer,
            "correct": None,
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "tokens_reasoning": resp.tokens_reasoning,
            "cost_usd": resp.cost_usd,
            "latency_s": round(resp.latency_s, 3),
            "tool_calls": tool_calls,
            "truncated": resp.truncated,
            "error": resp.error,
            "cached": resp.cached,
            "batch": resp.batch,
            "finish_reason": resp.finish_reason,
            "generation_id": resp.generation_id,
            "cache_key": key,
            "attempts": resp.attempts,
            "billing_uncertain": resp.billing_uncertain,
            "cost_estimated": resp.cost_estimated,
            "mode": resp.mode,
            "resource_cost_usd": resp.resource_cost_usd,
            "resource_latency_s": resp.resource_latency_s,
        })
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, file_lock(self.path.with_suffix(".jsonl.lock")):
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
