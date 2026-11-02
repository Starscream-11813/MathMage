"""Deterministic offline responses for exercising orchestration without paid calls.

These are plumbing fixtures, not model predictions or experiment results. They never
read reference answers, use the network, execute code, or infer correctness labels.
"""
from __future__ import annotations
import hashlib
import json
from mathmage.config import ModelSpec
from mathmage.llm.openrouter import build_body
from mathmage.llm.types import LLMResponse, ResolvedSettings, approx_tokens

class MockBackend:
    is_live = False

    def complete(self, settings: ResolvedSettings, messages: list[dict], response_format: dict | None,
                 spec: ModelSpec) -> tuple[LLMResponse, int]:
        payload = json.dumps(build_body(settings, messages, response_format), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        prompt = "\n".join(m.get("content", "") for m in messages)
        value = int(digest[:8], 16) % 1000
        if '"first_error_line"' in prompt:
            text = json.dumps({"verdict": "unsure", "first_error_line": None,
                               "diagnosis": "Mock review; no mathematical validation performed.",
                               "check_code": "", "confidence": 0.0})
        elif '"choice"' in prompt:
            text = '[MOCK audit; no mathematical validation performed.]\n' + json.dumps({"choice": 1, "confidence": 0.0})
        elif "Return only the program" in prompt:
            text = '```python\n# Offline fixture; does not solve this problem.\nprint("ANSWER: NONE")\n```'
        elif '"applies_when"' in prompt:
            text = json.dumps({"method": "Check each stated constraint.", "applies_when": "Offline fixture only.",
                               "pitfall": "Mock outputs do not establish correctness.", "checklist": ["Check definitions."]})
        elif "Do not state, guess, or compute the final answer" in prompt:
            text = "[MOCK] Check the definitions and break the argument into smaller justified steps."
        elif response_format and response_format.get("type") == "json_object":
            text = json.dumps({"mock": True, "fixture_id": digest[:12]})
        else:
            text = (f"[MOCK fixture {digest[:12]}; not a mathematical solution.]\n"
                    f"CERTIFICATE\n1. Offline orchestration fixture only.\n\\boxed{{{value}}}")
        response = LLMResponse(text=text, model_slug=settings.slug, provider="mock",
                               tokens_in=sum(approx_tokens(m.get("content", "")) for m in messages),
                               tokens_out=approx_tokens(text), finish_reason="stop", attempts=1, mode="mock")
        return response, 1
