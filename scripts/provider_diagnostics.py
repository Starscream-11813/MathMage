"""Operational provider diagnostics without changing frozen research code.

Only failed OpenRouter chat response metadata is recorded. Prompts, request
headers, credentials, and successful response bodies are never inspected.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit


def redact(value, limit=2000):
    """Bound error text and remove common credential formats before persisting."""
    value = str(value)
    # Providers sometimes nest serialized JSON inside metadata.raw.
    value = re.sub(r"\\([\"'])", r"\1", value)
    value = re.sub(r"(?i)\bBearer\s+[^\s\"',;)}]+", "Bearer [REDACTED]", value)
    value = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(
        r"(?i)([\"']?(?:authorization|api[_-]?key|access[_-]?token|secret)[\"']?\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
        r"\1[REDACTED]", value)
    return value[:limit]


def retry_after_seconds(value, *, now=None):
    """Accept RFC Retry-After delta seconds or an HTTP-date; reject bad hints."""
    if value is None:
        return None
    now = time.time() if now is None else now
    try:
        text = str(value).strip()
        if re.fullmatch(r"[0-9]+", text):
            return float(text)
        when = parsedate_to_datetime(text)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, when.timestamp() - now)
    except (ValueError, TypeError, OverflowError):
        return None


class ProviderDiagnostics:
    def __init__(self, root, qwen_max_attempts=1):
        if (isinstance(qwen_max_attempts, bool)
                or not isinstance(qwen_max_attempts, int) or qwen_max_attempts < 1):
            raise ValueError("qwen_max_attempts must be a positive integer")
        self.root = Path(root)
        self.qwen_max_attempts = qwen_max_attempts
        self._lock = threading.Lock()
        self._backend_locks = {}
        self._backoff_until = {}
        self._restored_backoff = False
        self.diagnostic_errors = 0

    def _backend_lock(self, backend):
        with self._lock:
            return self._backend_locks.setdefault(id(backend), threading.RLock())

    def last_provider_backoff(self, provider_or_settings):
        provider = getattr(provider_or_settings, "provider", provider_or_settings)
        with self._lock:
            if not self._restored_backoff:
                self._restore_backoff()
                self._restored_backoff = True
            return max(0.0, self._backoff_until.get(str(provider), 0.0) - time.time())

    def _restore_backoff(self):
        """Retain provider hints across a crash before the retry state is saved.

        Called under the state lock once per adapter. Only the persisted provider,
        response timestamp and parsed delay are used; error bodies stay unread.
        A partial or malformed JSONL record cannot erase earlier valid hints.
        """
        try:
            with (self.root / "provider_diagnostics.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        provider, delay = record["provider"], record["retry_after_seconds"]
                        if (not isinstance(provider, str) or type(delay) not in (int, float)
                                or not math.isfinite(delay) or delay < 0):
                            continue
                        stamp = datetime.fromisoformat(record["time_utc"])
                        if stamp.tzinfo is None:
                            continue
                        deadline = stamp.timestamp() + delay
                        if not math.isfinite(deadline):
                            continue
                        self._backoff_until[provider] = max(self._backoff_until.get(provider, 0.0), deadline)
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
        except FileNotFoundError:
            pass
        except Exception:
            self.diagnostic_errors += 1

    def _capture(self, response, settings):
        try:
            if response.status_code == 200:
                return
            url = getattr(response, "url", None)
            if not url:
                url = getattr(getattr(response, "request", None), "url", "")
            location = urlsplit(url)
            if (location.hostname != "openrouter.ai"
                    or location.path.rstrip("/") != "/api/v1/chat/completions"):
                return
            now = time.time()
            headers = {str(k).lower(): v for k, v in response.headers.items()}
            selected = {k: redact(value, 300) for k, value in headers.items()
                        if k == "retry-after" or k.startswith(("x-ratelimit-", "ratelimit-"))}
            backoff = retry_after_seconds(headers.get("retry-after"), now=now)
            record = {
                "time_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "http_status": int(response.status_code),
                "model_slug": redact(settings.slug, 200),
                "provider": redact(settings.provider, 200),
                "headers": selected,
                "retry_after_seconds": backoff,
            }
            try:
                payload = response.json()
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
                if isinstance(error, dict):
                    record["error"] = {k: redact(error[k], 2000) for k in ("code", "message")
                                       if k in error}
                    metadata = error.get("metadata")
                    if isinstance(metadata, dict):
                        record["metadata"] = {
                            k: redact(metadata[k], 4000 if k == "raw" else 500)
                            for k in ("error_type", "provider_name", "provider_code", "raw")
                            if k in metadata}
            except (ValueError, TypeError):
                record["error_body"] = "not JSON; body deliberately omitted"
            with self._lock:
                if backoff is not None:
                    name = str(settings.provider)
                    self._backoff_until[name] = max(self._backoff_until.get(name, 0.0), now + backoff)
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / "provider_diagnostics.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            # Diagnostic failures must not alter the transport result or billing.
            self.diagnostic_errors += 1

    def complete(self, original_backend, backend, settings, messages, response_format, spec):
        qwen = settings.provider == "SiliconFlow" and settings.slug == "qwen/qwen3-32b"
        # Qwen variants share a backend and the outside pacer already serializes
        # them. This lock additionally protects temporary attempt-limit changes.
        with self._backend_lock(backend) if qwen else nullcontext():
            old_attempts = getattr(backend, "max_attempts", None)
            hooks = None
            old_response_hooks = None
            had_response_hooks = False

            def capture(response, *args, **kwargs):
                try:
                    self._capture(response, settings)
                except Exception:
                    self.diagnostic_errors += 1
                return None  # requests keeps the original response unchanged.

            try:
                try:
                    session = backend._session()
                    hooks = session.hooks
                    had_response_hooks = "response" in hooks
                    old_response_hooks = hooks.get("response")
                    previous = old_response_hooks or []
                    if callable(previous):
                        previous = [previous]
                    # Capture before other hooks can replace the response. The
                    # original hook objects/list are restored exactly below.
                    hooks["response"] = [capture, *previous]
                except Exception:
                    self.diagnostic_errors += 1
                    hooks = None
                if qwen and old_attempts is not None:
                    backend.max_attempts = self.qwen_max_attempts
                return original_backend(backend, settings, messages, response_format, spec)
            finally:
                if qwen and old_attempts is not None:
                    backend.max_attempts = old_attempts
                if hooks is not None:
                    if had_response_hooks:
                        hooks["response"] = old_response_hooks
                    else:
                        hooks.pop("response", None)
