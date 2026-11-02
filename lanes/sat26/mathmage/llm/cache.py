"""Persistent response cache keyed by a hash of (model, provider, settings, prompt, sample tag).

Only successful responses are cached. The client combines this cache with a per-key
process lock and the ledger's unresolved-call blocks to avoid duplicate paid calls.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mathmage import paths
from mathmage.llm.types import LLMResponse

DEFAULT_CACHE = paths.CACHE / "llm_cache.sqlite"


class ResponseCache:
    def __init__(self, path: Path = DEFAULT_CACHE, namespace: str = "live"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.namespace = namespace  # "live" and "mock" responses never mix
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses (namespace TEXT, key TEXT, created TEXT, model_slug TEXT, "
                "provider TEXT, template_id TEXT, request_json TEXT, response_json TEXT, PRIMARY KEY(namespace, key))")
            self._conn.commit()

    def get(self, key: str) -> Optional[LLMResponse]:
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM responses WHERE namespace=? AND key=?",
                                     (self.namespace, key)).fetchone()
        if row is None:
            return None
        r = LLMResponse.from_json(row[0])
        r.cached = True
        r.cost_usd = 0.0       # a cache hit costs nothing now; the original cost is in the ledger
        r.latency_s = 0.0
        r.attempts = 0
        return r

    def put(self, key: str, response: LLMResponse, model_slug: str, provider: str, template_id: str,
            request_json: str) -> None:
        if not response.ok:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                (self.namespace, key, datetime.now(timezone.utc).isoformat(), model_slug, provider, template_id,
                 request_json, response.to_json()))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses WHERE namespace=?",
                                          (self.namespace,)).fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
