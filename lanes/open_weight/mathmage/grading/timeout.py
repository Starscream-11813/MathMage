"""Thread-based timeouts for symbolic computations.

math-verify's own timeouts use signals or subprocesses, which are unavailable or fragile
on Windows, so its internal timeouts are disabled and calls are wrapped here instead. A
computation that overruns keeps running in a daemon thread but no longer blocks.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logging.getLogger("math_verify").setLevel(logging.ERROR)
logging.getLogger("latex2sympy2_extended").setLevel(logging.ERROR)


def call_with_timeout(fn: Callable[..., T], *args: Any, timeout: float = 5.0, default: Any = None, **kwargs: Any) -> T:
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - grading must never crash a run
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or "error" in box:
        return default
    return box.get("value", default)
