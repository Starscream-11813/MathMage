"""Durable budget reservations shared by threads and processes.

Reservations are fsynced before dispatch. They survive a crash and continue to count
against both caps until explicitly reconciled. Old cost-only JSONL entries remain
readable. Corrupt ledgers fail closed.
"""
from __future__ import annotations
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from mathmage import paths
from mathmage.config import BudgetConfig, load_budget
from mathmage.llm.locking import file_lock

DEFAULT_LEDGER = paths.LOGS / "ledger.jsonl"

class BudgetExceeded(RuntimeError):
    pass

@dataclass(frozen=True)
class Reservation:
    token: str
    experiment: str
    amount: float
    request_key: Optional[str] = None

class BudgetLedger:
    def __init__(self, budget: Optional[BudgetConfig] = None, path: Path = DEFAULT_LEDGER, enforce: bool = True):
        self.budget = budget or load_budget()
        self.path = Path(path)
        self.enforce = enforce
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._pending: dict[str, Reservation] = {}
        self._unreconciled: set[str] = set()
        self.spent_by_exp: dict[str, float] = {}
        self.spent_total = 0.0
        with file_lock(self.lock_path):
            self._reload()

    def _reload(self):
        self._pending = {}
        self._unreconciled = set()
        self.spent_by_exp = {}
        self.spent_total = 0.0
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    event = entry.get("event", "commit")
                    if event == "reserve":
                        self._amount(float(entry["amount"]))
                        self._pending[entry["token"]] = Reservation(entry["token"], entry["experiment"],
                                                                   entry["amount"], entry.get("request_key"))
                    elif event == "release":
                        self._pending.pop(entry["token"], None)
                    elif event == "commit":
                        self._pending.pop(entry.get("token"), None)
                        cost = float(entry["cost_usd"])
                        self._amount(cost)
                        exp = entry["experiment"]
                        self.spent_by_exp[exp] = self.spent_by_exp.get(exp, 0.0) + cost
                        self.spent_total += cost
                        if entry.get("unreconciled") and entry.get("cache_key"):
                            self._unreconciled.add(entry["cache_key"])
                    else:
                        raise ValueError(f"unknown ledger event {event!r}")
                except (ValueError, KeyError, TypeError) as exc:
                    raise RuntimeError(f"budget ledger invalid at line {number}; reconcile before live use") from exc

    @staticmethod
    def _amount(amount: float):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("budget amounts must be finite and nonnegative")

    def _append(self, entry: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._reload()

    def line(self, experiment: str) -> float:
        if experiment not in self.budget.experiments:
            raise BudgetExceeded(f"experiment {experiment!r} has no budget line in configs/budget.yaml")
        return self.budget.experiments[experiment] * (1 + self.budget.overrun_tolerance)

    def _pending_total(self, experiment: Optional[str] = None) -> float:
        return sum(r.amount for r in self._pending.values() if experiment is None or r.experiment == experiment)

    def pending(self, experiment: Optional[str] = None) -> float:
        with file_lock(self.lock_path):
            self._reload()
            return self._pending_total(experiment)

    def remaining(self, experiment: str) -> float:
        with file_lock(self.lock_path):
            self._reload()
            return min(self.line(experiment) - self.spent_by_exp.get(experiment, 0) - self._pending_total(experiment),
                       self.budget.global_cap_usd - self.spent_total - self._pending_total())

    def unreconciled(self, request_key: str) -> bool:
        with file_lock(self.lock_path):
            self._reload()
            return request_key in self._unreconciled or any(
                r.request_key == request_key for r in self._pending.values())

    def reserve(self, experiment: str, amount: float, request_key: Optional[str] = None) -> Reservation:
        self._amount(amount)
        with file_lock(self.lock_path):
            self._reload()
            if self.enforce:
                exp_after = self.spent_by_exp.get(experiment, 0) + self._pending_total(experiment) + amount
                total_after = self.spent_total + self._pending_total() + amount
                if exp_after > self.line(experiment) + 1e-12:
                    raise BudgetExceeded(f"{experiment}: reserving ${amount:.4f} exceeds its budget line")
                if total_after > self.budget.global_cap_usd + 1e-12:
                    raise BudgetExceeded(f"reserving ${amount:.4f} exceeds the global cap")
            reservation = Reservation(uuid.uuid4().hex, experiment, amount, request_key)
            self._append({"event": "reserve", **reservation.__dict__})
            return reservation

    def commit(self, reservation: Reservation, cost_usd: float, record: dict) -> None:
        self._amount(cost_usd)
        with file_lock(self.lock_path):
            self._reload()
            if reservation.token not in self._pending:
                raise ValueError("reservation already finalized or unknown")
            self._append({**record, "event": "commit", "token": reservation.token,
                          "experiment": reservation.experiment, "cost_usd": float(cost_usd)})

    def release(self, reservation: Reservation) -> None:
        with file_lock(self.lock_path):
            self._reload()
            if reservation.token in self._pending:
                self._append({"event": "release", "token": reservation.token})

    def summary(self) -> dict:
        with file_lock(self.lock_path):
            self._reload()
            return {"spent_total": round(self.spent_total, 8), "pending_total": self._pending_total(),
                    "global_cap": self.budget.global_cap_usd,
                    "by_experiment": dict(sorted(self.spent_by_exp.items())),
                    "lines": dict(self.budget.experiments), "unreconciled_calls": len(self._unreconciled),
                    "pending_calls": len(self._pending)}
