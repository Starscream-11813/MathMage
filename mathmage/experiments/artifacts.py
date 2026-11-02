"""Content-addressed research artifacts, provenance, and atomic checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze(artifact: dict, path: Path) -> dict:
    artifact = {**artifact, "frozen": True}
    artifact.pop("sha256", None)
    artifact["sha256"] = digest(artifact)
    atomic_json(path, artifact)
    return artifact


def read_frozen(path: Path, *, live: bool = False) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = artifact.pop("sha256", None)
    if not artifact.get("frozen") or expected != digest(artifact):
        raise ValueError("offline artifact is not frozen or its content hash changed")
    if live and artifact.get("mode") != "live":
        raise ValueError("live evaluation requires artifacts measured with live models")
    provenance = artifact.get("provenance", {})
    train, dev = set(provenance.get("train_ids", [])), set(provenance.get("dev_ids", []))
    if train & dev:
        raise ValueError("training and development sets overlap")
    for card in artifact.get("cards", []):
        if card["source_item_id"] not in train:
            raise ValueError("strategy cards must come from the training split")
    return {**artifact, "sha256": expected}


def seal_record(payload: dict, *, plan_id: str, mode: str) -> dict:
    """Bind an agent-only observation to its immutable run plan and execution mode."""
    if "run_provenance" in payload:
        raise ValueError("record is already sealed")
    return {**payload, "run_provenance": {"plan_id": plan_id, "mode": mode, "sha256": digest(payload)}}


def read_record(path: Path, *, plan_id: str, mode: str, identity: dict | None = None) -> dict:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    provenance = stored.pop("run_provenance", {})
    if provenance != {"plan_id": plan_id, "mode": mode, "sha256": digest(stored)}:
        raise ValueError("saved observation provenance or content hash changed")
    if any(stored.get(key) != value for key, value in (identity or {}).items()):
        raise ValueError("saved observation belongs to a different item, seed, or condition")
    return stored
