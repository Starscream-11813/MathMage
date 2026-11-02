"""Repository paths. Everything is resolved relative to the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
REGISTRY = DATA / "registry"
CACHE = DATA / "cache"
LOGS = ROOT / "logs"
RUNS = ROOT / "runs"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
API_CONFIG = ROOT / "api" / "config.json"


def ensure_dirs() -> None:
    for p in (RAW, REGISTRY, CACHE, LOGS, RUNS, FIGURES, TABLES):
        p.mkdir(parents=True, exist_ok=True)
