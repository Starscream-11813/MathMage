"""Repository paths for the open-weight lane.

This lane runs from its own frozen code copy and configuration, but shares the main
repository's data, answer vault, response cache, budget ledger, call logs and run
directories, so one ledger enforces the single $75 authorization across both lanes.
"""

from __future__ import annotations

from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
ROOT = LANE.parents[1]                      # the main MathMage repository (shared state)
CONFIGS = LANE / "configs"                  # lane-specific model roster and budget lines
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
