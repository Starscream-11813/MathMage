"""Typed access to the frozen model roster and budget configuration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from mathmage import paths


class Price(BaseModel):
    """US dollars per million tokens."""

    model_config = {"populate_by_name": True, "allow_inf_nan": False}

    in_: float = Field(alias="in", ge=0)
    out: float = Field(ge=0)
    cache_read: Optional[float] = Field(default=None, ge=0)

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.in_ + tokens_out * self.out) / 1e6


class BatchSpec(BaseModel):
    slug: str
    price: Price


class ModelSpec(BaseModel):
    key: str = ""
    slug: str
    canonical_slug: str = ""
    provider: str
    provider_fallback_candidates: list[str] = []
    lineage: str
    role_notes: str = ""
    reasoning: Optional[dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: int = Field(gt=0)
    expected_out_tokens: dict[str, int] = {}
    price: Price
    batch: Optional[BatchSpec] = None
    knowledge_cutoff: Optional[str] = None
    supports_seed: bool = False

    def expected_out(self, kind: str) -> int:
        if kind in self.expected_out_tokens:
            return self.expected_out_tokens[kind]
        if self.expected_out_tokens:
            return max(self.expected_out_tokens.values())
        return self.max_tokens

    @property
    def reasoning_effort(self) -> Optional[str]:
        if self.reasoning and "effort" in self.reasoning:
            return str(self.reasoning["effort"])
        if self.reasoning and self.reasoning.get("enabled") is True:
            return "enabled"
        return None


class RosterSpec(BaseModel):
    members: list[str]
    checker: str
    card_writer: str
    default_judge: str


class ModelsConfig(BaseModel):
    frozen_at: str
    models: dict[str, ModelSpec]
    rosters: dict[str, RosterSpec]

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(f"unknown model key {key!r}; known: {sorted(self.models)}")
        return self.models[key]


class BudgetConfig(BaseModel):
    model_config = {"allow_inf_nan": False}

    global_cap_usd: float = Field(gt=0)
    experiments: dict[str, float]
    overrun_tolerance: float = Field(default=0.20, ge=0)
    estimate_safety_factor: float = Field(default=2.5, gt=0)

    @field_validator("experiments")
    @classmethod
    def nonnegative_lines(cls, lines):
        if any(value < 0 for value in lines.values()):
            raise ValueError("experiment budget lines must be nonnegative")
        return lines


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def load_models(path: Optional[str] = None) -> ModelsConfig:
    raw = _load_yaml(Path(path) if path else paths.CONFIGS / "models.yaml")
    for k, v in raw["models"].items():
        v["key"] = k
    return ModelsConfig(**raw)


@lru_cache(maxsize=4)
def load_budget(path: Optional[str] = None) -> BudgetConfig:
    return BudgetConfig(**_load_yaml(Path(path) if path else paths.CONFIGS / "budget.yaml"))


def load_api_key(name: str = "OPENROUTER_API_KEY") -> Optional[str]:
    """Environment variable first, then the gitignored api/config.json."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    if paths.API_CONFIG.exists():
        with open(paths.API_CONFIG, encoding="utf-8") as f:
            return json.load(f).get(name)
    return None
