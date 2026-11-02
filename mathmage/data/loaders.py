"""Download raw benchmark data and convert it into registry items plus vault records.

Sources (all public):
  * MathArena competition sets on the Hugging Face Hub (CC BY-NC-SA 4.0):
    AIME 2023 I/II, AIME 2024 I/II, AIME 2025, AIME 2026, HMMT February 2025 and 2026.
  * TheoremQA test set from the original release (github.com/wenhuchen/TheoremQA, MIT),
    which carries the ``field`` label needed to isolate the physics subset.
  * MATH-500 (HuggingFaceH4/MATH-500).
Raw files are cached under data/raw/ and never modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from mathmage import paths

TQA_URL = "https://raw.githubusercontent.com/wenhuchen/TheoremQA/main/theoremqa_test.json"

# Documented corrections of source transcriptions: item_id -> (wrong text, official text, note).
ERRATA = {
    "aime_2026/10": (
        r"so that ${}\overline{AC}$ is perpendicular $\overline{BC},$",
        r"so that $\overline{A'C'}$ is perpendicular to $\overline{BC},$",
        "MathArena's transcription drops the primes; the official AIME 2026 I Problem 10 rotates the triangle "
        "so that the image side A'C' is perpendicular to BC (official answer 156). With the garbled text "
        "MathArena also accepted 165; with the official text only 156 is correct.",
    ),
}


def apply_errata(item_id: str, problem: str) -> str:
    if item_id in ERRATA:
        wrong, right, _ = ERRATA[item_id]
        assert wrong in problem, f"erratum for {item_id} no longer matches the source text"
        problem = problem.replace(wrong, right)
    return problem


@dataclass
class RawItem:
    """An item before families and splits are assigned. ``answer`` goes to the vault."""

    item_id: str
    benchmark: str
    source: str
    problem: str
    answer: str
    answer_type: str          # integer | latex | float | list_float | list_integer
    answer_format: str
    domain: str
    item_date: str
    band: str
    kind: str
    solution: str = ""
    extra: Optional[dict[str, Any]] = None


def _hf_parquet(repo: str, filename: str = "data/train-00000-of-00001.parquet") -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                            cache_dir=str(paths.RAW / "hf"))
    return pd.read_parquet(local)


def _domain(v: Any) -> str:
    """MathArena stores problem types as a list (sometimes a string repr of one)."""
    if v is None:
        return ""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                import ast

                return "; ".join(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return s
        return s
    try:
        return "; ".join(str(x) for x in list(v))
    except TypeError:
        return str(v)


def load_aime(year: int) -> list[RawItem]:
    """Thirty AIME problems; ids 01-15 are AIME I and 16-30 are AIME II."""
    if year in (2025, 2026):
        df = _hf_parquet(f"MathArena/aime_{year}")
        rows = [(int(r["problem_idx"]), r) for r in df.to_dict("records")]
        source = f"MathArena/aime_{year}"
    else:
        rows = []
        for part, offset in (("I", 0), ("II", 15)):
            df = _hf_parquet(f"MathArena/aime_{year}_{part}")
            rows += [(offset + int(r["problem_idx"]), r) for r in df.to_dict("records")]
        source = f"MathArena/aime_{year}_I+II"
    out = []
    for idx, r in sorted(rows, key=lambda t: t[0]):
        iid = f"aime_{year}/{idx:02d}"
        out.append(RawItem(
            item_id=iid, benchmark=f"aime_{year}", source=source,
            problem=apply_errata(iid, str(r["problem"]).strip()), answer=str(r["answer"]).strip(), answer_type="integer",
            answer_format="aime_integer", domain=_domain(r.get("problem_type")),
            item_date=f"{year}-02", band="olympiad", kind="olympiad",
            extra={"exam": "I" if idx <= 15 else "II", "number": idx if idx <= 15 else idx - 15}))
    assert len(out) == 30, (year, len(out))
    return out


def load_hmmt_feb(year: int) -> list[RawItem]:
    df = _hf_parquet(f"MathArena/hmmt_feb_{year}")
    out = []
    for r in sorted(df.to_dict("records"), key=lambda r: int(r["problem_idx"])):
        idx = int(r["problem_idx"])
        ans = str(r["answer"]).strip()
        out.append(RawItem(
            item_id=f"hmmt_feb_{year}/{idx:02d}", benchmark=f"hmmt_feb_{year}", source=f"MathArena/hmmt_feb_{year}",
            problem=str(r["problem"]).strip(), answer=ans,
            answer_type="integer" if ans.lstrip("-").isdigit() else "latex", answer_format="exact",
            domain=_domain(r.get("problem_type")), item_date=f"{year}-02", band="olympiad", kind="olympiad"))
    return out


def _tqa_raw() -> list[dict]:
    dest = paths.RAW / "theoremqa_test.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(TQA_URL, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    with open(dest, encoding="utf-8") as f:
        return json.load(f)


def load_tqa_physics(include_pictures: bool = False) -> list[RawItem]:
    """TheoremQA physics items (131; 114 are text-only). Picture items are excluded by
    default because one roster member (DeepSeek-V3) is text-only."""
    tmap = {"float": ("float", "tqa_float"), "integer": ("integer", "tqa_integer"),
            "list of float": ("list_float", "tqa_list_float"), "list of integer": ("list_integer", "tqa_list_integer")}
    out = []
    for r in _tqa_raw():
        if r["field"] != "Physics" or (r.get("Picture") and not include_pictures):
            continue
        atype, fmt = tmap[r["Answer_type"]]
        tid = r["id"].replace(".json", "")
        out.append(RawItem(
            item_id=f"tqa_phys/{tid}", benchmark="tqa_phys", source="TheoremQA (wenhuchen/TheoremQA)",
            problem=str(r["Question"]).strip(), answer=json.dumps(r["Answer"]), answer_type=atype,
            answer_format=fmt, domain=str(r.get("subfield", "")), item_date="2023-05", band="physics",
            kind="physics", solution="" if r.get("explanation") in (None, "NONE") else str(r["explanation"]),
            extra={"theorem": r.get("theorem"), "tqa_id": r["id"]}))
    return out


def load_math500() -> list[RawItem]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id="HuggingFaceH4/MATH-500", filename="test.jsonl", repo_type="dataset",
                            cache_dir=str(paths.RAW / "hf"))
    out = []
    with open(local, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r["unique_id"].replace("test/", "").replace(".json", "")
            out.append(RawItem(
                item_id=f"math500/{uid}", benchmark="math500", source="HuggingFaceH4/MATH-500",
                problem=r["problem"].strip(), answer=r["answer"].strip(), answer_type="latex",
                answer_format="exact", domain=r["subject"], item_date="2021-03", band="foundations",
                kind="math500", solution=r["solution"], extra={"level": int(r["level"]), "subject": r["subject"]}))
    assert len(out) == 500
    return out


def load_all_raw() -> list[RawItem]:
    items: list[RawItem] = []
    for y in (2023, 2024, 2025, 2026):
        items += load_aime(y)
    for y in (2025, 2026):
        items += load_hmmt_feb(y)
    items += load_tqa_physics()
    items += load_math500()
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


def matharena_outputs(competition: str, columns: Optional[list[str]] = None,
                      local_files_only: bool = False) -> pd.DataFrame:
    """Public per-attempt MathArena logs (e.g. 'aime_2025', 'hmmt_feb_2026')."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=f"MathArena/{competition}_outputs",
                            filename="data/train-00000-of-00001.parquet", repo_type="dataset",
                            cache_dir=str(paths.RAW / "hf"), local_files_only=local_files_only)
    frame = pd.read_parquet(local, columns=columns)
    frame.attrs["source_repo"] = f"MathArena/{competition}_outputs"
    frame.attrs["source_revision"] = Path(local).parent.parent.name
    return frame


def write_raw_snapshot(items: list[RawItem], path: Path) -> None:
    """A human-readable snapshot of the raw items (without answers) for auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"item_id": it.item_id, "problem": it.problem}, ensure_ascii=False) + "\n")
