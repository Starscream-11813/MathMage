"""Problem families, train/dev/test splits, and named evaluation subsets.

Rules (research plan, "Family-level splits and ledgers"):
  * Near-duplicate problems (cosine similarity of sentence embeddings >= threshold, or
    identical normalized text) are grouped into one family before any split.
  * Learning (calibration, tutoring matrix, strategy cards, thresholds) uses only the
    training pool: AIME 2023 and HMMT February 2025. Every other item is test.
  * A training item that shares a family with any test item is excluded from learning.
  * The training pool is split by family into train and dev, stratified by competition.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Optional

import numpy as np

from mathmage.data.loaders import RawItem
from mathmage.data.registry import Item

TRAIN_POOL = {"aime_2023", "hmmt_feb_2025"}
DEV_FRACTION = 1 / 3


def _stable_hash(s: str, salt: str = "mathmage-v2") -> int:
    return int(hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:12], 16)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def embed(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))


def near_duplicate_families(ids: list[str], texts: list[str], threshold: float = 0.92,
                            embeddings: Optional[np.ndarray] = None) -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    """Return {item_id: family_id} and the list of near-duplicate pairs found."""
    n = len(ids)
    uf = _UnionFind(n)
    pairs: list[tuple[str, str, float]] = []
    norm = [_normalize(t) for t in texts]
    seen: dict[str, int] = {}
    for i, t in enumerate(norm):
        if t in seen:
            uf.union(seen[t], i)
            pairs.append((ids[seen[t]], ids[i], 1.0))
        else:
            seen[t] = i
    if embeddings is None:
        embeddings = embed(texts)
    sims = embeddings @ embeddings.T
    iu, ju = np.where(np.triu(sims, k=1) >= threshold)
    for i, j in zip(iu.tolist(), ju.tolist()):
        if norm[i] != norm[j]:
            pairs.append((ids[i], ids[j], float(sims[i, j])))
        uf.union(i, j)
    fam = {}
    for i in range(n):
        root = uf.find(i)
        fam[ids[i]] = f"fam-{ids[root]}"
    return fam, pairs


def assign_splits(raw: list[RawItem], families: dict[str, str]) -> list[Item]:
    test_families = {families[r.item_id] for r in raw if r.benchmark not in TRAIN_POOL}
    # Stratified family-level split of the training pool.
    by_bench: dict[str, list[str]] = {}
    for r in raw:
        if r.benchmark in TRAIN_POOL:
            by_bench.setdefault(r.benchmark, [])
            f = families[r.item_id]
            if f not in by_bench[r.benchmark]:
                by_bench[r.benchmark].append(f)
    dev_families: set[str] = set()
    for bench, fams in by_bench.items():
        clean = sorted((f for f in fams if f not in test_families), key=_stable_hash)
        k = int(round(len(clean) * DEV_FRACTION))
        dev_families.update(clean[:k])
    items = []
    for r in raw:
        f = families[r.item_id]
        if r.benchmark in TRAIN_POOL:
            split = "excluded" if f in test_families else ("dev" if f in dev_families else "train")
        else:
            split = "test"
        items.append(Item(
            item_id=r.item_id, benchmark=r.benchmark, source=r.source, problem=r.problem,
            answer_format=r.answer_format, domain=r.domain, family_id=f, split=split,
            item_date=r.item_date, band=r.band, kind=r.kind,
            extra=json.dumps(r.extra or {}, ensure_ascii=False)))
    return items


def stratified_sample(items: Iterable[Item], k: int, key, salt: str) -> list[str]:
    """Proportional stratified sample of k item ids (largest-remainder allocation)."""
    items = list(items)
    strata: dict[str, list[Item]] = {}
    for it in items:
        strata.setdefault(key(it), []).append(it)
    n = len(items)
    quotas = {s: k * len(v) / n for s, v in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = k - sum(alloc.values())
    for s in sorted(quotas, key=lambda s: (-(quotas[s] - alloc[s]), s))[:rest]:
        alloc[s] += 1
    chosen = []
    for s, v in sorted(strata.items()):
        v = sorted(v, key=lambda it: _stable_hash(it.item_id, salt))
        chosen += [it.item_id for it in v[: alloc[s]]]
    return sorted(chosen)


def make_subsets(items: list[Item]) -> dict[str, list[str]]:
    by = lambda b: [it for it in items if it.benchmark == b]  # noqa: E731
    math500 = by("math500")
    hmmt26 = by("hmmt_feb_2026")
    subsets = {
        # Xolver comparison: stratified by subject x level, as in the blueprint.
        "math500_100": stratified_sample(math500, 100, key=lambda it: f"{json.loads(it.extra)['subject']}|{json.loads(it.extra)['level']}", salt="math500"),
        # Ablations: AIME 2026 plus 15 HMMT 2026 problems stratified by problem type.
        "ablation_45": sorted([it.item_id for it in by("aime_2026")]
                              + stratified_sample(hmmt26, 15, key=lambda it: it.domain.split(";")[0], salt="hmmt26")),
        "sat_suite": sorted(it.item_id for it in items if it.benchmark in
                            {"aime_2024", "aime_2025", "aime_2026", "hmmt_feb_2026", "tqa_phys"}),
        "xolver_aime": sorted(it.item_id for it in items if it.benchmark in {"aime_2024", "aime_2025"}),
        "train": sorted(it.item_id for it in items if it.split == "train"),
        "dev": sorted(it.item_id for it in items if it.split == "dev"),
    }
    # Pilot: 20 training problems, balanced across the two training competitions.
    train = [it for it in items if it.split == "train"]
    subsets["pilot_20"] = stratified_sample(train, 20, key=lambda it: it.benchmark, salt="pilot")
    return subsets
