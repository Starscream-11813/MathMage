"""Equivalence between two answers, with no reference answer involved.

The team pipeline uses this to cluster candidate answers; the grader uses the same
parsing to compare a prediction with the reference. Answer formats come from the
registry (see FORMAT_INSTRUCTIONS).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Optional, Sequence

from mathmage.grading.extract import clean_answer, is_approximation, parse_aime_integer
from mathmage.grading.timeout import call_with_timeout

TQA_FORMATS = {"tqa_float", "tqa_integer", "tqa_list_float", "tqa_list_integer"}
CLUSTER_REL_TOL = 0.02   # two numeric candidates within 2% are treated as the same answer
_DECIMAL_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")


@lru_cache(maxsize=20000)
def mv_parse(ans: str):
    """math-verify parse of a bare answer (wrapped in \\boxed{} for robust extraction)."""
    from math_verify import parse

    if ans is None:
        return None
    return call_with_timeout(parse, "\\boxed{" + ans + "}", parsing_timeout=None, timeout=5.0, default=None)


def mv_equal(a: str, b: str) -> bool:
    from math_verify import verify

    pa, pb = mv_parse(a), mv_parse(b)
    if not pa or not pb:
        return False
    f = lambda x, y: bool(call_with_timeout(verify, x, y, timeout_seconds=None, timeout=5.0, default=False))  # noqa: E731
    return f(pa, pb) or f(pb, pa)


# ----------------------------------------------------------------------------- numbers
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SCI_RE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+))\s*(?:\\times|\\cdot|x|\*)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?")


def parse_number(s: Optional[str]) -> Optional[float]:
    """Best-effort float from an answer string; tolerant of units and LaTeX."""
    s = clean_answer(s)
    if not s:
        return None
    s = s.replace("{,}", "").replace("−", "-")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    m = _SCI_RE.search(s)
    if m:
        try:
            value = float(m.group(1)) * 10 ** int(m.group(2))
            return value if math.isfinite(value) else None
        except (ValueError, OverflowError):
            return None
    # A purely numeric string
    if re.fullmatch(r"\s*" + _NUM_RE.pattern + r"\s*", s):
        value = float(s)
        return value if math.isfinite(value) else None
    # Symbolic value (fractions, roots, pi): evaluate with math-verify/sympy
    p = mv_parse(s)
    if p:
        try:
            v = complex(p[0].evalf())
            if abs(v.imag) < 1e-12 and math.isfinite(v.real):
                return float(v.real)
            return None
        except (TypeError, ValueError, AttributeError):
            pass
    m = _NUM_RE.search(s)
    if m:
        try:
            value = float(m.group(0))
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    return None


def parse_number_list(s: Optional[str]) -> Optional[list[float]]:
    s = clean_answer(s)
    if not s:
        return None
    s = s.replace("\\left", "").replace("\\right", "").strip()
    # Remove one matching list delimiter, preserving commands such as \frac.
    if s.startswith(r"\{") and s.endswith(r"\}"):
        s = s[2:-2]
    elif len(s) >= 2 and (s[0], s[-1]) in (("[", "]"), ("(", ")"), ("{", "}")):
        s = s[1:-1]
    parts = [p for p in re.split(r"[,;]\s*", s) if p.strip()]
    vals = [parse_number(p) for p in parts]
    if not vals or any(v is None for v in vals):
        return None
    return vals  # type: ignore[return-value]


def rel_close(a: float, b: float, rel: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    scale = max(abs(a), abs(b))
    return abs(a - b) <= rel * scale if scale > 0 else True


# ----------------------------------------------------------------------------- exact answers
def _exact_decimal_equal(gold: str, pred: str) -> bool:
    """A finite decimal is an exact rational, never a tolerance-based approximation."""
    parsed = mv_parse(gold)
    if not parsed:
        return False

    def compare():
        import sympy

        return sympy.simplify(parsed[0] - sympy.Rational(pred.replace(" ", ""))) == 0

    return bool(call_with_timeout(compare, timeout=8.0, default=False))


def _numeric_equal(a: str, b: str) -> bool:
    pa, pb = mv_parse(a), mv_parse(b)
    if not pa or not pb:
        return False

    def ev():
        import sympy

        x = sympy.N(pa[0], 60)
        y = sympy.N(pb[0], 60)
        if not (x.is_number and y.is_number):
            return False
        xc, yc = complex(x), complex(y)
        if abs(xc.imag) > 1e-30 or abs(yc.imag) > 1e-30:
            return False
        err = abs(sympy.N(x - y, 60))
        scale = max(abs(x), abs(y))
        return bool(err == 0 or (scale > 0 and err <= 1e-12 * scale))

    return bool(call_with_timeout(ev, timeout=8.0, default=False))


def exact_correct(gold: str, pred: Optional[str]) -> bool:
    from math_verify import verify

    if pred is None:
        return False
    g, p = clean_answer(gold), clean_answer(pred)
    if not p:
        return False
    if is_approximation(p) and not is_approximation(g):
        return False
    gold_is_decimal = _DECIMAL_RE.search(g or "") is not None
    if not gold_is_decimal and _DECIMAL_RE.fullmatch(p.replace(" ", "")):
        # A decimal for a non-decimal exact answer counts only if it is exactly equal (e.g. 0.5 = 1/2).
        return _exact_decimal_equal(g, p)
    if g == p:
        return True
    pg, pp = mv_parse(g), mv_parse(p)
    if pg and pp and bool(call_with_timeout(verify, pg, pp, timeout_seconds=None, timeout=8.0, default=False)):
        return True
    if "," not in g:
        return _numeric_equal(g, p)
    return False


# ----------------------------------------------------------------------------- public API
def normalize(ans: Optional[str], answer_format: str) -> Optional[str]:
    """A display/canonical form; None when the answer cannot be read in this format."""
    if ans is None:
        return None
    if answer_format == "aime_integer":
        v = parse_aime_integer(ans)
        return None if v is None else str(v)
    if answer_format in ("tqa_float", "tqa_integer"):
        v = parse_number(ans)
        return None if v is None else f"{v:.6g}"
    if answer_format in ("tqa_list_float", "tqa_list_integer"):
        vs = parse_number_list(ans)
        return None if vs is None else "[" + ", ".join(f"{v:.6g}" for v in vs) + "]"
    c = clean_answer(ans)
    return c if c else None


def equivalent(a: Optional[str], b: Optional[str], answer_format: str) -> bool:
    if a is None or b is None:
        return False
    if answer_format == "aime_integer":
        va, vb = parse_aime_integer(a), parse_aime_integer(b)
        return va is not None and va == vb
    if answer_format == "tqa_integer":
        va, vb = parse_number(a), parse_number(b)
        return va is not None and vb is not None and round(va) == round(vb)
    if answer_format == "tqa_float":
        va, vb = parse_number(a), parse_number(b)
        return va is not None and vb is not None and rel_close(va, vb, CLUSTER_REL_TOL)
    if answer_format in ("tqa_list_float", "tqa_list_integer"):
        la, lb = parse_number_list(a), parse_number_list(b)
        if la is None or lb is None or len(la) != len(lb):
            return False
        return all(rel_close(x, y, CLUSTER_REL_TOL) for x, y in zip(sorted(la), sorted(lb)))
    return exact_correct(a, b) or exact_correct(b, a)


def cluster(answers: Sequence[Optional[str]], answer_format: str) -> list[int]:
    """Cluster labels (0, 1, ...) by equivalence; None answers get label -1."""
    labels = [-1] * len(answers)
    reps: list[int] = []
    for i, a in enumerate(answers):
        if a is None:
            continue
        for lab, j in enumerate(reps):
            if equivalent(answers[j], a, answer_format):
                labels[i] = lab
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    return labels
