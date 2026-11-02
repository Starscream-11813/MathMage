"""Final-answer extraction from model responses (no reference answers involved).

Two policies:
  * strict (agents): the last \\boxed{} (adjacent final boxes are joined, for answers that
    list several values) or an explicit "final answer: ..." line; otherwise no answer.
  * lenient (grading raw responses): strict, then MathArena's fallback for competitions
    with ``strict_parsing: false`` (the last integer in the text).
"""

from __future__ import annotations

import re
from typing import Optional

_BOX_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_UNICODE = [("−", "-"), ("–", "-"), ("×", "\\times "), ("·", "\\cdot "), ("⋅", "\\cdot "), ("π", "\\pi "),
            ("∞", "\\infty "), ("≤", "\\le "), ("≥", "\\ge "), ("…", "\\ldots"), ("⎧", ""), ("⎫", ""),
            ("⎩", ""), ("⎭", ""), ("\u00a0", " ")]
_SQRT_UNICODE_RE = re.compile(r"√\s*\(([^()]*)\)|√\s*([0-9A-Za-z]+)")


def _balanced(text: str, start: int) -> Optional[tuple[str, int]]:
    """(content, end_index) of the brace group whose opening brace is at text[start - 1]."""
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i
        i += 1
    return None


def boxed_spans(text: str) -> list[tuple[str, int, int]]:
    """Top-level boxes only (a box nested inside another box is part of the outer one)."""
    out: list[tuple[str, int, int]] = []
    for m in _BOX_RE.finditer(text or ""):
        if out and m.start() < out[-1][2]:
            continue
        r = _balanced(text, m.end())
        if r is not None:
            out.append((r[0].strip(), m.start(), r[1] + 1))
    return out


def all_boxed(text: str) -> list[str]:
    return [b for b, _, _ in boxed_spans(text)]


def _unwrap(ans: str) -> str:
    inner = all_boxed(ans)
    while inner:
        ans = inner[-1]
        inner = all_boxed(ans)
    return ans


def last_boxed(text: str, join_adjacent: bool = True, gap: int = 60) -> Optional[str]:
    """The last \\boxed{} content; boxes separated from it only by short connective text
    (", ", " and ", " or ", newlines) are joined with ", " (e.g. two roots)."""
    spans = boxed_spans(text)
    if not spans:
        return None
    group = [spans[-1]]
    if join_adjacent:
        for b in reversed(spans[:-1]):
            between = text[b[2]:group[0][1]]
            if len(between) <= gap and re.fullmatch(r"[\s,;$.]*(?:and|or|,|;|\\quad|\\qquad|\\text\{\s*(?:and|or)\s*\})?[\s,;$.]*", between):
                group.insert(0, b)
            else:
                break
    parts = [_unwrap(b[0]) for b in group]
    return ", ".join(p for p in parts if p != "") if len(parts) > 1 else parts[0]


_FINAL_LINE_RE = re.compile(
    r"(?:final\s+answer\s*(?:is|:|=)|the\s+answer\s+is|\banswer\s*\**\s*[:=])\s*\**\s*\$?([^\n$*]+?)\$?\**\s*(?:\.(?!\d)|$)",
    re.IGNORECASE | re.MULTILINE)


def extract_final(text: str, lenient: bool = False) -> Optional[str]:
    if not text:
        return None
    b = last_boxed(text)
    if b is not None and b.strip() != "":
        return b
    ms = list(_FINAL_LINE_RE.finditer(text[-1500:]))
    if ms:
        cand = ms[-1].group(1).strip()
        if cand:
            return cand
    if lenient:
        ints = re.findall(r"\b\d+\b", text)
        if ints:
            return ints[-1]
    return None


_TEXT_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathbf|textbf|operatorname)\s*\{([^{}]*)\}")
_SIZE_RE = re.compile(r"\\(?:bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|big|Big|bigg|Bigg|left|right|displaystyle|textstyle)\b")


def normalize_unicode(s: str) -> str:
    s = _SQRT_UNICODE_RE.sub(lambda m: "\\sqrt{" + (m.group(1) or m.group(2)) + "}", s)
    for a, b in _UNICODE:
        s = s.replace(a, b)
    return s


def clean_answer(s: Optional[str], take_rhs: bool = True) -> Optional[str]:
    """Strip formatting that never changes the value: text wrappers, spacing, delimiter
    sizing, dollar signs, degree marks, thousands separators, and a leading 'x =' prefix."""
    if s is None:
        return None
    s = normalize_unicode(s)
    s = _TEXT_WRAP_RE.sub(r"\1", s)
    s = _SIZE_RE.sub("", s)
    for tok in ("\\!", "\\,", "\\;", "\\:", "\\ ", "~"):
        s = s.replace(tok, " ")
    s = s.replace("$", "").replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "").replace("\\%", "%")
    s = s.strip().rstrip(".").strip()
    if take_rhs and "=" in s and not any(t in s for t in ("\\neq", "\\le", "\\ge", "<", ">")):
        segs = [p.strip() for p in s.split("=")]
        lhs, rhs = segs[0].replace(" ", ""), segs[-1]
        is_name = re.fullmatch(r"[A-Za-z]{1,4}(?:'|\^\{?\\prime\}?)*(?:_\{?[A-Za-z0-9\\ ]{1,10}\}?)?(?:\^\{?2\}?)?", lhs) is not None
        is_numeric = re.fullmatch(r"[\d\s+\-*/().\\a-z{}^]*", lhs) is not None and not re.search(r"[a-wyzA-Z]", re.sub(r"\\[a-z]+", "", lhs))
        if rhs and (is_name or is_numeric or len(segs) >= 3):
            s = rhs
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s.replace(" ", "")):
        s = s.replace(",", "").replace(" ", "")
    return s.strip()


def parse_aime_integer(s: Optional[str]) -> Optional[int]:
    """AIME answers are integers 0-999; accepts '070', '70', '\\text{70}', '70.0', 'n = 70'."""
    s = clean_answer(s)
    if s is None:
        return None
    s = s.replace(",", "").strip()
    m = re.fullmatch(r"[+]?0*(\d{1,4})(?:\.0+)?", s)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 999 else None


def is_approximation(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(t in s for t in ("\\ldots", "...", "\\approx", "≈", "\\cdots"))
