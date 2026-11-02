"""Grading against reference answers.

* AIME: integer equality (answers 0-999).
* HMMT and MATH-500 (exact answers): gold and prediction are cleaned identically; an
  approximation (ellipsis, \\approx, or a decimal when the gold is not a decimal) is
  wrong, as in MathArena; otherwise math-verify symbolic equivalence, with a 60-digit
  numeric fallback (relative error <= 1e-12) for equal closed forms sympy cannot simplify.
* TheoremQA-physics: the official TheoremQA rule (floats within 4% relative error,
  integers after rounding, lists compared sorted and elementwise); a 1% variant is
  recorded as a sensitivity check.

Validated against MathArena's own labels on 22,894 public attempts (scripts/validate_grader.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from mathmage.grading.equiv import exact_correct, parse_number, parse_number_list
from mathmage.grading.extract import extract_final, parse_aime_integer
from mathmage.grading.vault import AnswerVault, load_vault

TQA_TOL = 0.04
TQA_STRICT_TOL = 0.01


@dataclass(frozen=True)
class Grade:
    item_id: str
    extracted: Optional[str]
    correct: bool
    method: str
    strict_correct: Optional[bool] = None


# ----------------------------------------------------------------------------- TheoremQA
def _tqa_compare_number(pred: float, gt, tol: float) -> bool:
    if isinstance(gt, bool):
        return False
    if isinstance(gt, int):
        return round(pred) == gt
    return abs(pred - float(gt)) <= tol * abs(float(gt))


def _tqa_correct(pred_str: Optional[str], gt_json: str, tol: float) -> bool:
    gt = json.loads(gt_json)
    if isinstance(gt, list):
        pred = parse_number_list(pred_str)
        if pred is None or len(pred) != len(gt):
            return False
        return all(_tqa_compare_number(p, g, tol) for p, g in zip(sorted(pred), sorted(gt)))
    pred_v = parse_number(pred_str)
    if pred_v is None:
        return False
    return _tqa_compare_number(pred_v, gt, tol)


# ----------------------------------------------------------------------------- public API
def grade_answer(item_id: str, answer: Optional[str], vault: Optional[AnswerVault] = None,
                 method_suffix: str = "") -> Grade:
    vault = vault or load_vault()
    rec = vault[item_id]
    fmt = rec.answer_format
    if fmt == "aime_integer":
        v = parse_aime_integer(answer)
        return Grade(item_id, answer, v is not None and v == int(rec.answer), "aime_int" + method_suffix)
    if fmt.startswith("tqa_"):
        return Grade(item_id, answer, _tqa_correct(answer, rec.answer, TQA_TOL), "tqa_official" + method_suffix,
                     strict_correct=_tqa_correct(answer, rec.answer, TQA_STRICT_TOL))
    return Grade(item_id, answer, exact_correct(rec.answer, answer), "exact" + method_suffix)


def grade_response(item_id: str, response: Optional[str], vault: Optional[AnswerVault] = None) -> Grade:
    """Grade a raw response with MathArena-style lenient extraction (last-integer fallback)."""
    strict = extract_final(response or "", lenient=False)
    if strict is not None:
        return grade_answer(item_id, strict, vault)
    return grade_answer(item_id, extract_final(response or "", lenient=True), vault, method_suffix="+lastint")
