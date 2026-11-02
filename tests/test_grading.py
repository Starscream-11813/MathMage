"""Grading rules, including every disagreement class found against MathArena's labels."""

import json

import pytest

from mathmage.grading.equiv import cluster, equivalent, exact_correct, parse_number, parse_number_list, rel_close
from mathmage.grading.extract import clean_answer, extract_final, last_boxed, parse_aime_integer
from mathmage.grading.verify import _tqa_correct


# ----------------------------------------------------------------------------- extraction
def test_last_boxed_nested_and_adjacent():
    assert last_boxed(r"so \boxed{\boxed{12}}") == "12"
    assert last_boxed(r"first \boxed{1}. Later the answer is \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    joined = last_boxed(r"The roots are \boxed{\frac{-1+\sqrt{17}}{2}} and \boxed{\frac{-1-\sqrt{17}}{2}}.")
    assert joined == r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}"


def test_extract_final_fallbacks():
    assert extract_final("blah\nFinal answer: 42") == "42"
    assert extract_final("**Answer:** 588") == "588"
    assert extract_final("no answer here 17 and 23", lenient=False) is None
    assert extract_final("no answer here 17 and 23", lenient=True) == "23"


@pytest.mark.parametrize("text,answer", [
    ("Final answer: 0.955", "0.955"),
    ("The answer is 345.5 m/s.", "345.5 m/s"),
    ("**Answer:** 1.2e-3", "1.2e-3"),
    ("Final answer: [1.32, 2.0]", "[1.32, 2.0]"),
    ("Final answer: 0.955\nFinal answer: 0.97", "0.97"),
])
def test_final_line_preserves_decimal_points(text, answer):
    assert extract_final(text) == answer


@pytest.mark.parametrize("raw,val", [("070", 70), ("70", 70), (r"\text{70}", 70), ("70.0", 70), ("n=104", 104),
                                     ("50+32=82", 82), ("1000", None), ("-3", None), ("12.5", None)])
def test_aime_integer(raw, val):
    assert parse_aime_integer(raw) == val


def test_clean_answer_rhs_rules():
    assert clean_answer(r"X P = \sqrt{23}\;-\;2\sqrt{3}\,.") .replace(" ", "") == r"\sqrt{23}-2\sqrt{3}"
    assert clean_answer(r"w=10+4(1+\sqrt{37})=14+4\sqrt{37}").replace(" ", "") == r"14+4\sqrt{37}"
    # An equation-valued answer is kept whole.
    assert clean_answer("2x - y + 3z + 8 = 0") == "2x - y + 3z + 8 = 0"


# ----------------------------------------------------------------------------- exact answers
@pytest.mark.parametrize("gold,pred,ok", [
    (r"\sqrt{6}", "√6", True),
    (r"\sqrt{23}-2 \sqrt{3}", "√23−2√3", True),
    (r"\sqrt{23}-2 \sqrt{3}", r"X P = \sqrt{23}\;-\;2\sqrt{3}\,.", True),
    # 1000x too small: MathArena's label marks this correct, which is a label error.
    (r"2^{25} \cdot 26!", "13,532,215,908,553,332,190,020,108,288,000", False),
    (r"2^{25} \cdot 26!", "13,532,215,908,553,332,190,020,108,288,000,000", True),
    (r"1-\frac{2}{\pi}", r"\,0.363380228\ldots\,", False),
    (r"1-\frac{2}{\pi}", "0.36338", False),
    (r"1-\frac{2}{\pi}", r"1 - \frac{2}{\pi}", True),
    (r"(3+\sqrt{6})^{-1/3}", r"\frac{\sqrt[3]{27-9\sqrt{6}}}{3}", True),
    (r"50(1 - \frac{1}{2^{101} - 1})", r"\frac{100\bigl(2^{100}-1\bigr)}{2^{101}-1}", True),
    (r"74^\circ", "74°", True),
    (r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}", r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}", True),
    (r"\frac{1}{2}", "0.5", True),
    (r"\frac{1}{10}", "0.1", True),
    (r"\frac{1}{3}", "0.3333333333333333", False),
    (r"\sqrt{2}", "1.4142135623730951", False),
    ("1", "1.0000000000001", False),
    (r"-\frac{1}{21}", r"-\frac{1}{21}", True),
    (r"-\frac{1}{21}", r"\frac{1}{21}", False),
    ("2x - y + 3z + 8 = 0", "2x - y + 3z + 8 = 0", True),
    ("2x - y + 3z + 8 = 0", "0", False),
    ("3840", "3840", True),
    ("3840", "3480", False),
])
def test_exact_correct(gold, pred, ok):
    assert exact_correct(gold, pred) is ok


# ----------------------------------------------------------------------------- TheoremQA
def test_tqa_official_rule():
    assert _tqa_correct("0.955", json.dumps(0.955), 0.04)
    assert _tqa_correct("0.97", json.dumps(0.955), 0.04)          # 1.6% off: official rule accepts
    assert not _tqa_correct("0.97", json.dumps(0.955), 0.01)      # strict 1% rule rejects
    assert not _tqa_correct("1.2", json.dumps(0.955), 0.04)
    assert _tqa_correct("11760", json.dumps(11760), 0.04)
    assert _tqa_correct("[1.32, 2.0]", json.dumps([2.0, 1.32]), 0.04)   # lists compared sorted
    assert not _tqa_correct("[1.32]", json.dumps([2.0, 1.32]), 0.04)
    assert _tqa_correct(r"9.55 \times 10^{-1}", json.dumps(0.955), 0.04)
    assert _tqa_correct("345 m/s", json.dumps(345.0), 0.04)


def test_parse_number_variants():
    assert parse_number("1.2e-3") == pytest.approx(1.2e-3)
    assert parse_number(r"\frac{3}{4}") == pytest.approx(0.75)
    assert parse_number("1,234.5") == pytest.approx(1234.5)


@pytest.mark.parametrize("raw", ["1e309", "-1e309", r"1 \times 10^{309}"])
def test_nonfinite_number_is_rejected(raw):
    assert parse_number(raw) is None
    assert not _tqa_correct(raw, json.dumps(1), 0.04)
    assert not equivalent(raw, "1", "tqa_integer")
    assert not equivalent(raw, "1", "tqa_float")
    assert not rel_close(float("inf"), 1.0, 0.02)


@pytest.mark.parametrize("raw", [
    r"\frac{3}{4}, \frac{5}{6}",
    r"[\frac{3}{4}, \frac{5}{6}]",
    r"\left\{\frac{3}{4}, \frac{5}{6}\right\}",
])
def test_number_list_preserves_latex_commands(raw):
    assert parse_number_list(raw) == pytest.approx([0.75, 5 / 6])
    assert _tqa_correct(raw, json.dumps([0.75, 5 / 6]), 0.04)


# ----------------------------------------------------------------------------- clustering
def test_cluster_and_equivalence():
    assert cluster(["070", "70", "71", None, "70"], "aime_integer") == [0, 0, 1, -1, 0]
    assert equivalent("0.955", "0.96", "tqa_float")         # within the 2% clustering tolerance
    assert not equivalent("0.955", "1.0", "tqa_float")
    assert equivalent(r"\frac{1}{2}", "0.5", "exact")
    assert cluster([r"\frac{7}{2}", "3.5", r"\frac{2}{7}"], "exact") == [0, 0, 1]
