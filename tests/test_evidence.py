"""Forward programs may abstain on failure without disguising constant answers."""

import pytest

from mathmage.tools.evidence import computational_evidence


def test_forward_computation_with_early_abstention_is_accepted():
    program = """
def solve():
    solutions = [value for value in range(20) if value * value == 81]
    if not solutions:
        print('ANSWER: NONE')
        return
    answer = int(sum(solutions))
    print(f'ANSWER: {answer}')
solve()
"""
    assert computational_evidence(program, "independent_forward")[0]


@pytest.mark.parametrize("abstention", [
    "print('ANSWER: NONE')",
    "print('ANSWER:', 'NONE')",
    "print(f'ANSWER: NONE')",
    "print('ANSWER:', ' NONE', sep='', end='\\n')",
])
def test_literal_abstention_outputs_do_not_disqualify_computed_answers(abstention):
    program = f"{abstention}\nanswer = sum(range(5))\nprint('ANSWER:', answer)"
    assert computational_evidence(program, "independent_forward")[0]


@pytest.mark.parametrize("program", [
    "print('ANSWER: NONE')",
    "unused = sum(range(5))\nprint('ANSWER: NONE')",
    "print('ANSWER: NONE')\nprint('ANSWER: 10')",
    "unused = sum(range(5))\nprint('ANSWER: NONE')\nprint('ANSWER: 10')",
    "answer = sum(range(5))\nprint('ANSWER:', answer)\nprint('ANSWER: 10')",
    "print('ANSWER: NONE', end=' ANSWER: 10')",
    "answer = sum(range(5))\nprint('ANSWER:', answer)\nprint('ANSWER: NONE; ANSWER: 10')",
])
def test_abstention_cannot_hide_constant_or_absent_actual_answers(program):
    assert not computational_evidence(program, "independent_forward")[0]


def test_claim_check_rules_are_not_relaxed_by_abstention():
    assert not computational_evidence(
        "unused = 2 < 3\nprint('ANSWER: NONE')\nprint('CLAIM_FALSE')", "review_objection")[0]
    assert computational_evidence(
        "actual = 2 + 2\nif actual != 5:\n    print('CLAIM_FALSE')", "review_objection")[0]
