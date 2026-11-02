"""Frozen prompt templates.

Each template's id carries a short hash of its text (e.g. ``solve.v1#3fa2c1``), which is
logged with every call and is part of the cache key through the rendered prompt. Editing
a template therefore changes its id and can never silently reuse old responses.

Rules shared by every template that can carry help to a solver: help never states a
final answer; every note is framed as possibly incomplete or wrong (research plan,
"Controls"); reviewers and tutors never see reference answers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    name: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}#{hashlib.sha256(self.text.encode()).hexdigest()[:6]}"

    def render(self, **kw: str) -> str:
        return self.text.format(**kw)


CERT_RULES = (
    "After your solution, write a line containing only CERTIFICATE, followed by a numbered list of the "
    "load-bearing claims and intermediate values of your solution, each checkable on its own (for example "
    "by a short computation). Then give the final answer as \\boxed{{...}}."
)

SOLVE = Template("solve.v1", (
    "Solve the following problem. Reason step by step.\n"
    + CERT_RULES + "\n{format_instruction}\n"
    "{cards}"
    "\nProblem:\n{problem}\n"
))

SOLVE_SCAFFOLD = Template("solve_scaffold.v1", (
    "Solve the following problem. A note is attached; it may be incomplete or wrong, so use it only if it "
    "holds up under your own checking. Reason step by step.\n"
    + CERT_RULES + "\n{format_instruction}\n"
    "\nProblem:\n{problem}\n"
    "\nNote:\n{note}\n"
))

SOLVE_REVISE = Template("solve_revise.v1", (
    "You solved the problem below. A reviewer raised the concerns listed after your solution. For each "
    "concern, either accept it and fix your solution, or rebut it with a specific mathematical reason. "
    "Concerns may be wrong. Then write a full corrected solution.\n"
    + CERT_RULES + "\n{format_instruction}\n"
    "\nProblem:\n{problem}\n"
    "\nYour solution:\n{solution}\n"
    "\nConcerns:\n{concerns}\n"
))

CHECK_FORWARD = Template("check_forward.v1", (
    "Write a self-contained Python 3 program that computes the answer to the problem below by direct "
    "computation from its definitions: brute force, exhaustive enumeration, exact arithmetic (fractions, "
    "sympy), or careful numerical simulation, whichever is feasible within 20 seconds on one CPU core. "
    "Prefer enumeration over formulas you have not verified. Only the standard library, sympy, and numpy "
    "are available; there is no internet access and no input.\n"
    "The program must print exactly one final line of the form\n"
    "ANSWER: <value>\n"
    "where <value> is in the form the problem requires. If the answer cannot be computed this way, print "
    "ANSWER: NONE.\n"
    "{format_instruction}\n"
    "\nProblem:\n{problem}\n"
    "\nReturn only the program, in a single ```python code block."
))

CHECK_FORWARD_ALT = Template("check_forward_alt.v1", (
    "Write a self-contained Python 3 program that finds the answer to the problem below by an exhaustive "
    "search or simulation over a small, faithful model of the problem (scale parameters down only if you "
    "can extrapolate exactly, and say how in a comment). Do not rely on a closed-form formula. Only the "
    "standard library, sympy, and numpy are available; the program must finish within 20 seconds.\n"
    "Print exactly one final line: ANSWER: <value>   (or ANSWER: NONE if impossible).\n"
    "{format_instruction}\n"
    "\nProblem:\n{problem}\n"
    "\nReturn only the program, in a single ```python code block."
))

REVIEW = Template("review.v1", (
    "You are reviewing a proposed solution to the problem below. Do not solve the problem yourself and do "
    "not state, guess, or compute any final answer. Check the certificate line by line.\n"
    "Return a JSON object with these fields:\n"
    '  "verdict": "valid" | "invalid" | "unsure",\n'
    '  "first_error_line": the number of the first certificate line that is wrong, or null,\n'
    '  "diagnosis": what is wrong in that line, without fixing it (empty if none),\n'
    '  "check_code": optional Python code that recomputes the disputed quantity from the problem\'s own '
    'definitions and prints CLAIM_FALSE if the certificate line is wrong or CLAIM_TRUE otherwise (empty if '
    'no executable check is possible),\n'
    '  "confidence": a number between 0 and 1.\n'
    "\nProblem:\n{problem}\n"
    "\nProposed solution:\n{solution}\n"
))

JUDGE = Template("judge.v1", (
    "You are adjudicating between candidate solutions to one problem. You are auditing written "
    "certificates and verification evidence, NOT solving the problem.\n"
    "Rules:\n"
    "- Do not solve the problem yourself, and do not prefer a candidate because its answer matches your own "
    "intuition.\n"
    "- Judge only certificate quality and evidence: are the load-bearing steps written down and locally "
    "checkable, is the arithmetic correct as written, is the case analysis complete, and does the final "
    "answer follow from the established steps?\n"
    "- Executable checks and reviewer reports are listed with each candidate. A failed executable check is "
    "strong evidence of a defect; an unverified reviewer objection is weak evidence.\n"
    "- Reject or downgrade a candidate only by naming a specific local defect.\n"
    "First write an audit entry for every candidate, then decide.\n"
    'End with a JSON object on the last line: {{"choice": <candidate number>, "confidence": <0..1>}}.\n'
    "\nProblem:\n{problem}\n"
    "\nCandidates:\n{candidates}\n"
))

# SAT's final-answer selection prompt (SAT, Appendix D.1.1), used verbatim for ablation A2.
SAT_JUDGE = Template("sat_judge.v1", (
    "You are selecting the best proof-of-work certificate for one {benchmark_name} problem.\n"
    "You are given the problem and all candidate final team certificates produced by a strategy bank. Choose "
    "the candidate whose final answer is best supported by its written certificate.\n"
    "You are auditing written certificates, NOT solving the problem.\n"
    "Rules:\n"
    "- Do not solve the problem yourself, and do not select or reject a candidate because its answer matches "
    "or conflicts with your own solution, estimate, or intuition about the answer.\n"
    "- Judge only certificate quality: are the load-bearing steps written down and locally checkable, is the "
    "arithmetic correct as written, is the case analysis complete, and does the final transition to the "
    "stated answer follow from the established steps?\n"
    "- You may reject or downgrade a candidate ONLY by naming a specific local defect in its written "
    "certificate: a step that does not follow from the previous ones, an arithmetic error you can point to, a "
    "missing case, an unjustified assumption, or a final transition not supported by the steps. \"I believe "
    "the answer is different\" or \"it conflicts with another candidate's derivation\" are NOT defects.\n"
    "- For each candidate, record all such defects in a named_defects list; use an empty list if no specific "
    "defect is identified.\n"
    "- Do not use answer frequency while auditing; audit every certificate on its written content alone. "
    "Frequency enters ONLY through the mandatory tie-break rule in stage 2.\n"
    "- If every certificate is flawed, choose the least-bad candidate and name its remaining defect.\n"
    "Work in two stages, strictly in this order:\n"
    "1. AUDIT: write a structured audit entry for EVERY candidate, based only on its written certificate.\n"
    "2. DECIDE: only after all audits are written, compare the audits and select the certificate with the "
    "strongest written support. MANDATORY TIE-BREAK: if more than one candidate has an empty named_defects "
    "list, you MUST select from among those defect-free candidates the one whose final answer is the most "
    "common answer within that defect-free set; if frequencies are tied, prefer the certificate whose audit "
    "shows the most complete, explicit support. Do not override this tie-break with prose-quality "
    "preferences.\n"
    'End with a JSON object on the last line: {{"choice": <candidate number>}}.\n'
    "Problem ({benchmark_name}):\n{problem}\n"
    "Candidate certificates:\n{candidates}\n"
))

HINT = Template("hint.v1", (
    "A teammate attempted the problem below and their answer could not be verified. Write ONE {hint_kind} "
    "that would help them make progress. Do not state, guess, or compute the final answer or any quantity "
    "from which it follows in one step. Keep it under 120 words.\n"
    "\nProblem:\n{problem}\n"
    "\nTeammate's attempt (certificate only):\n{attempt}\n"
))

HINT_KINDS = {
    "strategic": "strategic hint (name the approach or key idea, without details)",
    "decomposition": "decomposition into ordered subgoals",
    "socratic": "set of at most three Socratic questions (questions only)",
    "diagnosis": "diagnosis of what is most likely wrong in the attempt (without fixing it)",
}

CARD = Template("card.v1", (
    "Below is a solved competition problem with a verified solution. Write a reusable strategy card for "
    "solving OTHER problems of this kind. Do not mention this problem's specific numbers or its answer.\n"
    "Return JSON with fields: \"method\" (one sentence), \"applies_when\" (one sentence), "
    "\"pitfall\" (one sentence), \"checklist\" (a list of at most three short checks).\n"
    "\nProblem:\n{problem}\n"
    "\nVerified solution:\n{solution}\n"
))

MISLEAD = Template("mislead.v1", (
    "Here is a problem and a correct solution. Write ONE plausible-sounding but FALSE claim about this "
    "problem that a strong solver might find tempting, in the style of a strategic hint. It must be false "
    "given the solution and must not state the answer. Return only the claim.\n"
    "\nProblem:\n{problem}\n"
    "\nCorrect solution:\n{solution}\n"
))

PLACEBO_NOTE = ("Read the problem carefully, keep track of every condition it states, and give your final "
                "answer in the requested form after checking it once more.")

TEMPLATES = {t.name: t for t in (SOLVE, SOLVE_SCAFFOLD, SOLVE_REVISE, CHECK_FORWARD, CHECK_FORWARD_ALT, REVIEW,
                                 JUDGE, SAT_JUDGE, HINT, CARD, MISLEAD)}


def render(name: str, **kw: str) -> tuple[str, str]:
    """(template_id, rendered_text)."""
    t = TEMPLATES[name]
    return t.id, t.render(**kw)


def template_family(template_id: str) -> str:
    return template_id.split("#")[0].split(".")[0]
