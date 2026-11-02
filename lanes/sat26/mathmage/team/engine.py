"""The staged MathMage-Team protocol.

Only Item.view() enters model prompts. This module deliberately has no grading,
answer-vault, reference-solution, or test-outcome dependency.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections import Counter

import numpy as np
from dataclasses import asdict, replace
from typing import Any

from mathmage.config import load_models
from mathmage.grading.extract import clean_answer, extract_final
from mathmage.grading.equiv import cluster, equivalent, normalize
from mathmage.llm.types import LLMRequest, LLMResponse
from mathmage.llm.ledger import BudgetExceeded
from mathmage.prompts.templates import (CHECK_FORWARD, CHECK_FORWARD_ALT, HINT, HINT_KINDS, JUDGE,
                                      REVIEW, SAT_JUDGE, SOLVE, SOLVE_REVISE,
                                      SOLVE_SCAFFOLD, Template)
from mathmage.scaffolds import HINT_LEVELS, residual_audit, structural_audit
from mathmage.team.artifacts import DEFAULT_BANK, STRATEGIES, load_artifacts
from mathmage.team.policy import Action, Policy, PolicyConfig, prompt_tokens
from mathmage.team.posterior import AgentProfile, Evidence, Obs, PosteriorParams
from mathmage.team.types import POLICY_DEFAULTS, Candidate, CheckEvidence, TeamConfig, TeamResult, ZoneEstimate
from mathmage.team.zones import estimate_zone
from mathmage.tools import ToolResult, computational_evidence
from mathmage.tools.local_sandbox import make_sandbox

# verify_forward is retained for older records; newly emitted candidates use tool_forward.
TOOL_PHASES = {"tool_forward", "verify_forward", "tool_alt"}

# Uncalibrated planning priors (logit abilities relative to a typical olympiad item). They are
# placeholders for mock runs only; live runs require posterior parameters calibrated offline.
PRIOR_ABILITY = {"o3-mini-low": 0.4, "haiku-4.5": 0.0, "deepseek-v3": -0.9, "qwen3-32b": 1.0,
                 "qwen3-32b-nothink": -0.2}


def default_posterior_params(members, checker) -> PosteriorParams:
    agents = {key: AgentProfile(PRIOR_ABILITY.get(key, 0.0)) for key in dict.fromkeys(members)}
    agents["tool"] = AgentProfile(0.0, abstain=0.4)
    return PosteriorParams(agents=agents, calibrated=False,
                           provenance={"source": "uncalibrated planning priors (mock/plumbing only)"})


RESIDUAL_AUDIT = Template("hint_residual_audit.v1", (
    "Audit whether this proposed hint leaves substantial mathematical work. Do not solve the problem "
    "or state its answer. The hint may be wrong. Reject a hint that states an answer, encodes one, or "
    "makes completion a single substitution or arithmetic step. Return JSON fields reveals_answer "
    "(boolean), one_step_completion (boolean), remaining_nontrivial_steps (list of distinct steps).\n"
    "Problem:\n{problem}\nHint:\n{hint}"
))
PARTIAL_CHECK = Template("check_certificate.v1", (
    "Write a self-contained Python standard-library program that checks computational intermediate "
    "claims in this certificate by independently recomputing them from the problem definitions. "
    "Do not assume any certificate claim is true. Print PASS:<line number> for a checked true claim, "
    "FAIL:<line number> for a checked false claim, or UNCHECKABLE if no independent check is feasible. "
    "Never print PASS without an actual computation/comparison. Limit work to 20 seconds, one CPU. "
    "Return one python code block.\nProblem:\n{problem}\nCertificate:\n{certificate}"
))
DISCUSS = Template("discussion.v1", (
    "Discuss competing approaches and the bottleneck with your teammates. Their proposals may be wrong. "
    "Write a full solution and a numbered CERTIFICATE, ending with a boxed final answer.\n"
    "{format_instruction}\nProblem:\n{problem}\nShared whiteboard:\n{whiteboard}"
))
DIVERSIFY = Template("diversify.v1", (
    "Attempt a substantially different mathematical method from the listed attempts. Treat all shared "
    "insights as fallible and recheck them. Write your full solution, a numbered CERTIFICATE, then a "
    "boxed final answer.\n{format_instruction}\nProblem:\n{problem}\nPrior methods and verified partial "
    "insights:\n{whiteboard}"
))
SAFE_FORWARD = Template("check_forward_stdlib.v1", CHECK_FORWARD.text.replace(
    "Only the standard library, sympy, and numpy are available", "Only Python's standard library is available"))
SAFE_FORWARD_ALT = Template("check_forward_alt_stdlib.v1", CHECK_FORWARD_ALT.text.replace(
    "standard library, sympy, and numpy are available", "Python standard library is available"))
TEAM_TEMPLATES = {template.name: template for template in (
    SOLVE, SOLVE_REVISE, SOLVE_SCAFFOLD, REVIEW, HINT, JUDGE, SAT_JUDGE,
    RESIDUAL_AUDIT, PARTIAL_CHECK, DISCUSS, DIVERSIFY, CHECK_FORWARD, CHECK_FORWARD_ALT,
    SAFE_FORWARD, SAFE_FORWARD_ALT)}


def parse_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    values = []
    for pos, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[pos:])
                if isinstance(value, dict):
                    values.append(value)
            except ValueError:
                pass
    return values[-1] if values else {}


def extract_answer(text: str) -> str | None:
    return clean_answer(extract_final(text))


def normalize_answer(answer: str) -> str:
    return clean_answer(answer) or ""


def certificate(text: str) -> str:
    text = text.split("CERTIFICATE", 1)[-1] if "CERTIFICATE" in text else text
    # Tutors receive the intermediate certificate only, never its boxed answer.
    text = re.split(r"\\boxed\s*\{|^ANSWER:", text, maxsplit=1, flags=re.M)[0]
    return text.strip()


def python_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return match.group(1).strip() if match else ""


class TeamEngine:
    def __init__(self, client, config: TeamConfig | None = None, verifier=None):
        self.client = client
        self.config = config or TeamConfig()
        self.models = load_models()
        roster = self.models.rosters[self.config.roster]
        self.members = list(self.config.members or roster.members)
        if not self.members:
            raise ValueError("a team needs at least one member")
        self.artifacts = load_artifacts(self.config.offline_artifacts, live=bool(getattr(client, "live", False)))
        frozen_policy = self.artifacts.get("policy_config", {})
        if not isinstance(frozen_policy, dict):
            raise ValueError("frozen policy_config must be an object")
        self.config = replace(self.config, **{
            key: getattr(self.config, key) if getattr(self.config, key) is not None else frozen_policy.get(key, default)
            for key, default in POLICY_DEFAULTS.items()})
        if any(getattr(self.config, key) is None for key in POLICY_DEFAULTS):
            raise ValueError("frozen policy_config limits cannot be null")
        self.best_member = self.artifacts.get("best_member", self.members[0])
        if self.best_member not in self.members:
            raise ValueError("offline best member is not on the configured roster")
        if self.config.condition in {"C1", "C2"}:
            self.members = [self.best_member] * len(self.members)
        self.member_ids = [f"{key}#{i}" for i, key in enumerate(self.members)]
        self.checker = self.config.checker or roster.checker
        self.default_judge = self.config.default_judge or roster.default_judge
        if self.config.condition == "C1":
            self.checker = self.default_judge = self.best_member
        self.effective_mu = self.config.mu if self.config.mu is not None else self.artifacts.get("mu_star", 0.0)
        if (isinstance(self.effective_mu, bool) or not isinstance(self.effective_mu, (int, float))
                or not math.isfinite(self.effective_mu) or self.effective_mu < 0):
            raise ValueError("frozen mu_star must be a finite nonnegative number")
        self.verifier = verifier
        self.sandbox = make_sandbox(self.config.sandbox)
        if self.config.sandbox != "local":
            self.sandbox.image = self.config.sandbox_image
        self.sandbox_backend = getattr(self.sandbox, "backend", "docker")
        # A local process sandbox has sympy/numpy; the default container image has the standard library only.
        self._forward_template = CHECK_FORWARD if self.sandbox_backend == "local-process" else SAFE_FORWARD
        self._forward_alt_template = CHECK_FORWARD_ALT if self.sandbox_backend == "local-process" else SAFE_FORWARD_ALT
        self._sandbox_available = None
        if getattr(client, "live", False) and self.config.condition != "A7" and verifier is None:
            if not self.config.tools_enabled:
                raise RuntimeError("live protocols without tools must use the explicitly named A7 condition")
            self._sandbox_available = self.sandbox.available
            if not self._sandbox_available:
                raise RuntimeError("live team protocol requires isolated Docker checks or an explicitly selected local sandbox before any model call: " + self.sandbox.reason)

    def _request(self, model, template, phase, role, **kwargs):
        prompt = template.render(**kwargs)
        sampling_seed = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{phase}:{role}".encode()).digest()[:4], "big") % (2 ** 31)
        return LLMRequest(model_key=model, messages=[{"role": "user", "content": prompt}],
                          template_id=template.id,
                          sample_tag=f"seed={self.config.seed}:{phase}:{role}", kind=self.item.kind,
                          seed=sampling_seed,
                          meta={**self.item_provenance, "item_id": self.item.item_id,
                                "condition": self.config.condition, "variant": self.config.condition,
                                "phase": phase, "stage": phase, "role": role, "seed": self.config.seed,
                                "attempt_idx": self._attempt_index(role), "checkpoint_id": self.checkpoint_id})

    def _record(self, request, response):
        entry = {"phase": request.meta["phase"], "role": request.meta["role"],
                           "model_key": request.model_key, "template_id": request.template_id,
                           "sample_tag": request.sample_tag, "request_meta": copy.deepcopy(request.meta), **asdict(response)}
        if not entry.get("resource_cost_usd") and (response.tokens_in or response.tokens_out):
            entry["resource_cost_usd"] = self.models.model(request.model_key).price.cost(response.tokens_in, response.tokens_out)
        if not entry.get("resource_latency_s"):
            entry["resource_latency_s"] = response.latency_s
        self.calls.append(entry)

    def _call(self, request):
        if len(self.calls) + self.inherited_costs.get("calls", 0) >= self.config.max_calls:
            self.trace.append({"event": "call_limit", "phase": request.meta["phase"]})
            return LLMResponse(text="", error="team call limit reached")
        if self.config.condition == "C1":
            request.messages = self.replay_history + request.messages
        try:
            response = self.client.complete(request)
        except BudgetExceeded:
            raise
        except Exception as exc:
            response = LLMResponse(text="", error=f"client exception: {type(exc).__name__}")
        if (response.error or "").startswith("BudgetExceeded:"):
            raise BudgetExceeded(response.error)
        self._record(request, response)
        if self.config.condition == "C1":
            self.replay_history = request.messages + [{"role": "assistant", "content": response.text}]
        return response

    def _independent_calls(self, requests):
        if self.config.condition == "C1":
            return [self._call(request) for request in requests]
        remaining = max(0, self.config.max_calls - len(self.calls) - self.inherited_costs.get("calls", 0))
        pending = requests[:remaining]
        if hasattr(self.client, "complete_many"):
            try:
                responses = self.client.complete_many(pending)
            except BudgetExceeded:
                raise
            except Exception as exc:
                responses = [LLMResponse(text="", error=f"batch client exception: {type(exc).__name__}") for _ in pending]
            if len(responses) != len(pending):
                raise ValueError("complete_many must return one response per request")
            denied = []
            for request, response in zip(pending, responses):
                if (response.error or "").startswith("BudgetExceeded:"):
                    denied.append(response.error)
                else:
                    self._record(request, response)
            if denied:
                # Concurrent calls can succeed on either side of a denied reservation.
                # Preserve all dispatched work, then abort the incomplete observation.
                raise BudgetExceeded("; ".join(denied))
        else:
            responses = [self._call(request) for request in pending]
        return responses + [LLMResponse(text="", error="team call limit reached") for _ in requests[remaining:]]

    def _candidate(self, response, index, phase, parent=None):
        ident = f"s0-{index}" if phase == "stage0" else f"c{len(self.candidates)}"
        candidate = Candidate(ident, self.member_ids[index], self.members[index], response.text,
                              normalize(extract_answer(response.text), self.answer_format), certificate(response.text), phase, parent,
                              valid=response.ok and not response.truncated,
                              error=response.error or ("truncated" if response.truncated else None))
        self.candidates.append(candidate)
        return candidate

    def _can_execute(self):
        if not self.config.tools_enabled or self.config.condition == "A7":
            return False
        if self._sandbox_available is None:
            self._sandbox_available = self.sandbox.available
            if not self._sandbox_available:
                self.trace.append({"event": "tools_unavailable", "reason": self.sandbox.reason})
        return self._sandbox_available

    def _execute(self, code, purpose, candidate_id=None):
        passed, rationale = computational_evidence(code, purpose)
        result = self.sandbox.run(code) if passed else ToolResult(True, error=rationale)
        self.trace.append({"event": "execution", "purpose": purpose, "candidate_id": candidate_id,
                           "code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                           "computational_structure_pass": passed, "structure_assessment": rationale,
                           **asdict(result)})
        return result

    def _verify(self, candidate):
        if not self.config.tools_enabled or self.config.condition == "A7":
            candidate.evidence = CheckEvidence(details="tools disabled by condition")
            return
        if not candidate.valid:
            return
        if self.verifier is not None:
            evidence = self.verifier.verify(self.item, candidate)
            candidate.evidence = evidence if isinstance(evidence, CheckEvidence) else CheckEvidence(**evidence)
            return
        if not self._can_execute():
            candidate.evidence = CheckEvidence(details=self.sandbox.reason)
            return
        if not self._forward_attempted:
            self._tool_forward()
        if self._forward_answer is not None and candidate.answer is not None:
            matched = equivalent(candidate.answer, self._forward_answer, self.answer_format)
            candidate.evidence = CheckEvidence("passed" if matched else "failed", "independent_forward_program",
                ["independently computed final answer agrees"] if matched else [],
                [] if matched else ["independently computed final answer disagrees"], matched,
                "Executable agreement is computational evidence, not a formal proof.")
            return
        response = self._call(self._request(self.checker, PARTIAL_CHECK, "verify_partial", candidate.id,
                                            problem=self.item.problem, certificate=candidate.certificate))
        code = python_code(response.text) if response.ok and not response.truncated else ""
        if not code:
            candidate.evidence = CheckEvidence(details="no executable certificate check returned")
            return
        result = self._execute(code, "partial_certificate", candidate.id)
        if result.error or result.exit_code != 0:
            candidate.evidence = CheckEvidence("error", "certificate_program", details=result.error or result.stderr)
            return
        passed = re.findall(r"^PASS:(\d+)\s*$", result.stdout, re.M)
        failed = re.findall(r"^FAIL:(\d+)\s*$", result.stdout, re.M)
        candidate.evidence = CheckEvidence("failed" if failed else "partial" if passed else "unavailable",
            "certificate_program", [f"certificate line {i}" for i in passed],
            [f"certificate line {i}" for i in failed], False,
            "Partial checks cannot establish correctness of a complete answer.")

    def _tool_forward(self, alt: bool = False):
        """Independent program: computes an answer from the problem's definitions (no candidate in view).

        Its answer is evidence for matching candidates and, with ``tool_candidate``, an
        annotator in the posterior. The alternative variant searches a faithful small model.
        """
        if not alt:
            self._forward_attempted = True
        template = self._forward_alt_template if alt else self._forward_template
        phase = "tool_alt" if alt else "verify_forward"
        response = self._call(self._request(self.checker, template, phase, "checker",
                                            problem=self.item.problem, format_instruction=self.item.format_instruction))
        code = python_code(response.text) if response.ok and not response.truncated else ""
        answer, stdout = None, ""
        if code and self._can_execute():
            result = self._execute(code, "independent_forward")
            stdout = result.stdout or ""
            if result.executed and result.exit_code == 0 and not result.error:
                lines = [ln for ln in stdout.splitlines() if ln.strip().upper().startswith("ANSWER:")]
                raw = lines[-1].split(":", 1)[1].strip() if lines else None
                if raw and raw.upper() != "NONE":
                    answer = raw
        if not alt:
            self._forward_answer = answer
        if self.config.tool_candidate and (code or response.ok):
            ident = "s0-tool" if not alt else f"c{len(self.candidates)}"
            self.candidates.append(Candidate(
                ident, f"tool#{1 if alt else 0}", self.checker,
                f"```python\n{code}\n```\nOUTPUT:\n{stdout[-1500:]}" if code else response.text,
                normalize(answer, self.answer_format) if answer else None, code, "tool_alt" if alt else "tool_forward",
                valid=answer is not None, error=None if answer is not None else "program produced no answer"))
        return answer

    def _cards(self):
        if self.config.condition == "A8":
            return ""
        cards = self.artifacts.get("cards", [])
        words = set(re.findall(r"[a-z]{4,}", self.item.problem.lower()))
        cards = sorted(cards, key=lambda c: -len(words & set(re.findall(r"[a-z]{4,}", c.get("applies_when", "").lower()))))
        clean = [{k: c[k] for k in ("method", "applies_when", "pitfall", "checklist") if k in c} for c in cards[:2]]
        return "\nFallible reusable strategy cards:\n" + json.dumps(clean) if clean else ""

    def _stage0(self):
        requests = [self._request(model, SOLVE, "stage0", f"member={i}", problem=self.item.problem,
                                 format_instruction=self.item.format_instruction, cards=self._cards())
                    for i, model in enumerate(self.members)]
        if self.config.condition == "C1":
            responses = [self._call(request) for request in requests]
        else:
            responses = self._independent_calls(requests)
        created = [self._candidate(response, i, "stage0") for i, response in enumerate(responses)]
        # Member proposals first, then one independent program (blind to the candidates), then checks.
        if (self.config.tool_candidate and self.verifier is None and self.config.tools_enabled
                and self.config.condition != "A7" and self._can_execute()):
            self._tool_forward()
        for candidate in created:
            self._verify(candidate)

    def _member_index(self, candidate):
        return self.member_ids.index(candidate.member)

    def _review(self):
        if self.config.condition == "A2":
            self.trace.append({"event": "review_skipped", "reason": "A2 certificate-only judge"})
            return
        for candidate in list(self.candidates[:len(self.members)]):
            if not candidate.valid:
                continue
            author = self._member_index(candidate)
            others = [i for i in range(len(self.members)) if i != author]
            different = [i for i in others if self.models.model(self.members[i]).lineage != self.models.model(candidate.model_key).lineage]
            if not others:
                continue
            reviewer = (different or others)[0]
            response = self._call(self._request(self.members[reviewer], REVIEW, "review", candidate.id,
                                                problem=self.item.problem, solution=candidate.text))
            report = parse_json(response.text) if response.ok and not response.truncated else {}
            located = (isinstance(report.get("first_error_line"), int)
                       and not isinstance(report.get("first_error_line"), bool) and report["first_error_line"] > 0)
            objection = report.get("verdict") == "invalid" and located and bool(report.get("diagnosis"))
            verified = False
            code = report.get("check_code", "")
            if objection and isinstance(code, str) and code.strip() and self._can_execute():
                result = self._execute(code, "review_objection", candidate.id)
                verified = (result.executed and result.exit_code == 0 and not result.error
                            and "CLAIM_FALSE" in result.stdout.splitlines())
            allowed = objection and (verified or self.config.condition == "A3")
            self.trace.append({"event": "review", "candidate_id": candidate.id, "reviewer": self.member_ids[reviewer],
                               "different_lineage": bool(different), "report": report,
                               "objection_verified": verified, "revision_allowed": allowed})
            if not allowed:
                continue
            # The original stays in the candidate pool, including its earlier evidence.
            concerns = json.dumps({"line": report["first_error_line"], "diagnosis": report["diagnosis"],
                                   "executable_warrant_verified": verified})
            response = self._call(self._request(candidate.model_key, SOLVE_REVISE, "rebuttal_revision", candidate.id,
                problem=self.item.problem, solution=candidate.text, concerns=concerns,
                format_instruction=self.item.format_instruction))
            revised = self._candidate(response, author, "rebuttal_revision", candidate.id)
            self._verify(revised)

    def _tutor(self, solver):
        options = [i for i in range(len(self.members)) if i != solver]
        if not options:
            return solver
        if self.config.condition == "A6":
            digest = hashlib.sha256(f"{self.config.seed}:{self.item.item_id}:{solver}".encode()).digest()
            return options[int.from_bytes(digest[:8], "big") % len(options)]
        matrix = self.artifacts.get("tutoring_matrix", {})
        row = matrix.get(self.member_ids[solver], matrix.get(self.members[solver], {}))
        return max(options, key=lambda i: (float(row.get(self.member_ids[i], row.get(self.members[i], 0))), -i))

    def _coach(self):
        if self.config.condition == "A5":
            self._discussion()
            return
        originals = self.candidates[:len(self.members)]
        for original in originals:
            if not original.valid or original.evidence.full_answer_verified:
                continue
            solver = self._member_index(original)
            tutor = self._tutor(solver)
            current = original
            for level in HINT_LEVELS[:self.config.max_hint_levels]:
                response = self._call(self._request(self.members[tutor], HINT, "hint", f"{original.id}:{level}",
                    problem=self.item.problem, attempt=current.certificate, hint_kind=HINT_KINDS[level]))
                hint = response.text if response.ok and not response.truncated else ""
                audit = structural_audit(hint)
                if audit.structural_pass:
                    auditors = [i for i in range(len(self.members)) if i != tutor]
                    auditor = auditors[0] if auditors else tutor
                    response = self._call(self._request(self.members[auditor], RESIDUAL_AUDIT, "hint_audit",
                        f"{original.id}:{level}", problem=self.item.problem, hint=hint))
                    audit = residual_audit(hint, parse_json(response.text) if response.ok and not response.truncated else {})
                self.trace.append({"event": "hint", "candidate_id": current.id, "solver": original.member,
                                   "tutor": self.member_ids[tutor], "level": level, "hint": hint,
                                   "audit": audit.to_dict(), "matching_measured": bool(self.artifacts.get("tutoring_matrix"))})
                if not audit.accepted:
                    continue
                response = self._call(self._request(current.model_key, SOLVE_SCAFFOLD, "coached_solve",
                    f"{original.id}:{level}", problem=self.item.problem,
                    format_instruction=self.item.format_instruction, note=hint))
                current = self._candidate(response, solver, "coached_solve", current.id)
                self._verify(current)
                if current.evidence.full_answer_verified:
                    break

    def _whiteboard(self):
        return json.dumps([{"candidate_id": c.id, "certificate": c.certificate,
                            "verified_claims": c.evidence.verified_claims,
                            "failed_claims": c.evidence.failed_claims} for c in self.candidates if c.valid])

    def _discussion(self):
        whiteboard = self._whiteboard()
        requests = [self._request(model, DISCUSS, "open_discussion", f"member={i}", problem=self.item.problem,
                                 format_instruction=self.item.format_instruction, whiteboard=whiteboard)
                    for i, model in enumerate(self.members)]
        for i, response in enumerate(self._independent_calls(requests)):
            self._verify(self._candidate(response, i, "open_discussion", f"s0-{i}"))

    def _diversify(self, phase="method_diversification"):
        index = next((i for i, m in enumerate(self.members) if m == self.best_member), 0)
        response = self._call(self._request(self.members[index], DIVERSIFY, phase, f"member={index}",
            problem=self.item.problem, format_instruction=self.item.format_instruction, whiteboard=self._whiteboard()))
        self._verify(self._candidate(response, index, phase))

    def _fresh(self):
        index = next((i for i, m in enumerate(self.members) if m == self.best_member), 0)
        for sample in range(self.config.fresh_samples):
            response = self._call(self._request(self.members[index], SOLVE, "fresh_verified_search", f"sample={sample}",
                problem=self.item.problem, format_instruction=self.item.format_instruction, cards=self._cards()))
            self._verify(self._candidate(response, index, "fresh_verified_search"))

    def _strategies(self, zone):
        if self.config.strategy_override is not None:
            if any(strategy not in STRATEGIES for strategy in self.config.strategy_override):
                raise ValueError("strategy_override contains an unknown template")
            return list(self.config.strategy_override)
        if self.config.condition == "A1":
            return list(STRATEGIES)
        if self.config.condition == "A4":
            return ["evidence_gated_review", "graded_coaching", "method_diversification", "fresh_verified_search"][:self.config.max_strategies]
        bank = self.artifacts.get("strategy_bank", DEFAULT_BANK)
        order = list(bank.get(zone.label, DEFAULT_BANK[zone.label]))
        # Fresh search always precedes remaining strategies in U.
        if zone.label == "U":
            order = ["fresh_verified_search"] + [s for s in order if s != "fresh_verified_search"]
        return order[:self.config.max_strategies]

    def _select(self, exclude_tools: bool = True):
        valid = [c for c in self.candidates if c.valid and c.answer is not None
                 and not (exclude_tools and c.phase in TOOL_PHASES)]
        if not valid:
            return None
        initial = [c for c in self.candidates[:len(self.members)] if c.valid and c.answer is not None]
        labels = cluster([c.answer for c in initial] + [c.answer for c in valid], self.answer_format)
        votes = Counter(labels[:len(initial)])
        vote_counts = {candidate.id: votes[label] for candidate, label in zip(valid, labels[len(initial):])}
        fallback = max(valid, key=lambda c: (c.evidence.full_answer_verified, not bool(c.evidence.failed_claims),
                                             vote_counts[c.id], -self.candidates.index(c)))
        if len(valid) == 1:
            return fallback.id
        scores = self.artifacts.get("discernment", {})
        judge = self.default_judge
        if self.config.condition != "A2" and scores:
            index = max(range(len(self.members)), key=lambda i: float(scores.get(self.member_ids[i], scores.get(self.members[i], 0))))
            judge = self.members[index]
        template = SAT_JUDGE if self.config.condition == "A2" else JUDGE
        pool = []
        for number, candidate in enumerate(valid, 1):
            row = {"number": number, "id": candidate.id, "solution": candidate.text}
            if self.config.condition != "A2":
                row["evidence"] = asdict(candidate.evidence)
                row["reviews"] = [t for t in self.trace if t["event"] == "review" and t["candidate_id"] == candidate.id]
            pool.append(row)
        response = self._call(self._request(judge, template, "selection", "judge", problem=self.item.problem,
            candidates=json.dumps(pool), benchmark_name=self.item.benchmark))
        verdict = parse_json(response.text) if response.ok and not response.truncated else {}
        choice = verdict.get("choice")
        selected = valid[choice - 1] if isinstance(choice, int) and not isinstance(choice, bool) and 1 <= choice <= len(valid) else fallback
        guarded = False
        if self.config.condition != "A3":
            verified = [c for c in valid if c.evidence.full_answer_verified]
            if verified and not selected.evidence.full_answer_verified:
                selected = max(verified, key=lambda c: (vote_counts[c.id], -self.candidates.index(c)))
                guarded = True
        self.trace.append({"event": "selection", "judge": judge, "discernment_measured": bool(scores),
                           "certificate_only": self.config.condition == "A2", "verdict": verdict,
                           "autonomy_guard_applied": guarded, "selected_id": selected.id,
                           "originals_retained": all(c.id in {x.id for x in self.candidates} for c in self.candidates[:len(self.members)])})
        return selected.id

    # ================================================================== posterior router
    def _posterior_params(self) -> PosteriorParams:
        raw = self.artifacts.get("posterior_params")
        return PosteriorParams.from_dict(raw) if raw else default_posterior_params(self.members, self.checker)

    def _validate_live_posterior(self):
        # Training can collect a stage0 checkpoint before calibration. Only routing an
        # evaluated live run requires the frozen, training-derived posterior.
        if not getattr(self.client, "live", False) or self.config.router != "posterior":
            return
        params = self._posterior_params()
        trained = set(params.provenance.get("train_ids", []))
        permitted = set(self.artifacts.get("provenance", {}).get("train_ids", []))
        if not params.calibrated or not trained or not trained <= permitted:
            raise ValueError("live posterior runs require calibrated training-derived posterior parameters; use the rule router for a prior-free pilot")
        needed = set(self.members) | ({"tool"} if self.config.tools_enabled and self.config.condition != "A7" else set())
        if needed - params.agents.keys():
            raise ValueError("live posterior parameters do not cover the configured agents")

    @staticmethod
    def _agent_key(candidate) -> str:
        return "tool" if candidate.phase in TOOL_PHASES else candidate.model_key

    def _included(self):
        return [c for c in self.candidates if c.id not in self._overwritten
                and c.error != "team call limit reached"]

    def _posterior_state(self):
        included = self._included()
        # A3 can retire a tutor's answer from selection after a revision. Its
        # historical truth reference remains necessary to interpret coached
        # responses already generated from it, including chains of coaching.
        needed = {c.id for c in included}
        pending = list(needed)
        while pending:
            candidate_id = pending.pop()
            if candidate_id in self._coach_source:
                source_id, _ = self._coach_source[candidate_id]
                if source_id not in needed:
                    needed.add(source_id)
                    pending.append(source_id)
        included = [c for c in self.candidates if c.id in needed]
        pool = [c for c in included if c.valid and c.answer is not None]
        labels = cluster([c.answer for c in pool], self.answer_format)
        label_of = {c.id: lab for c, lab in zip(pool, labels)}
        obs = []
        for c in included:
            lab = label_of.get(c.id, -1)
            if c.phase in TOOL_PHASES and lab < 0:
                continue                      # a program without an answer abstains
            source, source_lift = -1, None
            if c.id in self._coach_source:
                tutor_candidate, source_lift = self._coach_source[c.id]
                source = label_of.get(tutor_candidate, -1)
            obs.append(Obs(self._agent_key(c), lab, self._lifts.get(c.id, 0.0), source,
                           source_lift if source >= 0 else None))
        evidence = []
        for kind, target, options in self._evidence_records:
            if target not in label_of:
                continue
            if kind == "judge_pick":
                opts = tuple(sorted({label_of[o] for o in options if o in label_of}))
                evidence.append(Evidence(kind, label_of[target], opts))
            else:
                evidence.append(Evidence(kind, label_of[target]))
        for c in pool:
            if c.evidence.source == "certificate_program":
                if c.evidence.status == "failed":
                    evidence.append(Evidence("check_failed", label_of[c.id]))
                elif c.evidence.status == "partial":
                    evidence.append(Evidence("check_passed", label_of[c.id]))
        return label_of, obs, evidence, max(labels, default=-1) + 1

    def _representative(self, cluster_label, label_of, reviewable=False):
        members = [c for c in self._included() if label_of.get(c.id) == cluster_label]
        if reviewable:
            members = [c for c in members if c.phase not in TOOL_PHASES and c.certificate]
        ranked = sorted(members, key=lambda c: (c.phase in TOOL_PHASES, c.phase != "stage0", self.candidates.index(c)))
        return ranked[0] if ranked else None

    def _latest_candidate(self, member_index):
        mine = [c for c in self._included() if c.member == self.member_ids[member_index] and c.phase not in TOOL_PHASES]
        return mine[-1] if mine else None

    def _reviewer_for(self, candidate):
        if candidate.phase in TOOL_PHASES:
            return None
        author_lineage = self.models.model(candidate.model_key).lineage
        others = [i for i in range(len(self.members)) if self.member_ids[i] != candidate.member]
        different = [i for i in others if self.models.model(self.members[i]).lineage != author_lineage]
        pool = different or others
        if not pool:
            return None
        scores = self.artifacts.get("discernment", {})
        return max(pool, key=lambda i: (float(scores.get(self.members[i], 0.0)), -i))

    @staticmethod
    def _tutor_dependent(params) -> bool:
        return "coached_right" in params.lift and "coached_wrong" in params.lift

    def _coaching_spec(self, solver, tutor, params, label_of):
        """(lift, tutor's answer cluster, tutor-right lift) for one prospective coached response.

        With tutor-dependent lifts the response is modeled as strongly lifted only if the
        tutor's current answer is the truth; a tutor without a usable answer gives the
        tutor-wrong lift under every hypothesis.
        """
        if self.config.condition == "A5" or not self._tutor_dependent(params):
            return self._pair_lift(solver, tutor, params), -1, None
        latest = self._latest_candidate(tutor)
        source = label_of.get(latest.id, -1) if latest is not None else -1
        return params.lift["coached_wrong"], source, params.lift["coached_right"]

    def _pair_lift(self, solver, tutor, params) -> float:
        if self.config.condition == "A5":
            return params.lift.get("discussion", 0.0)
        base = params.lift.get("coached", 0.0)
        row = self.artifacts.get("tutoring_matrix", {}).get(self.members[solver], {})
        values = [float(v) for k, v in row.items() if k != self.members[solver]]
        pair = row.get(self.members[tutor])
        if pair is None or not values:
            return base
        # First-order conversion of a probability-scale deviation to logits (slope 1/4 at p = 0.5).
        return float(np.clip(base + 4.0 * (float(pair) - float(np.mean(values))), -1.0, 2.5))

    def _cost_solve(self, key, extra=""):
        spec = self.models.model(key)
        return spec.price.cost(prompt_tokens(self.item.problem, extra) + 260, spec.expected_out(self.item.kind))

    def _cost_short(self, key, tokens_in, tokens_out=None):
        spec = self.models.model(key)
        return spec.price.cost(tokens_in, tokens_out or spec.expected_out("short"))

    def _judge_model(self):
        scores = self.artifacts.get("discernment", {})
        if self.config.condition != "A2" and scores:
            index = max(range(len(self.members)), key=lambda i: float(scores.get(self.members[i], 0)))
            return self.members[index]
        return self.default_judge

    def _feasible_actions(self, label_of, C, result, counters, cfg, params=None):
        cond = self.config.condition
        params = params or self._posterior_params()
        tools = (self.config.tools_enabled and cond != "A7" and self.verifier is None and self._can_execute())
        actions = []
        for i, key in enumerate(self.members):
            if counters["samples"][i] < cfg.max_samples_per_member - 1:
                actions.append(Action("sample", member=i, cost=self._cost_solve(key)))
        # With tutor-dependent lifts every other member is a candidate tutor, so the policy
        # can prefer the one whose current answer is most probably right. A5/A6 and the
        # single-lift model keep one matched tutor per solver.
        every_tutor = self._tutor_dependent(params) and cond not in {"A5", "A6"}
        for s, key in enumerate(self.members):
            latest = self._latest_candidate(s)
            if counters["hints"][s] >= cfg.max_hints_per_solver or latest is None or not latest.certificate:
                continue
            tutors = [t for t in range(len(self.members)) if t != s] if every_tutor else [self._tutor(s)]
            for t in tutors:
                if t == s and len(self.members) > 1:
                    continue
                if cond == "A5":
                    actions.append(Action("coach", member=s, tutor=t,
                                          cost=self._cost_solve(key, self._whiteboard())))
                    continue
                cost = self._cost_short(self.members[t], prompt_tokens(self.item.problem, latest.certificate) + 200)
                if self.config.hint_audit == "residual":
                    cost += self._cost_short(self.members[(t + 1) % len(self.members)], prompt_tokens(self.item.problem) + 400)
                actions.append(Action("coach", member=s, tutor=t, cost=cost + self._cost_solve(key, " " * 1200)))
        if tools and counters["tool_alt"] < cfg.max_tool_alt:
            spec = self.models.model(self.checker)
            actions.append(Action("tool_alt", cost=spec.price.cost(prompt_tokens(self.item.problem) + 400,
                                                                   spec.expected_out("code"))))
        if counters["diversify"] < cfg.max_diversify:
            index = next((i for i, m in enumerate(self.members) if m == self.best_member), 0)
            actions.append(Action("diversify", member=index, cost=self._cost_solve(self.members[index], " " * 4000)))
        if cond != "A2" and C:
            order = [int(c) for c in np.argsort(-result.q)
                     if result.selectable is None or c in result.selectable]
            for c in order[:3]:
                rep = self._representative(c, label_of, reviewable=True)
                if result.q[c] < 0.02 or rep is None or counters["reviews"][rep.id] >= cfg.max_reviews_per_cluster:
                    continue
                reviewer = self._reviewer_for(rep)
                if reviewer is None:
                    continue
                # A verified objection can trigger a solver revision in this same action.
                # Reserve both calls; VOC credits only the review evidence, conservatively.
                review_cost = self._cost_short(self.members[reviewer],
                                               prompt_tokens(self.item.problem, rep.text) + 350, 900)
                revision_cost = self._cost_solve(rep.model_key, rep.text + " " * 1200)
                actions.append(Action("review", member=reviewer, cluster=c, cost=review_cost + revision_cost))
            options = tuple(c for c in order[:4] if result.q[c] >= 0.05)
            if len(options) >= 2 and counters["judges"] < cfg.max_judges:
                pool_text = sum(len(self._representative(c, label_of).text) for c in options)
                actions.append(Action("judge", options=options,
                                      cost=self._cost_short(self._judge_model(), int(pool_text / 3.6) + 900, 1400)))
        return [a for a in actions if a.kind in cfg.allow]

    def _answer_leak(self, hint: str) -> bool:
        """Syntactic filter: a hint must not contain any candidate's final answer (research plan, Leakage audit)."""
        for c in self._included():
            ans = (c.answer or "").strip()
            if len(ans) >= 2 and re.search(r"(?<![\w.])" + re.escape(ans) + r"(?![\w.])", hint):
                return True
        return False

    def _act_sample(self, i, counters):
        counters["samples"][i] += 1
        response = self._call(self._request(self.members[i], SOLVE, "sample", f"sample={counters['samples'][i]}:member={i}",
                                            problem=self.item.problem, format_instruction=self.item.format_instruction,
                                            cards=self._cards()))
        candidate = self._candidate(response, i, "sample")
        self._lifts[candidate.id] = 0.0

    def _act_coach(self, s, t, counters, params):
        current = self._latest_candidate(s)
        level = HINT_LEVELS[min(counters["hints"][s], self.config.max_hint_levels - 1, len(HINT_LEVELS) - 1)]
        counters["hints"][s] += 1
        if self.config.condition == "A5":
            response = self._call(self._request(self.members[s], DISCUSS, "open_discussion", f"member={s}:round={counters['hints'][s]}",
                                                problem=self.item.problem, format_instruction=self.item.format_instruction,
                                                whiteboard=self._whiteboard()))
            candidate = self._candidate(response, s, "open_discussion", current.id)
            self._lifts[candidate.id] = params.lift.get("discussion", 0.0)
            return
        tutor_answer = self._latest_candidate(t)
        response = self._call(self._request(self.members[t], HINT, "hint", f"{current.id}:{level}",
                                            problem=self.item.problem, attempt=current.certificate,
                                            hint_kind=HINT_KINDS[level]))
        hint = response.text if response.ok and not response.truncated else ""
        audit = structural_audit(hint)
        leak = bool(hint) and self._answer_leak(hint)
        if leak:
            audit.structural_pass = False
            audit.reasons.append("hint contains a candidate final answer")
        if audit.structural_pass and self.config.hint_audit == "residual":
            auditor = next((i for i in range(len(self.members)) if i != t), t)
            check = self._call(self._request(self.members[auditor], RESIDUAL_AUDIT, "hint_audit", f"{current.id}:{level}",
                                             problem=self.item.problem, hint=hint))
            audit = residual_audit(hint, parse_json(check.text) if check.ok and not check.truncated else {})
        elif audit.structural_pass:
            audit.accepted = True
        self.trace.append({"event": "hint", "candidate_id": current.id, "solver": self.member_ids[s],
                           "tutor": self.member_ids[t], "level": level, "hint": hint, "answer_leak_filtered": leak,
                           "audit": audit.to_dict(), "matching_measured": bool(self.artifacts.get("tutoring_matrix"))})
        if not audit.accepted:
            return
        response = self._call(self._request(self.members[s], SOLVE_SCAFFOLD, "coached_solve", f"{current.id}:{level}",
                                            problem=self.item.problem, format_instruction=self.item.format_instruction,
                                            note=hint))
        candidate = self._candidate(response, s, "coached_solve", current.id)
        if self._tutor_dependent(params):
            self._lifts[candidate.id] = params.lift["coached_wrong"]
            if tutor_answer is not None:
                self._coach_source[candidate.id] = (tutor_answer.id, params.lift["coached_right"])
        else:
            self._lifts[candidate.id] = self._pair_lift(s, t, params)

    def _act_review(self, candidate, reviewer, counters, params):
        counters["reviews"][candidate.id] += 1
        response = self._call(self._request(self.members[reviewer], REVIEW, "review", candidate.id,
                                            problem=self.item.problem, solution=candidate.text))
        report = parse_json(response.text) if response.ok and not response.truncated else {}
        located = (isinstance(report.get("first_error_line"), int)
                   and not isinstance(report.get("first_error_line"), bool) and report["first_error_line"] > 0)
        objection = report.get("verdict") == "invalid" and located and bool(report.get("diagnosis"))
        verified = False
        code = report.get("check_code", "")
        if objection and isinstance(code, str) and code.strip() and self.config.condition != "A7" and self._can_execute():
            result = self._execute(code, "review_objection", candidate.id)
            verified = (result.executed and result.exit_code == 0 and not result.error
                        and "CLAIM_FALSE" in result.stdout.splitlines())
        kind = ("review_objection_verified" if verified else "review_objection_unverified" if objection
                else "review_valid" if report.get("verdict") == "valid" else "review_unsure")
        self._evidence_records.append((kind, candidate.id, ()))
        allowed = objection and (verified or self.config.condition == "A3")
        self.trace.append({"event": "review", "candidate_id": candidate.id, "reviewer": self.member_ids[reviewer],
                           "report": report, "evidence": kind, "objection_verified": verified,
                           "revision_allowed": allowed})
        if not allowed:
            return
        concerns = json.dumps({"line": report["first_error_line"], "diagnosis": report["diagnosis"],
                               "executable_warrant_verified": verified})
        author = self._member_index(candidate)
        response = self._call(self._request(candidate.model_key, SOLVE_REVISE, "rebuttal_revision", candidate.id,
                                            problem=self.item.problem, solution=candidate.text, concerns=concerns,
                                            format_instruction=self.item.format_instruction))
        if response.error == "team call limit reached":
            return
        revised = self._candidate(response, author, "rebuttal_revision", candidate.id)
        self._lifts[revised.id] = params.lift.get("revision", 0.0)
        if self.config.condition == "A3":
            self._overwritten.add(candidate.id)    # unconditional revision replaces the original

    def _act_judge(self, options, label_of, counters):
        counters["judges"] += 1
        reps = [self._representative(c, label_of) for c in options]
        judge = self._judge_model()
        pool = [{"number": n, "solution": c.text, "evidence": asdict(c.evidence),
                 "reviews": [t["evidence"] for t in self.trace if t.get("event") == "review" and t["candidate_id"] == c.id]}
                for n, c in enumerate(reps, 1)]
        response = self._call(self._request(judge, JUDGE, "selection", "judge", problem=self.item.problem,
                                            candidates=json.dumps(pool), benchmark_name=self.item.benchmark))
        verdict = parse_json(response.text) if response.ok and not response.truncated else {}
        choice = verdict.get("choice")
        valid = isinstance(choice, int) and not isinstance(choice, bool) and 1 <= choice <= len(reps)
        if valid:
            self._evidence_records.append(("judge_pick", reps[choice - 1].id, tuple(r.id for r in reps)))
        self.trace.append({"event": "judge", "judge": judge, "options": [r.id for r in reps], "verdict": verdict,
                           "valid_choice": valid})

    def _execute_action(self, action, label_of, counters, params):
        if action.kind == "sample":
            self._act_sample(action.member, counters)
        elif action.kind == "coach":
            self._act_coach(action.member, action.tutor, counters, params)
        elif action.kind == "tool_alt":
            counters["tool_alt"] += 1
            self._tool_forward(alt=True)
        elif action.kind == "diversify":
            counters["diversify"] += 1
            before = len(self.candidates)
            self._diversify()
            for c in self.candidates[before:]:
                self._lifts[c.id] = params.lift.get("diversify", 0.0)
        elif action.kind == "review":
            self._act_review(self._representative(action.cluster, label_of, reviewable=True), action.member, counters, params)
        elif action.kind == "judge":
            self._act_judge(action.options, label_of, counters)

    def _fixed_order(self, actions, result):
        """A4: zone-agnostic, fixed action order (no value-of-computation ranking)."""
        rank = {"review": 0, "sample": 1, "coach": 2, "judge": 3, "tool_alt": 4, "diversify": 5}
        return sorted(actions, key=lambda a: (rank[a.kind], a.member if a.member is not None else 0,
                                              a.cluster if a.cluster is not None else 0))

    def _run_policy(self):
        params = self._posterior_params()
        cond = self.config.condition
        allow = {"sample", "coach", "tool_alt", "review", "judge", "diversify"}
        if cond == "A2":
            allow -= {"review", "judge"}
        cfg = PolicyConfig(mu=self.effective_mu, epsilon=self.config.voc_epsilon,
                           max_actions=self.config.max_policy_actions,
                           max_item_cost_usd=self.config.max_item_cost_usd, allow=allow,
                           uniform_budget=cond == "A1", screen_coaching=cond != "A5")
        policy = Policy(params, cfg)
        counters = {"samples": Counter(), "hints": Counter(), "reviews": Counter(), "tool_alt": 0, "diversify": 0,
                    "judges": 0}
        steps, zone = [], None
        while True:
            label_of, obs, evidence, C = self._posterior_state()
            selectable = {label_of[c.id] for c in self._included() if c.id in label_of}
            result = policy.posterior_of(obs, evidence, C, selectable=selectable)
            actions = policy.score(self._feasible_actions(label_of, C, result, counters, cfg, params), obs, evidence, C,
                                   lambda m: self.members[m], lambda s, t: self._coaching_spec(s, t, params, label_of),
                                   result)
            if zone is None:
                zone = self._posterior_zone(policy, result, actions, params)
            costs = self._costs()
            if cond == "A4":
                choice = next((a for a in self._fixed_order(actions, result)
                               if costs["resource_cost_usd"] + a.cost <= cfg.max_item_cost_usd), None)
                reason = "fixed order"
                if result.confidence >= 0.9 or len(steps) >= cfg.max_actions:
                    choice, reason = None, "confidence reached or budget exhausted (fixed order)"
            else:
                choice, reason = policy.choose(actions, costs["resource_cost_usd"], len(steps))
            if choice is not None and costs["calls"] >= self.config.max_calls:
                choice, reason = None, "call budget exhausted"
            rep = self._representative(result.best, label_of) if C else None
            steps.append({"step": len(steps), "calls": costs["calls"], "cost_usd": costs["resource_cost_usd"],
                          "marginal_cost_usd": costs["marginal_cost_usd"], "q": [float(x) for x in result.q],
                          "q_none": result.q_none, "confidence": result.confidence, "coverage": result.coverage,
                          "best_candidate": rep.id if rep else None, "best_answer": rep.answer if rep else None,
                          "chosen": choice.to_dict() if choice else None, "stop_reason": None if choice else reason,
                          "candidates": [a.to_dict() for a in sorted(actions, key=lambda a: -a.ratio)[:8]]})
            if choice is None:
                break
            self.trace.append({"event": "policy_action", **choice.to_dict()})
            self._execute_action(choice, label_of, counters, params)
        final = steps[-1]["best_candidate"]
        if cond == "A2":
            final = self._select(exclude_tools=True)
            # Certificate judging is part of this ablation, and changes both decision and cost.
            # It is not reconstructible by threshold truncation of the posterior actions.
            steps[-1].update(best_candidate=final,
                             best_answer=next((c.answer for c in self.candidates if c.id == final), None),
                             calls=self._costs()["calls"], cost_usd=self._costs()["resource_cost_usd"],
                             marginal_cost_usd=self._costs()["marginal_cost_usd"])
        return final, zone, steps, params

    def _posterior_zone(self, policy, result, actions, params):
        label = policy.zone(result, actions)
        members = {self.member_ids[i]: policy.posterior.member_zones(result, key) for i, key in enumerate(self.members)}
        budget = max(1, int(self.config.max_policy_actions))
        members_budget = {self.member_ids[i]: policy.posterior.member_zones(result, key, K=budget)
                          for i, key in enumerate(self.members)}
        reasons = {"A": "posterior confidence above 0.9: no action is worth its cost",
                   "L": "coverage likely but the answer is uncertain: selection evidence pays",
                   "P": "coverage unlikely and coaching has the best value per dollar",
                   "U": "coverage unlikely and fresh search has the best value (or nothing pays)"}
        generation = [a for a in actions if a.kind in {"sample", "coach", "tool_alt", "diversify"} and a.voc > 0]
        return ZoneEstimate(label, {}, not params.calibrated, params.calibrated, reasons[label], {
            "confidence": result.confidence, "coverage": result.coverage, "q": [float(x) for x in result.q],
            "member_zones": members, "member_zones_budget": {"K": budget, "zones": members_budget},
            "posterior": result.summary(),
            "best_generation_action": max(generation, key=lambda a: a.ratio).label() if generation else None,
            "probability_semantics": "label-free posterior (calibrated offline)" if params.calibrated
            else "uncalibrated planning priors"})

    def _costs(self):
        inherited = getattr(self, "inherited_costs", {})
        resource_cost = sum(c.get("resource_cost_usd", 0) for c in self.calls) + inherited.get("resource_cost_usd", 0)
        marginal_cost = sum(c["cost_usd"] for c in self.calls)
        return {"calls": len(self.calls) + inherited.get("calls", 0),
                "dispatch_calls": len(self.calls), "inherited_calls": inherited.get("calls", 0),
                "backend_calls": sum(c.get("attempts", 0) for c in self.calls),
                "cost_usd": marginal_cost, "marginal_cost_usd": marginal_cost,
                "resource_cost_usd": resource_cost, "full_method_cost_usd": resource_cost,
                "tokens_in": sum(c["tokens_in"] for c in self.calls) + inherited.get("tokens_in", 0),
                "tokens_out": sum(c["tokens_out"] for c in self.calls) + inherited.get("tokens_out", 0),
                "tokens_reasoning": sum(c["tokens_reasoning"] for c in self.calls) + inherited.get("tokens_reasoning", 0),
                "latency_s": sum(c["latency_s"] for c in self.calls),
                "summed_latency_s": sum(c["latency_s"] for c in self.calls),
                "wall_latency_s": time.monotonic() - self.started_at,
                "resource_latency_s": sum(c.get("resource_latency_s", 0) for c in self.calls) + inherited.get("resource_latency_s", 0),
                "cached_calls": sum(bool(c["cached"]) for c in self.calls),
                "errors": sum(bool(c["error"]) or c["truncated"] for c in self.calls),
                "executions": sum(t["event"] == "execution" and t.get("executed", False) for t in self.trace)
                              + inherited.get("executions", 0),
                "marginal_executions": sum(t["event"] == "execution" and t.get("executed", False) for t in self.trace)}

    def _start(self, item):
        self.started_at = time.monotonic()
        self.item_provenance = {field: getattr(item, field, None)
                                for field in ("family_id", "band", "source", "item_date", "benchmark")}
        self.item = item.view() if hasattr(item, "view") else item
        self.answer_format = getattr(self.item, "answer_format", "exact")
        required = ("item_id", "benchmark", "problem", "format_instruction", "kind")
        if any(not hasattr(self.item, field) for field in required):
            raise TypeError("run expects an Item or agent-visible ItemView")
        self.candidates, self.calls, self.trace = [], [], []
        self.replay_history = []
        self.inherited_costs = {}
        self._forward_attempted, self._forward_answer = False, None
        self.checkpoint_id = None
        self._lifts, self._evidence_records, self._overwritten = {}, [], set()
        # Coached candidate id -> (tutor's latest candidate id at coaching time, tutor-right lift).
        self._coach_source = {}

    def _evidence_context(self):
        return {"answer_format": self.answer_format,
                "format_instruction_sha256": hashlib.sha256(self.item.format_instruction.encode()).hexdigest(),
                "checker": self.checker, "sandbox_backend": self.sandbox_backend,
                "sandbox_image": self.sandbox.image,
                "sandbox_image_id": getattr(self.sandbox, "_image_id", None),
                "tools_enabled": self.config.tools_enabled and self.config.condition != "A7",
                "tool_candidate": self.config.tool_candidate,
                "verifier": None if self.verifier is None else type(self.verifier).__module__ + "." + type(self.verifier).__qualname__}

    def _snapshot(self):
        checkpoint = {"schema_version": 2, "item_id": self.item.item_id, "seed": self.config.seed,
                      "members": self.member_ids, "stage0": [asdict(c) for c in self.candidates],
                      "stage0_costs": self._costs(), "condition": self.config.condition,
                      "stage0_calls": copy.deepcopy(self.calls), "replay_history": copy.deepcopy(self.replay_history),
                      "forward_attempted": self._forward_attempted, "forward_answer": self._forward_answer,
                      "cards_sha256": hashlib.sha256(self._cards().encode()).hexdigest(),
                      "problem_sha256": hashlib.sha256(self.item.problem.encode()).hexdigest(),
                      "evidence_context": self._evidence_context(),
                      "mode": "live" if getattr(self.client, "live", False) else "mock",
                      "models_sha256": self._checkpoint_digest(self.models.model_dump()),
                      "prompt_hashes": {name: template.id for name, template in TEAM_TEMPLATES.items()}}
        checkpoint["sha256"] = self._checkpoint_digest(checkpoint)
        return checkpoint

    def create_checkpoint(self, item):
        """Generate and check initial proposals only, with no selection overhead."""
        self._start(item)
        self._stage0()
        return self._snapshot()

    def run(self, item, checkpoint: dict | None = None) -> TeamResult:
        self._validate_live_posterior()
        self._start(item)
        if checkpoint is None:
            self._stage0()
            checkpoint = self._snapshot()
        else:
            checkpoint = copy.deepcopy(checkpoint)
            if checkpoint.get("sha256") != self._checkpoint_digest(checkpoint):
                raise ValueError("checkpoint content hash is missing or changed")
            mode = "live" if getattr(self.client, "live", False) else "mock"
            if checkpoint.get("mode") != mode:
                raise ValueError("checkpoint mode does not match the run")
            if checkpoint.get("models_sha256") != self._checkpoint_digest(self.models.model_dump()):
                raise ValueError("checkpoint model settings do not match the run")
            if checkpoint.get("prompt_hashes") != {name: template.id for name, template in TEAM_TEMPLATES.items()}:
                raise ValueError("checkpoint prompt templates do not match the run")
            if (checkpoint.get("item_id") != self.item.item_id or checkpoint.get("seed") != self.config.seed
                    or checkpoint.get("members") != self.member_ids):
                raise ValueError("checkpoint item, seed, or members do not match the run")
            if checkpoint.get("problem_sha256") != hashlib.sha256(self.item.problem.encode()).hexdigest():
                raise ValueError("checkpoint problem text does not match the run")
            saved_context = checkpoint.get("evidence_context")
            if (isinstance(saved_context, dict) and saved_context.get("sandbox_image_id")
                    and self.config.condition != "A7" and self.verifier is None):
                self._can_execute()  # pin the current image before comparing executable evidence
            current_context = self._evidence_context()
            if not isinstance(saved_context, dict):
                raise ValueError("checkpoint evidence context is missing; regenerate stage0")
            always_bound = ("answer_format", "format_instruction_sha256", "checker")
            if any(saved_context.get(key) != current_context[key] for key in always_bound):
                raise ValueError("checkpoint evidence context or answer format does not match the run")
            if self.config.condition != "A7" and saved_context != current_context:
                raise ValueError("checkpoint tool/evidence context does not match the run; regenerate stage0")
            if checkpoint.get("cards_sha256") != hashlib.sha256(self._cards().encode()).hexdigest():
                raise ValueError("checkpoint strategy cards do not match the run; regenerate stage0")
            rows = checkpoint.get("stage0", [])
            member_rows = [r for r in rows if r.get("id") != "s0-tool"]
            if (len(member_rows) != len(self.members)
                    or [r.get("id") for r in member_rows] != [f"s0-{i}" for i in range(len(self.members))]
                    or len(rows) - len(member_rows) > 1):
                raise ValueError("checkpoint must contain exactly the initial member candidates (plus one program)")
            if (checkpoint.get("condition") in {"C1", "C2"} or self.config.condition in {"C1", "C2"}) and checkpoint.get("condition") != self.config.condition:
                raise ValueError("single-agent and homogeneous controls require their own stage0 contexts")
            for row in checkpoint["stage0"]:
                row = copy.deepcopy(row)
                row["evidence"] = CheckEvidence(**row["evidence"])
                self.candidates.append(Candidate(**row))
            self.inherited_costs = checkpoint.get("stage0_costs", {})
            self.replay_history = checkpoint.get("replay_history", [])
            self._forward_attempted = checkpoint.get("forward_attempted", False)
            self._forward_answer = checkpoint.get("forward_answer")
            if self.config.condition == "A7":
                self.candidates = [c for c in self.candidates if c.phase not in TOOL_PHASES]
                for candidate in self.candidates:
                    candidate.evidence = CheckEvidence(details="tools disabled by condition")
                saved_calls = self.calls
                self.calls = [c for c in checkpoint.get("stage0_calls", []) if c["phase"] == "stage0"]
                self.inherited_costs = {}
                self.inherited_costs = self._costs()
                self.calls = saved_calls
            self.trace.append({"event": "checkpoint_replay", "stage0_ids": [c.id for c in self.candidates],
                               "saved_stage0_costs": checkpoint.get("stage0_costs", {})})
        self.checkpoint_id = checkpoint["sha256"]
        anytime, params = [], None
        if self.config.router == "rule":
            members_only = [c for c in self.candidates if c.phase not in TOOL_PHASES]
            zone = estimate_zone(members_only, self.artifacts.get("zone_thresholds"), self.answer_format)
            self.trace.append({"event": "zone", **asdict(zone)})
            order = self._strategies(zone)
            for strategy in order:
                self.trace.append({"event": "strategy", "name": strategy, "zone": zone.label})
                if strategy == "verify_and_accept":
                    continue
                if strategy == "evidence_gated_review":
                    self._review()
                elif strategy == "graded_coaching":
                    self._coach()
                elif strategy == "method_diversification":
                    self._diversify()
                elif strategy == "divergence_reconciliation":
                    self._diversify("divergence_reconciliation")
                elif strategy == "fresh_verified_search":
                    self._fresh()
            if zone.label == "A" and self.config.condition not in {"A1", "A4"} and self.config.strategy_override is None:
                verified = [c for c in self.candidates if c.valid and c.evidence.full_answer_verified
                            and c.phase not in TOOL_PHASES]
                labels = cluster([c.answer for c in verified], self.answer_format)
                counts = Counter(labels)
                best = max(range(len(verified)), key=lambda i: (counts[labels[i]], -i))
                final_id = verified[best].id
            else:
                final_id = self._select()
        else:
            final_id, zone, anytime, params = self._run_policy()
            self.trace.append({"event": "zone", **asdict(zone)})
        return TeamResult(self.item.item_id, self.config.condition, self.config.seed, self.candidates,
            [c["id"] for c in checkpoint["stage0"] if c["id"] != "s0-tool"], final_id, checkpoint=checkpoint, zone=zone,
            meta={"roster": self.config.roster, "members": self.member_ids, "router": self.config.router,
                  "stage0_tool": [c["id"] for c in checkpoint["stage0"] if c["id"] == "s0-tool"
                                  and self.config.condition != "A7"],
                  "anytime": anytime, "mu": self.effective_mu,
                  "mu_source": "explicit" if self.config.mu is not None else "frozen development choice" if "mu_star" in self.artifacts else "mock maximal default",
                  "frontier_replay_valid": self.config.router == "posterior" and self.config.condition not in {"A1", "A2", "A4"},
                  "posterior_calibrated": bool(params and params.calibrated),
                  "sandbox_backend": self.sandbox_backend,
                  "offline_frozen": bool(self.artifacts), "offline_mode": self.artifacts.get("mode", "unmeasured"),
                  "empirical_hint_leakage": "unmeasured", "independent_stage0": self.config.condition != "C1",
                  "control_context": "sequential single-agent replay" if self.config.condition == "C1" else "independent instances",
                  "tools_requested": self.config.tools_enabled and self.config.condition != "A7",
                  "tools_status": ("disabled" if not self.config.tools_enabled or self.config.condition == "A7"
                                   else "injected_verifier" if self.verifier is not None
                                   else "available" if self._sandbox_available
                                   else "unavailable" if self._sandbox_available is False else "not_probed"),
                  "tools_reason": self.sandbox.reason,
                  "verification_semantics": "computational corroboration; generated program semantics are not formally certified",
                  "calls": self.calls}, trace=self.trace, costs=self._costs())

    def run_from_checkpoint(self, item, checkpoint):
        return self.run(item, checkpoint=checkpoint)

    @staticmethod
    def _attempt_index(role):
        match = re.search(r"(?:member=|sample=|s0-)(\d+)", role)
        return int(match.group(1)) if match else None

    @staticmethod
    def _checkpoint_digest(checkpoint):
        payload = {k: v for k, v in checkpoint.items() if k != "sha256"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
