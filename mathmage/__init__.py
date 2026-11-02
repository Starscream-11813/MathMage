"""MathMage v2: zone-aware multi-agent mathematical reasoning.

Subpackages
-----------
llm            OpenRouter client (sync + batch), response cache, budget ledger, call log, mock model
data           item registry, loaders (MathArena, TheoremQA, MATH-500), answer vault, splits
grading        answer extraction and equivalence checking (math-verify, TheoremQA tolerance)
tools          sandboxed Python execution
prompts        frozen, hashed prompt templates
team           the MathMage-Team pipeline (stages 0-3, zone estimation, branches, selection)
psychometrics  IRT / S-IRT, label-free zone posteriors, tutoring matrices
analysis       metrics, statistics, tables, and figures
"""

__version__ = "2.0.0"
