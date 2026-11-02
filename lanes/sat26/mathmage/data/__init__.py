"""Item registry, loaders, and splits.

Agents only ever see :class:`ItemView` objects (identifier, problem text, answer-format
instruction). Reference answers and solutions live in the answer vault
(:mod:`mathmage.grading.vault`), which only the grader and the mock model read.
"""

from mathmage.data.registry import Item, ItemView, Registry, load_registry  # noqa: F401
