"""Answer extraction, equivalence, and grading.

* :mod:`extract` and :mod:`equiv` never touch reference answers; the team pipeline uses
  them to read and cluster candidate answers.
* :mod:`vault` and :mod:`verify` hold and use reference answers; only grading code,
  analysis scripts, and the mock model import them (enforced by tests/test_isolation.py).
"""
