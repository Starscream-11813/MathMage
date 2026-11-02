"""Grading child for scripts/v6_worker_guarded.py (fourth-wave amendment; read-only).

Runs the frozen ``mathmage.grading.equiv.exact_correct`` of the given package copy on JSON-line requests
{"i": id, "g": gold, "p": pred} from stdin and answers {"i": id, "r": bool} on the original stdout. A request
{"i": id, "parse": s} answers whether the frozen ``mv_parse(s)`` is truthy (amendment 58: the guard's parse test).
Library output is diverted to stderr so the protocol stays clean.

    python scripts/v6_grading_child.py <package root>
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
REPLY = os.fdopen(os.dup(1), "w", encoding="utf-8")
sys.stdout = sys.stderr

from mathmage.grading.equiv import exact_correct, mv_parse  # noqa: E402

for line in sys.stdin:
    request = json.loads(line)
    try:
        if "parse" in request:
            result = bool(mv_parse(request["parse"]))
        else:
            result = bool(exact_correct(request["g"], request["p"]))
    except Exception:  # noqa: BLE001 - the frozen helpers never raise; be as safe here
        result = False
    REPLY.write(json.dumps({"i": request["i"], "r": result}) + "\n")
    REPLY.flush()
