"""Launch a v6 worker with a process-isolated grading timeout (fourth-wave amendment; no new API behaviour).

The frozen helper mathmage.grading.timeout.call_with_timeout runs math_verify in a daemon thread and gives up after
its timeout. A computation that holds the interpreter lock defeats that timeout and freezes the whole worker: the
first case was a big-integer power such as 5^(2^32) in a model's answer to HMMT November 2025 Guts #2. This launcher
runs the frozen exact_correct, unchanged, in a persistent child process (scripts/v6_grading_child.py). The child is
killed and restarted when a call exceeds a hard deadline, and the call then returns False, the value the frozen
helper returns on timeout. Every comparison that finishes within the deadline returns exactly what the frozen code
returns. Nothing under mathmage/ and no frozen script changes; scripts/v6_followup.py runs as __main__ with its own
arguments, so its plan and hash checks are unaffected.

    python scripts/v6_worker_guarded.py worker --package lane --wave F --index 0 --count 8
"""
import json
import queue
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "lane"
ROOT = REPO / "lanes" / ("sat26" if PACKAGE == "sat26" else "open_weight")
DEADLINE_S = 30.0         # the frozen math_verify timeout is 8 s, and exact_correct adds one numeric check
RESTART_AFTER_S = 8.0     # a call this slow may leave a runaway thread in the child, so the child is replaced

sys.path.insert(0, str(ROOT))
import mathmage.grading.equiv as EQ  # noqa: E402  (binds the package copy that v6_followup then uses)
import mathmage.grading.verify as VERIFY  # noqa: E402


class GradingChild:
    SPARES = 8                                   # warm replacements kept ready, so repeated kills cost no start-up

    def __init__(self):
        self.lock = threading.Lock()
        self.proc, self.replies, self.counter = None, None, 0
        self.stats = {"calls": 0, "killed": 0, "restarted": 0}
        self.spares, self.spare_lock, self.spare_pending = [], threading.Lock(), 0

    def _spawn(self):
        proc = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts" / "v6_grading_child.py"), str(ROOT)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", bufsize=1,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        replies = queue.Queue()
        threading.Thread(target=self._read, args=(proc, replies), daemon=True).start()
        return proc, replies

    def _make_spare(self):
        """Start and warm (imports done) replacement children in the background, up to SPARES of them."""
        with self.spare_lock:
            wanted = self.SPARES - len(self.spares) - self.spare_pending
            self.spare_pending += max(0, wanted)

        def build():
            proc, replies = self._spawn()
            try:
                proc.stdin.write(json.dumps({"i": 0, "g": "1", "p": "2"}) + "\n")
                proc.stdin.flush()
                replies.get(timeout=60)
            except (OSError, queue.Empty):
                proc.kill()
                proc = None
            with self.spare_lock:
                self.spare_pending -= 1
                if proc is not None:
                    self.spares.append((proc, replies))
        for _ in range(max(0, wanted)):
            threading.Thread(target=build, daemon=True).start()

    def _start(self):
        spare = None
        with self.spare_lock:
            while self.spares and spare is None:
                candidate = self.spares.pop(0)
                spare = candidate if candidate[0].poll() is None else None
        if spare is not None:
            self.proc, self.replies = spare
        else:
            self.proc, self.replies = self._spawn()
        self._make_spare()

    @staticmethod
    def _read(proc, replies):
        for line in proc.stdout:
            replies.put(line)

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc = None

    def shutdown(self):
        self.stop()
        with self.spare_lock:
            spares, self.spares = self.spares, []
        for proc, _ in spares:
            try:
                proc.kill()
            except OSError:
                pass

    def ask(self, gold, pred):
        return self.ask_with_reason(gold, pred)[0]

    def ask_with_reason(self, gold, pred):
        """(result, deterministic): deterministic is False only when the pipe itself failed."""
        if pred is None:                       # the frozen helper's first test
            return False, True
        result, status = self.request({"g": gold, "p": pred}, DEADLINE_S)
        return result, status != "pipe"

    def request(self, payload, deadline):
        """(result, status) with status 'ok', 'deadline' (child killed: the frozen timeout value) or 'pipe'."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.counter += 1
            self.stats["calls"] += 1
            started = time.monotonic()
            try:
                self.proc.stdin.write(json.dumps({"i": self.counter, **payload}) + "\n")
                self.proc.stdin.flush()
                while True:
                    reply = json.loads(self.replies.get(timeout=max(.01, deadline - (time.monotonic() - started))))
                    if reply["i"] == self.counter:
                        result = bool(reply["r"])
                        break
            except queue.Empty:
                self.stop()
                self.stats["killed"] += 1
                return False, "deadline"
            except (OSError, ValueError):
                self.stop()
                self.stats["killed"] += 1
                return False, "pipe"
            if time.monotonic() - started > RESTART_AFTER_S:
                self.stop()
                self.stats["restarted"] += 1
            return result, "ok"


CHILD = GradingChild()
FROZEN_EXACT_CORRECT = EQ.exact_correct
# Amendment (2026-10-24 23:20 UTC, operational): exact_correct is a pure function of two strings, and answer
# clustering asks the same pair many times; a pathological pair (one that runs into the deadline) then cost 30 s on
# every repeat, and one mechanism state spent over 50 minutes in grading. Outcomes are memoized per process, so each
# distinct pair is evaluated once; a repeat returns exactly the value the first evaluation returned.
_MEMO, _MEMO_LOCK = {}, threading.Lock()

# Amendment 58 (2026-10-25 00:35 UTC, operational): the frozen mv_parse caches its result per process, a timeout's
# None included, so in the frozen code an answer whose parse times out costs its 5 seconds once and every later
# comparison with it is immediate. The child is killed after a hang and loses that cache; one calibration item (power
# towers such as 4048^{2024^2}) then cost the full deadline on every pair. The parse result is now tested once per
# cleaned answer, in its own child with a deadline just above the frozen 5-second parse timeout, and kept here. With
# two distinct cleaned answers, a falsy parse of the gold answer, or of a non-decimal predicted answer, ends every
# branch of the frozen exact_correct in False, so such a pair returns False without being sent to the child.
PARSE_DEADLINE_S = 6.0
PARSE_CHILD = GradingChild()
_PARSE_MEMO = {}


def parse_ok(s):
    """Whether the frozen mv_parse(s) is truthy (False past its timeout); None if the test itself failed."""
    with _MEMO_LOCK:
        if s in _PARSE_MEMO:
            return _PARSE_MEMO[s]
    result, status = PARSE_CHILD.request({"parse": s}, PARSE_DEADLINE_S)
    if status == "pipe":
        return None
    ok = result if status == "ok" else False
    with _MEMO_LOCK:
        _PARSE_MEMO[s] = ok
    return ok


def parse_shortcut(gold, pred):
    """True when the frozen exact_correct(gold, pred) is False because one cleaned answer does not parse."""
    if pred is None:
        return False
    g, p = EQ.clean_answer(gold), EQ.clean_answer(pred)
    if not g or not p or g == p:
        return False
    if parse_ok(g) is False:
        return True
    return not EQ._DECIMAL_RE.fullmatch(p.replace(" ", "")) and parse_ok(p) is False


# Amendment 59 (2026-10-25 00:50 UTC, operational): an answer is pathological when the frozen check runs past the
# deadline against a plain partner ("0", either direction), i.e. it forces an exact evaluation that cannot finish
# (power towers such as 4048^{2024^2}). Against such an answer the frozen check returns True only on a fast path
# (string or structural equality, milliseconds); any other comparison needs that evaluation, which cannot finish within
# the frozen 8-second verify timeout, and the numeric check of a value beyond floating range fails, so the frozen
# intended result is False. Pairs with a known pathological answer therefore get a short deadline (2 s; 1.5 s since amendment 62, 0.75 s since amendment 63).
SHORT_DEADLINE_S = 0.75
_KIND = {}

# Amendment 64: results precomputed in parallel under the same rules (program_v6/grading_memo.json, written by the
# session's precompute_memo.py) are loaded at start-up, so a refit replays them instead of recomputing them in series.
_DISK_MEMO = REPO / "runs" / "live_20261018" / "program_v6" / "grading_memo.json"
if _DISK_MEMO.exists():
    try:
        _disk = json.loads(_DISK_MEMO.read_text(encoding="utf-8"))
        _MEMO.update({(a, b): bool(r) for a, b, r in _disk.get("pairs", [])})
        _KIND.update({s: "pathological" for s in _disk.get("pathological", [])})
    except (OSError, ValueError):
        pass


def classify(s):
    """'pathological' or 'normal' (cached): whether the frozen check hangs against the plain partner '0'."""
    with _MEMO_LOCK:
        if s in _KIND:
            return _KIND[s]
    kind = "normal"
    for g, p in (("0", s), (s, "0")):
        if CHILD.request({"g": g, "p": p}, DEADLINE_S)[1] == "deadline":
            kind = "pathological"
            break
    with _MEMO_LOCK:
        _KIND[s] = kind
    return kind


def guarded_exact_correct(gold, pred):
    key = (gold, pred)
    with _MEMO_LOCK:
        if key in _MEMO:
            return _MEMO[key]
    if pred is None:                           # the frozen helper's first test
        return False
    if parse_shortcut(gold, pred):
        with _MEMO_LOCK:
            _MEMO[key] = False
        return False
    known = _KIND.get(gold) == "pathological" or _KIND.get(pred) == "pathological"
    result, status = CHILD.request({"g": gold, "p": pred}, SHORT_DEADLINE_S if known else DEADLINE_S)
    if status == "deadline" and not known:
        # Amendment 62: both answers of a pair that ran past the full deadline are marked, whether or not either
        # hangs against '0' (some towers hang only against particular partners). A normal answer's comparisons
        # finish in milliseconds, so marking it costs nothing.
        with _MEMO_LOCK:
            for s in (gold, pred):
                _KIND[s] = "pathological"
    deterministic = status != "pipe"
    if deterministic:
        with _MEMO_LOCK:
            _MEMO[key] = result
    return result


def install():
    EQ.exact_correct = guarded_exact_correct
    VERIFY.exact_correct = guarded_exact_correct


if __name__ == "__main__":
    install()
    sys.argv = [str(REPO / "scripts" / "v6_followup.py")] + sys.argv[1:]
    try:
        runpy.run_path(str(REPO / "scripts" / "v6_followup.py"), run_name="__main__")
    finally:
        CHILD.shutdown()
        PARSE_CHILD.shutdown()
