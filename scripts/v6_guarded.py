"""Run a frozen script under the fourth wave's grading guard (see scripts/v6_worker_guarded.py).

    python scripts/v6_guarded.py scripts/v6_fit.py fit --package lane

The guard binds the package copy named by --package, routes exact_correct through a killable child process, and
runs the script unchanged as __main__ with the remaining arguments.
"""
import importlib.util
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    script, args = sys.argv[1], sys.argv[2:]
    sys.argv = [script] + args                     # the guard reads --package from sys.argv
    spec = importlib.util.spec_from_file_location("v6_worker_guarded", REPO / "scripts" / "v6_worker_guarded.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    try:
        runpy.run_path(str(Path(script).resolve()), run_name="__main__")
    finally:
        guard.CHILD.shutdown()
        guard.PARSE_CHILD.shutdown()
