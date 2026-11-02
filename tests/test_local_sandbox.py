"""The local process sandbox runs math code and blocks network, processes, secrets, and runaways."""

import os
import sys

import pytest

from mathmage import paths
from mathmage.tools.local_sandbox import LocalSandbox, make_sandbox

pytestmark = pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="sandbox limits implemented for Windows/Linux")


@pytest.fixture(scope="module")
def box():
    return LocalSandbox(timeout_s=20, memory_mb=768)


def test_math_programs_run_with_sympy_and_numpy(box):
    r = box.run("import sympy, numpy as np\n"
                "n = sum(1 for a in range(1, 50) for b in range(1, 50) if sympy.lcm(a, b) == 48)\n"
                "print('ANSWER:', n, int(np.prod([2, 3])))")
    assert r.executed and r.exit_code == 0 and r.error is None, r.stderr
    assert r.stdout.strip().startswith("ANSWER:")


@pytest.mark.parametrize("code", [
    "import socket\nprint('ANSWER: 1')",
    "import subprocess\nsubprocess.run(['cmd', '/c', 'echo hi'])",
    "import os\nos.system('echo hi')",
    "import ctypes\nprint('ANSWER: 1')",
    "__import__('urllib.request').request.urlopen('http://example.com')",
])
def test_network_processes_and_native_calls_are_blocked(box, code):
    r = box.run(code)
    assert r.executed and r.exit_code != 0
    assert "ANSWER" not in r.stdout


def test_secrets_and_files_outside_scratch_are_unreadable_and_unwritable(box):
    secret = (paths.ROOT / "api" / "config.json").as_posix()
    r = box.run(f"print(open({secret!r}).read())")
    assert r.exit_code != 0 and "OPENROUTER" not in r.stdout and "sk-" not in r.stdout
    target = (paths.ROOT / "sandbox_escape_test.txt").as_posix()
    r = box.run(f"open({target!r}, 'w').write('x')")
    assert r.exit_code != 0 and not os.path.exists(target)
    r = box.run("import os\nprint(os.environ.get('OPENROUTER_API_KEY'))")
    assert r.exit_code == 0 and r.stdout.strip() == "None"


def test_scratch_writes_are_allowed(box):
    r = box.run("open('notes.txt', 'w').write('ok')\nprint(open('notes.txt').read())")
    assert r.exit_code == 0 and r.stdout.strip() == "ok", r.stderr


def test_runaway_loops_and_memory_are_stopped():
    box = LocalSandbox(timeout_s=3, memory_mb=512)
    r = box.run("while True:\n    pass")
    assert r.error == "program timed out"
    r = box.run("x = bytearray(1600 * 1024 * 1024)\nprint('ANSWER: big')")
    assert r.exit_code != 0 and "ANSWER" not in r.stdout


def test_auto_never_selects_host_execution():
    from mathmage.tools.sandbox import DockerSandbox

    assert isinstance(make_sandbox("auto"), DockerSandbox)
    assert isinstance(make_sandbox("local"), LocalSandbox)
    with pytest.raises(ValueError, match="sandbox must"):
        make_sandbox("typo")


def test_directory_mutations_and_readonly_truncation_are_blocked(box, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding="utf-8")
    directory = tmp_path / "created"
    for code in (f"import os\nos.mkdir({str(directory)!r})",
                 f"import os\nos.chdir({str(tmp_path)!r})",
                 f"import os\nos.open({str(outside)!r}, os.O_RDONLY | os.O_TRUNC)"):
        result = box.run(code)
        assert result.exit_code != 0
    assert outside.read_text(encoding="utf-8") == "preserve" and not directory.exists()


@pytest.mark.parametrize("kwargs", [{"timeout_s": float("nan")}, {"timeout_s": float("inf")},
                                      {"timeout_s": 0}, {"memory_mb": 0}, {"max_output_bytes": -1}])
def test_invalid_resource_limits_fail_closed(kwargs):
    with pytest.raises(ValueError):
        LocalSandbox(**kwargs)
