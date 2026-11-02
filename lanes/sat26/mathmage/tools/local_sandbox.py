"""Process-level sandbox for generated verification programs (no Docker required).

Threat model: programs that language models write to check mathematics are not
adversarial; the realistic risks are infinite loops, memory blow-ups, accidental
file writes or reads, and occasional attempts to use the network or spawn
processes. Each program runs in a fresh Python process (isolated mode, minimal
environment without API keys, empty scratch working directory) under:

* a Windows Job Object (POSIX: rlimits) capping memory, CPU time, and child processes;
* a PEP 578 audit hook installed before the program starts, blocking network,
  subprocesses, native calls (ctypes), the registry, writes outside the scratch
  directory, and reads outside the Python installation and the scratch directory;
* a wall-clock timeout and an output cap enforced by the parent.

This is defense in depth for non-adversarial code, not a security boundary against
a determined attacker. The Docker backend remains preferred where Docker exists;
this backend is used only when explicitly selected and is recorded in every result.
"""

from __future__ import annotations

import os
import math
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from mathmage.tools.sandbox import ToolResult

BOOTSTRAP = r'''
import os, sys
SCRATCH = os.path.normcase(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
MEMORY_MB, CPU_SECONDS = int(sys.argv[1]), int(sys.argv[2])
if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rt", "wt", "ot")]
    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]
    class _EXT(ctypes.Structure):
        _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = _EXT()
    # Also pin the job to one permitted CPU; CPU seconds alone do not cap parallelism.
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    affinity_ok = k32.GetProcessAffinityMask(wintypes.HANDLE(k32.GetCurrentProcess()),
                                            ctypes.byref(process_mask), ctypes.byref(system_mask))
    info.Basic.LimitFlags = 0x2 | 0x8 | 0x10 | 0x100 | 0x400
    info.Basic.Affinity = process_mask.value & -process_mask.value
    info.Basic.PerProcessUserTimeLimit = CPU_SECONDS * 10_000_000
    info.Basic.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = MEMORY_MB * 1024 * 1024
    ok = affinity_ok and info.Basic.Affinity and job and k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    ok = ok and k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess()))
    if not ok:
        sys.stderr.write("SANDBOX: could not apply job limits\n")
        sys.exit(97)
    del ctypes, wintypes, k32, info
else:
    import resource
    try:
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    except (AttributeError, ValueError, OSError):
        sys.stderr.write("SANDBOX: could not apply CPU affinity\n")
        sys.exit(97)
    limit = MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))

# Heavy, trusted libraries are imported before the hook so their own optional imports work.
for _m in ("math", "cmath", "fractions", "decimal", "itertools", "functools", "collections", "heapq", "bisect",
           "statistics", "random", "re", "string", "numbers", "operator"):
    __import__(_m)
for _m in ("numpy", "sympy"):
    try:
        __import__(_m)
    except Exception:
        pass

READ_ROOTS = []
for _p in {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(sys.executable), SCRATCH}:
    if _p:
        READ_ROOTS.append(os.path.normcase(os.path.realpath(_p)))
BLOCKED_PREFIXES = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
                    "os.fork", "os.kill", "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function", "winreg.",
                    "_winapi.CreateProcess", "urllib.Request", "webbrowser.", "shutil.rmtree", "os.remove",
                    "os.rmdir", "os.rename", "os.truncate", "os.chmod", "os.symlink", "os.link", "os.putenv",
                    "os.unsetenv", "os.utime", "os.chown", "os.mknod", "os.mkfifo", "msvcrt.")
BLOCKED_IMPORTS = {"socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "multiprocessing",
                   "_multiprocessing", "ctypes", "_ctypes", "winreg", "_winapi", "asyncio", "urllib.request",
                   "http", "http.client", "ftplib", "smtplib", "telnetlib", "webbrowser", "pty",
                   "requests", "httpx", "openai"}

def _inside(path, roots):
    try:
        p = os.path.normcase(os.path.realpath(os.fspath(path)))
    except Exception:
        return False
    return any(p == r or p.startswith(r + os.sep) for r in roots)

def _hook(event, args):
    if event == "import":
        name = args[0] or ""
        if name in BLOCKED_IMPORTS or name.split(".")[0] in BLOCKED_IMPORTS:
            raise ImportError(f"sandbox: import of {name} is not allowed")
        return
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int) or path is None:
            return
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & write_flags)
        if writing:
            if not _inside(path, [SCRATCH]):
                raise PermissionError("sandbox: writes are limited to the scratch directory")
        elif not _inside(path, READ_ROOTS):
            raise PermissionError("sandbox: reads outside the Python installation are not allowed")
        return
    if event in ("os.listdir", "os.scandir") and args and args[0] is not None and not _inside(args[0], READ_ROOTS):
        raise PermissionError("sandbox: directory listing outside the Python installation is not allowed")
    if event in ("os.mkdir", "os.chdir") and args and not _inside(args[0], [SCRATCH]):
        raise PermissionError("sandbox: directory changes are limited to the scratch directory")
    if event.startswith(BLOCKED_PREFIXES):
        raise PermissionError(f"sandbox: {event} is not allowed")

with open(os.path.join(SCRATCH, "program.py"), encoding="utf-8") as _f:
    _source = _f.read()
for _name in ("ctypes", "_ctypes", "subprocess", "socket", "_socket"):
    sys.modules.pop(_name, None)
sys.addaudithook(_hook)
_code = compile(_source, "program.py", "exec")
del _source, _f
exec(_code, {"__name__": "__main__", "__builtins__": __builtins__})
'''


class LocalSandbox:
    """Fresh, limited Python process per program (see module docstring for the threat model)."""

    backend = "local-process"

    def __init__(self, timeout_s: float = 20, memory_mb: int = 1024, max_output_bytes: int = 65536,
                 python: str | None = None):
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("sandbox timeout must be finite and positive")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (memory_mb, max_output_bytes)):
            raise ValueError("sandbox memory and output limits must be positive integers")
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.max_output_bytes = max_output_bytes
        self.python = python or sys.executable
        self.reason = "local process sandbox: Job Object/rlimits + audit hook + timeout (not a hardened boundary)"
        self.image = f"local-python {sys.version.split()[0]}"
        self._image_id = None

    @property
    def available(self) -> bool:
        return Path(self.python).exists()

    def _env(self, scratch: str) -> dict:
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "TEMP": scratch, "TMP": scratch, "TMPDIR": scratch, "HOME": scratch, "USERPROFILE": scratch,
               "PATH": os.path.dirname(self.python)}
        for key in ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def run(self, code: str) -> ToolResult:
        if len(code.encode("utf-8")) > 100000:
            return ToolResult(True, error="program exceeds size limit")
        scratch = tempfile.mkdtemp(prefix="mathmage-sbx-")
        try:
            Path(scratch, "bootstrap.py").write_text(BOOTSTRAP, encoding="utf-8")
            Path(scratch, "program.py").write_text(code, encoding="utf-8")
            args = [self.python, "-I", "-B", os.path.join(scratch, "bootstrap.py"), str(self.memory_mb),
                    str(int(self.timeout_s) + 5)]
            flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
            output, errors = bytearray(), bytearray()
            overflow = threading.Event()

            def drain(pipe, target):
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        break
                    room = self.max_output_bytes - len(target)
                    target.extend(chunk[:max(0, room)])
                    if len(chunk) > room:
                        overflow.set()

            process = subprocess.Popen(args, cwd=scratch, env=self._env(scratch), stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
            threads = [threading.Thread(target=drain, args=(process.stdout, output), daemon=True),
                       threading.Thread(target=drain, args=(process.stderr, errors), daemon=True)]
            for t in threads:
                t.start()
            error = None
            try:
                process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                error = "program timed out"
            for t in threads:
                t.join(timeout=1)
            if overflow.is_set():
                error = "program output exceeded limit"
            if process.returncode == 97:
                error = "sandbox limits could not be applied"
            return ToolResult(True, True, output.decode("utf-8", "replace"), errors.decode("utf-8", "replace"),
                              process.returncode, error)
        except OSError as exc:
            return ToolResult(False, error=f"local sandbox launch failed: {type(exc).__name__}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def make_sandbox(kind: str = "auto", timeout_s: float = 20):
    """'docker' (container), 'local' (process sandbox), or 'auto' (docker if available, else unavailable).

    'auto' never falls back to host execution silently; the local backend must be
    selected explicitly, and every result records which backend ran.
    """
    from mathmage.tools.sandbox import DockerSandbox

    if kind not in {"auto", "docker", "local"}:
        raise ValueError("sandbox must be auto, docker or local")
    if kind == "local":
        return LocalSandbox(timeout_s=timeout_s)
    return DockerSandbox(timeout_s=timeout_s)
