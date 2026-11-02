"""Optional Linux-container execution; never run model code on the host.

No host directories or environment variables enter the container. Unavailable
Docker, a missing local image, or non-Linux containers disable execution.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass


@dataclass
class ToolResult:
    available: bool
    executed: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error: str | None = None


class DockerSandbox:
    def __init__(self, image: str = "python:3.11-alpine", timeout_s: float = 20,
                 max_output_bytes: int = 65536):
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("sandbox timeout must be finite and positive")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
            raise ValueError("sandbox output limit must be a positive integer")
        self.image = image
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self._image_id = None
        self.reason = "not probed"

    @property
    def available(self) -> bool:
        if self._image_id:
            return True
        docker = shutil.which("docker")
        if not docker:
            self.reason = "Docker is unavailable; host execution is prohibited"
            return False
        try:
            info = subprocess.run([docker, "info", "--format", "{{json .}}"], capture_output=True,
                                  text=True, timeout=5, check=True)
            data = json.loads(info.stdout)
            if data.get("OSType") != "linux":
                self.reason = "a Linux-container isolation boundary is required"
                return False
            local = subprocess.run([docker, "image", "inspect", self.image, "--format", "{{.Id}}"],
                                   capture_output=True, text=True, timeout=5, check=True)
            image_id = local.stdout.strip()
            if not image_id.startswith("sha256:"):
                self.reason = "could not pin the local sandbox image"
                return False
            self._image_id = image_id
            self.reason = "Linux container, no network or host mounts"
            return True
        except (OSError, ValueError, subprocess.SubprocessError):
            self.reason = "Docker or the locally installed sandbox image is unavailable"
            return False

    def run(self, code: str) -> ToolResult:
        if not self.available:
            return ToolResult(False, error=self.reason)
        if len(code.encode("utf-8")) > 100000:
            return ToolResult(True, error="program exceeds size limit")
        name = "mathmage-check-" + uuid.uuid4().hex
        args = ["docker", "run", "--rm", "--pull", "never", "--name", name,
                "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "64",
                "--memory", "128m", "--memory-swap", "128m", "--cpus", "1",
                "--user", "65534:65534", "--log-driver", "none",
                "--ulimit", "nofile=32:32", "-i", self._image_id,
                "python", "-I", "-S", "-B", "-c", code]
        process = None
        output, errors = bytearray(), bytearray()
        overflow = threading.Event()

        def drain(pipe, target):
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                remaining = self.max_output_bytes - len(target)
                target.extend(chunk[:max(0, remaining)])
                if len(chunk) > remaining:
                    overflow.set()
                    # Continue draining to avoid blocking before timeout cleanup.

        try:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=False)
            threads = [threading.Thread(target=drain, args=(process.stdout, output), daemon=True),
                       threading.Thread(target=drain, args=(process.stderr, errors), daemon=True)]
            for thread in threads:
                thread.start()
            try:
                process.wait(timeout=self.timeout_s)
                error = None
            except subprocess.TimeoutExpired:
                process.kill()
                error = "container execution timed out"
            for thread in threads:
                thread.join(timeout=1)
            if overflow.is_set():
                error = "container output exceeded limit"
            return ToolResult(True, True, output.decode("utf-8", "replace"),
                              errors.decode("utf-8", "replace"), process.returncode, error)
        except OSError as exc:
            return ToolResult(False, error=f"container launch failed: {type(exc).__name__}")
        finally:
            # Kill a detached container too; killing the Docker CLI alone is insufficient.
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            if process is not None and process.poll() is None:
                process.kill()

