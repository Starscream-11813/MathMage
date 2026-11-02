"""Check this release against the SHA-256 bindings the study's frozen plans and manifests recorded (no API calls).

    python scripts/verify_provenance.py

results/provenance.json lists every package digest and source file that a recorded plan or manifest bound before
its model calls. Files are hashed byte for byte. Each package digest is recomputed by that package's own
_protocol_digest() in a separate interpreter, so the root package and the lane copies never share an import.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIGEST = ("import sys; sys.path.insert(0, sys.argv[1]); import mathmage; "
          "from mathmage.experiments.runner import _protocol_digest; print(mathmage.__file__); print(_protocol_digest())")


def package_digest(package: Path) -> str:
    """The package's own protocol digest, from an interpreter that can import only that copy of mathmage."""
    lines = subprocess.run([sys.executable, "-c", DIGEST, str(package.parent)], cwd=package.parent, check=True,
                           capture_output=True, text=True).stdout.split()
    if Path(lines[0]).resolve().parent != package.resolve():
        raise RuntimeError(f"{package} resolved to another copy of mathmage: {lines[0]}")
    return lines[1]


def main() -> int:
    record = json.loads((REPO / "results" / "provenance.json").read_text(encoding="utf-8"))
    checks = [("package", entry["package"], entry["sha256"], package_digest(REPO / entry["package"]))
              for entry in record["packages"]]
    checks += [("file", entry["path"], entry["sha256"], hashlib.sha256((REPO / entry["path"]).read_bytes()).hexdigest()
                if (REPO / entry["path"]).is_file() else "missing") for entry in record["files"]]
    for kind, target, expected, actual in checks:
        print(f"{'ok' if actual == expected else 'MISMATCH':8} {kind:7} {target:42} {expected[:12]}"
              + ("" if actual == expected else f"  now {actual[:12]}"))
    failed = sum(actual != expected for *_, expected, actual in checks)
    print(f"\n{len(checks) - failed} of {len(checks)} recorded bindings reproduced.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
