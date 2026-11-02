"""Add S6 (MathMage-R) to an existing results/tables/v6_results.json without recomputing the rest (no API calls).

    python analysis/v6_results_s6.py [--samples 2000]

Runs v6_results.mathmage_r once wave R is complete and rewrites v6_results.json and v6_results.md with it; every other
block is left exactly as the last full run of analysis/v6_results.py wrote it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v6_results as V  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    path = V.OUT_T / "v6_results.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    s6 = V.mathmage_r(V.Registry6(), args.samples)
    if s6 is None:
        raise SystemExit("wave R or the MathMage-R price is not available yet")
    out["S6"] = V.to_jsonable(s6)
    out["status"] = {"sat26": V.status("sat26"), "lane": V.status("lane")}
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    V.write(out, V.OUT_T / "v6_results.md")
    print(json.dumps({k: v for k, v in out["S6"].items() if k != "methods"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
