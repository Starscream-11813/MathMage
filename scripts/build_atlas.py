"""Build the public solo-sampling atlas from cached MathArena logs (no model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathmage import paths  # noqa: E402
from mathmage.analysis.atlas import COMPETITIONS, build_atlas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", help="Allow missing public datasets to be downloaded")
    parser.add_argument("--competitions", nargs="+", choices=COMPETITIONS, default=COMPETITIONS)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--horizon-k", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-errata", action="store_true", help="Include source attempts on known transcription errata")
    parser.add_argument("--output-dir", type=Path, default=paths.TABLES / "atlas")
    args = parser.parse_args()
    tables, report = build_atlas(competitions=args.competitions, allow_download=args.allow_download,
                                tau=args.tau, horizon_k=args.horizon_k, confidence=args.confidence,
                                bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                                exclude_errata=not args.include_errata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "atlas_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"n_analyzed_attempts": report["n_analyzed_attempts"],
                      "n_item_configuration_pairs": report["n_item_configuration_pairs"],
                      "n_public_configurations": report["n_public_configurations"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
