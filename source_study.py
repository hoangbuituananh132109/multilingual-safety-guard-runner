from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.source_study_data import build_arm, default_source_paths, validate_arm


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the matched 80k data-source study.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", choices=["nemotron_v3_9lang", "wildguardtrain_en", "sea_cultural_vi"], required=True)
    build.add_argument("--source-path", type=Path)
    build.add_argument("--benchmark-root", type=Path, default=ROOT / "work" / "benchmarks")
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--seed", type=int, default=3407)
    build.add_argument(
        "--smoke",
        action="store_true",
        help="Build a 32-train/8-validation contract smoke set instead of the 80k/1k arm.",
    )
    validate = sub.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        report = validate_arm(args.data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["valid"] else 2)

    source_path = args.source_path or default_source_paths(ROOT)[args.source]
    output_dir = args.output_dir or (
        ROOT / "work" / "source-study" / ("_smoke" if args.smoke else "") / args.source
    )
    manifest = build_arm(
        args.source,
        source_path,
        args.benchmark_root,
        output_dir,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
