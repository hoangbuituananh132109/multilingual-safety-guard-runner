from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.source_study_natural_data import (
    SOURCES,
    build_natural_arm,
    build_sea_pair,
    derive_smoke_tree,
    natural_source_path,
    validate_natural_arm,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate source-prior-preserving safety training arms."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", choices=SOURCES, required=True)
    build.add_argument("--source-path", type=Path)
    build.add_argument("--benchmark-root", type=Path, default=ROOT / "work" / "benchmarks")
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--seed", type=int, default=3407)
    build.add_argument("--smoke", action="store_true")
    sea_pair = sub.add_parser("build-sea-pair")
    sea_pair.add_argument("--source-path", type=Path)
    sea_pair.add_argument("--benchmark-root", type=Path, default=ROOT / "work" / "benchmarks")
    sea_pair.add_argument("--output-root", type=Path, default=ROOT / "work" / "source-study-natural")
    sea_pair.add_argument("--seed", type=int, default=3407)
    sea_pair.add_argument("--smoke", action="store_true")
    derive = sub.add_parser("derive-smoke")
    derive.add_argument("--source-root", type=Path, default=ROOT / "work" / "source-study-natural")
    derive.add_argument("--output-root", type=Path, default=ROOT / "work" / "source-study-natural" / "_smoke")
    derive.add_argument("--seed", type=int, default=3407)
    validate = sub.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        report = validate_natural_arm(args.data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["valid"] else 2)

    if args.command == "build-sea-pair":
        source_path = args.source_path or natural_source_path("sea_cultural_vi_natural", ROOT)
        output_root = args.output_root / "_smoke" if args.smoke else args.output_root
        manifests = build_sea_pair(
            source_path,
            args.benchmark_root,
            output_root,
            seed=args.seed,
            smoke=args.smoke,
        )
        print(json.dumps(manifests, ensure_ascii=False, indent=2))
        return

    if args.command == "derive-smoke":
        manifests = derive_smoke_tree(args.source_root, args.output_root, seed=args.seed)
        print(json.dumps(manifests, ensure_ascii=False, indent=2))
        return

    source_path = args.source_path or natural_source_path(args.source, ROOT)
    output_dir = args.output_dir or (
        ROOT / "work" / "source-study-natural" / ("_smoke" if args.smoke else "") / args.source
    )
    manifest = build_natural_arm(
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
