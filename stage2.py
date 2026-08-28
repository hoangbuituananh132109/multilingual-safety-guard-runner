from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.stage2_data import audit_translations, build_dataset, inventory_sources, load_stage2_config, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit, build, and validate the Phase-2 safety-guard dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-translations")
    audit.add_argument("--config", type=Path, default=Path("stage2_config.yaml"))
    audit.add_argument("--output", type=Path)

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--config", type=Path, default=Path("stage2_config.yaml"))
    inventory.add_argument("--translation-source", choices=("gemini", "luna_sol"), required=True)
    inventory.add_argument("--output", type=Path)

    build = sub.add_parser("build")
    build.add_argument("--config", type=Path, default=Path("stage2_config.yaml"))
    build.add_argument("--translation-source", choices=("gemini", "luna_sol"), required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--smoke-per-source", type=int, default=0)
    build.add_argument("--allow-incomplete", action="store_true")
    build.add_argument(
        "--exclude-source",
        action="append",
        choices=("v3", "vi", "wildguard", "reasoning", "nemotron35"),
        default=[],
        help="Repeat for controlled ablation builds; exclusions are recorded in the manifest.",
    )

    validate = sub.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "validate":
        result = validate_dataset(args.data_dir)
    else:
        cfg, root = load_stage2_config(args.config)
        if args.command == "audit-translations":
            result = audit_translations(cfg, root)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif args.command == "inventory":
            result = inventory_sources(cfg, root, args.translation_source)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            result = build_dataset(
                cfg,
                root,
                args.translation_source,
                args.output_dir,
                smoke_per_source=args.smoke_per_source,
                allow_incomplete=args.allow_incomplete,
                excluded_sources=set(args.exclude_source),
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "validate" and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
