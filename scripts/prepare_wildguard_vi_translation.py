"""Export the matched WildGuard arm for Gemini translation and materialize its VI twin.

The actual API runner is the UID-preserving ``translator`` package in the
workspace root.  This adapter keeps the selected study rows, labels, splits and
group IDs stable so English versus Vietnamese WildGuard is a paired experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.source_study_data import validate_arm
from core.stage2_data import Normalized, read_jsonl, render


def rows(path: Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def export(source_dir: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for split in ("train", "validation"):
            for row in rows(source_dir / f"{split}.jsonl"):
                payload = {
                    "record_uid": row["example_id"],
                    "prompt": row["prompt"],
                    "response": row.get("response"),
                    "study_split": split,
                    "source_id": row["source_id"],
                    "source_group_id": (row.get("metadata") or {}).get("study_group_id"),
                    "prompt_safety_label": row["prompt_safety_label"],
                    "response_safety_label": row["safety_label"] if row["view"] == "PR" else None,
                    "view": row["view"],
                    "source_example_id": row["example_id"],
                    "source_content_sha256": row["content_sha256"],
                    "source_metadata": row.get("metadata") or {},
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1
    report = {"source_dir": str(source_dir), "output": str(output), "records": count}
    if count != 81_000:
        raise ValueError(f"Expected 81,000 paired translation records, got {count}")
    return report


def materialize(translated: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    rendered: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for row in rows(translated):
        split = str(row.get("study_split") or "")
        if split not in rendered:
            raise ValueError(f"Invalid study_split for {row.get('record_uid')}: {split!r}")
        prompt_vi = str(row.get("prompt_vi") or "").strip()
        response_vi = row.get("response_vi")
        response_vi = str(response_vi).strip() if response_vi is not None else None
        view = str(row.get("view") or "")
        if not prompt_vi or (view == "PR" and not response_vi):
            raise ValueError(f"Blank translated field for {row.get('record_uid')}")
        prompt_label = str(row.get("prompt_safety_label") or "")
        response_label = str(row.get("response_safety_label") or "") or None
        normalized = Normalized(
            source="wildguardtrain_vi_gemini",
            source_id=str(row["source_id"]),
            split=split,
            language="vi",
            prompt=prompt_vi,
            response=response_vi if view == "PR" else None,
            prompt_label=prompt_label,
            response_label=response_label if view == "PR" else None,
            categories=[],
            force_taxonomy_off=True,
            metadata={
                **(row.get("source_metadata") or {}),
                "study_group_id": row.get("source_group_id"),
                "source_example_id": row.get("source_example_id"),
                "source_content_sha256": row.get("source_content_sha256"),
                "translation_provider": row.get("translation_provider"),
                "translation_model": row.get("translation_model"),
                "translation_prompt_version": row.get("translation_prompt_version"),
                "translation_text_sha256": row.get("translation_text_sha256"),
            },
        )
        rendered[split].append(render(normalized, seed, taxonomy_mode="off", thinking_mode="no_think"))

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rendered[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = validate_arm(output_dir)
    if report["errors"]:
        raise ValueError("VI WildGuard materialization failed validation")
    manifest = {
        "study": "qwen3_4b_data_source_screen_80k",
        "source": "wildguardtrain_vi_gemini",
        "paired_with": "wildguardtrain_en",
        "translation_input": str(translated),
        "train_examples": len(rendered["train"]),
        "validation_examples": len(rendered["validation"]),
        "taxonomy_mode": "off",
        "thinking_mode": "no_think",
        "license_note": "Derived from gated WildGuardMix. Do not publish a bundle that bypasses upstream access controls.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def merge_shards(source: Path, shard_glob: str, output: Path) -> dict[str, Any]:
    translated_by_uid: dict[str, dict[str, Any]] = {}
    shard_paths = sorted(Path().glob(shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No translation shards match {shard_glob}")
    for path in shard_paths:
        for row in rows(path):
            row.pop("_line", None)
            uid = str(row.get("record_uid") or "")
            if not uid or uid in translated_by_uid:
                raise ValueError(f"Missing or duplicate translated UID {uid!r} in {path}")
            translated_by_uid[uid] = row
    ordered: list[dict[str, Any]] = []
    for row in rows(source):
        uid = str(row["record_uid"])
        translated = translated_by_uid.pop(uid, None)
        if translated is None:
            raise ValueError(f"Missing translated UID: {uid}")
        ordered.append(translated)
    if translated_by_uid:
        raise ValueError(f"Translation shards contain {len(translated_by_uid)} unknown UIDs")
    if len(ordered) != 81_000:
        raise ValueError(f"Expected 81,000 merged translations, got {len(ordered)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"source": str(source), "shards": [str(path) for path in shard_paths], "output": str(output), "records": len(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export")
    p_export.add_argument("--source-dir", type=Path, default=Path("work/source-study/wildguardtrain_en"))
    p_export.add_argument("--output", type=Path, default=Path("work/source-study/wildguard-vi/translation_input.jsonl"))
    p_materialize = sub.add_parser("materialize")
    p_materialize.add_argument("--translated", type=Path, required=True)
    p_materialize.add_argument("--output-dir", type=Path, default=Path("work/source-study/wildguardtrain_vi_gemini"))
    p_materialize.add_argument("--seed", type=int, default=3407)
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--source", type=Path, default=Path("work/source-study/wildguard-vi/translation_input.jsonl"))
    p_merge.add_argument("--shard-glob", default="work/source-study/wildguard-vi/shard_[0-4].jsonl")
    p_merge.add_argument("--output", type=Path, default=Path("work/source-study/wildguard-vi/translated_81000.jsonl"))
    args = parser.parse_args()
    if args.command == "export":
        report = export(args.source_dir, args.output)
    elif args.command == "materialize":
        report = materialize(args.translated, args.output_dir, args.seed)
    else:
        report = merge_shards(args.source, args.shard_glob, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
