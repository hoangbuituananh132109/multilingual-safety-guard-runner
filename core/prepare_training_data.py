from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {"safe", "unsafe"}


def clean(value: Any) -> Any:
    return None if isinstance(value, float) and math.isnan(value) else value


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    value["_source_path"] = str(path)
                    value["_source_line"] = line_number
                    yield value


def canonical(source_id: str, split: str, language: str, view: str, prompt: str, response: str | None, label: str, prompt_label: str, categories: list[str], tag: str | None = None) -> dict[str, Any]:
    text = f"Prompt: {prompt}" + (f"\nResponse: {response}" if response is not None else "")
    return {
        "example_id": f"nemotron_v3:{source_id}:{view}:{language}",
        "source_id": source_id,
        "source_split": split,
        "dataset_source": "nemotron_v3",
        "tag": tag,
        "language": language,
        "view": view,
        "text": text,
        "prompt": prompt,
        "response": response,
        "safety_label": label,
        "prompt_safety_label": prompt_label,
        "categories": categories,
        "category_scope": "interaction" if view == "PR" and categories else "unavailable",
    }


def rows(root: Path, split: str, languages: set[str]) -> Iterable[dict[str, Any]]:
    names = {f"{split}.jsonl"}
    if split == "valid":
        names.add("validation.jsonl")
    files = sorted(path for path in root.rglob("*.jsonl") if path.name in names)
    if not files:
        raise FileNotFoundError(f"No {split} JSONL files found under {root}")
    for path in files:
        for row in read_jsonl(path):
            prompt = clean(row.get("prompt"))
            response = clean(row.get("response"))
            prompt_label = str(clean(row.get("prompt_label")) or "").lower()
            response_label = str(clean(row.get("response_label")) or "").lower()
            raw_language = str(clean(row.get("language")) or path.parent.name)
            language = {"zh-CN": "zh", "zh_CN": "zh", "zh-cn": "zh"}.get(raw_language, raw_language)
            if language not in languages:
                continue
            source_id = str(clean(row.get("id")) or digest(f"{path}:{row['_source_line']}"))
            tag = str(clean(row.get("tag")) or "").strip() or None
            categories = [value.strip() for value in str(clean(row.get("violated_categories")) or "").split(",") if value.strip()]
            if prompt and prompt_label in VALID_LABELS:
                yield canonical(source_id, split, language, "P", str(prompt), None, prompt_label, prompt_label, [], tag)
            if prompt and response and response_label in VALID_LABELS:
                safe_prompt_label = prompt_label if prompt_label in VALID_LABELS else "safe"
                yield canonical(source_id, split, language, "PR", str(prompt), str(response), response_label, safe_prompt_label, categories, tag)


def write_split(path: Path, values: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in values:
            key = (row["source_id"], row["view"], row["language"])
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            for field in ("language", "view", "safety_label"):
                counts[f"{field}:{row[field]}"] += 1
            if row.get("tag"):
                counts[f"tag:{row['tag']}"] += 1
    identities = {(source_id, language) for source_id, _view, language in seen}
    return {"path": str(path), "sha256": file_digest(path), **dict(counts)}, identities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    languages = set(args.languages)
    report: dict[str, Any] = {
        "source": "nvidia/Nemotron-Safety-Guard-Dataset-v3",
        "revision": args.revision,
        "languages": sorted(languages),
        "splits": {},
    }
    identities: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "valid", "test"):
        values = rows(args.root, split, languages)
        if args.limit > 0:
            values = islice(values, args.limit)
        split_report, split_ids = write_split(args.output_dir / f"{split}.jsonl", values)
        report["splits"][split] = split_report
        identities[split] = split_ids
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"Cross-split leakage {left}/{right}: {len(overlap)}")
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
