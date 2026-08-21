from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as parquet


POLYGUARD_LANGUAGES = {
    "English": "en",
    "Arabic": "ar",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Japanese": "ja",
    "Thai": "th",
    "Chinese": "zh",
}
MULTIJAIL_LANGUAGES = {"en", "ar", "th", "zh"}
POLYGUARD_REVISION = "c5b466a95b64ff121db4398246b6abb7672696ec"
MULTIJAIL_REVISION = "69d16bdef53a0a061d7e56d2b6edb361df7f1507"


def parquet_rows(root: Path) -> Iterable[dict[str, Any]]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    for path in files:
        yield from parquet.read_table(path).to_pylist()


def safety_label(*values: Any) -> str | None:
    for value in values:
        label = str(value or "").strip().casefold()
        if label in {"harmful", "unsafe"}:
            return "unsafe"
        if label in {"unharmful", "safe", "benign"}:
            return "safe"
    return None


def category_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [text]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            counts[f"language:{row['language']}"] += 1
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
    return {"path": str(path), **dict(counts)}


def polyguard_rows(root: Path) -> Iterable[dict[str, Any]]:
    for row in parquet_rows(root):
        language = POLYGUARD_LANGUAGES.get(str(row.get("language")))
        if language is None:
            continue
        prompt = str(row.get("prompt") or "").strip()
        response = str(row.get("response") or "").strip()
        source_id = f"{row.get('id')}:{language}"
        prompt_label = safety_label(row.get("prompt_harm_label"), row.get("prompt_label"))
        response_label = safety_label(row.get("response_harm_label"), row.get("response_label"))
        subset = "adversarial" if bool(row.get("adversarial")) else "standard"
        if prompt and prompt_label:
            yield {
                "example_id": f"polyguard:{source_id}:P",
                "source_id": source_id,
                "source_split": "test",
                "dataset_source": "ToxicityPrompts/PolyGuardPrompts",
                "language": language,
                "view": "P",
                "subset": subset,
                "text": f"Prompt: {prompt}",
                "prompt": prompt,
                "response": None,
                "safety_label": prompt_label,
                "prompt_safety_label": prompt_label,
                "categories": [],
                "category_scope": "prompt",
            }
        if prompt and response and response_label:
            yield {
                "example_id": f"polyguard:{source_id}:PR",
                "source_id": source_id,
                "source_split": "test",
                "dataset_source": "ToxicityPrompts/PolyGuardPrompts",
                "language": language,
                "view": "PR",
                "subset": subset,
                "text": f"Prompt: {prompt}\nResponse: {response}",
                "prompt": prompt,
                "response": response,
                "safety_label": response_label,
                "prompt_safety_label": prompt_label,
                "categories": [],
                "category_scope": "response",
            }


def multijail_rows(root: Path) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(parquet_rows(root)):
        language = str(row.get("language") or "").strip().casefold()
        prompt = str(row.get("text") or "").strip()
        if language not in MULTIJAIL_LANGUAGES or not prompt:
            continue
        yield {
            "example_id": f"multijail:{language}:{index}:P",
            "source_id": str(index),
            "source_split": "test",
            "dataset_source": "ToxicityPrompts/DAMO-MultiJail",
            "language": language,
            "view": "P",
            "subset": "ALL",
            "text": f"Prompt: {prompt}",
            "prompt": prompt,
            "response": None,
            "safety_label": "unsafe",
            "prompt_safety_label": "unsafe",
            "categories": category_list(row.get("tags")),
            "category_scope": "prompt",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optional public benchmarks used by the Nemotron v3 paper.")
    parser.add_argument("--polyguard-root", type=Path)
    parser.add_argument("--multijail-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("work/benchmarks"))
    args = parser.parse_args()
    if args.polyguard_root is None and args.multijail_root is None:
        raise ValueError("Provide --polyguard-root and/or --multijail-root")

    manifest: dict[str, Any] = {
        "sources": {
            "ToxicityPrompts/PolyGuardPrompts": POLYGUARD_REVISION,
            "ToxicityPrompts/DAMO-MultiJail": MULTIJAIL_REVISION,
        },
        "benchmarks": {},
    }
    if args.polyguard_root is not None:
        manifest["benchmarks"]["polyguard_prompts"] = write_jsonl(
            args.output_dir / "polyguard_prompts_9lang.jsonl",
            polyguard_rows(args.polyguard_root),
        )
    if args.multijail_root is not None:
        manifest["benchmarks"]["multijail"] = write_jsonl(
            args.output_dir / "multijail_4lang.jsonl",
            multijail_rows(args.multijail_root),
        )
    output = args.output_dir / "published_benchmark_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
