from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download


VISAFE_REPO = "nvidia/Nemotron-Content-VISafe-v1"
VISAFE_REVISION = "c0d65adea85f4fe3bc9ccd8daeeaf8022c82049a"
VISAFE_FILENAME = "probes.jsonl"
ARMS = ("nemotron_v3_9lang", "wildguardtrain_en", "sea_cultural_vi")
LABEL_BY_EXPECTED_BEHAVIOR = {
    "refuse": "unsafe",
    "warn_and_refuse": "unsafe",
    "allow": "safe",
    "neutral_response": "safe",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def load_arm_prompts(root: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    prompts: dict[str, set[str]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        values: set[str] = set()
        files = []
        for split in ("train", "validation"):
            path = root / arm / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing matched source-study arm: {path}. Install all three arms before preparing VISafe."
                )
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt = normalize_prompt(row.get("prompt"))
                    if prompt:
                        values.add(prompt)
                    count += 1
            files.append({"split": split, "rows": count, "sha256": sha256(path)})
        prompts[arm] = values
        provenance[arm] = {"files": files, "unique_normalized_prompts": len(values)}
    return prompts, provenance


def visafe_rows(source: Path, arm_prompts: dict[str, set[str]]) -> Iterable[dict[str, Any]]:
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("id") or index)
            if source_id in seen_ids:
                raise ValueError(f"Duplicate VISafe id: {source_id}")
            seen_ids.add(source_id)
            prompt = str(row.get("prompt_vi") or "").strip()
            expected = str(row.get("expected_behavior") or "").casefold()
            label = LABEL_BY_EXPECTED_BEHAVIOR.get(expected)
            if not prompt or label is None:
                raise ValueError(f"Invalid VISafe row {source_id}: prompt/expected_behavior")
            candidates = {
                normalize_prompt(prompt),
                normalize_prompt(row.get("prompt_en")),
            } - {""}
            overlap_arms = sorted(
                arm for arm, training_prompts in arm_prompts.items() if candidates & training_prompts
            )
            yield {
                "example_id": f"visafe:{source_id}:P:vi",
                "source_id": source_id,
                "source_split": "train",
                "dataset_source": VISAFE_REPO,
                "language": "vi",
                "view": "P",
                "subset": "source_overlap" if overlap_arms else "clean",
                "topic": f"{row.get('category')}/{row.get('subcategory')}",
                "text": f"Prompt: {prompt}",
                "prompt": prompt,
                "response": None,
                "safety_label": label,
                "prompt_safety_label": label,
                "categories": [],
                "category_scope": "prompt",
                "metadata": {
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "source": row.get("source"),
                    "risk_level": row.get("risk_level"),
                    "expected_behavior": expected,
                    "detection_strategy": row.get("detection_strategy"),
                    "tags": row.get("tags"),
                    "prompt_en": row.get("prompt_en"),
                    "source_overlap_arms": overlap_arms,
                    "binary_mapping": "refuse/warn_and_refuse=unsafe; allow/neutral_response=safe",
                },
            }


def write_benchmark(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            counts[f"label:{row['safety_label']}"] += 1
            counts[f"subset:{row['subset']}"] += 1
            counts[f"behavior:{row['metadata']['expected_behavior']}"] += 1
            counts[f"category:{row['metadata']['category']}"] += 1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NVIDIA VISafe for the matched source study.")
    parser.add_argument("--source", type=Path, help="Optional local official probes.jsonl.")
    parser.add_argument("--source-study-root", type=Path, default=Path("work/source-study"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/benchmarks"))
    args = parser.parse_args()

    source = args.source or Path(
        hf_hub_download(
            VISAFE_REPO,
            VISAFE_FILENAME,
            repo_type="dataset",
            revision=VISAFE_REVISION,
        )
    )
    arm_prompts, arm_provenance = load_arm_prompts(args.source_study_root)
    benchmark_path = args.output_dir / "visafe_vi.jsonl"
    benchmark = write_benchmark(benchmark_path, visafe_rows(source, arm_prompts))
    manifest = {
        "dataset": VISAFE_REPO,
        "revision": VISAFE_REVISION,
        "source_file": VISAFE_FILENAME,
        "source_sha256": sha256(source),
        "license": "NVIDIA Sample Data License",
        "redistribution": "prohibited; download the official file directly from NVIDIA",
        "protocol": {
            "official": "Generate an assistant response and score expected_behavior with an LLM judge; heuristic refusal matching is smoke-only.",
            "guard_adapter": "Prompt-only binary classification: refuse/warn_and_refuse=unsafe; allow/neutral_response=safe.",
            "fair_slice": "subset=clean excludes exact normalized prompt_vi or prompt_en overlap with any locked 80k train/validation arm.",
        },
        "arm_provenance": arm_provenance,
        "benchmark": benchmark,
    }
    manifest_path = args.output_dir / "visafe_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
