"""CPU smoke for source-study manifests, configs, Qwen chat rendering and tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.prompt import render_instruction


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                value["_line"] = line_number
                yield value


def validate_arm(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    for split, expected in (
        ("train", int(manifest["train_examples"])),
        ("validation", int(manifest["validation_examples"])),
    ):
        count = 0
        for row in read_jsonl(directory / f"{split}.jsonl"):
            count += 1
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in seen:
                errors.append(f"{split}:{row['_line']}: missing or duplicate example_id")
            seen.add(example_id)
            if row.get("taxonomy_mode") != "off" or row.get("thinking_mode") != "no_think":
                errors.append(f"{split}:{row['_line']}: expected taxonomy-off/no-think")
            metadata = row.get("metadata") or {}
            split_groups[split].add(str(metadata.get("study_group_id") or ""))
            split_content[split].add(str(row.get("content_sha256") or ""))
        if count != expected:
            errors.append(f"{split}: expected {expected}, got {count}")
    if split_groups["train"] & split_groups["validation"]:
        errors.append("study groups cross train/validation")
    if split_content["train"] & split_content["validation"]:
        errors.append("content hashes cross train/validation")
    return {"valid": not errors, "errors": errors}


DEFAULT_CONFIGS = (
    ROOT / "source_study_train_qwen3_4b_nemotron_v3.yaml",
    ROOT / "source_study_train_qwen3_4b_wildguard_en.yaml",
    ROOT / "source_study_train_qwen3_4b_sea_vi.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
    return value


def check_shared_recipe(configs: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    first_path, first = configs[0]
    expected_model = first["model"]
    expected_training = first["training"]
    expected_max_length = int(first["data"]["max_length"])
    for path, cfg in configs[1:]:
        if cfg["model"] != expected_model:
            raise ValueError(f"Model recipe differs: {first_path} vs {path}")
        if cfg["training"] != expected_training:
            raise ValueError(f"Training recipe differs: {first_path} vs {path}")
        if int(cfg["data"]["max_length"]) != expected_max_length:
            raise ValueError(f"max_length differs: {first_path} vs {path}")
    return {
        "model": expected_model,
        "training": expected_training,
        "max_length": expected_max_length,
        "effective_global_batch_4gpu": int(expected_training["per_device_batch_size"])
        * int(expected_training["gradient_accumulation_steps"])
        * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, dest="configs")
    parser.add_argument("--sample-per-arm", type=int, default=64)
    parser.add_argument("--report", type=Path, default=ROOT / "work" / "source-study" / "contract_smoke.json")
    args = parser.parse_args()
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    configs = [(path, load_yaml(path)) for path in config_paths]
    recipe = check_shared_recipe(configs)
    tokenizer = AutoTokenizer.from_pretrained(
        recipe["model"]["id"],
        revision=recipe["model"].get("revision", "main"),
        trust_remote_code=recipe["model"].get("trust_remote_code", False),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    report: dict[str, Any] = {"status": "passed", "recipe": recipe, "arms": {}}
    max_length = recipe["max_length"]
    for config_path, cfg in configs:
        data_dir = (ROOT / Path(cfg["data"]["train"])).resolve().parent
        validation = validate_arm(data_dir)
        if not validation["valid"]:
            raise ValueError(f"Data validation failed for {data_dir}: {validation['errors'][:5]}")
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            expected_hash = manifest["splits"][split]["sha256"]
            actual_hash = file_hash(data_dir / f"{split}.jsonl")
            if actual_hash != expected_hash:
                raise ValueError(f"Hash mismatch for {data_dir / f'{split}.jsonl'}")

        counts: Counter[str] = Counter()
        lengths: list[int] = []
        prompt_lengths: list[int] = []
        target_lengths: list[int] = []
        for index, row in enumerate(read_jsonl(data_dir / "train.jsonl")):
            if index >= args.sample_per_arm:
                break
            if row.get("taxonomy_mode") != "off" or row.get("thinking_mode") != "no_think":
                raise ValueError(f"Mode mismatch in {data_dir}: {row.get('example_id')}")
            target = json.loads(row["target"])
            expected_keys = {"User Safety", "Response Safety"} if row["view"] == "PR" else {"User Safety"}
            if set(target) != expected_keys:
                raise ValueError(f"Wrong target keys in {row.get('example_id')}: {sorted(target)}")
            prompt_text = render_instruction(tokenizer, row["instruction"], enable_thinking=False)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(target_ids) >= max_length:
                raise ValueError(f"Target too long in {row.get('example_id')}: {len(target_ids)}")
            combined = min(len(prompt_ids), max_length - len(target_ids)) + len(target_ids)
            prompt_lengths.append(len(prompt_ids))
            target_lengths.append(len(target_ids))
            lengths.append(combined)
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
            if len(prompt_ids) + len(target_ids) > max_length:
                counts["prompt_truncated"] += 1
        report["arms"][manifest["source"]] = {
            "config": str(config_path),
            "data_dir": str(data_dir),
            "sampled": len(lengths),
            "counts": dict(sorted(counts.items())),
            "prompt_tokens_min_max": [min(prompt_lengths), max(prompt_lengths)],
            "target_tokens_min_max": [min(target_lengths), max(target_lengths)],
            "combined_tokens_min_max": [min(lengths), max(lengths)],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
