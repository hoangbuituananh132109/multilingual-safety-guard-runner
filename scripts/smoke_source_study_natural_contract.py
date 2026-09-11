#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from transformers import AutoTokenizer

from core.prompt import render_instruction


SOURCES = (
    "nemotron_v3_9lang_natural",
    "wildguardtrain_en_natural",
    "sea_cultural_vi_natural",
    "sea_cultural_bilingual_id50_natural",
)
THREE_WAY_LABELS = {"safe", "sensitive", "unsafe"}


CONFIGS = {
    "nemotron_v3_9lang_natural": Path("source_study_natural_train_qwen3_4b_nemotron_v3.yaml"),
    "wildguardtrain_en_natural": Path("source_study_natural_train_qwen3_4b_wildguard_en.yaml"),
    "sea_cultural_vi_natural": Path("source_study_natural_train_qwen3_4b_sea_vi.yaml"),
    "sea_cultural_bilingual_id50_natural": Path("source_study_natural_train_qwen3_4b_sea_bilingual.yaml"),
}


def _target_contract(row: dict[str, Any], family: str) -> None:
    target = str(row.get("target") or "")
    label = str(row.get("safety_label") or "")
    view = str(row.get("view") or "")
    if family == "sea_guard":
        if target not in THREE_WAY_LABELS or target != label:
            raise ValueError(f"Invalid SEA target for {row.get('example_id')}: {target!r}")
        return
    payload = json.loads(target)
    key = "Response Safety" if view == "PR" else "User Safety"
    if payload.get(key) != label:
        raise ValueError(f"Invalid binary target for {row.get('example_id')}: {target!r}")


def _common_config_contract(configs: dict[str, dict[str, Any]]) -> None:
    reference_arm = next(iter(configs))
    reference = configs[reference_arm]
    for arm, config in configs.items():
        if config["training"] != reference["training"]:
            raise ValueError(f"Training controls differ between {reference_arm} and {arm}")
        model = dict(config["model"])
        reference_model = dict(reference["model"])
        model.pop("family", None)
        reference_model.pop("family", None)
        if model != reference_model:
            raise ValueError(f"Model/LoRA controls differ between {reference_arm} and {arm}")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_number
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline prompt/target/tokenization contract smoke")
    parser.add_argument("--model", required=True, help="Existing local tokenizer/model directory")
    parser.add_argument("--data-root", type=Path, default=Path("work/source-study-natural/_smoke"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = Path(os.path.expandvars(args.model)).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model/tokenizer directory is missing: {model_path}")

    configs: dict[str, dict[str, Any]] = {}
    for arm, path in CONFIGS.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if Path(config["data"]["train"]).parent.name != arm:
            raise ValueError(f"Config data path does not match arm: {path}")
        configs[arm] = config
    _common_config_contract(configs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    report: dict[str, Any] = {
        "status": "pass",
        "model": str(model_path),
        "data_root": str(args.data_root.resolve()),
        "max_length": args.max_length,
        "arms": {},
    }
    for arm in SOURCES:
        family = str(configs[arm]["model"]["family"])
        counts = {"rows": 0, "truncated_prompts": 0, "max_prompt_tokens": 0, "max_target_tokens": 0}
        for split in ("train", "validation"):
            path = args.data_root / arm / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in _read_jsonl(path):
                _target_contract(row, family)
                rendered = render_instruction(tokenizer, str(row["instruction"]), enable_thinking=False)
                if rendered.rstrip().endswith("<think>"):
                    raise ValueError(f"no_think row opens a reasoning prefill: {row.get('example_id')}")
                target_ids = tokenizer.encode(str(row["target"]), add_special_tokens=False) + [tokenizer.eos_token_id]
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                if len(target_ids) >= args.max_length:
                    raise ValueError(f"Target cannot fit max_length for {row.get('example_id')}")
                counts["rows"] += 1
                counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], len(prompt_ids))
                counts["max_target_tokens"] = max(counts["max_target_tokens"], len(target_ids))
                counts["truncated_prompts"] += int(len(prompt_ids) + len(target_ids) > args.max_length)
        report["arms"][arm] = {"family": family, **counts}

    output_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text, end="")


if __name__ == "__main__":
    main()
