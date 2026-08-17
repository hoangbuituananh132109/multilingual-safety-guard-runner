from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def overall(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("benchmark") == "sea_vi" and row.get("language") == "ALL" and row.get("view") == "ALL" and row.get("subset") == "ALL":
            return row
    return None


def run_directory(runs_root: Path, model: str, checkpoint: str, method: str, mode: str) -> Path:
    if checkpoint == "before":
        return runs_root / model / "base"
    return runs_root / model / f"{method}_{mode}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", choices=["before", "after"], default="before")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_root = Path(config["paths"]["runs_root"])
    rows: list[dict[str, Any]] = []
    for specification in config["models"]:
        model = str(specification["name"])
        directory = run_directory(runs_root, model, args.checkpoint, args.method, args.run_mode)
        likelihood = overall(load_rows(directory / "likelihood" / "likelihood_metrics.json"))
        guard = overall(load_rows(directory / "guard" / "metrics.json"))
        rows.append({
            "model": model,
            "run_directory": str(directory),
            "likelihood_ready": likelihood is not None,
            "guard_ready": guard is not None,
            "bits_per_byte": likelihood.get("bits_per_byte") if likelihood else None,
            "perplexity": likelihood.get("perplexity") if likelihood else None,
            "tokens_scored": likelihood.get("tokens_scored") if likelihood else None,
            "guard_macro_f1": guard.get("macro_f1") if guard else None,
            "guard_unsafe_recall": guard.get("unsafe_recall") if guard else None,
            "guard_unsafe_f1": guard.get("unsafe_f1") if guard else None,
            "guard_parse_rate": guard.get("parse_rate") if guard else None,
        })
    bpb_ranked = sorted((row for row in rows if row["bits_per_byte"] is not None), key=lambda row: float(row["bits_per_byte"]))
    guard_ranked = sorted((row for row in rows if row["guard_macro_f1"] is not None), key=lambda row: (-float(row["guard_macro_f1"]), -float(row["guard_unsafe_recall"] or 0.0)))
    for rank, row in enumerate(bpb_ranked, 1):
        row["bpb_rank"] = rank
    for rank, row in enumerate(guard_ranked, 1):
        row["guard_rank"] = rank
    for row in rows:
        row.setdefault("bpb_rank", None)
        row.setdefault("guard_rank", None)
    default_output = runs_root / f"model_comparison_{args.checkpoint}.csv"
    output = (args.output or default_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "method": args.method if args.checkpoint == "after" else None,
        "run_mode": args.run_mode if args.checkpoint == "after" else None,
        "comparison_csv": str(output),
        "ranking_rule": {
            "bpb": "Lower bits_per_byte is better for cross-tokenizer Vietnamese raw-text likelihood.",
            "guard": "Higher macro_f1, then higher unsafe_recall, is better for SEA-VI safety classification.",
            "selection": "Do not choose solely from perplexity or BPB. Use guard metrics as the primary decision and BPB as a Vietnamese-language diagnostic.",
        },
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
