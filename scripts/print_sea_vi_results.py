#!/usr/bin/env python3
"""Print every discovered SEA-VI evaluation result as copyable TSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


FIELDS = (
    "examples",
    "parsed",
    "parse_errors",
    "parse_rate",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "unsafe_f1",
    "unsafe_precision",
    "unsafe_recall",
    "tp",
    "tn",
    "fp",
    "fn",
)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def first_prediction(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def infer_modes(directory: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    taxonomy = manifest.get("taxonomy_mode")
    thinking = manifest.get("thinking_mode")
    prediction = first_prediction(directory / "predictions.jsonl")
    taxonomy = taxonomy or prediction.get("taxonomy_mode")
    thinking = thinking or prediction.get("thinking_mode")
    # Older Nemotron evaluator versions had no switches: taxonomy was always
    # enabled and Qwen thinking was always disabled.
    if str(manifest.get("family", "")).casefold() == "nemotron":
        taxonomy = taxonomy or "on"
        thinking = thinking or "no_think"
    lowered = str(directory).casefold()
    if not taxonomy:
        if "tax_off" in lowered or "taxonomy_off" in lowered:
            taxonomy = "off"
        elif "tax_on" in lowered or "taxonomy_on" in lowered:
            taxonomy = "on"
    if not thinking:
        if "nothink" in lowered or "no_think" in lowered:
            thinking = "no_think"
        elif "think" in lowered:
            thinking = "think"
    return str(taxonomy or "unknown"), str(thinking or "unknown")


def overall_sea_row(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, list):
        return None
    candidates = [row for row in payload if isinstance(row, dict) and str(row.get("benchmark", "")).casefold() == "sea_vi"]
    if not candidates:
        return None
    for row in candidates:
        if all(str(row.get(field, "ALL")).upper() == "ALL" for field in ("language", "view", "subset")):
            return row
    return candidates[0]


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def relative(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("runs"), Path("runs-stage2")])
    args = parser.parse_args()
    cwd = Path.cwd()

    metric_files: set[Path] = set()
    progress_files: set[Path] = set()
    for root in args.roots:
        if root.exists():
            metric_files.update(path.resolve() for path in root.rglob("metrics.json"))
            progress_files.update(path.resolve() for path in root.rglob("progress.json"))

    rows: list[list[str]] = []
    completed_directories: set[Path] = set()
    for metrics_path in sorted(metric_files, key=str):
        metric = overall_sea_row(load_json(metrics_path))
        if metric is None:
            continue
        directory = metrics_path.parent
        completed_directories.add(directory)
        manifest_value = load_json(directory / "run_manifest.json")
        manifest = manifest_value if isinstance(manifest_value, dict) else {}
        progress_value = load_json(directory / "progress.json")
        progress = progress_value if isinstance(progress_value, dict) else {}
        taxonomy, thinking = infer_modes(directory, manifest)
        rows.append([
            relative(directory, cwd),
            str(progress.get("status") or "metrics_present"),
            taxonomy,
            thinking,
            str(manifest.get("base_model") or ""),
            *[format_value(metric.get(field)) for field in FIELDS],
        ])

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(("result_dir", "status", "taxonomy", "thinking", "base_model", *FIELDS))
    writer.writerows(rows)
    print(f"\nSEA_VI_COMPLETED_RESULTS={len(rows)}")

    incomplete: list[tuple[str, dict[str, Any]]] = []
    for progress_path in sorted(progress_files, key=str):
        directory = progress_path.parent
        if directory in completed_directories:
            continue
        progress_value = load_json(progress_path)
        if not isinstance(progress_value, dict):
            continue
        if "eval" not in str(directory).casefold() and progress_value.get("current_benchmark") != "sea_vi":
            continue
        incomplete.append((relative(directory, cwd), progress_value))

    if incomplete:
        print("\nINCOMPLETE_OR_FAILED_EVALUATIONS")
        writer.writerow(("result_dir", "status", "phase", "completed", "total", "percent", "current_benchmark"))
        for directory, progress in incomplete:
            writer.writerow((
                directory,
                progress.get("status", "unknown"),
                progress.get("phase", ""),
                progress.get("completed", ""),
                progress.get("total", ""),
                progress.get("percent", ""),
                progress.get("current_benchmark", ""),
            ))


if __name__ == "__main__":
    main()
