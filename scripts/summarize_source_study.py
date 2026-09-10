from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact Qwen3-4B data-source comparison table.")
    parser.add_argument("--eval-root", type=Path, default=Path("runs-source-study/qwen3_4b/evaluations"))
    parser.add_argument("--output", type=Path, default=Path("runs-source-study/qwen3_4b/source_study_results.md"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in sorted(args.eval_root.glob("*/metrics.json")):
        progress = path.parent / "progress.json"
        if not progress.is_file() or json.loads(progress.read_text(encoding="utf-8")).get("status") != "complete":
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for metric in metrics:
            if metric.get("language") != "ALL" or metric.get("view") != "ALL" or metric.get("subset") != "ALL":
                continue
            rows.append(
                {
                    "model": path.parent.name,
                    "benchmark": str(metric["benchmark"]),
                    "n": str(metric["examples"]),
                    "balanced_acc": percent(metric.get("balanced_accuracy")),
                    "macro_f1": percent(metric.get("macro_f1")),
                    "unsafe_recall": percent(metric.get("unsafe_recall")),
                    "parse_rate": percent(metric.get("parse_rate")),
                }
            )
    if not rows:
        raise SystemExit(f"No completed evaluations under {args.eval_root}")
    columns = list(rows[0])
    lines = [
        "# Qwen3-4B matched 80k data-source study",
        "",
        "All metrics are percentages; every run uses taxonomy-off and no-think.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row[key] for key in columns) + " |" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
