from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "corrected_vi_1e": "Qwen3-8B Stage-1 5E + corrected Gemini-VI 1E",
    "corrected_reasoning_1e": "Qwen3-8B Stage-1 5E + corrected reasoning 1E",
    "corrected_full_1e": "Qwen3-8B Stage-1 5E + corrected full 1E",
}


def percentage(value: Any) -> float | None:
    return round(100.0 * float(value), 4) if value is not None else None


def nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def find_metric(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("benchmark") == benchmark
            and row.get("language") == "ALL"
            and row.get("view") == "ALL"
            and row.get("subset") == "ALL"
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one paper-aligned table from corrected Stage-2 evaluations.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("runs-stage2/qwen3_8b/corrected_study/evaluations"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("runs-stage2/qwen3_8b/corrected_study/results"),
    )
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(args.eval_root.glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        progress_path = run_dir / "progress.json"
        report_path = run_dir / "nvidia_report.json"
        manifest_path = run_dir / "run_manifest.json"
        if not progress_path.is_file() or not report_path.is_file() or not manifest_path.is_file():
            continue
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("status") != "complete":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sea = find_metric(metrics, "sea_vi") or {}
        aggregate_rows = [row for row in metrics if row.get("language") == "ALL" and row.get("view") == "ALL"]
        model_tag = run_dir.parent.name
        output_rows.append(
            {
                "model": MODEL_LABELS.get(model_tag, model_tag),
                "run": model_tag,
                "taxonomy": manifest.get("taxonomy_mode"),
                "thinking": manifest.get("thinking_mode"),
                "CG-P": percentage(nested(report, "cultureguard_standard", "Prompt", "average_harmful_f1")),
                "CG-PR": percentage(nested(report, "cultureguard_standard", "Response", "average_harmful_f1")),
                "JB-P": percentage(nested(report, "cultureguard_jb", "Prompt", "average_harmful_f1")),
                "JB-PR": percentage(nested(report, "cultureguard_jb", "Response", "average_harmful_f1")),
                "Poly-P": percentage(nested(report, "polyguard_prompts", "Prompt", "average_harmful_f1")),
                "Poly-PR": percentage(nested(report, "polyguard_prompts", "Response", "average_harmful_f1")),
                "XS-Recall": percentage(nested(report, "xsafety", "harmful_recall", "average")),
                "MJ-Recall": percentage(nested(report, "multijail", "Prompt", "average_harmful_recall")),
                "SEA-Acc": percentage(sea.get("accuracy")),
                "SEA-Macro-F1": percentage(sea.get("macro_f1")),
                "SEA-Unsafe-Recall": percentage(sea.get("unsafe_recall")),
                "Parse-min": percentage(min((row.get("parse_rate") for row in aggregate_rows if row.get("parse_rate") is not None), default=0.0)),
            }
        )

    mode_order = {("on", "no_think"): 0, ("off", "no_think"): 1, ("on", "think"): 2, ("off", "think"): 3}
    model_order = {name: index for index, name in enumerate(MODEL_LABELS)}
    output_rows.sort(key=lambda row: (mode_order.get((row["taxonomy"], row["thinking"]), 9), model_order.get(row["run"], 9)))
    if not output_rows:
        raise SystemExit(f"No completed evaluations found below {args.eval_root}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    columns = list(output_rows[0])
    lines = ["# Corrected Stage-2 full-benchmark results", "", "All numbers are percentages.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in output_rows:
        values = []
        for column in columns:
            value = row[column]
            values.append("-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(output_rows), "json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
