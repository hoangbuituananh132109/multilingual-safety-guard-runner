from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
GUARD_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "unsafe_precision", "unsafe_recall", "unsafe_f1", "parse_rate")


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_trainer_state(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [run_dir / "trainer_state.json"]
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=checkpoint_step,
        reverse=True,
    )
    candidates.extend(path / "trainer_state.json" for path in checkpoints)
    for path in candidates:
        value = load_json(path)
        if isinstance(value, dict):
            return value, path
    return None, None


def overall_rows(path: Path) -> dict[str, dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            continue
        if row.get("language") == "ALL" and row.get("view") == "ALL" and row.get("subset") == "ALL":
            rows[str(row.get("benchmark"))] = row
    return rows


def loss_rows(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    rows = []
    for entry in state.get("log_history") or []:
        if not isinstance(entry, dict) or "loss" not in entry:
            continue
        rows.append({
            "step": int(entry.get("step") or 0),
            "epoch": float(entry["epoch"]) if entry.get("epoch") is not None else None,
            "loss": float(entry["loss"]),
            "grad_norm": float(entry["grad_norm"]) if entry.get("grad_norm") is not None else None,
            "learning_rate": float(entry["learning_rate"]) if entry.get("learning_rate") is not None else None,
        })
    rows.sort(key=lambda row: row["step"])
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loss_svg(path: Path, rows: list[dict[str, Any]], model: str, smooth_window: int) -> None:
    width, height = 1200, 680
    left, right, top, bottom = 90, 35, 65, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    steps = [row["step"] for row in rows]
    losses = [row["loss"] for row in rows]
    smooth = moving_average(losses, smooth_window)
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses + smooth), max(losses + smooth)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    padding = (y_max - y_min) * 0.08
    y_min, y_max = max(0.0, y_min - padding), y_max + padding

    def x(value: float) -> float:
        return left + (value - x_min) / max(1, x_max - x_min) * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / max(1e-12, y_max - y_min) * plot_height

    raw_points = " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in zip(steps, losses))
    smooth_points = " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in zip(steps, smooth))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{html.escape(model)} training loss</text>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y_pos = y(value)
        elements.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left-12}" y="{y_pos+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.4f}</text>')
    for index in range(6):
        value = round(x_min + (x_max - x_min) * index / 5)
        x_pos = x(value)
        elements.append(f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{height-bottom}" stroke="#f3f4f6"/>')
        elements.append(f'<text x="{x_pos:.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="sans-serif" font-size="13">{value}</text>')
    elements.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>',
        f'<polyline points="{raw_points}" fill="none" stroke="#93c5fd" stroke-width="1.4" opacity="0.85"/>',
        f'<polyline points="{smooth_points}" fill="none" stroke="#dc2626" stroke-width="3"/>',
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="15">Optimizer step</text>',
        f'<text x="24" y="{height/2}" text-anchor="middle" transform="rotate(-90 24 {height/2})" font-family="sans-serif" font-size="15">Loss</text>',
        f'<line x1="{width-300}" y1="48" x2="{width-260}" y2="48" stroke="#93c5fd" stroke-width="2"/><text x="{width-250}" y="53" font-family="sans-serif" font-size="13">logged loss</text>',
        f'<line x1="{width-160}" y1="48" x2="{width-120}" y2="48" stroke="#dc2626" stroke-width="3"/><text x="{width-110}" y="53" font-family="sans-serif" font-size="13">moving avg ({smooth_window})</text>',
        "</svg>",
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def metric_rows(before_dir: Path, after_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    before_guard_path = before_dir / "guard" / "metrics.json"
    after_guard_path = after_dir / "guard" / "metrics.json"
    before_guard = overall_rows(before_guard_path)
    after_guard = overall_rows(after_guard_path)
    if not before_guard:
        missing.append(str(before_guard_path))
    if not after_guard:
        missing.append(str(after_guard_path))
    rows: list[dict[str, Any]] = []
    for benchmark in sorted(set(before_guard) | set(after_guard)):
        before = before_guard.get(benchmark, {})
        after = after_guard.get(benchmark, {})
        before_examples = before.get("examples")
        after_examples = after.get("examples")
        for metric in GUARD_METRICS:
            before_value = before.get(metric)
            after_value = after.get(metric)
            comparison_status = comparison_status_for(
                before_value,
                after_value,
                before_examples,
                after_examples,
            )
            rows.append({
                "task": "guard",
                "benchmark": benchmark,
                "metric": metric,
                "before_examples": before_examples,
                "after_examples": after_examples,
                "before": before_value,
                "after": after_value,
                "delta": float(after_value) - float(before_value) if comparison_status == "comparable" else None,
                "better_direction": "higher",
                "comparison_status": comparison_status,
            })
    before_like_path = before_dir / "likelihood" / "likelihood_metrics.json"
    after_like_path = after_dir / "likelihood" / "likelihood_metrics.json"
    before_like = overall_rows(before_like_path).get("sea_vi")
    after_like = overall_rows(after_like_path).get("sea_vi")
    if not before_like:
        missing.append(str(before_like_path))
    if not after_like:
        missing.append(str(after_like_path))
    for metric in ("bits_per_byte", "perplexity"):
        before_value = before_like.get(metric) if before_like else None
        after_value = after_like.get(metric) if after_like else None
        before_examples = before_like.get("examples") if before_like else None
        after_examples = after_like.get("examples") if after_like else None
        comparison_status = comparison_status_for(
            before_value,
            after_value,
            before_examples,
            after_examples,
        )
        rows.append({
            "task": "likelihood",
            "benchmark": "sea_vi",
            "metric": metric,
            "before_examples": before_examples,
            "after_examples": after_examples,
            "before": before_value,
            "after": after_value,
            "delta": float(after_value) - float(before_value) if comparison_status == "comparable" else None,
            "better_direction": "lower",
            "comparison_status": comparison_status,
        })
    return rows, missing


def comparison_status_for(
    before_value: Any,
    after_value: Any,
    before_examples: Any,
    after_examples: Any,
) -> str:
    if before_value is None and after_value is None:
        return "metric_unavailable"
    if before_value is None:
        return "missing_before"
    if after_value is None:
        return "missing_after"
    if before_examples is None or after_examples is None:
        return "sample_size_unknown"
    if int(before_examples) != int(after_examples):
        return "sample_size_mismatch"
    return "comparable"


def metric_svg(path: Path, rows: list[dict[str, Any]], model: str) -> None:
    selected = [
        row for row in rows
        if row["task"] == "guard"
        and row["metric"] in {"macro_f1", "unsafe_recall", "unsafe_f1"}
        and row["comparison_status"] == "comparable"
    ]
    if not selected:
        return
    width = 1280
    row_height = 62
    height = 100 + len(selected) * row_height
    left, right = 260, 70
    plot_width = width - left - right
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{html.escape(model)} guard metrics: before vs after</text>',
    ]
    for tick in range(6):
        value = tick / 5
        x_pos = left + value * plot_width
        elements.append(f'<line x1="{x_pos:.1f}" y1="55" x2="{x_pos:.1f}" y2="{height-35}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{x_pos:.1f}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    for index, row in enumerate(selected):
        y_base = 68 + index * row_height
        label = f"{row['benchmark']} / {row['metric']}"
        before = float(row["before"])
        after = float(row["after"])
        elements.append(f'<text x="{left-12}" y="{y_base+26}" text-anchor="end" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
        elements.append(f'<rect x="{left}" y="{y_base+4}" width="{before*plot_width:.1f}" height="18" rx="3" fill="#94a3b8"/>')
        elements.append(f'<rect x="{left}" y="{y_base+28}" width="{after*plot_width:.1f}" height="18" rx="3" fill="#2563eb"/>')
        elements.append(f'<text x="{left+before*plot_width+7:.1f}" y="{y_base+18}" font-family="sans-serif" font-size="12">{before:.4f}</text>')
        elements.append(f'<text x="{left+after*plot_width+7:.1f}" y="{y_base+42}" font-family="sans-serif" font-size="12">{after:.4f}</text>')
    elements.extend([
        f'<rect x="{width-260}" y="14" width="16" height="12" fill="#94a3b8"/><text x="{width-238}" y="25" font-family="sans-serif" font-size="12">before</text>',
        f'<rect x="{width-170}" y="14" width="16" height="12" fill="#2563eb"/><text x="{width-148}" y="25" font-family="sans-serif" font-size="12">after</text>',
        "</svg>",
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def markdown_report(
    path: Path,
    model: str,
    run_dir: Path,
    state_path: Path | None,
    losses: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    missing: list[str],
) -> None:
    lines = [
        f"# Training analysis: {model}",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Trainer state: `{state_path}`" if state_path else "- Trainer state: missing",
        f"- Logged loss points: {len(losses)}",
    ]
    sample_mismatches = {
        (row["benchmark"], row["before_examples"], row["after_examples"])
        for row in metrics
        if row["comparison_status"] == "sample_size_mismatch"
    }
    if sample_mismatches:
        lines.extend([
            "",
            "## Comparison warning",
            "",
            "The following before/after results use different sample counts. Their deltas are intentionally omitted:",
            "",
        ])
        lines.extend(
            f"- `{benchmark}`: before={before_examples}, after={after_examples}"
            for benchmark, before_examples, after_examples in sorted(sample_mismatches)
        )
        lines.extend([
            "",
            "Re-run both checkpoints with the same full benchmark or the same fixed `--sample` value before interpreting change.",
        ])
    if losses:
        lines.extend([
            f"- First loss: {losses[0]['loss']:.6f} at step {losses[0]['step']}",
            f"- Last loss: {losses[-1]['loss']:.6f} at step {losses[-1]['step']}",
            f"- Minimum logged loss: {min(row['loss'] for row in losses):.6f}",
            "",
            "![Training loss](train_loss.svg)",
        ])
    lines.extend([
        "",
        "## Before vs after",
        "",
        "| Benchmark | Metric | N before | N after | Before | After | Delta | Status | Better |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in metrics:
        before = "missing" if row["before"] is None else f"{float(row['before']):.6f}"
        after = "missing" if row["after"] is None else f"{float(row['after']):.6f}"
        delta = "missing" if row["delta"] is None else f"{float(row['delta']):+.6f}"
        before_examples = row["before_examples"] if row["before_examples"] is not None else "?"
        after_examples = row["after_examples"] if row["after_examples"] is not None else "?"
        lines.append(
            f"| {row['benchmark']} | {row['metric']} | {before_examples} | {after_examples} | "
            f"{before} | {after} | {delta} | {row['comparison_status']} | {row['better_direction']} |"
        )
    if any(row["comparison_status"] == "comparable" for row in metrics):
        lines.extend(["", "![Guard metrics](guard_before_after.svg)"])
    if missing:
        lines.extend(["", "## Missing inputs", ""])
        lines.extend(f"- `{value}`" for value in sorted(set(missing)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create training-loss plots and before/after evaluation reports.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_root = Path(config["paths"]["runs_root"])
    if not runs_root.is_absolute():
        runs_root = config_path.parent / runs_root
    before_dir = runs_root / args.model / "base"
    run_dir = runs_root / args.model / f"{args.method}_{args.run_mode}"
    output_dir = args.output_dir or run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    state, state_path = load_trainer_state(run_dir)
    losses = loss_rows(state)
    metrics, missing = metric_rows(before_dir, run_dir)
    if not losses:
        missing.append(str(run_dir / "trainer_state.json"))
    write_csv(output_dir / "train_loss.csv", losses)
    write_csv(output_dir / "before_after_metrics.csv", metrics)
    (output_dir / "before_after_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if losses:
        loss_svg(output_dir / "train_loss.svg", losses, args.model, max(1, args.smooth_window))
    metric_svg(output_dir / "guard_before_after.svg", metrics, args.model)
    markdown_report(output_dir / "REPORT.md", args.model, run_dir, state_path, losses, metrics, missing)
    result = {
        "model": args.model,
        "run_directory": str(run_dir),
        "output_directory": str(output_dir),
        "loss_points": len(losses),
        "loss_chart": str(output_dir / "train_loss.svg") if losses else None,
        "metrics_chart": str(output_dir / "guard_before_after.svg") if (output_dir / "guard_before_after.svg").is_file() else None,
        "report": str(output_dir / "REPORT.md"),
        "missing_inputs": sorted(set(missing)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
