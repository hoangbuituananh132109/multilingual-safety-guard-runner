#!/usr/bin/env python3
"""Audit a SEA-Safeguard-Train-Cultural-v3 Parquet split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq


LABELS = ("Safe", "Sensitive", "Unsafe")


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def stable_key(row: dict[str, Any]) -> str:
    payload = "\n".join(
        (
            text(row.get("id")),
            text(row.get("language")),
            text(row.get("prompt")),
            text(row.get("response")),
            text(row.get("response_model")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in counter.most_common()}


def risk_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 6),
    }


def risk_matches_dataset_v3(label: str, score: float) -> bool:
    if label == "Safe":
        return score < 0.4
    if label == "Sensitive":
        return 0.4 <= score <= 0.6
    if label == "Unsafe":
        return score > 0.6
    return False


def risk_matches_paper(label: str, score: float) -> bool:
    if label == "Safe":
        return score < 0.33
    if label == "Sensitive":
        return 0.33 <= score <= 0.66
    if label == "Unsafe":
        return score > 0.66
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-per-cell", type=int, default=3)
    args = parser.parse_args()

    source_files: list[Path] = []
    for value in args.parquet:
        if "*" in str(value):
            source_files.extend(sorted(value.parent.glob(value.name)))
        elif value.is_dir():
            source_files.extend(sorted(value.glob("*.parquet")))
        else:
            source_files.append(value)
    if not source_files or not all(path.is_file() for path in source_files):
        raise SystemExit(f"No complete Parquet input set found: {args.parquet}")
    rows = pq.read_table([str(path) for path in source_files]).to_pylist()
    counts: dict[str, Counter[Any]] = {
        "language": Counter(),
        "view": Counter(),
        "quality_assurance": Counter(),
        "topic": Counter(),
        "prompt_label": Counter(),
        "response_label": Counter(),
        "effective_label": Counter(),
        "response_model": Counter(),
        "binary_target": Counter(),
    }
    cross: Counter[tuple[str, str, str]] = Counter()
    ids: Counter[str] = Counter()
    prompts: Counter[str] = Counter()
    task_fingerprints: Counter[str] = Counter()
    id_prompts: dict[str, set[str]] = defaultdict(set)
    id_prompt_labels: dict[str, set[str]] = defaultdict(set)
    risk_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    dataset_threshold_mismatch_count = 0
    dataset_threshold_mismatch_examples: list[dict[str, Any]] = []
    paper_threshold_mismatch_count = 0
    invalid_labels: Counter[str] = Counter()
    review_cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    ordered_rows = sorted(rows, key=stable_key)
    for row in ordered_rows:
        language = text(row.get("language")) or "<missing>"
        response_raw = row.get("response")
        response = text(response_raw)
        view = "P" if response_raw is None else "PR"
        prompt_label = text(row.get("prompt_label")) or "<missing>"
        response_label = text(row.get("response_label")) or "<missing>"
        effective_label = response_label if view == "PR" else prompt_label
        row_id = text(row.get("id")) or "<missing>"
        prompt = text(row.get("prompt"))

        counts["language"][language] += 1
        counts["view"][view] += 1
        counts["quality_assurance"][text(row.get("quality_assurance")) or "<missing>"] += 1
        counts["topic"][text(row.get("topic")) or "<missing>"] += 1
        counts["prompt_label"][prompt_label] += 1
        counts["response_label"][response_label] += 1
        counts["effective_label"][effective_label] += 1
        counts["response_model"][text(row.get("response_model")) or "<missing>"] += 1
        cross[(language, view, effective_label)] += 1
        ids[row_id] += 1
        prompts[prompt] += 1
        id_prompts[row_id].add(prompt)
        id_prompt_labels[row_id].add(prompt_label)

        fingerprint_payload = json.dumps(
            {
                "language": language,
                "prompt": prompt,
                "prompt_label": prompt_label,
                "response": response,
                "response_label": response_label,
                "response_model": text(row.get("response_model")),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        task_fingerprints[hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()] += 1

        if effective_label not in LABELS:
            invalid_labels[effective_label] += 1
        score_field = "response_risk_score" if view == "PR" else "prompt_risk_score"
        raw_score = row.get(score_field)
        if raw_score is not None and effective_label in LABELS:
            score = float(raw_score)
            risk_values[(view, effective_label)].append(score)
            if not risk_matches_dataset_v3(effective_label, score):
                dataset_threshold_mismatch_count += 1
                if len(dataset_threshold_mismatch_examples) < 100:
                    dataset_threshold_mismatch_examples.append(
                        {"id": row_id, "view": view, "label": effective_label, "score": score}
                    )
            if not risk_matches_paper(effective_label, score):
                paper_threshold_mismatch_count += 1

        if view == "P":
            binary_target = "safe" if prompt_label in {"Safe", "Sensitive"} else "unsafe"
        else:
            binary_target = "safe" if response_label == "Safe" else "unsafe"
        counts["binary_target"][binary_target] += 1

        cell = (language, view, effective_label)
        if len(review_cells[cell]) < args.review_per_cell:
            review_cells[cell].append(
                {
                    "id": row_id,
                    "language": language,
                    "topic": text(row.get("topic")),
                    "quality_assurance": text(row.get("quality_assurance")),
                    "view": view,
                    "prompt": prompt,
                    "prompt_label": prompt_label,
                    "prompt_risk_score": row.get("prompt_risk_score"),
                    "response": response or None,
                    "response_label": None if response_label == "<missing>" else response_label,
                    "response_risk_score": row.get("response_risk_score"),
                    "response_model": text(row.get("response_model")) or None,
                }
            )

    multiplicity = Counter(ids.values())
    summary = {
        "source_files": [str(path.resolve()) for path in source_files],
        "source_bytes": sum(path.stat().st_size for path in source_files),
        "rows": len(rows),
        "distinct_ids": len(ids),
        "distinct_prompts": len(prompts),
        "duplicate_task_rows": sum(value - 1 for value in task_fingerprints.values()),
        "ids_with_multiple_distinct_prompts": sum(len(value) > 1 for value in id_prompts.values()),
        "ids_with_inconsistent_prompt_labels": sum(
            len(value) > 1 for value in id_prompt_labels.values()
        ),
        "id_row_multiplicity": counter_dict(multiplicity),
        "distributions": {name: counter_dict(value) for name, value in counts.items()},
        "language_view_effective_label": {
            "|".join(key): value for key, value in sorted(cross.items())
        },
        "risk_scores": {
            f"{view}|{label}": risk_summary(values)
            for (view, label), values in sorted(risk_values.items())
        },
        "dataset_v3_thresholds": "Safe <0.4; Sensitive 0.4-0.6; Unsafe >0.6",
        "dataset_v3_threshold_mismatch_count": dataset_threshold_mismatch_count,
        "dataset_v3_threshold_mismatch_examples": dataset_threshold_mismatch_examples,
        "paper_thresholds": "Safe <0.33; Sensitive 0.33-0.66; Unsafe >0.66",
        "paper_threshold_mismatch_count": paper_threshold_mismatch_count,
        "invalid_effective_labels": counter_dict(invalid_labels),
        "notes": {
            "view": "P means response is empty; PR means response is present.",
            "effective_label": "prompt_label for P; response_label for PR.",
            "binary_target": "SEA-Guard evaluation mapping: Sensitive prompt->safe, Sensitive response->unsafe.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path = args.output_dir / "manual_review_sample.jsonl"
    with review_path.open("w", encoding="utf-8") as handle:
        for cell in sorted(review_cells):
            for row in review_cells[cell]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Rows: {len(rows):,}; distinct ids: {len(ids):,}; distinct prompts: {len(prompts):,}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Review sample: {review_path.resolve()}")


if __name__ == "__main__":
    main()
