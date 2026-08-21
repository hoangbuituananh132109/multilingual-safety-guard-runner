"""Generate a paper-aligned summary from a run's guard metrics.json.

The paper (arXiv 2508.01710) reports average harmful-F1 across languages, with
Prompt (P) and Response (PR) reported separately. This script reads a run's
metrics.json and produces:

  - CultureGuard standard: mean harmful-F1 over 9 langs, P and PR separately
  - CultureGuard-JB:       mean harmful-F1 over 9 langs, P and PR separately
  - XSafety:               project binary harmful-F1 and harmful-recall over 7 langs

The public XSafety data contains only harmful prompts. Its standard binary F1 is
therefore not numerically equivalent to NVIDIA's published XSafety F1 protocol;
harmful recall is the directly comparable diagnostic after using all 2,800 rows
per language.

Usage:
  python nvidia_report.py --metrics runs/<model>/<checkpoint>/guard/metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CG_STANDARD_LANGS = ["en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh"]
XS_7_LANGS = ["en", "ar", "de", "fr", "hi", "ja", "zh"]
MULTIJAIL_4_LANGS = ["en", "ar", "th", "zh"]


def load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def by_view_lang(rows: list[dict[str, Any]], benchmark: str, view: str, langs: list[str], metric: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for lang in langs:
        match = next(
            (
                row
                for row in rows
                if row.get("benchmark") == benchmark
                and row.get("language") == lang
                and row.get("view") == view
                and row.get("subset") == "ALL"
            ),
            None,
        )
        result[lang] = match.get(metric) if match else None
    return result


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Path to guard/metrics.json")
    parser.add_argument("--out", type=Path, help="JSON output path (optional)")
    args = parser.parse_args()

    rows = load(args.metrics)
    if not rows:
        raise SystemExit(f"No metrics found at {args.metrics}")

    report: dict[str, Any] = {
        "metrics_file": str(args.metrics),
        "paper_reference_fraction": {
            "cultureguard_standard": {"Prompt": 0.8515, "Response": 0.8548},
            "cultureguard_jb": {"Prompt": 0.9177, "Response": 0.9435},
            "polyguard_prompts": {"Prompt": 0.7925, "Response": 0.7289},
            "multijail": {"harmful_f1": 0.9536, "harmful_recall": 0.9228},
            "xsafety": {"harmful_f1": 0.6697, "harmful_recall": 0.5327},
        },
        "protocol": {
            "cultureguard": "Average per-language harmful-F1, matching the paper aggregation.",
            "xsafety_recall": "Comparable only when the prepared data has 2,800 prompts per language.",
            "xsafety_f1": "Project binary F1 on an unsafe-only set; do not compare directly with NVIDIA Table 15.",
            "multijail": "Public data is harmful-only; recall is the safest direct comparison with the paper.",
        },
    }

    for benchmark in ("cultureguard_standard", "cultureguard_jb"):
        block: dict[str, Any] = {}
        for view, label in (("P", "Prompt"), ("PR", "Response")):
            per_lang = by_view_lang(rows, benchmark, view, CG_STANDARD_LANGS, "harmful_f1")
            block[label] = {"per_language": per_lang, "average_harmful_f1": mean(list(per_lang.values()))}
        report[benchmark] = block

    xs_f1 = by_view_lang(rows, "xsafety", "P", XS_7_LANGS, "harmful_f1")
    xs_recall = by_view_lang(rows, "xsafety", "P", XS_7_LANGS, "unsafe_recall")
    report["xsafety"] = {
        "project_binary_harmful_f1": {"per_language": xs_f1, "average": mean(list(xs_f1.values()))},
        "harmful_recall": {"per_language": xs_recall, "average": mean(list(xs_recall.values()))},
    }

    for benchmark, languages, views in (
        ("polyguard_prompts", CG_STANDARD_LANGS, (("P", "Prompt"), ("PR", "Response"))),
        ("multijail", MULTIJAIL_4_LANGS, (("P", "Prompt"),)),
    ):
        if any(row.get("benchmark") == benchmark for row in rows):
            block: dict[str, Any] = {}
            for view, label in views:
                per_lang = by_view_lang(rows, benchmark, view, languages, "harmful_f1")
                block[label] = {"per_language": per_lang, "average_harmful_f1": mean(list(per_lang.values()))}
                if benchmark == "multijail":
                    per_lang_recall = by_view_lang(rows, benchmark, view, languages, "unsafe_recall")
                    block[label]["per_language_harmful_recall"] = per_lang_recall
                    block[label]["average_harmful_recall"] = mean(list(per_lang_recall.values()))
            report[benchmark] = block

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
