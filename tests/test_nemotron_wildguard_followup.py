from __future__ import annotations

import unittest
import json
from pathlib import Path
import subprocess
import sys

import scripts.build_scaled_nemotron_wildguard as scaled_builder

from core.source_study_data import StudyRow
from scripts.build_scaled_nemotron_wildguard import (
    LANGUAGES,
    fit_mixed_budgets,
    language_balance,
    select_balanced_nemotron,
)
from scripts.select_nemotron_wildguard_ratio import BENCHMARK_GROUPS, score_candidate


def row(group: str, language: str) -> StudyRow:
    return StudyRow(
        source="nemotron_v3_9lang",
        source_id=f"{group}:{language}",
        pool="train",
        group_id=group,
        language=language,
        prompt=f"{group}:{language}",
        response=None,
        prompt_label="safe",
        response_label=None,
        categories=[],
        metadata={},
    )


class FollowupTests(unittest.TestCase):
    def test_group_completeness_audit_detects_partial_upstream_id(self) -> None:
        self.assertTrue(hasattr(scaled_builder, "group_completeness_audit"))
        pool = [row("shared", "en"), row("shared", "ar"), row("other", "en")]
        audit = scaled_builder.group_completeness_audit(pool, [pool[0], pool[2]])
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["incomplete_groups"], 1)
        self.assertEqual(audit["missing_rows"], 1)

    def test_scaled_validation_accepts_tiny_shortfall_instead_of_splitting_ids(self) -> None:
        self.assertTrue(hasattr(scaled_builder, "select_scaled_nemotron_validation"))
        rows = [row(f"full-{index}", language) for index in range(40) for language in LANGUAGES]
        selected, selected_ids = scaled_builder.select_scaled_nemotron_validation(
            rows, 300, seed=3407
        )
        self.assertEqual(len(selected), 297)
        for group_id in selected_ids:
            self.assertEqual(
                sum(value.group_id == group_id for value in selected),
                sum(value.group_id == group_id for value in rows),
            )

    def test_scaled_builder_cli_runs_from_repository_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/build_scaled_nemotron_wildguard.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scaled_recipes_request_every_valid_wildguard_row(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "source_study_scaled_selection_30_70.json": (30, 70),
            "source_study_scaled_selection_70_30.json": (70, 30),
        }
        for filename, (nemotron_percent, wildguard_percent) in expected.items():
            value = json.loads((root / filename).read_text(encoding="utf-8"))
            requested = value["requested_scaled_train_rows"]
            self.assertEqual(requested["wildguard"], 86_745)
            self.assertEqual(
                requested["nemotron"],
                round(86_745 * nemotron_percent / wildguard_percent),
            )

    def test_balanced_selector_preserves_complete_groups(self) -> None:
        rows = [row(f"full-{index}", language) for index in range(20) for language in LANGUAGES]
        rows.extend(row(f"single-{language}-{index}", language) for language in LANGUAGES for index in range(3))
        selected, selected_ids = select_balanced_nemotron(rows, 90, seed=3407, split="train")
        self.assertEqual(len(selected), 90)
        self.assertEqual(set(language_balance(selected)["counts"].values()), {10})
        for group_id in selected_ids:
            self.assertEqual(
                sum(value.group_id == group_id for value in selected),
                sum(value.group_id == group_id for value in rows),
            )

    def test_research_score_weights_groups_equally(self) -> None:
        metrics = {}
        group_values = {
            "multilingual_binary": 0.8,
            "vietnamese_binary": 0.6,
            "english_binary": 0.9,
            "unsafe_only": 0.5,
        }
        for group, benchmarks in BENCHMARK_GROUPS.items():
            for benchmark in benchmarks:
                metrics[benchmark] = {
                    "balanced_accuracy": group_values[group],
                    "unsafe_recall": group_values[group],
                    "parse_rate": 1.0,
                }
        report = score_candidate(metrics)
        self.assertAlmostEqual(report["research_score"], 0.7)
        self.assertAlmostEqual(report["group_scores"]["english_binary"], 0.9)

    def test_mixed_budget_clamps_after_cross_source_cleaning(self) -> None:
        result = fit_mixed_budgets(
            requested_nemotron=34_286,
            requested_wildguard=80_000,
            nemotron_percent=30,
            wildguard_percent=70,
            nemo_capacity=420_000,
            wild_capacity=79_368,
            balanced_nemo_capacity=420_000,
        )
        self.assertEqual(result, {"nemotron": 34_015, "wildguard": 79_368})


if __name__ == "__main__":
    unittest.main()
