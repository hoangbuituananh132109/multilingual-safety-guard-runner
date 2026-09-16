from __future__ import annotations

import unittest

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
