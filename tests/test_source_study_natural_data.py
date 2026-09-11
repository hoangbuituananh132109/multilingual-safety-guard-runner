from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.prompt import parse_sea_guard_label, sea_guard_instruction
from core.source_study_data import StudyRow
from core.source_study_natural_data import (
    _select_groups_exact,
    _wildguard_natural_rows,
    _switch_half_ids_to_english,
)


def row(group: str, language: str, index: int) -> StudyRow:
    response = None if index == 0 else f"response {group} {language} {index}"
    return StudyRow(
        source="test",
        source_id=f"{group}-{language}-{index}",
        pool="unsplit",
        group_id=group,
        language=language,
        prompt=f"prompt {group} {language}",
        response=response,
        prompt_label="sensitive",
        response_label="safe" if response else None,
        categories=[],
        metadata={},
    )


class NaturalSourceStudyTests(unittest.TestCase):
    def test_wildguard_pr_row_is_not_expanded_into_extra_prompt_task(self) -> None:
        records = [
            {
                "prompt": "same prompt",
                "response": None,
                "prompt_harm_label": "unharmful",
            },
            {
                "prompt": "another prompt",
                "response": "answer",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "compliance",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wildguard.jsonl"
            path.write_text("".join(json.dumps(value) + "\n" for value in records), encoding="utf-8")
            rows = list(_wildguard_natural_rows(path))
        self.assertEqual([value.view for value in rows], ["P", "PR"])
        self.assertEqual([value.label for value in rows], ["safe", "safe"])

    def test_exact_budget_never_splits_group(self) -> None:
        groups = {
            "g1": [row("g1", "en", index) for index in range(3)],
            "g2": [row("g2", "en", index) for index in range(2)],
            "g3": [row("g3", "en", 0)],
            "g4": [row("g4", "en", 0)],
        }
        selected, selected_ids = _select_groups_exact(
            groups, 6, seed=3407, source="test", split="train"
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual({value.group_id for value in selected}, selected_ids)
        for group_id in selected_ids:
            self.assertEqual(
                sum(value.group_id == group_id for value in selected),
                len(groups[group_id]),
            )

    def test_bilingual_switch_is_id_level_and_balanced(self) -> None:
        vi = [row(f"g{index}", "vi", item) for index in range(6) for item in range(2)]
        all_rows = vi + [row(f"g{index}", "en", item) for index in range(6) for item in range(3)]
        selected, audit = _switch_half_ids_to_english(vi, all_rows, seed=3407, split="train")
        languages = {}
        for value in selected:
            languages.setdefault(value.group_id, set()).add(value.language)
        self.assertTrue(all(len(values) == 1 for values in languages.values()))
        self.assertEqual(audit["english_ids"], 3)
        self.assertEqual(audit["vietnamese_ids"], 3)

    def test_sea_prompt_and_binary_eval_mapping(self) -> None:
        instruction = sea_guard_instruction("question", "answer")
        self.assertIn("assistant response", instruction)
        self.assertIn("sensitive", instruction)
        self.assertEqual(parse_sea_guard_label("sensitive", "P")[:2], ("sensitive", "safe"))
        self.assertEqual(parse_sea_guard_label("Label: Sensitive.", "PR")[:2], ("sensitive", "unsafe"))
        self.assertEqual(parse_sea_guard_label("maybe", "P")[:2], (None, None))


if __name__ == "__main__":
    unittest.main()
