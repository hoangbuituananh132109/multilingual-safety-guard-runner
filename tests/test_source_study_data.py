from __future__ import annotations

import unittest

from core.source_study_data import (
    LANGUAGES,
    SMOKE_TRAIN_TARGETS,
    SMOKE_VALIDATION_TARGETS,
    StudyRow,
    select_rows,
)


class SourceStudySelectionTests(unittest.TestCase):
    def test_multilingual_smoke_skips_zero_language_quotas(self) -> None:
        rows: list[StudyRow] = []
        for pool in ("train", "validation"):
            for language in LANGUAGES:
                for view in ("P", "PR"):
                    for label in ("safe", "unsafe"):
                        for index in range(4):
                            uid = f"{pool}-{language}-{view}-{label}-{index}"
                            rows.append(
                                StudyRow(
                                    source="nemotron_v3_9lang",
                                    source_id=uid,
                                    pool=pool,
                                    group_id=uid,
                                    language=language,
                                    prompt=f"prompt {uid}",
                                    response=f"response {uid}" if view == "PR" else None,
                                    prompt_label=label,
                                    response_label=label if view == "PR" else None,
                                    categories=[],
                                    metadata={},
                                )
                            )

        train, validation, _ = select_rows(
            rows,
            "nemotron_v3_9lang",
            3407,
            multilingual=True,
            train_cell_targets=SMOKE_TRAIN_TARGETS,
            validation_cell_targets=SMOKE_VALIDATION_TARGETS,
        )

        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertFalse({row.group_id for row in train} & {row.group_id for row in validation})


if __name__ == "__main__":
    unittest.main()
