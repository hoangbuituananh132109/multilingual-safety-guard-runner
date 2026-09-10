from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core.prepare_source_study_benchmarks import linguasafe_vi_rows
from core.linguasafe_metrics import linguasafe_severity_stats
from core.prepare_visafe_benchmark import visafe_rows


class LinguaSafeAdapterTests(unittest.TestCase):
    def test_filters_vietnamese_and_preserves_severity(self) -> None:
        table = pa.Table.from_pylist(
            [
                {
                    "id": 1.0,
                    "lang": "vi",
                    "level": 0,
                    "prompt": "Câu hỏi lành tính",
                    "source": None,
                    "specific": False,
                    "subtype": "Benign",
                    "type": "Normal",
                },
                {
                    "id": 2.0,
                    "lang": "vi",
                    "level": 3,
                    "prompt": "Câu hỏi nguy hiểm",
                    "source": {"content": "x", "dataset": "source"},
                    "specific": True,
                    "subtype": "Severe",
                    "type": "Harm",
                },
                {
                    "id": 3.0,
                    "lang": "en",
                    "level": 3,
                    "prompt": "English row",
                    "source": None,
                    "specific": False,
                    "subtype": "Severe",
                    "type": "Harm",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linguasafe.parquet"
            pq.write_table(table, path)
            rows = list(linguasafe_vi_rows(path))

        self.assertEqual([row["safety_label"] for row in rows], ["safe", "unsafe"])
        self.assertEqual([row["metadata"]["severity_level"] for row in rows], [0, 3])
        self.assertEqual(rows[1]["metadata"]["language_specific"], True)
        self.assertEqual({row["language"] for row in rows}, {"vi"})

    def test_severity_weighted_metrics_follow_alpha_point_six(self) -> None:
        rows = [
            {"severity_level": 0, "prediction_for_metrics": "safe"},
            {"severity_level": 1, "prediction_for_metrics": "unsafe"},
            {"severity_level": 2, "prediction_for_metrics": "safe"},
            {"severity_level": 3, "prediction_for_metrics": "unsafe"},
        ]
        metrics = linguasafe_severity_stats(rows)
        self.assertAlmostEqual(metrics["linguasafe_severity_weighted_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["linguasafe_severity_weighted_fpr"], 1.0 / 3.0)

    def test_severity_metrics_are_absent_for_other_benchmarks(self) -> None:
        metrics = linguasafe_severity_stats(
            [{"severity_level": None, "prediction_for_metrics": "safe"}]
        )
        self.assertIsNone(metrics["linguasafe_severity_weighted_f1"])
        self.assertIsNone(metrics["linguasafe_severity_weighted_fpr"])


class VISafeAdapterTests(unittest.TestCase):
    def test_maps_expected_behavior_and_marks_cross_language_overlap(self) -> None:
        records = [
            {
                "id": "unsafe-1",
                "prompt_vi": "Yêu cầu nguy hiểm",
                "prompt_en": "Dangerous request",
                "expected_behavior": "refuse",
                "category": "cybercrime",
                "subcategory": "phishing",
            },
            {
                "id": "safe-1",
                "prompt_vi": "Câu hỏi chính trị trung lập",
                "prompt_en": "Neutral political question",
                "expected_behavior": "neutral_response",
                "category": "political",
                "subcategory": "neutrality",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "probes.jsonl"
            source.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
                encoding="utf-8",
            )
            rows = list(
                visafe_rows(
                    source,
                    {
                        "nemotron_v3_9lang": {"dangerous request"},
                        "wildguardtrain_en": set(),
                        "sea_cultural_vi": set(),
                    },
                )
            )

        self.assertEqual([row["safety_label"] for row in rows], ["unsafe", "safe"])
        self.assertEqual([row["subset"] for row in rows], ["source_overlap", "clean"])
        self.assertEqual(
            rows[0]["metadata"]["source_overlap_arms"], ["nemotron_v3_9lang"]
        )


if __name__ == "__main__":
    unittest.main()
