from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from core.hosted_benchmark_eval import (
    build_request,
    extract_bundle,
    metric_rows,
    normalize_row,
    parse_host_response,
)


class HostedBenchmarkEvalTests(unittest.TestCase):
    def test_normalize_three_field_row_infers_benchmark_from_id(self) -> None:
        row = normalize_row(
            {
                "id": "cultureguard_standard:nemotron:abc:P:vi",
                "prompt": "classify this",
                "ground_truth": "unsafe",
            }
        )
        self.assertEqual(row["benchmark"], "cultureguard_standard")
        self.assertEqual(row["language"], "vi")
        self.assertEqual(row["view"], "P")
        self.assertEqual(row["expected_output"], "unsafe")

    def test_parse_openai_compatible_and_plain_responses(self) -> None:
        self.assertEqual(parse_host_response({"choices": [{"message": {"content": "unsafe"}}]}), "unsafe")
        self.assertEqual(parse_host_response({"output": "safe"}), "safe")
        self.assertEqual(parse_host_response({"response": "classification: unsafe"}), "classification: unsafe")
        self.assertIsNone(parse_host_response({"choices": []}))

    def test_metric_rows_uses_binary_ground_truth(self) -> None:
        rows = [
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "safe", "prediction": "safe"},
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "safe", "prediction": "unsafe"},
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "unsafe", "prediction": "unsafe"},
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "unsafe", "prediction": "safe"},
        ]
        result = metric_rows(rows)
        self.assertEqual(result[0]["examples"], 4)
        self.assertEqual(result[0]["balanced_accuracy"], 50.0)
        self.assertEqual(result[0]["parse_rate"], 100.0)

    def test_extract_bundle_rejects_zip_slip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.jsonl", "{}\n")
            with self.assertRaises(ValueError):
                extract_bundle(archive, root / "out")

    def test_build_request_is_deterministic(self) -> None:
        request = build_request("http://host/v1/chat/completions", "demo", "hello", 16)
        self.assertEqual(request["model"], "demo")
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 16)
        self.assertEqual(request["messages"][0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
