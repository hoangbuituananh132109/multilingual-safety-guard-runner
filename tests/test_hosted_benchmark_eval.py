from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.hosted_benchmark_eval import (
    build_request,
    extract_bundle,
    metric_rows,
    normalize_endpoint,
    normalize_row,
    parse_host_response,
    run,
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
        benchmark_row = next(row for row in result if row["group"] == "demo|ALL|ALL")
        self.assertEqual(benchmark_row["examples"], 4)
        self.assertEqual(benchmark_row["balanced_accuracy"], 50.0)

    def test_parse_errors_are_incorrect_for_both_ground_truth_classes(self) -> None:
        rows = [
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "safe", "prediction": None},
            {"benchmark": "demo", "language": "vi", "view": "P", "expected_output": "unsafe", "prediction": None},
        ]
        result = metric_rows(rows)[0]
        self.assertEqual(result["safe_recall"], 0.0)
        self.assertEqual(result["unsafe_recall"], 0.0)
        self.assertEqual(result["balanced_accuracy"], 0.0)
        self.assertEqual(result["parse_rate"], 0.0)

    def test_normalize_endpoint_accepts_vllm_base_or_full_chat_url(self) -> None:
        self.assertEqual(
            normalize_endpoint("http://model-host:8000/v1"),
            "http://model-host:8000/v1/chat/completions",
        )
        self.assertEqual(
            normalize_endpoint("http://model-host:8000/v1/chat/completions"),
            "http://model-host:8000/v1/chat/completions",
        )

    def test_extract_bundle_rejects_zip_slip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.jsonl", "{}\n")
            with self.assertRaises(ValueError):
                extract_bundle(archive, root / "out")

    def test_build_request_is_deterministic(self) -> None:
        request = build_request("http://host/v1/chat/completions", "demo", "hello", 16, enable_thinking=False)
        self.assertEqual(request["model"], "demo")
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 16)
        self.assertEqual(request["messages"][0]["content"], "hello")
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})

    def test_build_request_can_enable_qwen_thinking_explicitly(self) -> None:
        request = build_request("http://host/v1", "demo", "hello", 64, enable_thinking=True)
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": True})

    def test_run_uses_bounded_parallel_http_and_preserves_row_order(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            active = 0
            maximum_active = 0
            lock = threading.Lock()

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                with self.lock:
                    type(self).active += 1
                    type(self).maximum_active = max(type(self).maximum_active, type(self).active)
                time.sleep(0.03)
                payload = json.dumps({"choices": [{"message": {"content": "safe"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                with self.lock:
                    type(self).active -= 1

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "benchmark.zip"
                rows = [
                    {"id": f"demo:{index}:P:vi", "prompt": f"prompt {index}", "ground_truth": "safe"}
                    for index in range(6)
                ]
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr(
                        "qwen3_safety_benchmark_total.jsonl",
                        "".join(json.dumps(row) + "\n" for row in rows),
                    )
                args = argparse.Namespace(
                    endpoint=f"http://127.0.0.1:{server.server_port}/v1",
                    dry_run=False,
                    zip=str(archive),
                    extract_dir=str(root / "extract"),
                    data_file=None,
                    benchmark=None,
                    limit_per_benchmark=None,
                    limit=None,
                    output_dir=str(root / "output"),
                    model="demo",
                    max_tokens=4,
                    api_key_env="HOSTED_MODEL_API_KEY",
                    timeout=5.0,
                    retries=0,
                    concurrency=3,
                    progress_every=0,
                    thinking_mode="no_think",
                )
                result = run(args)
                predictions = [
                    json.loads(line)
                    for line in (root / "output" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual([row["id"] for row in predictions], [row["id"] for row in rows])
                self.assertGreaterEqual(Handler.maximum_active, 2)
                self.assertEqual(result["manifest"]["concurrency"], 3)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
