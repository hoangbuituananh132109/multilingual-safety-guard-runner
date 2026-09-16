from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HostedEvalRunnerTests(unittest.TestCase):
    def test_show_config_loads_one_hosted_model_settings_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            settings = Path(temp) / "hosted.env"
            settings.write_bytes((
                "\n".join(
                    (
                        'export HOSTED_MODEL_ENDPOINT="http://model-host:8000/v1/chat/completions"',
                        'export HOSTED_MODEL_NAME="Qwen3-235B-A22B"',
                        'export HOSTED_EVAL_CONCURRENCY="12"',
                        'export HOSTED_MODEL_THINKING_MODE="no_think"',
                    )
                )
                + "\n"
            ).encode("utf-8"))
            relative_settings = settings.relative_to(ROOT).as_posix()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"HOSTED_EVAL_SETTINGS_FILE={shlex.quote(relative_settings)} "
                    "bash scripts/run_hosted_benchmark_eval.sh show-config",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["endpoint"], "http://model-host:8000/v1/chat/completions")
            self.assertEqual(payload["model"], "Qwen3-235B-A22B")
            self.assertEqual(payload["concurrency"], 12)
            self.assertEqual(payload["thinking_mode"], "no_think")
            self.assertFalse(payload["api_key_set"])

    def test_dry_run_extracts_local_zip_without_endpoint(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            root = Path(temp)
            archive = root / "benchmark.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr(
                    "qwen3_safety_benchmark_total.jsonl",
                    json.dumps(
                        {
                            "id": "demo:0:P:vi",
                            "prompt": "Return safe or unsafe.",
                            "ground_truth": "safe",
                        }
                    )
                    + "\n",
                )
            output_root = root / "runs"
            extract_dir = root / "extract"
            settings = root / "hosted.env"
            archive_rel = archive.relative_to(ROOT).as_posix()
            extract_rel = extract_dir.relative_to(ROOT).as_posix()
            output_rel = output_root.relative_to(ROOT).as_posix()
            settings.write_bytes((
                "\n".join(
                    (
                        f'export HOSTED_BENCHMARK_ZIP="{archive_rel}"',
                        f'export HOSTED_BENCHMARK_EXTRACT_DIR="{extract_rel}"',
                        f'export HOSTED_EVAL_OUTPUT_ROOT="{output_rel}"',
                        'export HOSTED_MODEL_ENDPOINT=""',
                    )
                )
                + "\n"
            ).encode("utf-8"))
            relative_settings = settings.relative_to(ROOT).as_posix()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"HOSTED_EVAL_SETTINGS_FILE={shlex.quote(relative_settings)} "
                    "bash scripts/run_hosted_benchmark_eval.sh dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output_root / "dry-run" / "dry_run.json").read_text(encoding="utf-8"))
            self.assertFalse(report["network_called"])
            self.assertEqual(report["selected_examples"], 1)


if __name__ == "__main__":
    unittest.main()
