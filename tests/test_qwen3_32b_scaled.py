from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class Qwen32BScaledTests(unittest.TestCase):
    def test_show_config_loads_settings_and_keeps_global_batch_constant(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            settings = Path(temp) / "qwen32.env"
            settings.write_bytes((
                "\n".join(
                    (
                        'export SOURCE_STUDY_32B_TRAIN_GPUS="2,4,6,7"',
                        'export SOURCE_STUDY_32B_EVAL_GPU="5"',
                        'export SOURCE_STUDY_32B_MERGE_GPU="6"',
                        'export SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH="32"',
                        'export SOURCE_STUDY_32B_MICROBATCH="2"',
                    )
                )
                + "\n"
            ).encode("utf-8"))
            relative_settings = settings.relative_to(ROOT).as_posix()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"SOURCE_STUDY_32B_SETTINGS_FILE={shlex.quote(relative_settings)} "
                    "bash scripts/run_qwen3_32b_scaled.sh show-config",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["train_gpus"], "2,4,6,7")
            self.assertEqual(payload["train_gpu_count"], 4)
            self.assertEqual(payload["eval_gpu"], "5")
            self.assertEqual(payload["merge_gpu"], "6")
            self.assertEqual(payload["gradient_accumulation_steps"], 4)
            self.assertEqual(payload["effective_global_batch"], 32)

    def test_show_config_rejects_gpu_count_that_changes_batch_contract(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-c",
                "SOURCE_STUDY_32B_SETTINGS_FILE='' "
                "SOURCE_STUDY_32B_TRAIN_GPUS='0,1,2' "
                "SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH='32' "
                "SOURCE_STUDY_32B_MICROBATCH='2' "
                "bash scripts/run_qwen3_32b_scaled.sh show-config",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot preserve target global batch", result.stderr)

    def test_config_changes_model_scale_but_preserves_common_contract(self) -> None:
        path = ROOT / "source_study_scaled_train_qwen3_32b.yaml"
        self.assertTrue(path.is_file(), path)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["id"], "$SOURCE_STUDY_32B_MODEL_PATH")
        self.assertEqual(config["model"]["family"], "nemotron")
        self.assertEqual(config["model"]["tuning"], "lora")
        self.assertEqual(config["model"]["lora_r"], 8)
        self.assertEqual(config["model"]["lora_alpha"], 32)
        self.assertEqual(config["model"]["target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(config["data"]["max_length"], 2048)
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertEqual(config["training"]["learning_rate"], 1.0e-5)
        effective_batch = (
            config["training"]["per_device_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
            * 8
        )
        self.assertEqual(effective_batch, 32)
        self.assertEqual(config["training"]["seed"], 3407)

    def test_runner_is_offline_and_requires_explicit_train_opt_in(self) -> None:
        path = ROOT / "scripts" / "run_qwen3_32b_scaled.sh"
        self.assertTrue(path.is_file(), path)
        script = path.read_text(encoding="utf-8")
        for setting in (
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
        ):
            self.assertIn(setting, script)
        for forbidden in ("wget ", "curl ", "git pull", "hf download"):
            self.assertNotIn(forbidden, script)
        self.assertIn("SOURCE_STUDY_ALLOW_32B_TRAIN", script)
        self.assertIn("eval-base", script)
        self.assertIn("smoke-train", script)
        self.assertIn("eval-trained", script)


if __name__ == "__main__":
    unittest.main()
