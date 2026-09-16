from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class Qwen32BScaledTests(unittest.TestCase):
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
