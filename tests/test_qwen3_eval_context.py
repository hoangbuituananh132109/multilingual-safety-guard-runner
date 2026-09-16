from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.qwen235b_eval import _tokenize_bounded_prompt, run_evaluation


class CharacterTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Prompt length must be measured without extra special tokens")
        return [ord(character) for character in text]


class Qwen3EvalContextTests(unittest.TestCase):
    def test_long_prompt_preserves_instruction_head_and_target_tail(self) -> None:
        prompt, original, truncated = _tokenize_bounded_prompt(
            CharacterTokenizer(), "ABCDEFGHIJ", max_model_len=9, max_new_tokens=2
        )
        self.assertEqual(original, 10)
        self.assertTrue(truncated)
        self.assertEqual("".join(map(chr, prompt["prompt_token_ids"])), "ABCGHIJ")

    def test_short_prompt_remains_unchanged(self) -> None:
        prompt, original, truncated = _tokenize_bounded_prompt(
            CharacterTokenizer(), "ABC", max_model_len=9, max_new_tokens=2
        )
        self.assertEqual(original, 3)
        self.assertFalse(truncated)
        self.assertEqual(prompt["prompt_token_ids"], [65, 66, 67])

    def test_invalid_context_budget_fails_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            _tokenize_bounded_prompt(CharacterTokenizer(), "ABC", max_model_len=2, max_new_tokens=2)

    def test_evaluator_sends_bounded_token_ids_and_reports_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.jsonl"
            bundle.write_text(json.dumps({"id": "demo:one:P:vi", "prompt": "ABCDEFGHIJ", "ground_truth": "safe"}) + "\n", encoding="utf-8")
            captured = []

            class FakeLLM:
                def __init__(self, **_kwargs):
                    pass

                def generate(self, prompts, _sampling):
                    captured.extend(prompts)
                    return [SimpleNamespace(outputs=[SimpleNamespace(text="safe")]) for _ in prompts]

            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: CharacterTokenizer()))
            fake_vllm = SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
            args = argparse.Namespace(
                model="local-model", bundle=bundle, output_dir=root / "out", benchmark=None,
                limit=None, limit_per_benchmark=None, batch_size=2, tensor_parallel_size=1,
                gpu_memory_utilization=0.9, max_model_len=9, max_new_tokens=2,
                thinking_mode="no_think", seed=1, enforce_eager=True, trust_remote_code=False,
            )
            with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
                result = run_evaluation(args)

            self.assertEqual(len(captured), 1)
            self.assertEqual("".join(map(chr, captured[0]["prompt_token_ids"])), "ABCGHIJ")
            record = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["prompt_truncated"])
            self.assertEqual(record["prompt_tokens_original"], 10)
            self.assertEqual(record["prompt_tokens_used"], 7)
            self.assertEqual(result["manifest"]["truncated_examples"], 1)

    def test_existing_predictions_are_not_overwritten_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.jsonl"
            bundle.write_text(json.dumps({"id": "demo:one:P:vi", "prompt": "ABC", "ground_truth": "safe"}) + "\n", encoding="utf-8")
            output_dir = root / "out"
            output_dir.mkdir()
            predictions = output_dir / "predictions.jsonl"
            predictions.write_text("PREVIOUS RESULTS\n", encoding="utf-8")
            args = argparse.Namespace(
                model="local-model", bundle=bundle, output_dir=output_dir, benchmark=None,
                limit=None, limit_per_benchmark=None, batch_size=2, tensor_parallel_size=1,
                gpu_memory_utilization=0.9, max_model_len=9, max_new_tokens=2,
                thinking_mode="no_think", seed=1, enforce_eager=True, trust_remote_code=False,
            )
            class FakeLLM:
                def __init__(self, **_kwargs):
                    pass

                def generate(self, prompts, _sampling):
                    return [SimpleNamespace(outputs=[SimpleNamespace(text="safe")]) for _ in prompts]

            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: CharacterTokenizer()))
            fake_vllm = SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
            with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
                with self.assertRaisesRegex(FileExistsError, "new --output-dir"):
                    run_evaluation(args)
            self.assertEqual(predictions.read_text(encoding="utf-8"), "PREVIOUS RESULTS\n")


if __name__ == "__main__":
    unittest.main()
