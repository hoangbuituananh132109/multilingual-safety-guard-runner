from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.prompt import nemotron_instruction
from core.stage2_data import N23, Normalized, output_payload, render, render_views, validate_dataset, v3_records, vi_records


class Stage2DataTests(unittest.TestCase):
    def test_taxonomy_off_instruction_and_target_omit_categories(self) -> None:
        instruction = nemotron_instruction("prompt", None, taxonomy_mode="off")
        target = output_payload("unsafe", None, [N23[0]], "off")
        self.assertNotIn("UNSAFE CONTENT CATEGORIES", instruction)
        self.assertNotIn("Safety Categories", target)

    def test_unknown_category_forces_taxonomy_off(self) -> None:
        row = Normalized(
            source="n35",
            source_id="one",
            split="train",
            language="en",
            prompt="test",
            response=None,
            prompt_label="unsafe",
            response_label=None,
            categories=["Economic Harm"],
        )
        value = render(row, 3407)
        self.assertEqual(value["taxonomy_mode"], "off")
        self.assertEqual(value["unknown_categories"], ["Economic Harm"])
        self.assertNotIn("Safety Categories", value["target"])

    def test_think_target_keeps_json_after_reasoning(self) -> None:
        row = Normalized(
            source="reasoning",
            source_id="two",
            split="train",
            language="en",
            prompt="test",
            response=None,
            prompt_label="safe",
            response_label=None,
            categories=[N23[0]],
            reasoning="<think>short rationale</think>\nPrompt harm: unharmful",
        )
        value = render(row, 3407, taxonomy_mode="on", thinking_mode="think")
        self.assertTrue(value["target"].startswith("short rationale\n</think>\n\n{"))
        self.assertFalse(value["target"].startswith("<think>"))

    def test_v3_renders_one_stable_taxonomy_view(self) -> None:
        row = Normalized("nemotron_v3_replay", "id", "train", "en", "p", None, "safe", None, [])
        values = render_views(row, 3407)
        self.assertEqual(len(values), 1)
        self.assertIn(values[0]["taxonomy_mode"], {"on", "off"})
        self.assertEqual({value["thinking_mode"] for value in values}, {"no_think"})
        self.assertEqual(len({value["semantic_id"] for value in values}), 1)
        self.assertEqual(len({value["example_id"] for value in values}), 1)

    def test_v3_safe_rows_are_actually_sampled_on_and_off(self) -> None:
        modes = Counter()
        for index in range(512):
            row = Normalized("nemotron_v3_replay", f"safe-{index}", "train", "en", "p", None, "safe", None, [])
            modes[render_views(row, 3407)[0]["taxonomy_mode"]] += 1
        self.assertEqual(set(modes), {"on", "off"})
        self.assertGreater(modes["on"] / sum(modes.values()), 0.65)
        self.assertLess(modes["on"] / sum(modes.values()), 0.85)

    def test_reasoning_uses_one_stable_native_mode(self) -> None:
        modes = set()
        for index in range(128):
            row = Normalized(
                "nemotron_reasoning_28k", f"id-{index}", "train", "en", "p", None,
                "safe", None, [N23[0]], "<think>because it is benign</think>",
            )
            values = render_views(row, 3407)
            self.assertEqual(len(values), 1)
            modes.add(values[0]["thinking_mode"])
        self.assertEqual(modes, {"think", "no_think"})

    def test_safe_no_category_can_use_taxonomy_on(self) -> None:
        row = Normalized("n35", "none", "train", "en", "p", None, "safe", None, [])
        self.assertEqual(render(row, 3407)["taxonomy_mode"], "on")
        self.assertNotIn("Safety Categories", render(row, 3407)["target"])

    def test_safe_auxiliary_category_is_omitted_from_target(self) -> None:
        target = output_payload("safe", None, ["Needs Caution"], "on")
        self.assertNotIn("Safety Categories", target)

    def test_unsafe_known_category_can_use_taxonomy_on(self) -> None:
        row = Normalized("n35", "known", "train", "en", "p", None, "unsafe", None, [N23[0]])
        value = render(row, 3407)
        self.assertEqual(value["taxonomy_mode"], "on")
        self.assertIn("Safety Categories", value["target"])

    def test_vi_prompt_view_retains_category_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {
                "record_uid": "one", "prompt_en": "unsafe prompt", "prompt_vi": "noi dung nguy hiem",
                "response_vi": None, "prompt_label": "unsafe", "response_label": None,
                "violated_categories": N23[0],
            }
            for filename in ("nemotron_train_en_vi_v10_final.jsonl", "nemotron_valid_en_vi_v10_final.jsonl"):
                (root / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
            values = list(vi_records(root, "vi_gemini"))
            self.assertEqual(values[0].categories, [N23[0]])

    def test_v3_records_selected_language_not_last_configured_language(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for language in ("en", "zh"):
                (root / language).mkdir()
                for filename in ("train.jsonl", "valid.jsonl"):
                    row = {
                        "id": "same-id", "prompt": f"{language} prompt", "response": None,
                        "prompt_label": "safe", "response_label": None, "violated_categories": "",
                    }
                    (root / language / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
            values = list(v3_records(root, ["en", "zh"], 3407))
            self.assertTrue(all(value.prompt.startswith(value.language) for value in values))

    def test_validator_detects_cross_split_semantic_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = render(Normalized("source", "id", "train", "en", "p", None, "safe", None, []), 3407)
            for split in ("train", "validation"):
                (root / f"{split}.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
            result = validate_dataset(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("cross-split semantic_id" in error for error in result["errors"]))

    def test_validator_detects_cross_split_content_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = render(Normalized("a", "one", "train", "en", "p", None, "safe", None, []), 3407)
            second = render(Normalized("b", "two", "validation", "en", "p", None, "safe", None, []), 3407)
            (root / "train.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
            (root / "validation.jsonl").write_text(json.dumps(second) + "\n", encoding="utf-8")
            result = validate_dataset(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("cross-split content_sha256" in error for error in result["errors"]))

    def test_validator_detects_same_content_label_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            safe = render(Normalized("a", "one", "train", "en", "p", None, "safe", None, []), 3407)
            unsafe = render(Normalized("b", "two", "train", "en", "p", None, "unsafe", None, [N23[0]]), 3407)
            (root / "train.jsonl").write_text(json.dumps(safe) + "\n" + json.dumps(unsafe) + "\n", encoding="utf-8")
            (root / "validation.jsonl").write_text("", encoding="utf-8")
            result = validate_dataset(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("conflicting labels for content_sha256" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
