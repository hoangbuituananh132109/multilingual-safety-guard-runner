from __future__ import annotations

import unittest
from pathlib import Path

from core.train_resume_gate import prepare_resume


class FakeState:
    def __init__(self) -> None:
        self.barriers = 0

    def wait_for_everyone(self) -> None:
        self.barriers += 1


class TrainResumeGateTests(unittest.TestCase):
    def test_fresh_training_does_not_wait_at_resume_barrier(self) -> None:
        state = FakeState()

        def unexpected_sync(*_args: object) -> None:
            self.fail("fresh training must not touch checkpoint state")

        result = prepare_resume(state, True, Path("unused"), None, {}, unexpected_sync)
        self.assertIsNone(result)
        self.assertEqual(state.barriers, 0)

    def test_resume_writer_synchronizes_before_other_ranks_read(self) -> None:
        state = FakeState()
        calls: list[tuple[str, int]] = []

        def sync(_output: Path, resume_arg: str, _config: dict[str, object]) -> str:
            calls.append((resume_arg, state.barriers))
            return "checkpoint-123"

        result = prepare_resume(state, True, Path("unused"), "auto", {}, sync)
        self.assertEqual(result, "checkpoint-123")
        self.assertEqual(calls, [("auto", 0)])
        self.assertEqual(state.barriers, 1)

        state = FakeState()
        calls.clear()
        result = prepare_resume(state, False, Path("unused"), "auto", {}, sync)
        self.assertEqual(result, "checkpoint-123")
        self.assertEqual(calls, [("auto", 1)])
        self.assertEqual(state.barriers, 1)


if __name__ == "__main__":
    unittest.main()
