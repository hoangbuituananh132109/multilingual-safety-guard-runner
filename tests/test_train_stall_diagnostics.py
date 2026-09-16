from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from core.train_stall_diagnostics import dump_stack_if_stalled


class TrainStallDiagnosticsTests(unittest.TestCase):
    def test_stalled_phase_writes_python_stack_to_rank_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with dump_stack_if_stalled(Path(directory), "trainer_init", rank=3, timeout_seconds=0.05) as path:
                time.sleep(0.15)
            contents = path.read_text(encoding="utf-8")
            self.assertIn("stage=trainer_init rank=3", contents)
            self.assertIn("test_stalled_phase_writes_python_stack_to_rank_file", contents)

    def test_completed_phase_cancels_pending_dump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with dump_stack_if_stalled(Path(directory), "trainer_init", rank=4, timeout_seconds=0.05) as path:
                pass
            time.sleep(0.10)
            contents = path.read_text(encoding="utf-8")
            self.assertIn("stage=trainer_init rank=4", contents)
            self.assertNotIn("Current thread", contents)


if __name__ == "__main__":
    unittest.main()
