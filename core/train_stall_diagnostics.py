"""Per-rank Python stack capture for slow distributed training phases."""

from __future__ import annotations

import faulthandler
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def dump_stack_if_stalled(
    output_dir: Path, stage: str, rank: int, timeout_seconds: float = 90
) -> Iterator[Path]:
    """Write a stack trace if a rank remains inside a phase past the timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / f"{stage}_rank{rank}.stack.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"stage={stage} rank={rank} pid={os.getpid()} timeout_seconds={timeout_seconds}\n")
        handle.flush()
        faulthandler.dump_traceback_later(timeout_seconds, repeat=False, file=handle)
        try:
            yield path
        finally:
            faulthandler.cancel_dump_traceback_later()
