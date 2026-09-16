"""Keep checkpoint metadata synchronization out of fresh DDP runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def prepare_resume(
    distributed_state: Any,
    is_world_process_zero: bool,
    output: Path,
    resume_arg: str | None,
    train_cfg: dict[str, Any],
    sync_intervals: Callable[[Path, str | None, dict[str, Any]], str | None],
) -> str | None:
    """Serialize checkpoint-state edits only when a checkpoint is requested.

    A fresh run has no shared trainer_state.json to edit, so making rank zero
    wait at an extra NCCL barrier only creates another distributed failure
    point before Trainer's own synchronization.
    """
    if resume_arg is None:
        return None
    resume = sync_intervals(output, resume_arg, train_cfg) if is_world_process_zero else None
    distributed_state.wait_for_everyone()
    if not is_world_process_zero:
        resume = sync_intervals(output, resume_arg, train_cfg)
    return resume
