"""Resume five Gemini shards with cooldowns until the paired 81k set is complete."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/source-study/wildguard-vi/translation_input.jsonl"))
    parser.add_argument("--api-key-file", type=Path, default=Path("../API.txt"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/source-study/wildguard-vi"))
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=40)
    args = parser.parse_args()
    if args.shards != 5:
        raise ValueError("This audited assignment expects exactly five shards and 15 key slots")
    total = line_count(args.input)
    if total != 81_000:
        raise ValueError(f"Expected 81,000 translation inputs, got {total}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logs = args.work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    env["PYTHONPATH"] = str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")

    for round_index in range(1, args.max_rounds + 1):
        processes: list[tuple[int, subprocess.Popen, object, object]] = []
        for shard in range(args.shards):
            checkpoint = args.work_dir / f"shard_{shard}_checkpoint.jsonl"
            expected = (total + args.shards - 1 - shard) // args.shards
            if line_count(checkpoint) >= expected:
                continue
            first_slot = shard * 3 + 1
            command = [
                sys.executable,
                "-m",
                "translator.cli",
                "translate",
                "--input",
                str(args.input),
                "--output",
                str(args.work_dir / f"shard_{shard}.jsonl"),
                "--checkpoint",
                str(checkpoint),
                "--failed-output",
                str(args.work_dir / f"shard_{shard}_round_{round_index:02d}_failed.jsonl"),
                "--provider",
                "gemini",
                "--model",
                args.model,
                "--api-key-file",
                str(args.api_key_file),
                "--api-key-slots",
                f"{first_slot},{first_slot + 1},{first_slot + 2}",
                "--shard-index",
                str(shard),
                "--shard-count",
                str(args.shards),
                "--confirm-real-api",
            ]
            stdout = (logs / f"shard_{shard}_round_{round_index:02d}.out.log").open("w", encoding="utf-8")
            stderr = (logs / f"shard_{shard}_round_{round_index:02d}.err.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=repo_root, env=env, stdout=stdout, stderr=stderr)
            processes.append((shard, process, stdout, stderr))
        if not processes:
            print("All five translation checkpoints are complete.", flush=True)
            return
        failed = []
        for shard, process, stdout, stderr in processes:
            code = process.wait()
            stdout.close()
            stderr.close()
            checkpoint = args.work_dir / f"shard_{shard}_checkpoint.jsonl"
            expected = (total + args.shards - 1 - shard) // args.shards
            count = line_count(checkpoint)
            print(f"round={round_index} shard={shard} exit={code} completed={count}/{expected}", flush=True)
            if count < expected:
                failed.append(shard)
        if not failed:
            print("All five translation checkpoints are complete.", flush=True)
            return
        if round_index == args.max_rounds:
            raise SystemExit(f"Incomplete shards after {args.max_rounds} rounds: {failed}")
        print(f"Cooling down {args.cooldown_seconds}s before resuming shards {failed}", flush=True)
        time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
