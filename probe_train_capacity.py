#!/usr/bin/env python3
"""Find the largest training microbatch that survives a short real Trainer run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_batches(value: str) -> list[int]:
    batches = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not batches or any(batch < 1 for batch in batches):
        raise argparse.ArgumentTypeError("--batches must contain positive comma-separated integers")
    return sorted(set(batches), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["lora", "full"], required=True)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--target-global-batch", type=int, required=True)
    parser.add_argument("--batches", type=parse_batches, required=True)
    parser.add_argument("--vram-fraction", type=float, default=0.90)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--optimizer", choices=["adamw_torch", "adamw_torch_fused"])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "capacity_probes")
    parser.add_argument("--keep-going", action="store_true", help="Test every divisible candidate instead of stopping at the first success")
    args = parser.parse_args()

    if args.gpus < 1 or args.target_global_batch < 1 or args.max_length < 1:
        parser.error("GPU count, target global batch, and max length must be positive")
    if not 0.0 < args.vram_fraction <= 1.0:
        parser.error("--vram-fraction must be greater than 0 and at most 1")

    trainer_command = "train_qwen35" if args.model.startswith("qwen35") else "train"
    optimizer = args.optimizer or ("adamw_torch_fused" if args.method == "full" else "adamw_torch")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.results_dir / f"{args.model}_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for batch in args.batches:
        denominator = args.gpus * batch
        if args.target_global_batch % denominator:
            records.append({"batch": batch, "status": "skipped", "reason": "target global batch is not divisible"})
            continue
        accumulation = args.target_global_batch // denominator
        log_path = output_dir / f"batch_{batch}_accum_{accumulation}.log"
        command = [
            sys.executable,
            str(ROOT / "run.py"),
            trainer_command,
            "--model", args.model,
            "--method", args.method,
            "--mode", "smoke",
            "--gpus", str(args.gpus),
            "--per-device-batch-size", str(batch),
            "--gradient-accumulation-steps", str(accumulation),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--max-length", str(args.max_length),
            "--optimizer", optimizer,
            "--vram-fraction", str(args.vram_fraction),
            "--no-checkpoints",
            "--no-final-save",
        ]
        print(f"[probe] batch={batch} accumulation={accumulation} global_batch={args.target_global_batch}", flush=True)
        print("[probe] " + " ".join(command), flush=True)
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        started = time.time()
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
            return_code = process.wait()
        elapsed = time.time() - started
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        status = "success" if return_code == 0 else "oom" if "out of memory" in log_text else "failed"
        record = {
            "batch": batch,
            "gradient_accumulation_steps": accumulation,
            "effective_global_batch": args.target_global_batch,
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "log": str(log_path),
        }
        records.append(record)
        (output_dir / "summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"[probe] result={status} elapsed={elapsed:.1f}s log={log_path}", flush=True)
        if status == "success" and not args.keep_going:
            break

    successful = [record for record in records if record.get("status") == "success"]
    if not successful:
        raise SystemExit(f"No candidate succeeded. Inspect {output_dir / 'summary.json'}")
    best = successful[0]
    print(json.dumps({"recommended": best, "results_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
