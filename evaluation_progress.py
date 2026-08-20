from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def evaluation_directory(config: dict[str, Any], model: str, checkpoint: str, method: str, mode: str) -> Path:
    runs_root = Path(config["paths"]["runs_root"])
    suffix = "base" if checkpoint == "before" else f"{method}_{mode}"
    return runs_root / model / suffix / "guard"


def find_processes(output_dir: Path) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,%cpu=,%mem=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    markers = {
        output_dir.as_posix(),
        str(output_dir).replace("\\", "/"),
        "/".join(output_dir.parts[-4:]),
    }
    rows = []
    for line in completed.stdout.splitlines():
        normalized = line.replace("\\", "/")
        if not any(marker in normalized for marker in markers):
            continue
        parts = line.strip().split(maxsplit=4)
        if len(parts) != 5 or "core/evaluate.py" not in parts[4]:
            continue
        rows.append({
            "pid": int(parts[0]),
            "elapsed_seconds": int(parts[1]),
            "cpu_percent": float(parts[2]),
            "memory_percent": float(parts[3]),
            "command": parts[4],
        })
    return rows


def duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def snapshot(output_dir: Path) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    progress = load_json(progress_path)
    processes = find_processes(output_dir)
    metrics_ready = (output_dir / "metrics.json").is_file()
    predictions_ready = (output_dir / "predictions.jsonl").is_file()
    if progress.get("status") == "complete" and metrics_ready:
        status = "complete"
    elif processes and progress:
        status = "running"
    elif processes:
        status = "running_without_progress_file"
    elif metrics_ready:
        status = "complete"
    elif progress:
        status = "stopped_or_failed"
    else:
        status = "not_started"
    return {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "output_directory": str(output_dir),
        "status": status,
        "processes": processes,
        "phase": progress.get("phase"),
        "backend": progress.get("backend"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "percent": progress.get("percent"),
        "rows_per_second": progress.get("rows_per_second"),
        "eta_seconds": progress.get("eta_seconds"),
        "eta_human": duration(progress.get("eta_seconds")),
        "current_benchmark": progress.get("current_benchmark"),
        "progress_updated_at": progress.get("updated_at_utc"),
        "progress_file": str(progress_path) if progress_path.is_file() else None,
        "metrics_ready": metrics_ready,
        "predictions_ready": predictions_ready,
    }


def print_snapshot(value: dict[str, Any]) -> None:
    print(f"[eval-progress] {value['checked_at']} status={value['status']} phase={value['phase'] or 'unknown'}")
    print(
        f"  completed={value['completed'] if value['completed'] is not None else '?'}/"
        f"{value['total'] if value['total'] is not None else '?'} "
        f"percent={value['percent'] if value['percent'] is not None else '?'}% "
        f"rate={value['rows_per_second'] if value['rows_per_second'] is not None else '?'} rows/s "
        f"ETA={value['eta_human']}"
    )
    print(
        f"  benchmark={value['current_benchmark'] or 'unknown'} "
        f"metrics_ready={value['metrics_ready']} predictions_ready={value['predictions_ready']}"
    )
    if value["processes"]:
        for process in value["processes"]:
            print(
                f"  process pid={process['pid']} elapsed={duration(process['elapsed_seconds'])} "
                f"cpu={process['cpu_percent']}% mem={process['memory_percent']}%"
            )
    else:
        print("  process=not found")
    if value["status"] == "running_without_progress_file":
        print("  exact_percent=unavailable because this job started with older code; GPU/process activity is still detectable")
    print(f"  progress_file={value['progress_file'] or 'none'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect guard-evaluation progress from another terminal.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", choices=["before", "after"], default="before")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--watch", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_root = Path(config["paths"]["runs_root"])
    if not runs_root.is_absolute():
        config["paths"]["runs_root"] = str(config_path.parent / runs_root)
    output_dir = evaluation_directory(config, args.model, args.checkpoint, args.method, args.run_mode)
    while True:
        value = snapshot(output_dir)
        if args.json:
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        else:
            print_snapshot(value)
        if args.watch <= 0 or value["status"] in {"complete", "stopped_or_failed", "not_started"}:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
