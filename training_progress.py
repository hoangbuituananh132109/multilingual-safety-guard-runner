from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return value


def run_directory(config: dict[str, Any], model: str, method: str, mode: str) -> Path:
    return Path(config["paths"]["runs_root"]) / model / f"{method}_{mode}"


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir() and checkpoint_step(path) >= 0),
        key=checkpoint_step,
    )


def load_state(run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = [run_dir / "trainer_state.json"]
    candidates.extend(path / "trainer_state.json" for path in reversed(checkpoints(run_dir)))
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value, path
    return {}, None


def find_processes(run_dir: Path) -> list[dict[str, Any]]:
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
        run_dir.as_posix(),
        str(run_dir).replace("\\", "/"),
        f"{run_dir.parent.name}/{run_dir.name}",
    }
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not any(marker in line.replace("\\", "/") for marker in markers):
            continue
        parts = line.strip().split(maxsplit=4)
        if len(parts) != 5 or not any(name in parts[4] for name in ("core/train.py", "train_qwen35.py")):
            continue
        rows.append({
            "pid": int(parts[0]),
            "elapsed_seconds": int(parts[1]),
            "cpu_percent": float(parts[2]),
            "memory_percent": float(parts[3]),
            "command": parts[4],
        })
    return rows


def estimate_seconds_per_step(paths: list[Path], global_step: int, processes: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    if len(paths) >= 2:
        first, last = paths[-2], paths[-1]
        step_delta = checkpoint_step(last) - checkpoint_step(first)
        time_delta = last.stat().st_mtime - first.stat().st_mtime
        if step_delta > 0 and time_delta > 0:
            return time_delta / step_delta, "checkpoint timestamps"
    if processes and global_step > 0:
        return processes[0]["elapsed_seconds"] / global_step, "process elapsed/global step"
    return None, None


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def snapshot(config: dict[str, Any], model: str, method: str, mode: str) -> dict[str, Any]:
    run_dir = run_directory(config, model, method, mode)
    state, state_path = load_state(run_dir)
    paths = checkpoints(run_dir)
    processes = find_processes(run_dir)
    global_step = int(state.get("global_step") or (checkpoint_step(paths[-1]) if paths else 0))
    max_steps = int(state.get("max_steps") or 0)
    progress = global_step / max_steps if max_steps > 0 else None
    seconds_per_step, estimate_source = estimate_seconds_per_step(paths, global_step, processes)
    remaining_steps = max(0, max_steps - global_step) if max_steps else None
    eta_seconds = remaining_steps * seconds_per_step if remaining_steps is not None and seconds_per_step else None
    latest_checkpoint = paths[-1] if paths else None
    final_dir = run_dir / "final"
    final_ready = final_dir.is_dir() and any(final_dir.iterdir())
    if processes:
        status = "running"
    elif final_ready:
        status = "complete"
    elif max_steps and global_step >= max_steps:
        status = "training_finished_but_final_save_missing"
    elif latest_checkpoint:
        status = "stopped_or_interrupted"
    else:
        status = "not_started"
    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "method": method,
        "run_mode": mode,
        "run_directory": str(run_dir),
        "status": status,
        "processes": processes,
        "global_step": global_step,
        "max_steps": max_steps or None,
        "progress_percent": round(progress * 100, 2) if progress is not None else None,
        "epoch": state.get("epoch"),
        "remaining_steps": remaining_steps,
        "seconds_per_step": round(seconds_per_step, 3) if seconds_per_step is not None else None,
        "estimate_source": estimate_source,
        "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
        "eta_human": format_duration(eta_seconds),
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "latest_checkpoint_age": format_duration(time.time() - latest_checkpoint.stat().st_mtime) if latest_checkpoint else None,
        "trainer_state": str(state_path) if state_path else None,
        "final_ready": final_ready,
    }


def print_snapshot(value: dict[str, Any]) -> None:
    print(f"[progress] {value['checked_at']} model={value['model']} status={value['status']}")
    print(
        f"  step={value['global_step']}/{value['max_steps'] or '?'} "
        f"progress={value['progress_percent'] if value['progress_percent'] is not None else '?'}% "
        f"epoch={value['epoch'] if value['epoch'] is not None else '?'}"
    )
    print(
        f"  latest_checkpoint={value['latest_checkpoint'] or 'none'} "
        f"age={value['latest_checkpoint_age'] or 'unknown'}"
    )
    print(
        f"  speed={value['seconds_per_step'] if value['seconds_per_step'] is not None else '?'}s/step "
        f"source={value['estimate_source'] or 'unknown'} remaining={value['remaining_steps'] if value['remaining_steps'] is not None else '?'} "
        f"ETA={value['eta_human']}"
    )
    if value["processes"]:
        for process in value["processes"]:
            print(
                f"  process pid={process['pid']} elapsed={format_duration(process['elapsed_seconds'])} "
                f"cpu={process['cpu_percent']}% mem={process['memory_percent']}%"
            )
    else:
        print("  process=not found")
    print(f"  final_ready={value['final_ready']} trainer_state={value['trainer_state'] or 'none'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect training progress without relying on the original terminal.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds; 0 prints once.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    runs_root = Path(config["paths"]["runs_root"])
    if not runs_root.is_absolute():
        config["paths"]["runs_root"] = str(config_path.parent / runs_root)
    while True:
        value = snapshot(config, args.model, args.method, args.run_mode)
        if args.json:
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        else:
            print_snapshot(value)
        if args.watch <= 0 or value["status"] in {"complete", "training_finished_but_final_save_missing", "not_started"}:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
