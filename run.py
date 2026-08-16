from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return value


def execute(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=os.environ.copy())


def nested_root(root: Path, marker: Path) -> Path:
    if (root / marker).exists():
        return root
    matches = [path.parent for path in root.rglob(marker.name) if path.as_posix().endswith(marker.as_posix())]
    if len(matches) == 1:
        candidate = matches[0]
        for _ in marker.parts[:-1]:
            candidate = candidate.parent
        return candidate
    raise FileNotFoundError(f"Cannot locate {marker} under {root}")


def selected_model(config: dict[str, Any], name: str) -> dict[str, Any]:
    models = {str(item["name"]): item for item in config["models"]}
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name]


def revision_args(model: dict[str, Any]) -> list[str]:
    revision = model.get("revision")
    return ["--revision", str(revision)] if revision else []


def preflight(config: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps({"dry_run": True, "models": [item["name"] for item in config["models"]]}, indent=2))
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    devices = []
    for index in range(torch.cuda.device_count()):
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "bf16": torch.cuda.is_bf16_supported(),
        })
    left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    _ = left @ right
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "devices": devices, "bf16_matmul": "ok"}, indent=2))


def prepare(config: dict[str, Any], limit: int, dry_run: bool) -> None:
    data = config["data"]
    benchmarks = config["benchmarks"]
    nemotron_input = Path(data["nemotron_root"])
    sea_input = Path(benchmarks["sea_root"])
    xsafety_input = Path(benchmarks["xsafety_root"])
    nemotron = nemotron_input if dry_run else nested_root(nemotron_input, Path("en/train.jsonl"))
    sea = sea_input if dry_run else nested_root(sea_input, Path("seahelm_tasks/task_config.yaml"))
    xsafety = xsafety_input if dry_run else nested_root(xsafety_input, Path("en"))
    execute([
        sys.executable,
        str(CORE / "prepare_training_data.py"),
        "--root", str(nemotron),
        "--revision", str(data["revision"]),
        "--languages", *[str(value) for value in data["languages"]],
        "--output-dir", str(data["output_dir"]),
        "--limit", str(limit),
    ], dry_run)
    execute([
        sys.executable,
        str(CORE / "prepare_sea.py"),
        "--repo-root", str(sea),
        "--revision", str(benchmarks["sea_revision"]),
        "--output-file", str(benchmarks["sea_output"]),
    ], dry_run)
    execute([
        sys.executable,
        str(CORE / "prepare_benchmarks.py"),
        "--cultureguard-test", str(Path(data["output_dir"]) / "test.jsonl"),
        "--sea-source", str(benchmarks["sea_output"]),
        "--xsafety-root", str(xsafety),
        "--xsafety-revision", str(benchmarks["xsafety_revision"]),
        "--xsafety-languages", *[str(value) for value in benchmarks["xsafety_languages"]],
        "--output-dir", str(benchmarks["output_dir"]),
    ], dry_run)


def benchmark_args(config: dict[str, Any]) -> list[str]:
    root = Path(config["benchmarks"]["output_dir"])
    result: list[str] = []
    for name, filename in (
        ("cultureguard", "cultureguard_test_9lang.jsonl"),
        ("xsafety", "xsafety_multilingual.jsonl"),
        ("sea_vi", "sea_safeguard_vi.jsonl"),
    ):
        result.extend(["--benchmark", f"{name}={root / filename}"])
    return result


def evaluate(config: dict[str, Any], model: dict[str, Any], stage: str, limit: int | None, dry_run: bool) -> None:
    output = Path(config["paths"]["output_root"]) / f"eval_{stage}" / str(model["name"])
    command = [
        sys.executable,
        str(CORE / "evaluate.py"),
        "--base-model", str(model["id"]),
        *revision_args(model),
        "--family", str(model["family"]),
        "--output-dir", str(output),
        "--batch-size", str(config["evaluation"]["batch_size"]),
        *benchmark_args(config),
    ]
    if stage == "after":
        command.extend(["--adapter", str(Path(config["paths"]["output_root"]) / "train" / str(model["name"]) / "final")])
    if config["evaluation"].get("load_in_4bit"):
        command.append("--load-in-4bit")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    execute(command, dry_run)


def likelihood(config: dict[str, Any], model: dict[str, Any], stage: str, limit: int | None, dry_run: bool) -> None:
    benchmark = Path(config["benchmarks"]["output_dir"]) / "sea_safeguard_vi.jsonl"
    output = Path(config["paths"]["output_root"]) / f"likelihood_{stage}" / str(model["name"])
    command = [
        sys.executable,
        str(CORE / "likelihood.py"),
        "--base-model", str(model["id"]),
        *revision_args(model),
        "--benchmark", f"sea_vi={benchmark}",
        "--output-dir", str(output),
        "--language", "vi",
        "--field", str(config["evaluation"]["likelihood_field"]),
        "--max-length", str(config["evaluation"]["likelihood_max_length"]),
        "--stride", str(config["evaluation"]["likelihood_stride"]),
    ]
    if stage == "after":
        command.extend(["--adapter", str(Path(config["paths"]["output_root"]) / "train" / str(model["name"]) / "final")])
    if config["evaluation"].get("load_in_4bit"):
        command.append("--load-in-4bit")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    execute(command, dry_run)


def train(config: dict[str, Any], model: dict[str, Any], mode: str, resume: bool, dry_run: bool) -> None:
    output = Path(config["paths"]["output_root"]) / "train" / str(model["name"])
    settings = config["training"]
    generated = {
        "model": {
            "id": model["id"],
            "revision": model.get("revision"),
            "family": model["family"],
            "tuning": model["tuning"],
            "attention": model.get("attention", "sdpa"),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "lora_r": int(settings["lora_r"]),
            "lora_alpha": int(settings["lora_alpha"]),
            "lora_dropout": float(settings["lora_dropout"]),
            "target_modules": list(model["target_modules"]),
        },
        "data": {
            "train": str(Path(config["data"]["output_dir"]) / "train.jsonl"),
            "validation": str(Path(config["data"]["output_dir"]) / "valid.jsonl"),
            "max_length": int(settings["max_length"]),
        },
        "training": {key: value for key, value in settings.items() if key not in {"max_length", "smoke_max_steps", "pilot_max_steps"}},
        "output_dir": str(output),
    }
    generated_path = output / "train_config.yaml"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    command = [sys.executable, str(CORE / "train.py"), "--config", str(generated_path)]
    if mode == "smoke":
        command.extend(["--max-steps", str(settings["smoke_max_steps"])])
    elif mode == "pilot":
        command.extend(["--max-steps", str(settings["pilot_max_steps"])])
    if resume:
        command.append("--resume")
    execute(command, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/workspace/project/config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--limit", type=int, default=0)
    for stage in ("evaluate", "likelihood"):
        value = subparsers.add_parser(stage)
        value.add_argument("--model", required=True)
        value.add_argument("--checkpoint", choices=["before", "after"], default="before")
        value.add_argument("--limit", type=int)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    train_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        preflight(config, args.dry_run)
    elif args.command == "prepare":
        prepare(config, args.limit, args.dry_run)
    else:
        model = selected_model(config, args.model)
        if args.command == "evaluate":
            evaluate(config, model, args.checkpoint, args.limit, args.dry_run)
        elif args.command == "likelihood":
            likelihood(config, model, args.checkpoint, args.limit, args.dry_run)
        elif args.command == "train":
            train(config, model, args.mode, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
