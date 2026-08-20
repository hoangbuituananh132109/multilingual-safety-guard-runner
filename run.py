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
        raise ValueError(f"Unknown model: {name}. Available: {', '.join(models)}")
    return models[name]


def model_id(config: dict[str, Any], model: dict[str, Any]) -> str:
    """Resolve the base-model path/id for a model.

    Precedence:
      1. Env var MODEL_PATH_<NAME_UPPER> (e.g. MODEL_PATH_QWEN3_8B) - lets a
         company machine point at an existing local model directory without
         committing that path to git.
      2. The `id` field in config.yaml.
    """
    env_key = "MODEL_PATH_" + str(model["name"]).upper()
    override = os.environ.get(env_key)
    if override:
        return override
    return str(model["id"])


def revision_args(model: dict[str, Any]) -> list[str]:
    revision = model.get("revision")
    return ["--revision", str(revision)] if revision else []


def run_root(config: dict[str, Any], model: dict[str, Any], method: str, mode: str) -> Path:
    return Path(config["paths"]["runs_root"]) / str(model["name"]) / f"{method}_{mode}"


def preflight(config: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps({"dry_run": True, "models": [item["name"] for item in config["models"]]}, indent=2))
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the current Python environment")
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
    print(json.dumps({"python": sys.executable, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "devices": devices, "bf16_matmul": "ok"}, indent=2))


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


def evaluation_source(config: dict[str, Any], model: dict[str, Any], checkpoint: str, method: str, mode: str) -> tuple[str, Path | None, Path, list[str]]:
    if checkpoint == "before":
        destination = Path(config["paths"]["runs_root"]) / str(model["name"]) / "base"
        return model_id(config, model), None, destination, revision_args(model)
    trained = run_root(config, model, method, mode)
    final = trained / "final"
    if method == "lora":
        return model_id(config, model), final, trained, revision_args(model)
    return str(final), None, trained, []


def evaluate(config: dict[str, Any], model: dict[str, Any], checkpoint: str, method: str, mode: str, limit: int | None, sample: int | None, dry_run: bool, backend: str = "transformers") -> None:
    base_model, adapter, destination, revisions = evaluation_source(config, model, checkpoint, method, mode)
    command = [
        sys.executable,
        str(CORE / "evaluate.py"),
        "--base-model", base_model,
        *revisions,
        "--family", str(model["family"]),
        "--output-dir", str(destination / "guard"),
        "--batch-size", str(config["evaluation"]["batch_size"]),
        "--backend", backend,
        *benchmark_args(config),
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if config["evaluation"].get("load_in_4bit"):
        command.append("--load-in-4bit")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if sample is not None:
        command.extend(["--sample", str(sample)])
    execute(command, dry_run)


def likelihood(config: dict[str, Any], model: dict[str, Any], checkpoint: str, method: str, mode: str, limit: int | None, sample: int | None, dry_run: bool) -> None:
    base_model, adapter, destination, revisions = evaluation_source(config, model, checkpoint, method, mode)
    benchmark = Path(config["benchmarks"]["output_dir"]) / "sea_safeguard_vi.jsonl"
    command = [
        sys.executable,
        str(CORE / "likelihood.py"),
        "--base-model", base_model,
        *revisions,
        "--benchmark", f"sea_vi={benchmark}",
        "--output-dir", str(destination / "likelihood"),
        "--language", "vi",
        "--field", str(config["evaluation"]["likelihood_field"]),
        "--max-length", str(config["evaluation"]["likelihood_max_length"]),
        "--stride", str(config["evaluation"]["likelihood_stride"]),
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if config["evaluation"].get("load_in_4bit"):
        command.append("--load-in-4bit")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if sample is not None:
        command.extend(["--sample", str(sample)])
    execute(command, dry_run)


def train(config: dict[str, Any], model: dict[str, Any], method: str, mode: str, resume: bool, dry_run: bool) -> None:
    output = run_root(config, model, method, mode)
    common = dict(config["training"]["common"])
    method_settings = dict(config["training"][method])
    settings = {**common, **method_settings}
    model_settings: dict[str, Any] = {
        "id": model_id(config, model),
        "revision": model.get("revision"),
        "family": model["family"],
        "tuning": method,
        "attention": model.get("attention", "sdpa"),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    if method == "lora":
        model_settings.update({
            "lora_r": int(settings["lora_r"]),
            "lora_alpha": int(settings["lora_alpha"]),
            "lora_dropout": float(settings["lora_dropout"]),
            "target_modules": list(model["target_modules"]),
        })
    training_settings = {key: value for key, value in settings.items() if key not in {"max_length", "smoke_max_steps", "pilot_max_steps", "lora_r", "lora_alpha", "lora_dropout"}}
    generated = {
        "run_name": f"{model['name']}-{method}-{mode}",
        "model": model_settings,
        "data": {
            "train": str(Path(config["data"]["output_dir"]) / "train.jsonl"),
            "validation": str(Path(config["data"]["output_dir"]) / "valid.jsonl"),
            "max_length": int(settings["max_length"]),
        },
        "training": training_settings,
        "output_dir": str(output),
    }
    generated_path = output / "train_config.yaml"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    command = [sys.executable, str(CORE / "train.py"), "--config", str(generated_path)]
    # Skip eval during training by default: eval on the full valid set is very
    # expensive and makes training look stuck. Run eval separately afterwards.
    command.append("--skip-eval")
    if mode == "smoke":
        command.extend(["--max-steps", str(settings["smoke_max_steps"])])
    elif mode == "pilot":
        command.extend(["--max-steps", str(settings["pilot_max_steps"])])
    if resume:
        command.append("--resume")
    execute(command, dry_run)


def train_qwen35(config: dict[str, Any], model: dict[str, Any], method: str, mode: str, resume: bool, dry_run: bool) -> None:
    """Train a multimodal Qwen3.5-4B text-only via the standalone trainer."""
    output = run_root(config, model, method, mode)
    common = dict(config["training"]["common"])
    method_settings = dict(config["training"][method])
    settings = {**common, **method_settings}
    model_settings: dict[str, Any] = {
        "id": model_id(config, model),
        "revision": model.get("revision"),
        "family": model["family"],
        "tuning": method,
        "attention": model.get("attention", "sdpa"),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    if method == "lora":
        model_settings.update({
            "lora_r": int(settings["lora_r"]),
            "lora_alpha": int(settings["lora_alpha"]),
            "lora_dropout": float(settings["lora_dropout"]),
            "target_modules": list(model["target_modules"]),
        })
    training_settings = {key: value for key, value in settings.items() if key not in {"max_length", "smoke_max_steps", "pilot_max_steps", "lora_r", "lora_alpha", "lora_dropout"}}
    generated = {
        "run_name": f"{model['name']}-{method}-{mode}",
        "model": model_settings,
        "data": {
            "train": str(Path(config["data"]["output_dir"]) / "train.jsonl"),
            "validation": str(Path(config["data"]["output_dir"]) / "valid.jsonl"),
            "max_length": int(settings["max_length"]),
        },
        "training": training_settings,
        "output_dir": str(output),
    }
    generated_path = output / "train_config.yaml"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    command = [sys.executable, str(ROOT / "train_qwen35.py"), "--config", str(generated_path)]
    if mode == "smoke":
        command.extend(["--max-steps", str(settings["smoke_max_steps"])])
    elif mode == "pilot":
        command.extend(["--max-steps", str(settings["pilot_max_steps"])])
    if resume:
        command.append("--resume")
    execute(command, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--checkpoint", choices=["before", "after"], default="before")
    compare_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    compare_parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    unpack_parser = subparsers.add_parser("unpack")
    unpack_parser.add_argument("--replace", action="store_true")
    unpack_parser.add_argument("--strict", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--limit", type=int, default=0)
    for command in ("evaluate", "likelihood"):
        value = subparsers.add_parser(command)
        value.add_argument("--model", required=True)
        value.add_argument("--checkpoint", choices=["before", "after"], default="before")
        value.add_argument("--method", choices=["lora", "full"], default="lora")
        value.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
        value.add_argument("--limit", type=int)
        value.add_argument("--sample", type=int)
        if command == "evaluate":
            value.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    train_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    train_parser.add_argument("--resume", action="store_true")
    train35_parser = subparsers.add_parser("train_qwen35")
    train35_parser.add_argument("--model", required=True)
    train35_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    train35_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    train35_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    os.chdir(config_path.parent)
    if args.command == "preflight":
        preflight(config, args.dry_run)
    elif args.command == "status":
        execute([sys.executable, str(ROOT / "status.py"), "--config", str(config_path)], args.dry_run)
    elif args.command == "compare":
        command = [sys.executable, str(ROOT / "compare_models.py"), "--config", str(config_path), "--checkpoint", args.checkpoint]
        if args.checkpoint == "after":
            command.extend(["--method", args.method, "--run-mode", args.run_mode])
        execute(command, args.dry_run)
    elif args.command == "unpack":
        command = [sys.executable, str(ROOT / "unpack_zips.py")]
        if args.replace:
            command.append("--replace")
        if args.strict:
            command.append("--strict")
        execute(command, args.dry_run)
    elif args.command == "prepare":
        prepare(config, args.limit, args.dry_run)
    else:
        model = selected_model(config, args.model)
        if args.command == "evaluate":
            evaluate(config, model, args.checkpoint, args.method, args.run_mode, args.limit, args.sample, args.dry_run, getattr(args, "backend", "transformers"))
        elif args.command == "likelihood":
            likelihood(config, model, args.checkpoint, args.method, args.run_mode, args.limit, args.sample, args.dry_run)
        elif args.command == "train":
            train(config, model, args.method, args.mode, args.resume, args.dry_run)
        elif args.command == "train_qwen35":
            train_qwen35(config, model, args.method, args.mode, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
