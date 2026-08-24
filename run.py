from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
        ("cultureguard_standard", "cultureguard_standard_9lang.jsonl"),
        ("cultureguard_jb", "cultureguard_jb_9lang.jsonl"),
        ("xsafety", "xsafety_multilingual.jsonl"),
        ("sea_vi", "sea_safeguard_vi.jsonl"),
    ):
        result.extend(["--benchmark", f"{name}={root / filename}"])
    for name, filename in (
        ("polyguard_prompts", "polyguard_prompts_9lang.jsonl"),
        ("multijail", "multijail_4lang.jsonl"),
    ):
        path = root / filename
        if path.is_file():
            result.extend(["--benchmark", f"{name}={path}"])
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



def adapter_signature(adapter: Path) -> str:
    """Hash the PEFT config and weights used to build a merged vLLM model."""
    files = [adapter / "adapter_config.json", *sorted(adapter.glob("adapter_model.*"))]
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No LoRA adapter files found under {adapter}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def merged_adapter_path(base_model: str, adapter: Path, destination: Path, dry_run: bool) -> Path:
    """Return (and lazily create) a merged weights dir for a LoRA adapter.

    vLLM cannot load a PeftModel directly, so for after+lora eval we merge the
    adapter into the base weights once and point vLLM at the merged dir.
    """
    merged = destination / "merged"
    marker = merged / "merge_manifest.json"
    if dry_run:
        return merged
    signature = adapter_signature(adapter)
    if marker.is_file():
        try:
            manifest = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if manifest.get("base_model") == base_model and manifest.get("adapter_sha256") == signature:
            return merged
        print(f"adapter changed; rebuilding stale merged model at {merged}", flush=True)
        shutil.rmtree(merged)
    print(f"merging LoRA adapter {adapter} into {merged} for vLLM eval...", flush=True)
    execute([
        sys.executable,
        str(ROOT / "merge_adapter.py"),
        "--base-model", base_model,
        "--adapter", str(adapter),
        "--output", str(merged),
        "--adapter-sha256", signature,
    ], dry_run=False)
    return merged

def evaluate(config: dict[str, Any], model: dict[str, Any], checkpoint: str, method: str, mode: str, limit: int | None, sample: int | None, dry_run: bool, backend: str = "transformers", decoding_profile: str = "greedy", output_tag: str = "guard", tensor_parallel_size: int = 1) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_tag):
        raise ValueError("--output-tag may contain only letters, numbers, dot, underscore, and dash")
    base_model, adapter, destination, revisions = evaluation_source(config, model, checkpoint, method, mode)
    if backend == "vllm" and adapter is not None:
        merged = merged_adapter_path(base_model, adapter, destination, dry_run)
        base_model = str(merged)
        adapter = None
    command = [
        sys.executable,
        str(CORE / "evaluate.py"),
        "--base-model", base_model,
        *revisions,
        "--family", str(model["family"]),
        "--output-dir", str(destination / output_tag),
        "--batch-size", str(config["evaluation"]["batch_size"]),
        "--vllm-chunk-size", str(config["evaluation"].get("vllm_chunk_size", 1000)),
        "--gpu-memory-utilization", str(config["evaluation"].get("gpu_memory_utilization", 0.92)),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--backend", backend,
        "--decoding-profile", decoding_profile,
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


def distributed_training_command(
    script: Path,
    script_args: list[str],
    gpus: int,
    nnodes: int,
    node_rank: int,
    master_addr: str | None,
    master_port: int,
) -> list[str]:
    if gpus < 1:
        raise ValueError("--gpus must be at least 1")
    if nnodes < 1:
        raise ValueError("--nnodes must be at least 1")
    if not 0 <= node_rank < nnodes:
        raise ValueError("--node-rank must be between 0 and nnodes-1")
    if gpus == 1 and nnodes == 1:
        return [sys.executable, str(script), *script_args]
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={gpus}",
    ]
    if nnodes == 1:
        launcher.append("--standalone")
    else:
        if not master_addr:
            raise ValueError("--master-addr is required when --nnodes is greater than 1")
        launcher.extend([
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
        ])
    return [*launcher, str(script), *script_args]


def write_training_config(path: Path, generated: dict[str, Any], dry_run: bool, nnodes: int, node_rank: int) -> None:
    if dry_run:
        return
    content = yaml.safe_dump(generated, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if nnodes == 1 or node_rank == 0:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        return
    deadline = time.time() + 300
    while time.time() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for node 0 to write {path}")


def apply_training_overrides(
    settings: dict[str, Any],
    *,
    per_device_batch_size: int | None,
    gradient_accumulation_steps: int | None,
    epochs: float | None,
    learning_rate: float | None,
    save_steps: int | None,
    save_total_limit: int | None,
    max_length: int | None,
    optimizer: str | None,
) -> None:
    positive_ints = {
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "max_length": max_length,
    }
    for key, value in positive_ints.items():
        if value is not None:
            if value < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be at least 1")
            settings[key] = value
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("--epochs must be greater than 0")
        settings["epochs"] = epochs
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("--learning-rate must be greater than 0")
        settings["learning_rate"] = learning_rate
    if optimizer is not None:
        settings["optim"] = optimizer


def train(
    config: dict[str, Any],
    model: dict[str, Any],
    method: str,
    mode: str,
    resume: bool,
    dry_run: bool,
    no_checkpoints: bool = False,
    gpus: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int = 29500,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    epochs: float | None = None,
    learning_rate: float | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    max_length: int | None = None,
    optimizer: str | None = None,
    vram_fraction: float | None = None,
    no_final_save: bool = False,
) -> None:
    output = run_root(config, model, method, mode)
    common = dict(config["training"]["common"])
    method_settings = dict(config["training"][method])
    settings = {**common, **method_settings}
    apply_training_overrides(
        settings,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_length=max_length,
        optimizer=optimizer,
    )
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
    write_training_config(generated_path, generated, dry_run, nnodes, node_rank)
    script_args = ["--config", str(generated_path)]
    # Skip eval during training by default: eval on the full valid set is very
    # expensive and makes training look stuck. Run eval separately afterwards.
    script_args.append("--skip-eval")
    if no_checkpoints:
        script_args.append("--no-checkpoints")
    if no_final_save:
        script_args.append("--no-final-save")
    if vram_fraction is not None:
        if not 0.0 < vram_fraction <= 1.0:
            raise ValueError("--vram-fraction must be greater than 0 and at most 1")
        script_args.extend(["--vram-fraction", str(vram_fraction)])
    if mode == "smoke":
        script_args.extend(["--max-steps", str(settings["smoke_max_steps"])])
    elif mode == "pilot":
        script_args.extend(["--max-steps", str(settings["pilot_max_steps"])])
    if resume:
        script_args.append("--resume")
    command = distributed_training_command(CORE / "train.py", script_args, gpus, nnodes, node_rank, master_addr, master_port)
    execute(command, dry_run)


def train_qwen35(
    config: dict[str, Any],
    model: dict[str, Any],
    method: str,
    mode: str,
    resume: bool,
    dry_run: bool,
    gpus: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int = 29500,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    epochs: float | None = None,
    learning_rate: float | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    max_length: int | None = None,
    optimizer: str | None = None,
    vram_fraction: float | None = None,
    no_checkpoints: bool = False,
    no_final_save: bool = False,
) -> None:
    """Train a multimodal Qwen3.5-4B text-only via the standalone trainer."""
    output = run_root(config, model, method, mode)
    common = dict(config["training"]["common"])
    method_settings = dict(config["training"][method])
    settings = {**common, **method_settings}
    apply_training_overrides(
        settings,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_length=max_length,
        optimizer=optimizer,
    )
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
    write_training_config(generated_path, generated, dry_run, nnodes, node_rank)
    script_args = ["--config", str(generated_path)]
    if no_checkpoints:
        script_args.append("--no-checkpoints")
    if no_final_save:
        script_args.append("--no-final-save")
    if vram_fraction is not None:
        if not 0.0 < vram_fraction <= 1.0:
            raise ValueError("--vram-fraction must be greater than 0 and at most 1")
        script_args.extend(["--vram-fraction", str(vram_fraction)])
    if mode == "smoke":
        script_args.extend(["--max-steps", str(settings["smoke_max_steps"])])
    elif mode == "pilot":
        script_args.extend(["--max-steps", str(settings["pilot_max_steps"])])
    if resume:
        script_args.append("--resume")
    command = distributed_training_command(ROOT / "train_qwen35.py", script_args, gpus, nnodes, node_rank, master_addr, master_port)
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
    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--model", required=True)
    progress_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    progress_parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    progress_parser.add_argument("--watch", type=float, default=0.0)
    progress_parser.add_argument("--json", action="store_true")
    eval_progress_parser = subparsers.add_parser("eval-progress")
    eval_progress_parser.add_argument("--model", required=True)
    eval_progress_parser.add_argument("--checkpoint", choices=["before", "after"], default="before")
    eval_progress_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    eval_progress_parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    eval_progress_parser.add_argument("--watch", type=float, default=0.0)
    eval_progress_parser.add_argument("--json", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--model", required=True)
    analyze_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    analyze_parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    analyze_parser.add_argument("--smooth-window", type=int, default=25)
    analyze_parser.add_argument("--output-dir", type=Path)
    unpack_parser = subparsers.add_parser("unpack")
    unpack_parser.add_argument("--replace", action="store_true")
    unpack_parser.add_argument("--strict", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--limit", type=int, default=0)
    nvidia_parser = subparsers.add_parser("nvidia-report", help="Paper-aligned harmful-F1/recall summary from a run's guard metrics.")
    nvidia_parser.add_argument("--model", required=True)
    nvidia_parser.add_argument("--checkpoint", choices=["before", "after"], default="before")
    nvidia_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    nvidia_parser.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
    for command in ("evaluate", "likelihood"):
        value = subparsers.add_parser(command)
        value.add_argument("--model", required=True)
        value.add_argument("--checkpoint", choices=["before", "after"], default="before")
        value.add_argument("--method", choices=["lora", "full"], default="lora")
        value.add_argument("--run-mode", choices=["smoke", "pilot", "full"], default="full")
        value.add_argument("--limit", type=int)
        value.add_argument("--sample", type=int)
        if command == "evaluate":
            value.add_argument("--backend", choices=["transformers", "vllm"])
            value.add_argument("--decoding-profile", choices=["greedy", "nemotron_model_card"])
            value.add_argument("--output-tag", default="guard")
            value.add_argument("--gpus", type=int, default=1)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    train_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--no-checkpoints", action="store_true", help="Only save final model, skip periodic checkpoints")
    train_parser.add_argument("--gpus", type=int, default=1)
    train_parser.add_argument("--nnodes", type=int, default=1)
    train_parser.add_argument("--node-rank", type=int, default=0)
    train_parser.add_argument("--master-addr")
    train_parser.add_argument("--master-port", type=int, default=29500)
    train_parser.add_argument("--per-device-batch-size", type=int)
    train_parser.add_argument("--gradient-accumulation-steps", type=int)
    train35_parser = subparsers.add_parser("train_qwen35")
    train35_parser.add_argument("--model", required=True)
    train35_parser.add_argument("--method", choices=["lora", "full"], default="lora")
    train35_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    train35_parser.add_argument("--resume", action="store_true")
    train35_parser.add_argument("--gpus", type=int, default=1)
    train35_parser.add_argument("--nnodes", type=int, default=1)
    train35_parser.add_argument("--node-rank", type=int, default=0)
    train35_parser.add_argument("--master-addr")
    train35_parser.add_argument("--master-port", type=int, default=29500)
    train35_parser.add_argument("--per-device-batch-size", type=int)
    train35_parser.add_argument("--gradient-accumulation-steps", type=int)
    train35_parser.add_argument("--no-checkpoints", action="store_true", help="Disable periodic checkpoints for disposable smoke/probe runs")
    for value in (train_parser, train35_parser):
        value.add_argument("--epochs", type=float)
        value.add_argument("--learning-rate", type=float)
        value.add_argument("--save-steps", type=int)
        value.add_argument("--save-total-limit", type=int)
        value.add_argument("--max-length", type=int)
        value.add_argument("--optimizer", choices=["adamw_torch", "adamw_torch_fused"])
        value.add_argument("--vram-fraction", type=float)
        value.add_argument("--no-final-save", action="store_true", help="Skip final model export; only for disposable probes")
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
    elif args.command == "progress":
        command = [
            sys.executable,
            str(ROOT / "training_progress.py"),
            "--config", str(config_path),
            "--model", args.model,
            "--method", args.method,
            "--run-mode", args.run_mode,
        ]
        if args.watch > 0:
            command.extend(["--watch", str(args.watch)])
        if args.json:
            command.append("--json")
        execute(command, args.dry_run)
    elif args.command == "eval-progress":
        command = [
            sys.executable,
            str(ROOT / "evaluation_progress.py"),
            "--config", str(config_path),
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--method", args.method,
            "--run-mode", args.run_mode,
        ]
        if args.watch > 0:
            command.extend(["--watch", str(args.watch)])
        if args.json:
            command.append("--json")
        execute(command, args.dry_run)
    elif args.command == "analyze":
        command = [
            sys.executable,
            str(ROOT / "analyze_training.py"),
            "--config", str(config_path),
            "--model", args.model,
            "--method", args.method,
            "--run-mode", args.run_mode,
            "--smooth-window", str(args.smooth_window),
        ]
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
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
    elif args.command == "nvidia-report":
        model = selected_model(config, args.model)
        runs_root = Path(config["paths"]["runs_root"])
        if not runs_root.is_absolute():
            runs_root = config_path.parent / runs_root
        if args.checkpoint == "before":
            metrics = runs_root / str(model["name"]) / "base" / "guard" / "metrics.json"
        else:
            metrics = runs_root / str(model["name"]) / f"{args.method}_{args.run_mode}" / "guard" / "metrics.json"
        out = metrics.parent / "nvidia_report.json"
        execute([sys.executable, str(ROOT / "nvidia_report.py"), "--metrics", str(metrics), "--out", str(out)], args.dry_run)
    else:
        model = selected_model(config, args.model)
        if args.command == "evaluate":
            backend = args.backend or str(config["evaluation"].get("backend", "transformers"))
            decoding_profile = args.decoding_profile or str(config["evaluation"].get("decoding_profile", "greedy"))
            evaluate(config, model, args.checkpoint, args.method, args.run_mode, args.limit, args.sample, args.dry_run, backend, decoding_profile, args.output_tag, getattr(args, "gpus", 1))
        elif args.command == "likelihood":
            likelihood(config, model, args.checkpoint, args.method, args.run_mode, args.limit, args.sample, args.dry_run)
        elif args.command == "train":
            train(
                config,
                model,
                args.method,
                args.mode,
                args.resume,
                args.dry_run,
                getattr(args, "no_checkpoints", False),
                args.gpus,
                args.nnodes,
                args.node_rank,
                args.master_addr,
                args.master_port,
                args.per_device_batch_size,
                args.gradient_accumulation_steps,
                args.epochs,
                args.learning_rate,
                args.save_steps,
                args.save_total_limit,
                args.max_length,
                args.optimizer,
                args.vram_fraction,
                args.no_final_save,
            )
        elif args.command == "train_qwen35":
            train_qwen35(
                config,
                model,
                args.method,
                args.mode,
                args.resume,
                args.dry_run,
                args.gpus,
                args.nnodes,
                args.node_rank,
                args.master_addr,
                args.master_port,
                args.per_device_batch_size,
                args.gradient_accumulation_steps,
                args.epochs,
                args.learning_rate,
                args.save_steps,
                args.save_total_limit,
                args.max_length,
                args.optimizer,
                args.vram_fraction,
                args.no_checkpoints,
                args.no_final_save,
            )


if __name__ == "__main__":
    main()
