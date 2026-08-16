from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from unpack_zips import ROOT, SPECS, match_archive, validate_destination


def package_status() -> list[dict[str, Any]]:
    names = ("torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes", "scikit-learn", "PyYAML")
    result = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result.append({"name": name, "version": version, "ready": version is not None})
    return result


def gpu_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"ready": False, "error": str(error)}
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    return {
        "ready": bool(devices),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported() if devices else False,
        "devices": devices,
    }


def inputs_status(zip_root: Path) -> list[dict[str, Any]]:
    result = []
    for spec in SPECS:
        destination = ROOT / spec.destination
        ready, detail = validate_destination(destination, spec)
        archive = match_archive(zip_root, spec)
        result.append({
            "key": spec.key,
            "ready": ready,
            "detail": detail,
            "destination": str(destination),
            "zip": str(archive) if archive else None,
        })
    return result


def prepared_status(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected = (
        ("training_train", Path(config["data"]["output_dir"]) / "train.jsonl"),
        ("training_valid", Path(config["data"]["output_dir"]) / "valid.jsonl"),
        ("training_test", Path(config["data"]["output_dir"]) / "test.jsonl"),
        ("training_manifest", Path(config["data"]["output_dir"]) / "manifest.json"),
        ("sea_vi", Path(config["benchmarks"]["sea_output"])),
        ("benchmark_manifest", Path(config["benchmarks"]["output_dir"]) / "benchmark_manifest.json"),
    )
    return [{"key": key, "ready": path.is_file(), "path": str(path), "bytes": path.stat().st_size if path.is_file() else None} for key, path in expected]


def runs_status(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["paths"]["runs_root"])
    result = []
    for model in config["models"]:
        name = str(model["name"])
        base = root / name / "base"
        result.append({"model": name, "run": "base", "guard_eval": (base / "guard" / "metrics.json").is_file(), "likelihood": (base / "likelihood" / "likelihood_metrics.json").is_file()})
        for method in ("lora", "full"):
            for mode in ("smoke", "pilot", "full"):
                path = root / name / f"{method}_{mode}"
                final = path / "final"
                result.append({
                    "model": name,
                    "run": f"{method}_{mode}",
                    "started": (path / "resolved_config.json").is_file(),
                    "complete": final.is_dir() and any(final.iterdir()),
                    "guard_eval": (path / "guard" / "metrics.json").is_file(),
                    "likelihood": (path / "likelihood" / "likelihood_metrics.json").is_file(),
                    "path": str(path),
                })
    return result


def next_actions(report: dict[str, Any]) -> list[str]:
    actions = []
    missing_packages = [item["name"] for item in report["packages"] if not item["ready"]]
    if missing_packages:
        actions.append("Container is missing packages: " + ", ".join(missing_packages))
    if not report["gpu"]["ready"]:
        actions.append("GPU is not visible from the current Python environment")
    required_inputs = {"nemotron_9lang", "sea_helm", "xsafety"}
    missing_inputs = [item["key"] for item in report["inputs"] if item["key"] in required_inputs and not item["ready"]]
    if missing_inputs:
        actions.append("Place and unpack required archives: " + ", ".join(missing_inputs))
    if not all(item["ready"] for item in report["prepared"]):
        actions.append("Run: python run.py prepare")
    ready_models = [item["key"] for item in report["inputs"] if item.get("kind") == "model" and item["ready"]]
    if not ready_models:
        actions.append("Unpack at least one model before evaluation or training")
    return actions


def render_text(report: dict[str, Any]) -> str:
    lines = ["SAFETY GUARD WORKSPACE STATUS", ""]
    lines.append(f"Python: {report['python']['executable']}")
    lines.append(f"GPU ready: {report['gpu']['ready']}")
    lines.append("")
    lines.append("INPUTS")
    for item in report["inputs"]:
        lines.append(f"[{'OK' if item['ready'] else 'MISSING'}] {item['key']} -> {item['destination']}")
    lines.append("")
    lines.append("PREPARED DATA")
    for item in report["prepared"]:
        lines.append(f"[{'OK' if item['ready'] else 'MISSING'}] {item['key']} -> {item['path']}")
    lines.append("")
    lines.append("NEXT ACTIONS")
    for action in report["next_actions"] or ["No blocking preparation step detected"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    previous = Path.cwd()
    try:
        import os

        os.chdir(config_path.parent)
        inputs = inputs_status(Path(config["paths"]["zip_root"]))
        spec_by_key = {spec.key: spec for spec in SPECS}
        for item in inputs:
            item["kind"] = spec_by_key[item["key"]].kind
        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
            "packages": package_status(),
            "gpu": gpu_status(),
            "inputs": inputs,
            "prepared": prepared_status(config),
            "runs": runs_status(config),
        }
        report["next_actions"] = next_actions(report)
        work = Path(config["paths"]["work_root"])
        work.mkdir(parents=True, exist_ok=True)
        (work / "status_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text = render_text(report)
        (work / "status_report.txt").write_text(text, encoding="utf-8")
        print(text)
    finally:
        import os

        os.chdir(previous)


if __name__ == "__main__":
    main()
