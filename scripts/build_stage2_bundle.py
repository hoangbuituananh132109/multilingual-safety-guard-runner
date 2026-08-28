"""Create a portable Stage-2 data bundle without copying model weights or tests."""
from __future__ import annotations

import argparse
import json
import zipfile
from hashlib import sha256
from pathlib import Path


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-full-ready", action="store_true", help="Also require benchmark leakage inputs, not just complete training sources.")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness_key = "full_ready" if args.require_full_ready else "training_ready"
    if not manifest.get(readiness_key) and not args.allow_incomplete:
        raise SystemExit(
            f"Refusing to bundle a dataset without {readiness_key}=true. Inspect manifest blockers "
            "or pass --allow-incomplete for a disposable smoke transfer."
        )
    files = [data_dir / "train.jsonl", data_dir / "validation.jsonl", manifest_path]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle_manifest = {
        "bundle_schema": 1,
        "source_manifest": manifest,
        "files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size} for path in files},
        "contains_model_weights": False,
        "contains_benchmark_tests": False,
        "note": "Training-only rendered data. WildGuard and benchmark test data are not embedded unless present in the source manifest.",
    }
    readme = (
        "# Stage-2 rendered training bundle\n\n"
        "This archive contains rendered train/validation JSONL and its manifest. "
        "It intentionally excludes model weights and benchmark test data. Verify the "
        "SHA-256 values in bundle_manifest.json before training.\n"
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(data_dir / "train.jsonl", "data/train.jsonl")
        archive.write(data_dir / "validation.jsonl", "data/validation.jsonl")
        archive.write(manifest_path, "data/manifest.json")
        archive.writestr("bundle_manifest.json", json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n")
        archive.writestr("README.md", readme)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, **bundle_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
