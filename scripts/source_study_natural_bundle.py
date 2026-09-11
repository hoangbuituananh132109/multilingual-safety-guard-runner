from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.source_study_natural_data import SOURCES, validate_natural_arm


FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "validation_report.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tree(root: Path) -> None:
    for prefix in (Path(), Path("_smoke")):
        for arm in SOURCES:
            directory = root / prefix / arm
            report = validate_natural_arm(directory)
            if not report["valid"]:
                raise ValueError(f"Invalid arm {directory}: {report['errors'][:5]}")


def create(source_root: Path, output: Path) -> dict[str, Any]:
    validate_tree(source_root)
    members: dict[str, dict[str, Any]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="natural-bundle-") as temporary:
        manifest_path = Path(temporary) / "bundle_manifest.json"
        for prefix in (Path(), Path("_smoke")):
            for arm in SOURCES:
                for filename in FILES:
                    path = source_root / prefix / arm / filename
                    relative = (prefix / arm / filename).as_posix()
                    members[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        bundle_manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "members": members,
            "license_warning": "Contains gated WildGuardMix and SEA-Safeguard derivatives. Transfer privately; do not publish as a public Hugging Face dataset.",
        }
        manifest_path.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        if temporary_output.exists():
            temporary_output.unlink()
        with zipfile.ZipFile(temporary_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(manifest_path, "bundle_manifest.json")
            for relative in sorted(members):
                archive.write(source_root / Path(relative), relative)
        temporary_output.replace(output)
    return {"status": "created", "output": str(output), "sha256": sha256(output), "bytes": output.stat().st_size, "members": len(members)}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def install(bundle: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        try:
            validate_tree(output)
        except Exception as exc:
            raise FileExistsError(f"Refusing to overwrite non-empty invalid directory {output}: {exc}") from exc
        return {"status": "already_installed", "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.installing-", dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            _safe_extract(archive, staging)
        bundle_manifest_path = staging / "bundle_manifest.json"
        bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        for relative, expected in bundle_manifest["members"].items():
            path = staging / Path(relative)
            if not path.is_file() or sha256(path) != expected["sha256"]:
                raise ValueError(f"Bundle member verification failed: {relative}")
        bundle_manifest_path.unlink()
        validate_tree(staging)
        if output.exists():
            output.rmdir()
        shutil.move(str(staging), str(output))
    return {"status": "installed", "output": str(output), "bundle_sha256": sha256(bundle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/install the offline natural source-study data bundle.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--source-root", type=Path, default=Path("work/source-study-natural"))
    make.add_argument("--output", type=Path, default=Path("zip/source_study_natural_v1.gated.zip"))
    get = sub.add_parser("install")
    get.add_argument("--zip", type=Path, required=True)
    get.add_argument("--output", type=Path, default=Path("work/source-study-natural"))
    args = parser.parse_args()
    result = create(args.source_root, args.output) if args.command == "create" else install(args.zip, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
