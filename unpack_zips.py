from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3


@dataclass(frozen=True)
class ArchiveSpec:
    key: str
    names: tuple[str, ...]
    destination: str
    marker: str
    kind: str


SPECS = (
    ArchiveSpec("nemotron_9lang", ("nemotron_9lang.zip", "nemotron-9lang.zip"), "input/nemotron", "en/train.jsonl", "nemotron"),
    ArchiveSpec("sea_helm", ("sea-helm.zip", "sea-helm-*.zip"), "input/sea-helm", "seahelm_tasks/task_config.yaml", "sea"),
    ArchiveSpec("xsafety", ("xsafety.zip", "xsafety-*.zip", "multilingual_safety_benchmark-*.zip"), "input/xsafety", "en", "xsafety"),
    ArchiveSpec("qwen3_8b", ("qwen3-8b.zip",), "input/models/qwen3-8b", "config.json", "model"),
    ArchiveSpec("qwen3_4b", ("qwen3-4b.zip",), "input/models/qwen3-4b", "config.json", "model"),
    ArchiveSpec("qwen3_1_7b", ("qwen3-1.7b.zip", "qwen3-1_7b.zip"), "input/models/qwen3-1.7b", "config.json", "model"),
    ArchiveSpec("qwen3_0_6b", ("qwen3-0.6b.zip", "qwen3-0_6b.zip"), "input/models/qwen3-0.6b", "config.json", "model"),
    ArchiveSpec("llama31_8b", ("llama31-8b-instruct.zip", "llama-3.1-8b-instruct.zip"), "input/models/llama31-8b-instruct", "config.json", "model"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def match_archive(zip_root: Path, spec: ArchiveSpec) -> Path | None:
    patterns = tuple(value.casefold() for value in spec.names)
    matches = [path.resolve() for path in zip_root.glob("*.zip") if path.is_file() and any(fnmatch.fnmatch(path.name.casefold(), pattern) for pattern in patterns)]
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(f"Multiple ZIP files match {spec.key}: {unique}")
    return unique[0] if unique else None


def validate_members(archive: Path) -> dict[str, int]:
    entries = 0
    compressed = 0
    uncompressed = 0
    with zipfile.ZipFile(archive) as value:
        if len(value.infolist()) > 200000:
            raise ValueError(f"Too many entries in {archive}")
        for member in value.infolist():
            normalized = member.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
            unix_mode = member.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ValueError(f"Symbolic links are not allowed in {archive}: {member.filename}")
            if member.file_size < 0 or member.compress_size < 0:
                raise ValueError(f"Invalid size in {archive}: {member.filename}")
            entries += 1
            compressed += member.compress_size
            uncompressed += member.file_size
    if uncompressed > 100 * GIB:
        raise ValueError(f"Archive is larger than 100 GiB after extraction: {archive}")
    if compressed and uncompressed / compressed > 10000:
        raise ValueError(f"Suspicious compression ratio in {archive}")
    return {"entries": entries, "compressed_bytes": compressed, "uncompressed_bytes": uncompressed}


def find_root(extracted: Path, marker: str) -> Path:
    marker_path = Path(marker)
    direct = extracted / marker_path
    if direct.exists():
        return extracted
    matches = [path for path in extracted.rglob(marker_path.name) if path.as_posix().endswith(marker_path.as_posix())]
    matches = [path for path in matches if "__MACOSX" not in path.parts and ".cache" not in path.parts]
    roots: list[Path] = []
    for match in matches:
        candidate = match
        for _ in marker_path.parts:
            candidate = candidate.parent
        roots.append(candidate)
    roots = sorted(set(roots), key=lambda value: (len(value.parts), value.as_posix()))
    if len(roots) != 1:
        raise ValueError(f"Expected one root containing {marker}, found {roots}")
    return roots[0]


def validate_destination(path: Path, spec: ArchiveSpec) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if spec.kind == "nemotron":
        missing = [f"{language}/{split}.jsonl" for language in ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh") for split in ("train", "valid", "test") if not (path / language / f"{split}.jsonl").is_file()]
        return (not missing, "ok" if not missing else "missing: " + ", ".join(missing))
    if spec.kind == "sea":
        marker = path / "seahelm_tasks" / "task_config.yaml"
        return marker.is_file(), "ok" if marker.is_file() else f"missing: {marker}"
    if spec.kind == "xsafety":
        csv_files = list((path / "en").glob("*.csv")) if (path / "en").is_dir() else []
        return bool(csv_files), "ok" if csv_files else "missing English CSV files"
    config = path / "config.json"
    tokenizer = (path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()
    weights = list(path.glob("*.safetensors"))
    try:
        config_value = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else None
    except (OSError, json.JSONDecodeError):
        config_value = None
    def valid_safetensors_header(value: Path) -> bool:
        try:
            with value.open("rb") as handle:
                header_length = int.from_bytes(handle.read(8), byteorder="little", signed=False)
                if header_length < 2 or header_length > min(100 * 1024 * 1024, value.stat().st_size - 8):
                    return False
                header = json.loads(handle.read(header_length).decode("utf-8"))
                return isinstance(header, dict)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False

    valid_weights = bool(weights) and all(valid_safetensors_header(value) for value in weights)
    valid = isinstance(config_value, dict) and tokenizer and valid_weights
    return valid, "ok" if valid else "model requires a valid config.json, tokenizer, and valid safetensors at its root"


def backup_destination(destination: Path, work_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = work_root / "unpack_backups" / f"{destination.name}-{stamp}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(destination), str(backup))
    return backup


def extract_one(archive: Path, spec: ArchiveSpec, work_root: Path, replace: bool) -> dict[str, Any]:
    destination = (ROOT / spec.destination).resolve()
    if ROOT.resolve() not in destination.parents:
        raise ValueError(f"Unsafe destination: {destination}")
    valid, message = validate_destination(destination, spec)
    if valid and not replace:
        return {"key": spec.key, "status": "already_ready", "destination": str(destination), "validation": message}
    backup = None
    if destination.exists():
        if not replace:
            raise ValueError(f"Destination exists but is invalid: {destination}: {message}. Use --replace to move it to a backup.")
    archive_stats = validate_members(archive)
    temp_parent = work_root / "unpack_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"{spec.key}-", dir=temp_parent))
    try:
        with zipfile.ZipFile(archive) as value:
            value.extractall(temp_root)
        source_root = find_root(temp_root, spec.marker)
        valid, message = validate_destination(source_root, spec)
        if not valid:
            raise ValueError(f"Extracted content failed validation: {message}")
        if destination.exists():
            backup = backup_destination(destination, work_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_root), str(destination))
        valid, message = validate_destination(destination, spec)
        if not valid:
            raise ValueError(f"Extracted destination failed validation: {message}")
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    return {
        "key": spec.key,
        "status": "extracted",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "destination": str(destination),
        "backup": str(backup) if backup else None,
        "validation": message,
        **archive_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-dir", type=Path, default=ROOT / "zip")
    parser.add_argument("--only", choices=[spec.key for spec in SPECS], action="append")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    zip_root = args.zip_dir.resolve()
    zip_root.mkdir(parents=True, exist_ok=True)
    for relative in ("input/nemotron", "input/sea-helm", "input/xsafety", "input/models", "work", "runs"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    selected = [spec for spec in SPECS if not args.only or spec.key in args.only]
    report: dict[str, Any] = {"zip_root": str(zip_root), "results": [], "missing": []}
    for spec in selected:
        archive = match_archive(zip_root, spec)
        destination = (ROOT / spec.destination).resolve()
        valid, message = validate_destination(destination, spec)
        if archive is None:
            if valid:
                report["results"].append({"key": spec.key, "status": "already_ready", "destination": str(destination), "validation": message})
            else:
                report["missing"].append({"key": spec.key, "accepted_names": list(spec.names), "destination": str(destination)})
            continue
        report["results"].append(extract_one(archive, spec, ROOT / "work", args.replace))
    report_path = ROOT / "work" / "unpack_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report["missing"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
