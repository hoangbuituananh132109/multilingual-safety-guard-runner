"""Create or safely install a reproducible source-study data bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "validation_report.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack(source: Path, output: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    members = []
    for name in FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        members.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    bundle = {
        "schema_version": 1,
        "source": manifest["source"],
        "train_examples": manifest["train_examples"],
        "validation_examples": manifest["validation_examples"],
        "license_note": manifest["license_note"],
        "members": members,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for member in members:
            archive.write(source / member["name"], arcname=member["name"])
        archive.writestr("bundle_manifest.json", json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    bundle.update({"archive": str(output), "archive_bytes": output.stat().st_size, "archive_sha256": sha256(output)})
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return bundle


def install(archive_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if names != set(FILES) | {"bundle_manifest.json"}:
            raise ValueError(f"Unexpected archive members: {sorted(names)}")
        bundle = json.loads(archive.read("bundle_manifest.json"))
        expected = {item["name"]: item for item in bundle["members"]}
        for info in archive.infolist():
            target = (output / info.filename).resolve()
            if output.resolve() not in target.parents and target != output.resolve():
                raise ValueError(f"Unsafe archive path: {info.filename}")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output)
    for name, item in expected.items():
        path = output / name
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Bundle verification failed: {name}")
    return {"status": "installed", "source": bundle["source"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    unpack = sub.add_parser("install")
    unpack.add_argument("--zip", type=Path, required=True)
    unpack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.source, args.output) if args.command == "create" else install(args.zip, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
