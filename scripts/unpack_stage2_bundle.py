"""Safely unpack a Stage-2 bundle and verify its file hashes."""
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
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"Refusing to overwrite non-empty directory: {output}; use --force explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        names = set(archive.namelist())
        required = {"bundle_manifest.json", "data/train.jsonl", "data/validation.jsonl", "data/manifest.json"}
        if not required.issubset(names):
            raise SystemExit("Bundle is missing required files: " + ", ".join(sorted(required - names)))
        for member in archive.infolist():
            target = (output / member.filename).resolve()
            if not target.is_relative_to(output):
                raise SystemExit(f"Refusing path-traversal archive member: {member.filename}")
        archive.extractall(output)
    bundle = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in bundle["files"].items():
        path = output / "data" / name
        if digest(path) != expected["sha256"]:
            raise SystemExit(f"SHA-256 mismatch after extraction: {path}")
    print(json.dumps({"output_dir": str(output), "verified": True, "source_status": bundle["source_manifest"].get("status")}, indent=2))


if __name__ == "__main__":
    main()
