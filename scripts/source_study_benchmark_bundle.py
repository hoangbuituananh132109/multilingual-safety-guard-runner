"""Create/install the three extra source-study benchmarks with hash checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


EXPECTED = {
    "xstest_en.jsonl": 450,
    "wildguardtest_en.jsonl": 3_408,
    "sealsbench_vi.jsonl": 26_644,
}
MANIFEST = "source_study_benchmark_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def nonempty_lines(value: bytes) -> int:
    return sum(1 for line in value.splitlines() if line.strip())


def create(source: Path, output: Path) -> dict:
    files = [*EXPECTED, MANIFEST]
    members = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            value = (source / name).read_bytes()
            if name in EXPECTED and nonempty_lines(value) != EXPECTED[name]:
                raise ValueError(f"{name}: expected {EXPECTED[name]} rows, got {nonempty_lines(value)}")
            archive.writestr(name, value)
            members.append({"name": name, "bytes": len(value), "sha256": sha256_bytes(value)})
        bundle = {"schema_version": 1, "members": members, "expected_rows": EXPECTED}
        archive.writestr("bundle_manifest.json", json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    result = {**bundle, "archive": str(output), "archive_bytes": output.stat().st_size}
    result["archive_sha256"] = sha256_bytes(output.read_bytes())
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def install(archive_path: Path, output: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        bundle = json.loads(archive.read("bundle_manifest.json"))
        members = {item["name"]: item for item in bundle["members"]}
        if set(members) != set(EXPECTED) | {MANIFEST}:
            raise ValueError("Unexpected benchmark bundle members")
        output.mkdir(parents=True, exist_ok=True)
        for name, item in members.items():
            value = archive.read(name)
            if len(value) != item["bytes"] or sha256_bytes(value) != item["sha256"]:
                raise ValueError(f"Hash mismatch in archive: {name}")
            target = output / name
            if target.exists() and target.read_bytes() != value:
                raise FileExistsError(f"Refusing to overwrite different benchmark: {target}")
            target.write_bytes(value)
    return {"status": "installed", "output": str(output), "benchmarks": EXPECTED}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("create")
    pack.add_argument("--source", type=Path, default=Path("work/benchmarks"))
    pack.add_argument("--output", type=Path, default=Path("zip/source-study/source_study_benchmarks_v1.gated.zip"))
    unpack = sub.add_parser("install")
    unpack.add_argument("--zip", type=Path, required=True)
    unpack.add_argument("--output", type=Path, default=Path("work/benchmarks"))
    args = parser.parse_args()
    result = create(args.source, args.output) if args.command == "create" else install(args.zip, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
