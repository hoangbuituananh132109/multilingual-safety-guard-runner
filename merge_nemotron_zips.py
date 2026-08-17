"""Merge the three nemotron_12lang_partN.zip archives into one full 12-language zip."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ("nemotron_12lang_part1.zip", "nemotron_12lang_part2.zip", "nemotron_12lang_part3.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=ROOT / "zip")
    parser.add_argument("--out", type=Path, default=None, help="Output merged zip path (default: zip/nemotron_12lang.zip)")
    args = parser.parse_args()

    zip_root = args.zip_dir.resolve()
    missing = [name for name in PARTS if not (zip_root / name).is_file()]
    if missing:
        raise SystemExit(f"Missing part archives in {zip_root}: {', '.join(missing)}")

    out = (args.out or (zip_root / "nemotron_12lang.zip")).resolve()
    expected = {f"{lang}/{split}.jsonl" for lang in ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh", "it", "ko", "nl") for split in ("train", "valid", "test")}

    seen: set[str] = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as dst:
        for name in PARTS:
            src = zip_root / name
            with zipfile.ZipFile(src) as zf:
                for member in zf.infolist():
                    if member.filename in seen:
                        raise SystemExit(f"Duplicate entry {member.filename!r} while merging")
                    seen.add(member.filename)
                    dst.writestr(member, zf.read(member.filename))

    missing_entries = expected - seen
    if missing_entries:
        raise SystemExit(f"Merged archive missing entries: {sorted(missing_entries)}")

    print(f"Merged {len(PARTS)} parts -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Entries: {len(seen)}")
    print(f"sha256: {sha256(out)}")


if __name__ == "__main__":
    main()
