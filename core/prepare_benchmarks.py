from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LANGUAGE_DIRS = {
    "ar": "ar",
    "de": "de",
    "en": "en",
    "es": "sp",
    "fr": "fr",
    "hi": "hi",
    "ja": "ja",
    "zh": "zh",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            required = {"example_id", "language", "view", "prompt", "safety_label"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in {path}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            counts[f"language:{row['language']}"] += 1
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
    return {"path": str(path), "sha256": sha256_file(path), **dict(counts)}


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as value:
        for member in value.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        value.extractall(destination)


def download_xsafety(revision: str, cache_root: Path) -> tuple[Path, dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"xsafety-{revision.replace('/', '_')}.zip"
    url = f"https://codeload.github.com/Jarviswang94/Multilingual_safety_benchmark/zip/{revision}"
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    extract_root = cache_root / f"xsafety-{revision.replace('/', '_')}"
    if not extract_root.exists():
        with tempfile.TemporaryDirectory(dir=cache_root) as temp:
            temp_root = Path(temp)
            safe_extract(archive, temp_root)
            children = [path for path in temp_root.iterdir() if path.is_dir()]
            if len(children) != 1:
                raise ValueError(f"Unexpected XSafety archive structure: {children}")
            shutil.move(str(children[0]), extract_root)
    return extract_root, {"url": url, "revision": revision, "archive_sha256": sha256_file(archive)}


def xsafety_rows(root: Path, languages: list[str]) -> Iterable[dict[str, Any]]:
    for language in languages:
        directory_name = LANGUAGE_DIRS.get(language)
        if not directory_name:
            raise ValueError(f"XSafety does not support configured language: {language}")
        language_root = root / directory_name
        if not language_root.exists():
            raise FileNotFoundError(f"XSafety language directory is missing: {language_root}")
        files = sorted(path for path in language_root.glob("*.csv") if "sense" not in path.stem.lower())
        if not files:
            raise FileNotFoundError(f"No XSafety CSV files found in {language_root}")
        for source_path in files:
            category = source_path.stem
            with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for line_number, cells in enumerate(csv.reader(handle), 1):
                    prompt = ",".join(cells).strip()
                    if not prompt:
                        continue
                    source_id = hashlib.sha256(
                        f"{language}:{source_path.name}:{line_number}:{prompt}".encode("utf-8")
                    ).hexdigest()[:20]
                    yield {
                        "example_id": f"xsafety:{source_id}:P:{language}",
                        "source_id": source_id,
                        "source_split": "test",
                        "dataset_source": "xsafety",
                        "language": language,
                        "view": "P",
                        "text": f"Prompt: {prompt}",
                        "prompt": prompt,
                        "response": None,
                        "safety_label": "unsafe",
                        "prompt_safety_label": "unsafe",
                        "categories": [category],
                        "category_scope": "prompt",
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the three-benchmark evaluation suite.")
    parser.add_argument("--cultureguard-test", type=Path, required=True)
    parser.add_argument("--sea-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xsafety-root", type=Path)
    parser.add_argument("--download-xsafety", action="store_true")
    parser.add_argument("--xsafety-revision", default="main")
    parser.add_argument("--xsafety-cache", type=Path, default=Path("/cache/benchmarks"))
    parser.add_argument("--xsafety-languages", nargs="+", default=list(LANGUAGE_DIRS))
    args = parser.parse_args()

    xsafety_root = args.xsafety_root
    xsafety_source: dict[str, Any] = {"local_root": str(xsafety_root) if xsafety_root else None}
    if args.download_xsafety:
        xsafety_root, xsafety_source = download_xsafety(args.xsafety_revision, args.xsafety_cache)
    if xsafety_root is None:
        raise ValueError("Provide --xsafety-root or --download-xsafety")

    outputs = {
        "cultureguard": args.output_dir / "cultureguard_test_9lang.jsonl",
        "sea_vi": args.output_dir / "sea_safeguard_vi.jsonl",
        "xsafety": args.output_dir / "xsafety_multilingual.jsonl",
    }
    manifest = {
        "suite": "safety_guard_three_benchmarks_v1",
        "notes": {
            "cultureguard": "Official held-out test split; prompt and response guard classification.",
            "sea_vi": "Vietnamese-only SEA-HELM safeguard rows; prompt and response guard classification.",
            "xsafety": "Unsafe-prompt recall diagnostic; XSafety has no Vietnamese and is not a balanced classification set.",
        },
        "sources": {
            "cultureguard_test": str(args.cultureguard_test),
            "sea_source": str(args.sea_source),
            "xsafety": xsafety_source,
        },
        "benchmarks": {},
    }
    manifest["benchmarks"]["cultureguard"] = write_jsonl(outputs["cultureguard"], read_jsonl(args.cultureguard_test))
    manifest["benchmarks"]["sea_vi"] = write_jsonl(
        outputs["sea_vi"],
        (row for row in read_jsonl(args.sea_source) if str(row.get("language")) == "vi"),
    )
    manifest["benchmarks"]["xsafety"] = write_jsonl(
        outputs["xsafety"],
        xsafety_rows(xsafety_root, args.xsafety_languages),
    )
    manifest_path = args.output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
