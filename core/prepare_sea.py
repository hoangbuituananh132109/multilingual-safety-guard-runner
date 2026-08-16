from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as value:
        for member in value.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        value.extractall(destination)


def download_repo(repo: str, revision: str, cache_root: Path) -> tuple[Path, dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = f"{repo.replace('/', '--')}--{revision}"
    archive = cache_root / f"{safe_name}.zip"
    url = f"https://codeload.github.com/{repo}/zip/{revision}"
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=180) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    extract_root = cache_root / safe_name
    if not extract_root.exists():
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            temporary_root = Path(temporary)
            safe_extract(archive, temporary_root)
            children = [path for path in temporary_root.iterdir() if path.is_dir()]
            if len(children) != 1:
                raise ValueError(f"Unexpected SEA-HELM archive structure: {children}")
            shutil.move(str(children[0]), extract_root)
    return extract_root, {
        "repo": repo,
        "revision": revision,
        "url": url,
        "archive_sha256": sha256_file(archive),
    }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield line_number, value


def normalize_label(value: Any) -> str:
    label = " ".join(str(value or "").strip().casefold().split())
    mapping = {"safe": "safe", "harmful": "unsafe", "an toàn": "safe", "có hại": "unsafe"}
    if label not in mapping:
        raise ValueError(f"Unsupported SEA Safeguard label: {value!r}")
    return mapping[label]


def canonical(
    subset: str,
    line_number: int,
    view: str,
    prompt: str,
    response: str | None,
    label: str,
    prompt_label: str,
    source_path: Path,
    repo_root: Path,
    revision: str,
    task_name: str,
    registered_in_suite: bool,
) -> dict[str, Any]:
    if not prompt.strip() or (view == "PR" and not str(response or "").strip()):
        raise ValueError(f"Empty SEA text: {subset}:{line_number}:{view}")
    return {
        "example_id": f"sea:{subset}:{line_number:04d}:{view}:vi",
        "dataset_source": "sea_helm_safeguard",
        "source_revision": revision,
        "source_path": source_path.relative_to(repo_root).as_posix(),
        "source_line": line_number,
        "source_split": "evaluation",
        "seahelm_task_name": task_name,
        "registered_in_pinned_seahelm_vi_suite": registered_in_suite,
        "subset": subset,
        "language": "vi",
        "view": view,
        "prompt": prompt,
        "response": response if view == "PR" else None,
        "text": f"Prompt: {prompt}" + (f"\nResponse: {response}" if view == "PR" else ""),
        "safety_label": normalize_label(label),
        "prompt_safety_label": normalize_label(prompt_label),
        "categories": [],
        "category_scope": "unavailable",
    }


def build_rows(repo_root: Path, revision: str) -> list[dict[str, Any]]:
    safeguard = repo_root / "seahelm_tasks" / "safety" / "safeguard"
    task_config_path = repo_root / "seahelm_tasks" / "task_config.yaml"
    task_config = yaml.safe_load(task_config_path.read_text(encoding="utf-8")) if task_config_path.exists() else {}
    registered_tasks = set(((task_config.get("seahelm") or {}).get("vi") or []))
    rows: list[dict[str, Any]] = []
    for subset in ("general", "cultural_content_generation"):
        source = safeguard / subset / "data" / f"vi_{subset}.jsonl"
        for line_number, row in iter_jsonl(source):
            item = (row.get("prompts") or [{}])[0]
            prompt = str(item.get("prompt_text") or "")
            response = str(item.get("response_text") or "")
            prompt_label = str(row.get("prompt_label") or "")
            prompt_task = f"safeguard_{subset}_prompt"
            response_task = f"safeguard_{subset}_response"
            rows.append(canonical(subset, line_number, "P", prompt, None, prompt_label, prompt_label, source, repo_root, revision, prompt_task, prompt_task in registered_tasks))
            rows.append(canonical(subset, line_number, "PR", prompt, response, str(row.get("response_label") or ""), prompt_label, source, repo_root, revision, response_task, response_task in registered_tasks))

    subset = "cultural_in_the_wild"
    source = safeguard / subset / "data" / "vi_cultural_in_the_wild.jsonl"
    for line_number, row in iter_jsonl(source):
        item = (row.get("prompts") or [{}])[0]
        prompt = str(item.get("local_prompt") or "")
        label = str(row.get("label") or "")
        task_name = "safeguard_cultural_in_the_wild"
        rows.append(canonical(subset, line_number, "P", prompt, None, label, label, source, repo_root, revision, task_name, task_name in registered_tasks))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pinned SEA-HELM revision and build the Vietnamese SEA Safeguard manifest.")
    parser.add_argument("--repo", default="aisingapore/SEA-HELM")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/cache/benchmarks"))
    parser.add_argument("--repo-root", type=Path, help="Use an already copied SEA-HELM tree instead of downloading.")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=1840)
    args = parser.parse_args()

    if args.repo_root:
        repo_root = args.repo_root
        source = {"repo": args.repo, "revision": args.revision, "local_root": str(repo_root)}
    else:
        repo_root, source = download_repo(args.repo, args.revision, args.cache_root)
    rows = build_rows(repo_root, args.revision)
    if args.expected_examples and len(rows) != args.expected_examples:
        raise ValueError(f"Expected {args.expected_examples:,} Vietnamese SEA examples, got {len(rows):,}; inspect the pinned upstream revision before continuing.")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate SEA example_id")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            for field in ("subset", "view", "safety_label"):
                counts[f"{field}:{row[field]}"] += 1
            counts[f"registered_in_pinned_seahelm_vi_suite:{str(row['registered_in_pinned_seahelm_vi_suite']).lower()}"] += 1
    manifest = {
        "source": source,
        "output_file": str(args.output_file),
        "output_sha256": sha256_file(args.output_file),
        "counts": dict(sorted(counts.items())),
        "protocol_note": "This file contains all Vietnamese SEA Safeguard tasks supported by the pinned source. The registered_in_pinned_seahelm_vi_suite flag distinguishes the current default suite. Official scoring still requires the pinned localized prompts and upstream harness.",
    }
    manifest_path = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
