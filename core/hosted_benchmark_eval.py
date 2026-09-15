"""Evaluate a three-field safety benchmark through a hosted HTTP model.

The client never loads a model. It extracts a local ZIP, validates rows with
``id``, ``prompt`` and ``ground_truth``, and only performs HTTP requests when
an endpoint is supplied without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


_LABEL_RE = re.compile(r"(?i)\b(safe|unsafe)\b")
_ANSWER_RE = re.compile(r"(?i)(?:answer|label|classification|prediction|output)\s*[:=-]\s*(safe|unsafe)\b")
_LANGUAGES = {"ar", "de", "en", "es", "fr", "hi", "ja", "th", "vi", "zh"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_bundle(archive: Path, output_dir: Path) -> list[Path]:
    """Extract a local ZIP while rejecting traversal and symlink members."""

    archive = Path(archive)
    output_dir = Path(output_dir)
    if not archive.is_file():
        raise FileNotFoundError(f"Missing local ZIP: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a ZIP archive: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if not name or name.startswith("/") or ":" in parts[0] or ".." in parts:
                raise ValueError(f"Unsafe ZIP member path: {info.filename!r}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink ZIP member is not allowed: {info.filename!r}")
            target = (output_dir / Path(*parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"ZIP member escapes extraction directory: {info.filename!r}") from exc
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info, "r") as source, target.open("wb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
            extracted.append(target)
    return extracted


def find_data_file(extract_dir: Path, requested: str | None = None) -> Path:
    extract_dir = Path(extract_dir).resolve()
    if requested:
        candidate = (extract_dir / requested).resolve()
        try:
            candidate.relative_to(extract_dir)
        except ValueError as exc:
            raise ValueError("--data-file must be inside --extract-dir") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing extracted data file: {candidate}")
        return candidate
    preferred = extract_dir / "qwen3_safety_benchmark_total.jsonl"
    if preferred.is_file():
        return preferred
    candidates = sorted(extract_dir.rglob("*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"Cannot choose a JSONL automatically; found {len(candidates)} files")
    return candidates[0]


def _label_from_ground_truth(value: Any) -> str:
    if isinstance(value, str):
        label = value.strip().casefold()
    elif isinstance(value, dict):
        label = str(value.get("label") or value.get("expected_output") or "").strip().casefold()
    else:
        label = ""
    if label not in {"safe", "unsafe"}:
        raise ValueError(f"ground_truth must resolve to safe/unsafe, got {value!r}")
    return label


def normalize_row(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize a three-field row or the older rich row used by the repo."""

    if not isinstance(source, dict):
        raise ValueError("benchmark row must be an object")
    identifier = str(source.get("id") or "").strip()
    prompt = str(source.get("prompt") or "").strip()
    if not identifier or not prompt:
        raise ValueError(f"row has empty id/prompt: {source!r}")
    label = _label_from_ground_truth(source.get("ground_truth", source.get("expected_output")))
    parts = identifier.split(":")
    benchmark = str(source.get("benchmark") or parts[0] or "unknown")
    view = str(source.get("view") or next((p for p in parts if p.upper() in {"P", "PR"}), "unknown")).upper()
    language = str(source.get("language") or next((p for p in reversed(parts) if p.casefold() in _LANGUAGES), "unknown")).casefold()
    return {
        "id": identifier,
        "prompt": prompt,
        "expected_output": label,
        "benchmark": benchmark,
        "language": language,
        "view": view,
        "subset": source.get("subset"),
    }


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            yield normalize_row(source)


def parse_binary_output(text: str) -> tuple[str | None, str]:
    stripped = str(text or "").strip().casefold()
    if stripped in {"safe", "unsafe"}:
        return stripped, "strict"
    answer_matches = _ANSWER_RE.findall(str(text or ""))
    if answer_matches:
        return answer_matches[-1].casefold(), "repaired"
    matches = _LABEL_RE.findall(str(text or ""))
    if matches:
        return matches[-1].casefold(), "repaired"
    return None, "parse_error"


def _text_from_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                content = message["content"]
                if isinstance(content, list):
                    return "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
                return str(content)
            if first.get("text") is not None:
                return str(first["text"])
    for key in ("output", "response", "text", "content"):
        if payload.get(key) is not None:
            value = payload[key]
            return str(value) if not isinstance(value, list) else "".join(str(x) for x in value)
    return None


def parse_host_response(payload: Any) -> str | None:
    return _text_from_payload(payload)


def build_request(endpoint: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    del endpoint
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if model:
        body["model"] = model
    return body


def _post_json(endpoint: str, body: dict[str, Any], api_key: str | None, timeout: float) -> Any:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _percentage(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 4)


def _group_metric(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = len(rows)
    parsed = [row for row in rows if row.get("prediction") in {"safe", "unsafe"}]
    y = [1 if row.get("expected_output") == "unsafe" else 0 for row in rows]
    p = [1 if row.get("prediction") == "unsafe" else 0 for row in rows]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    safe_total, unsafe_total = tn + fp, tp + fn
    safe_recall = tn / safe_total if safe_total else None
    unsafe_recall = tp / unsafe_total if unsafe_total else None
    safe_precision = tn / (tn + fn) if tn + fn else 0.0
    unsafe_precision = tp / (tp + fp) if tp + fp else 0.0
    safe_f1 = 2 * safe_precision * safe_recall / (safe_precision + safe_recall) if safe_recall is not None and safe_precision + safe_recall else 0.0
    unsafe_f1 = 2 * unsafe_precision * unsafe_recall / (unsafe_precision + unsafe_recall) if unsafe_recall is not None and unsafe_precision + unsafe_recall else 0.0
    balanced = (safe_recall + unsafe_recall) / 2 if safe_recall is not None and unsafe_recall is not None else None
    return {
        "group": group,
        "examples": examples,
        "parsed": len(parsed),
        "parse_rate": _percentage(len(parsed) / examples if examples else None),
        "accuracy": _percentage((tn + tp) / examples if examples else None),
        "balanced_accuracy": _percentage(balanced),
        "safe_recall": _percentage(safe_recall),
        "unsafe_recall": _percentage(unsafe_recall),
        "safe_f1": _percentage(safe_f1),
        "unsafe_f1": _percentage(unsafe_f1),
        "macro_f1": _percentage((safe_f1 + unsafe_f1) / 2),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('benchmark', 'unknown')}|{row.get('language', 'unknown')}|{row.get('view', 'unknown')}"].append(row)
    result = [_group_metric(group, members) for group, members in sorted(groups.items())]
    if result:
        result.insert(0, _group_metric("ALL", rows))
    return result


def _api_error_text(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read(1000).decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return f"HTTP {exc.code}: {detail}".strip()
    return f"{type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = str(args.endpoint or os.environ.get("HOSTED_MODEL_ENDPOINT", "")).strip()
    if not args.dry_run and not endpoint:
        raise ValueError("Provide --endpoint (or HOSTED_MODEL_ENDPOINT), or use --dry-run")
    extracted = extract_bundle(Path(args.zip), Path(args.extract_dir))
    data_path = find_data_file(Path(args.extract_dir), args.data_file)
    rows: list[dict[str, Any]] = []
    seen_per_benchmark: Counter[str] = Counter()
    for row in read_rows(data_path):
        benchmark = str(row["benchmark"])
        if args.benchmark and benchmark not in args.benchmark:
            continue
        if args.limit_per_benchmark is not None and seen_per_benchmark[benchmark] >= args.limit_per_benchmark:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        rows.append(row)
        seen_per_benchmark[benchmark] += 1
    if not rows:
        raise ValueError("No rows selected after filters")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        smoke = {
            "dry_run": True,
            "zip": str(Path(args.zip).resolve()),
            "zip_sha256": sha256(Path(args.zip)),
            "extracted_files": [str(p) for p in extracted],
            "data_file": str(data_path.resolve()),
            "selected_examples": len(rows),
            "benchmarks": dict(sorted(seen_per_benchmark.items())),
            "sample_request": build_request(endpoint, args.model, rows[0]["prompt"], args.max_tokens),
            "network_called": False,
        }
        (output_dir / "dry_run.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return smoke

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    predictions_path = output_dir / "predictions.jsonl"
    records: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            body = build_request(args.endpoint, args.model, row["prompt"], args.max_tokens)
            raw_output = ""
            prediction: str | None = None
            parse_status = "request_error"
            error = None
            for attempt in range(args.retries + 1):
                try:
                    payload = _post_json(endpoint, body, api_key, args.timeout)
                    raw_output = parse_host_response(payload) or json.dumps(payload, ensure_ascii=False)
                    prediction, parse_status = parse_binary_output(raw_output)
                    break
                except Exception as exc:
                    error = _api_error_text(exc)
                    if attempt < args.retries:
                        time.sleep(min(30.0, 2.0**attempt))
            record = {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "language": row["language"],
                "view": row["view"],
                "expected_output": row["expected_output"],
                "prediction": prediction,
                "parse_status": parse_status,
                "raw_output": raw_output,
            }
            if error:
                record["request_error"] = error
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if args.progress_every and index % args.progress_every == 0:
                print(f"processed={index}/{len(rows)}", flush=True)
    metrics = metric_rows(records)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if metrics:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    manifest = {
        "contract": "three_field_binary_safety_v1",
        "endpoint": endpoint,
        "model": args.model,
        "zip": str(Path(args.zip).resolve()),
        "zip_sha256": sha256(Path(args.zip)),
        "data_file": str(data_path.resolve()),
        "data_sha256": sha256(data_path),
        "examples": len(records),
        "benchmarks": dict(sorted(seen_per_benchmark.items())),
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "output_sha256": sha256(predictions_path),
        "network_policy": "hosted endpoint only; no model download or local model loading",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate a local ZIP benchmark through a hosted model endpoint")
    parser.add_argument("--zip", required=True, help="Local ZIP containing the JSONL benchmark")
    parser.add_argument("--extract-dir", default="work/eval-hosted/benchmark-total")
    parser.add_argument("--data-file", help="JSONL path inside extract-dir; auto-detected by default")
    parser.add_argument("--endpoint", default="", help="OpenAI-compatible endpoint; empty only with --dry-run")
    parser.add_argument("--model", default="", help="Hosted model name, if required by endpoint")
    parser.add_argument("--output-dir", default="runs/hosted-benchmark")
    parser.add_argument("--benchmark", action="append", help="Restrict to benchmark prefix; may repeat")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-benchmark", type=int)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--api-key-env", default="HOSTED_MODEL_API_KEY")
    parser.add_argument("--dry-run", action="store_true", help="Extract/validate and write sample request; never call endpoint")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit_per_benchmark is not None and args.limit_per_benchmark < 1:
        raise ValueError("--limit-per-benchmark must be positive")
    if args.max_tokens < 1 or args.timeout <= 0 or args.retries < 0:
        raise ValueError("max-tokens/timeout/retries have invalid values")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
