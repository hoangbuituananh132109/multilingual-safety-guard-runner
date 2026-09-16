"""Offline vLLM evaluator for the Qwen3-235B binary safety bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .qwen235b_eval_bundle import QWEN235B_SYSTEM_PROMPT, sha256


_LABEL_RE = re.compile(r"(?i)\b(safe|unsafe)\b")
_ANSWER_RE = re.compile(r"(?i)(?:answer|label|classification|prediction|output)\s*[:=-]\s*(safe|unsafe)\b")


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


def _percentage(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 4)


def _group_metric(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = len(rows)
    parsed = [row for row in rows if row.get("prediction") in {"safe", "unsafe"}]
    # A malformed model answer is counted as the opposite of the gold label
    # for the conservative full-set metric, while parse_rate exposes it.
    y = [1 if row.get("expected_output") == "unsafe" else 0 for row in rows]
    p = []
    for row, gold in zip(rows, y):
        prediction = row.get("prediction")
        if prediction not in {"safe", "unsafe"}:
            prediction = "safe" if gold == 1 else "unsafe"
        p.append(1 if prediction == "unsafe" else 0)
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y, p))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(y, p))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y, p))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y, p))
    safe_total = tn + fp
    unsafe_total = tp + fn
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
        "safe_examples": safe_total,
        "unsafe_examples": unsafe_total,
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
        benchmark = str(row.get("benchmark") or "unknown")
        language = str(row.get("language") or "unknown")
        view = str(row.get("view") or "unknown")
        groups[f"{benchmark}|{language}|{view}"].append(row)
    if not groups:
        return []
    result = [_group_metric(group, members) for group, members in sorted(groups.items())]
    result.insert(0, _group_metric("ALL", rows))
    return result


def _read_bundle(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Bundle row is not an object at {path}:{line_number}")
            yield row


def _render_messages(tokenizer: Any, messages: list[dict[str, str]], thinking_mode: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking_mode == "think", **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _expected_output(row: dict[str, Any]) -> str:
    """Read the normalized gold label from either supported bundle schema.

    The original Qwen bundle stores ``expected_output`` and ``messages``.
    The unified benchmark transfer bundle intentionally stores only
    ``id``, ``prompt`` and ``ground_truth``.  Supporting both here keeps the
    evaluator usable with the exact aggregate ZIP without rebuilding it.
    """
    value: Any = row.get("expected_output")
    if value is None:
        value = row.get("ground_truth")
        if isinstance(value, dict):
            value = value.get("label")
            if value is None and "binary" in row.get("ground_truth", {}):
                value = "unsafe" if int(row["ground_truth"]["binary"]) else "safe"
    label = str(value or "").strip().casefold()
    if label not in {"safe", "unsafe"}:
        raise ValueError(f"Bundle row {row.get('id')} has invalid ground truth: {value!r}")
    return label


def _render_row_prompt(tokenizer: Any, row: dict[str, Any], thinking_mode: str) -> str:
    """Render either message-based rows or the aggregate bundle's raw prompt."""
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        return _render_messages(tokenizer, messages, thinking_mode)
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        # Aggregate transfer rows already contain the complete classifier
        # instruction, conversation, target marker, and output marker.
        return prompt
    raise ValueError(f"Bundle row {row.get('id')} has neither messages nor prompt")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    bundle_path = Path(args.bundle)
    output_dir = Path(args.output_dir)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing local Qwen235B bundle: {bundle_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, seed=args.seed)
    predictions_path = output_dir / "predictions.jsonl"
    records: list[dict[str, Any]] = []
    seen_per_benchmark: dict[str, int] = defaultdict(int)
    total_seen = 0
    pending_rows: list[dict[str, Any]] = []
    pending_prompts: list[str] = []
    prediction_handle: Any = None

    def flush() -> None:
        if not pending_rows:
            return
        outputs = llm.generate(pending_prompts, sampling)
        for row, result in zip(pending_rows, outputs):
            raw = result.outputs[0].text if result.outputs else ""
            prediction, parse_status = parse_binary_output(raw)
            record = {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "language": row.get("language"),
                "view": row.get("view"),
                "subset": row.get("subset"),
                "expected_output": row["expected_output"],
                "prediction": prediction,
                "parse_status": parse_status,
                "raw_output": raw,
            }
            records.append(record)
            if prediction_handle is not None:
                prediction_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                prediction_handle.flush()
        pending_rows.clear()
        pending_prompts.clear()

    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        prediction_handle = handle
        for row in _read_bundle(bundle_path):
            benchmark = str(row.get("benchmark") or "unknown")
            if args.benchmark and benchmark not in args.benchmark:
                continue
            if args.limit_per_benchmark is not None and seen_per_benchmark[benchmark] >= args.limit_per_benchmark:
                continue
            if args.limit is not None and total_seen >= args.limit:
                break
            expected_output = _expected_output(row)
            pending_rows.append(row)
            pending_prompts.append(_render_row_prompt(tokenizer, row, args.thinking_mode))
            row["expected_output"] = expected_output
            seen_per_benchmark[benchmark] += 1
            total_seen += 1
            if len(pending_rows) >= args.batch_size:
                flush()
        flush()

    metrics = metric_rows(records)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if metrics:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    manifest = {
        "contract": "qwen3_235b_binary_v1",
        "model": str(args.model),
        "bundle": str(bundle_path.resolve()),
        "bundle_sha256": sha256(bundle_path),
        "system_prompt_sha256": hashlib.sha256(QWEN235B_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "thinking_mode": args.thinking_mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "examples": len(records),
        "benchmarks": dict(sorted(seen_per_benchmark.items())),
        "output_sha256": sha256(predictions_path),
        "offline_policy": "local model and local bundle only",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-235B on the local unified safety bundle with vLLM")
    parser.add_argument("--model", required=True, help="Local model directory; no Hub lookup is attempted")
    parser.add_argument("--bundle", type=Path, default=Path("work/eval-qwen235b/qwen235b_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen235b-eval"))
    parser.add_argument("--benchmark", action="append", help="Restrict to a benchmark name; may be repeated")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-benchmark", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--thinking-mode", choices=["no_think", "think"], default="no_think")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit_per_benchmark is not None and args.limit_per_benchmark < 1:
        raise ValueError("--limit-per-benchmark must be positive")
    result = run_evaluation(args)
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
