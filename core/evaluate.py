from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[evaluate] {stamp} {message}", flush=True)

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from prompt import N23, NEMOTRON_PROMPT_TEMPLATE, render_prompt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_progress(output_dir: Path, **values: Any) -> None:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "progress.json")


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


CATEGORY_BY_NAME = {normalized_name(value): value for value in N23}
CATEGORY_BY_CODE = {f"s{index}": value for index, value in enumerate(N23, 1)}


def normalize_categories(value: Any) -> tuple[list[str], list[str]]:
    if value is None:
        return [], []
    candidates = value if isinstance(value, list) else re.split(r"[,;\n]+", str(value))
    known: list[str] = []
    unknown: list[str] = []
    for raw in candidates:
        candidate = str(raw).strip().strip(". ")
        if not candidate:
            continue
        code = re.match(r"^(S\d{1,2})\s*[:\-]?\s*(.*)$", candidate, flags=re.I)
        category = CATEGORY_BY_CODE.get(code.group(1).casefold()) if code else None
        if category is None:
            category = CATEGORY_BY_NAME.get(normalized_name(candidate))
        if category is None:
            unknown.append(candidate)
        elif category not in known:
            known.append(category)
    return known, unknown


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return None, False
    candidate = stripped[start:end + 1]
    try:
        value = json.loads(candidate)
        return (value, stripped == candidate) if isinstance(value, dict) else (None, False)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'(:\s*)(safe|unsafe)(\s*[,}])',
            lambda match: f'{match.group(1)}"{match.group(2)}"{match.group(3)}',
            candidate,
            flags=re.I,
        )
        try:
            value = json.loads(repaired)
            return (value, False) if isinstance(value, dict) else (None, False)
        except json.JSONDecodeError:
            return None, False


def parse_output(text: str, family: str, view: str) -> dict[str, Any]:
    if family == "qwen_binary":
        match = re.search(r"(?im)^\s*Safety\s*:\s*(Safe|Unsafe|Controversial)\s*$", text)
        if not match:
            return {"prediction": None, "parse_status": "PARSE_ERROR", "payload": None, "categories": [], "unknown_categories": []}
        value = match.group(1).lower()
        return {"prediction": "unsafe" if value in {"unsafe", "controversial"} else "safe", "parse_status": "strict", "payload": {"Safety": value}, "categories": [], "unknown_categories": []}
    value, strict = extract_json(text)
    if value is None:
        return {"prediction": None, "parse_status": "PARSE_ERROR", "payload": None, "categories": [], "unknown_categories": []}
    key = "Response Safety" if view == "PR" else "User Safety"
    label = str(value.get(key, "")).lower()
    categories, unknown = normalize_categories(value.get("Safety Categories"))
    return {
        "prediction": label if label in {"safe", "unsafe"} else None,
        "parse_status": "strict_json" if strict and label in {"safe", "unsafe"} else "repaired_json" if label in {"safe", "unsafe"} else "PARSE_ERROR",
        "payload": value,
        "categories": categories,
        "unknown_categories": unknown,
    }


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["prediction"] in {"safe", "unsafe"}]
    y = [1 if row["gold"] == "unsafe" else 0 for row in rows]
    p = [1 if row["prediction_for_metrics"] == "unsafe" else 0 for row in rows]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    has_both_classes = len(set(y)) == 2
    valid_y = [1 if row["gold"] == "unsafe" else 0 for row in valid]
    valid_p = [1 if row["prediction"] == "unsafe" else 0 for row in valid]
    valid_has_both = len(set(valid_y)) == 2
    return {
        "examples": len(rows), "parsed": len(valid), "parse_errors": len(rows) - len(valid),
        "parse_rate": len(valid) / len(rows) if rows else None,
        "safe_examples": sum(value == 0 for value in y), "unsafe_examples": sum(value == 1 for value in y),
        "accuracy": accuracy_score(y, p) if y else None,
        "balanced_accuracy": balanced_accuracy_score(y, p) if y and has_both_classes else None,
        "macro_f1": f1_score(y, p, average="macro", zero_division=0) if y and has_both_classes else None,
        "unsafe_f1": f1_score(y, p, pos_label=1, average="binary", zero_division=0) if y else None,
        "harmful_f1": f1_score(y, p, pos_label=1, average="binary", zero_division=0) if y else None,
        "unsafe_precision": precision_score(y, p, zero_division=0) if y else None,
        "unsafe_recall": recall_score(y, p, zero_division=0) if y else None,
        "parsed_only_accuracy": accuracy_score(valid_y, valid_p) if valid_y else None,
        "parsed_only_balanced_accuracy": balanced_accuracy_score(valid_y, valid_p) if valid_y and valid_has_both else None,
        "parsed_only_macro_f1": f1_score(valid_y, valid_p, average="macro", zero_division=0) if valid_y and valid_has_both else None,
        "parsed_only_unsafe_f1": f1_score(valid_y, valid_p, pos_label=1, average="binary", zero_division=0) if valid_y else None,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--family", choices=["nemotron", "qwen_binary"], required=True)
    parser.add_argument("--benchmark", action="append", required=True, help="NAME=/path/file.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vllm-chunk-size", type=int, default=1000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-input-tokens", type=int, default=8064)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int, help="Stratified random sample of N rows per benchmark (by language+view, fixed seed)")
    parser.add_argument("--parse-error-policy", choices=["incorrect", "unsafe", "exclude"], default="incorrect")
    args = parser.parse_args()
    if args.vllm_chunk_size < 1:
        raise ValueError("--vllm-chunk-size must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_progress(
        args.output_dir,
        status="running",
        phase="loading_tokenizer",
        backend=args.backend,
        completed=0,
        total=None,
        percent=None,
    )

    tokenizer_source = args.adapter / "tokenizer" if args.adapter and (args.adapter / "tokenizer").exists() else args.base_model
    tokenizer_kwargs = {} if args.adapter and (args.adapter / "tokenizer").exists() else {"revision": args.revision}
    log(f"loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    log(f"tokenizer loaded, vocab={getattr(tokenizer, 'vocab_size', '?')}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = None
    vllm_llm = None
    if args.backend == "vllm":
        log("backend=vllm: deferring model load until prompts are collected")
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
        log(f"loading base model from {args.base_model} (dtype=bf16, 4bit={args.load_in_4bit})...")
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision, torch_dtype=torch.bfloat16, quantization_config=quant, device_map="auto", attn_implementation="sdpa")
        if args.adapter:
            log(f"loading LoRA adapter from {args.adapter}...")
            model = PeftModel.from_pretrained(model, args.adapter)
            log("adapter loaded")
        log(f"base model loaded in {time.time()-load_start:.1f}s, device={next(model.parameters()).device}")
        model.eval()

    all_predictions: list[dict[str, Any]] = []
    vllm_batches: list[tuple[str, list[dict[str, Any]], list[str]]] = []
    total_prepared = 0
    for specification in args.benchmark:
        name, path_text = specification.split("=", 1)
        rows = read_jsonl(Path(path_text))
        if args.sample:
            import random as _random
            _rng = _random.Random(3407)
            groups: dict[tuple, list] = {}
            for row in rows:
                key = (str(row.get("language", "")), str(row.get("view", "")))
                groups.setdefault(key, []).append(row)
            sampled = []
            for key, members in groups.items():
                _rng.shuffle(members)
                n = max(1, round(args.sample * len(members) / len(rows))) if len(rows) else 0
                sampled.extend(members[:n])
            # fill remainder to reach exactly args.sample
            if len(sampled) < args.sample:
                pool = [r for r in rows if r not in sampled]
                _rng.shuffle(pool)
                sampled.extend(pool[: args.sample - len(sampled)])
            rows = sampled[:args.sample]
        elif args.limit:
            rows = rows[:args.limit]
        log(f"benchmark={name} rows={len(rows)} batch_size={args.batch_size}")
        bench_start = time.time()
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            prompts = []
            for row in batch_rows:
                prompt = row.get("prompt")
                response = row.get("response")
                if prompt is None:
                    text = str(row["text"])
                    first, marker, second = text.partition("\nResponse:")
                    prompt = first.removeprefix("Prompt:").lstrip()
                    response = second.lstrip() if marker else None
                rendered = render_prompt(tokenizer, args.family, str(prompt), str(response) if response is not None else None)
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if len(ids) > args.max_input_tokens:
                    half = args.max_input_tokens // 2
                    rendered = tokenizer.decode(ids[:half] + ids[-(args.max_input_tokens-half):], skip_special_tokens=True)
                prompts.append(rendered)
            if args.backend == "vllm":
                existing = next((b for b in vllm_batches if b[0] == name), None)
                if existing is None:
                    vllm_batches.append((name, list(batch_rows), list(prompts)))
                else:
                    existing[1].extend(batch_rows)
                    existing[2].extend(prompts)
                total_prepared += len(batch_rows)
                write_progress(
                    args.output_dir,
                    status="running",
                    phase="preparing_prompts",
                    backend=args.backend,
                    completed=total_prepared,
                    total=None,
                    percent=None,
                    current_benchmark=name,
                )
                continue
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            input_width = encoded["input_ids"].shape[1]
            for row, sequence in zip(batch_rows, generated):
                raw = tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
                parsed = parse_output(raw, args.family, str(row["view"]))
                prediction = parsed["prediction"]
                gold = str(row.get("safety_label"))
                if prediction in {"safe", "unsafe"}:
                    prediction_for_metrics = prediction
                elif args.parse_error_policy == "unsafe":
                    prediction_for_metrics = "unsafe"
                elif args.parse_error_policy == "incorrect":
                    prediction_for_metrics = "safe" if gold == "unsafe" else "unsafe"
                else:
                    prediction_for_metrics = None
                all_predictions.append({
                    "benchmark": name,
                    "example_id": row.get("example_id"),
                    "language": row.get("language"),
                    "view": row.get("view"),
                    "subset": row.get("subset") or "ALL",
                    "topic": row.get("topic"),
                    "gold": gold,
                    "prediction": prediction,
                    "prediction_for_metrics": prediction_for_metrics,
                    "parse_status": parsed["parse_status"],
                    "parsed_payload": parsed["payload"],
                    "gold_categories": list(row.get("categories") or []),
                    "predicted_categories": parsed["categories"],
                    "unknown_categories": parsed["unknown_categories"],
                    "raw_output": raw,
                })
            write_progress(
                args.output_dir,
                status="running",
                phase="generating",
                backend=args.backend,
                completed=len(all_predictions),
                total=None,
                percent=None,
                current_benchmark=name,
            )
            if len(all_predictions) % (args.batch_size * 5) == 0 or start + args.batch_size >= len(rows):
                elapsed = time.time() - bench_start
                done = min(start + args.batch_size, len(rows))
                rate = done / elapsed if elapsed > 0 else 0
                log(f"  [{name}] {done}/{len(rows)} ({100.0*done/len(rows):.1f}%) elapsed={elapsed:.0f}s rate={rate:.2f} rows/s")
        if len(rows):
            action = "prompts prepared" if args.backend == "vllm" else "done"
            log(f"  [{name}] {action} {len(rows)} rows in {time.time()-bench_start:.0f}s")

    if args.backend == "vllm" and vllm_batches:
        total_prompts = sum(len(batch_rows) for _, batch_rows, _ in vllm_batches)
        log(f"running vLLM generation on {total_prompts} prompts in chunks of {args.vllm_chunk_size}...")
        write_progress(
            args.output_dir,
            status="running",
            phase="loading_model",
            backend=args.backend,
            completed=0,
            total=total_prompts,
            percent=0.0,
        )
        from vllm import LLM, SamplingParams
        load_start = time.time()
        vllm_llm = LLM(model=args.base_model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=args.gpu_memory_utilization)
        log(f"vLLM engine loaded in {time.time()-load_start:.1f}s")
        sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
        generation_start = time.time()
        total_completed = 0
        for name, batch_rows, prompts in vllm_batches:
            gen_start = time.time()
            for start in range(0, len(prompts), args.vllm_chunk_size):
                chunk_rows = batch_rows[start:start + args.vllm_chunk_size]
                chunk_prompts = prompts[start:start + args.vllm_chunk_size]
                results = vllm_llm.generate(chunk_prompts, sampling)
                outputs = [result.outputs[0].text for result in results]
                for row, raw in zip(chunk_rows, outputs):
                    parsed = parse_output(raw, args.family, str(row["view"]))
                    prediction = parsed["prediction"]
                    gold = str(row.get("safety_label"))
                    if prediction in {"safe", "unsafe"}:
                        prediction_for_metrics = prediction
                    elif args.parse_error_policy == "unsafe":
                        prediction_for_metrics = "unsafe"
                    elif args.parse_error_policy == "incorrect":
                        prediction_for_metrics = "safe" if gold == "unsafe" else "unsafe"
                    else:
                        prediction_for_metrics = None
                    all_predictions.append({
                        "benchmark": name,
                        "example_id": row.get("example_id"),
                        "language": row.get("language"),
                        "view": row.get("view"),
                        "subset": row.get("subset") or "ALL",
                        "topic": row.get("topic"),
                        "gold": gold,
                        "prediction": prediction,
                        "prediction_for_metrics": prediction_for_metrics,
                        "parse_status": parsed["parse_status"],
                        "parsed_payload": parsed["payload"],
                        "gold_categories": list(row.get("categories") or []),
                        "predicted_categories": parsed["categories"],
                        "unknown_categories": parsed["unknown_categories"],
                        "raw_output": raw,
                    })
                total_completed += len(chunk_rows)
                elapsed = time.time() - generation_start
                rate = total_completed / elapsed if elapsed > 0 else 0.0
                remaining = total_prompts - total_completed
                eta_seconds = remaining / rate if rate > 0 else None
                write_progress(
                    args.output_dir,
                    status="running",
                    phase="generating",
                    backend=args.backend,
                    completed=total_completed,
                    total=total_prompts,
                    percent=round(100.0 * total_completed / total_prompts, 2),
                    current_benchmark=name,
                    current_benchmark_completed=min(start + len(chunk_rows), len(batch_rows)),
                    current_benchmark_total=len(batch_rows),
                    rows_per_second=round(rate, 3),
                    eta_seconds=round(eta_seconds) if eta_seconds is not None else None,
                )
                log(
                    f"  [{name}] {min(start + len(chunk_rows), len(batch_rows))}/{len(batch_rows)} "
                    f"overall={total_completed}/{total_prompts} ({100.0*total_completed/total_prompts:.1f}%) "
                    f"rate={rate:.2f} rows/s"
                )
            log(f"  [{name}] vLLM done {len(batch_rows)} rows in {time.time()-gen_start:.0f}s")
        log(f"vLLM generation complete")

    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric_predictions = [row for row in all_predictions if row["prediction_for_metrics"] in {"safe", "unsafe"}]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_predictions:
        benchmark, language, view, subset = str(row["benchmark"]), str(row["language"]), str(row["view"]), str(row["subset"])
        groups[(benchmark, "ALL", "ALL", "ALL")].append(row)
        groups[(benchmark, language, "ALL", "ALL")].append(row)
        groups[(benchmark, language, view, "ALL")].append(row)
        groups[(benchmark, language, view, subset)].append(row)
    result_rows = [{"benchmark": key[0], "language": key[1], "view": key[2], "subset": key[3], **stats(value)} for key, value in sorted(groups.items())]
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "family": args.family,
        "backend": args.backend,
        "parse_error_policy": args.parse_error_policy,
        "prompt_template_sha256": hashlib.sha256(NEMOTRON_PROMPT_TEMPLATE.encode("utf-8")).hexdigest() if args.family == "nemotron" else None,
        "protocol_note": "SEA rows evaluated with the configured model-native guard prompt; this is not the official SEA-HELM prompt/leaderboard protocol.",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        status="complete",
        phase="complete",
        backend=args.backend,
        completed=len(all_predictions),
        total=len(all_predictions),
        percent=100.0,
        predictions_file=str(args.output_dir / "predictions.jsonl"),
        metrics_file=str(args.output_dir / "metrics.json"),
    )


if __name__ == "__main__":
    main()
