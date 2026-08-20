from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[likelihood] {stamp} {message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_ids(model: Any, ids: list[int], max_length: int, stride: int) -> tuple[float, int]:
    import torch

    if len(ids) < 2:
        return 0.0, 0
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    for begin in range(0, len(ids), stride):
        end = min(begin + max_length, len(ids))
        target_length = end - previous_end
        input_ids = torch.tensor([ids[begin:end]], dtype=torch.long, device=model.device)
        labels = input_ids.clone()
        labels[:, :-target_length] = -100
        valid_tokens = int((labels[:, 1:] != -100).sum().item())
        if valid_tokens:
            with torch.inference_mode():
                loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            total_nll += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
        previous_end = end
        if end == len(ids):
            break
    return total_nll, total_tokens


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_nll = sum(float(row["nll_nats"]) for row in rows)
    total_tokens = sum(int(row["tokens_scored"]) for row in rows)
    total_bytes = sum(int(row["utf8_bytes"]) for row in rows)
    mean_nll = total_nll / total_tokens if total_tokens else None
    return {
        "examples": len(rows),
        "tokens_scored": total_tokens,
        "utf8_bytes": total_bytes,
        "nll_nats": total_nll,
        "mean_nll_per_token": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50.0)) if mean_nll is not None else None,
        "bits_per_byte": total_nll / math.log(2) / total_bytes if total_bytes else None,
    }


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser(
        description="Compute raw-text NLL/perplexity and tokenizer-fairer bits-per-byte diagnostics. This is not an official SEA-HELM score."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--benchmark", action="append", required=True, help="NAME=/path/file.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", default="vi")
    parser.add_argument("--field", choices=["text", "prompt", "response"], default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_length < 2 or not 0 < args.stride < args.max_length:
        raise ValueError("Require max_length >= 2 and 0 < stride < max_length")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.adapter / "tokenizer" if args.adapter and (args.adapter / "tokenizer").exists() else args.base_model
    tokenizer_kwargs = {} if args.adapter and (args.adapter / "tokenizer").exists() else {"revision": args.revision}
    log(f"loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    log(f"tokenizer loaded, vocab={getattr(tokenizer, 'vocab_size', '?')}")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.load_in_4bit else None
    log(f"loading base model from {args.base_model} (dtype=bf16, 4bit={args.load_in_4bit})...")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if args.adapter:
        log(f"loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
        log("adapter loaded")
    log(f"base model loaded in {time.time()-load_start:.1f}s, device={next(model.parameters()).device}")
    model.eval()

    results: list[dict[str, Any]] = []
    for specification in args.benchmark:
        name, path_text = specification.split("=", 1)
        rows = [row for row in read_jsonl(Path(path_text)) if args.language == "all" or str(row.get("language")) == args.language]
        if args.limit is not None:
            rows = rows[:args.limit]
        log(f"benchmark={name} language={args.language} rows={len(rows)} max_length={args.max_length} stride={args.stride}")
        bench_start = time.time()
        for index, row in enumerate(rows, 1):
            text = row.get(args.field)
            if text is None or not str(text).strip():
                continue
            value = str(text)
            ids = tokenizer.encode(value, add_special_tokens=False)
            nll, tokens = score_ids(model, ids, args.max_length, args.stride)
            result = {
                "benchmark": name,
                "example_id": row.get("example_id"),
                "language": row.get("language"),
                "view": row.get("view"),
                "subset": row.get("subset") or "ALL",
                "field": args.field,
                "input_tokens": len(ids),
                "tokens_scored": tokens,
                "utf8_bytes": len(value.encode("utf-8")),
                "nll_nats": nll,
                "perplexity": math.exp(min(nll / tokens, 50.0)) if tokens else None,
                "bits_per_byte": nll / math.log(2) / len(value.encode("utf-8")) if value else None,
            }
            results.append(result)
            if index % 25 == 0:
                (args.output_dir / "progress.json").write_text(json.dumps({"completed": len(results)}) + "\n", encoding="utf-8")
                elapsed = time.time() - bench_start
                rate = index / elapsed if elapsed > 0 else 0
                log(f"  [{name}] {index}/{len(rows)} ({100.0*index/len(rows):.1f}%) elapsed={elapsed:.0f}s rate={rate:.2f} rows/s")
        if len(rows):
            elapsed = time.time() - bench_start
            log(f"  [{name}] done {len(rows)} rows in {elapsed:.0f}s")

    with (args.output_dir / "likelihood_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        key = (str(row["benchmark"]), str(row["language"]), str(row["view"]), str(row["subset"]))
        groups[(key[0], "ALL", "ALL", "ALL")].append(row)
        groups[(key[0], key[1], "ALL", "ALL")].append(row)
        groups[(key[0], key[1], key[2], "ALL")].append(row)
        groups[key].append(row)
    metrics = [
        {"benchmark": key[0], "language": key[1], "view": key[2], "subset": key[3], **aggregate(value)}
        for key, value in sorted(groups.items())
    ]
    (args.output_dir / "likelihood_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if metrics:
        with (args.output_dir / "likelihood_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "language": args.language,
        "field": args.field,
        "max_length": args.max_length,
        "stride": args.stride,
        "interpretation": {
            "perplexity": "Raw token-level diagnostic; do not directly rank models with different tokenizers from this value alone.",
            "bits_per_byte": "Preferred cross-tokenizer raw-text likelihood diagnostic; lower is better.",
            "official_sea_helm": False,
        },
    }
    (args.output_dir / "likelihood_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
