"""Merge a LoRA adapter into a base model so vLLM can serve the tuned weights.

Usage:
  python merge_adapter.py --base-model <BASE> --adapter <ADAPTER_DIR> --output <OUT_DIR>

Works for any HuggingFace causal LM (Llama, Qwen3, Qwen3.5 text-only, Gemma).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(message: str) -> None:
    print(f"[merge] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--adapter-sha256")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"loading base model from {args.base_model} (dtype={args.dtype})...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    log(f"base model loaded in {time.time()-start:.1f}s")
    log(f"loading LoRA adapter from {args.adapter}...")
    start = time.time()
    model = PeftModel.from_pretrained(model, args.adapter)
    log(f"adapter loaded in {time.time()-start:.1f}s")
    log("merging adapter into base weights...")
    start = time.time()
    merged = model.merge_and_unload()
    log(f"merge done in {time.time()-start:.1f}s")
    merged = merged.to(dtype)
    merged.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    log(f"saving merged model to {args.output}...")
    start = time.time()
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter / "tokenizer" if (args.adapter / "tokenizer").exists() else args.base_model)
    tokenizer.save_pretrained(args.output)
    log(f"saved in {time.time()-start:.1f}s")
    manifest = {
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "adapter_sha256": args.adapter_sha256,
        "dtype": args.dtype,
        "output": str(args.output),
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"done -> {args.output}")


if __name__ == "__main__":
    main()
