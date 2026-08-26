"""Merge a LoRA adapter into a base model so vLLM can serve the tuned weights.

Usage:
  python merge_adapter.py --base-model <BASE> --adapter <ADAPTER_DIR> --output <OUT_DIR>

Works for any HuggingFace causal LM (Llama, Qwen3, Qwen3.5 multimodal, Gemma).
Fix for Qwen3.5-4B hybrid (Qwen3_5ForConditionalGeneration): tries ImageTextToText first.
Handles CPU offload (needs offload_dir on machines without enough GPU RAM).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import os, tempfile
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def log(message: str) -> None:
    print(f"[merge] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def get_offload_dir():
    if os.name == "nt":
        pp = Path("D:/Downloads/Safety Dataset/tmp/offload")
        try:
            pp.mkdir(parents=True, exist_ok=True)
            return str(pp)
        except Exception:
            pass
    for cand in [Path("/tmp/offload_qwen35"), Path(tempfile.gettempdir()) / "offload_qwen35", Path("tmp/offload")]:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            tt = cand / ".writetest"
            tt.write_text("ok", encoding="utf-8")
            tt.unlink(missing_ok=True)
            return str(cand)
        except Exception:
            continue
    return str(Path(tempfile.gettempdir()) / "offload_qwen35")


def load_base_model(base_model: str, dtype, attn_impl: str = "sdpa"):
    """Try Qwen3.5 multimodal first, fallback to CausalLM."""
    # On CPU-only machines, force CPU to avoid meta-offload dispatch issues with PEFT
    if not torch.cuda.is_available():
        log("no CUDA, forcing device_map=cpu (no offload)")
        try:
            m = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype, device_map="cpu", attn_implementation=attn_impl, trust_remote_code=True, low_cpu_mem_usage=True)
            log(f"loaded via AutoModelForImageTextToText ({type(m).__name__}) on CPU")
            return m
        except Exception as e:
            log(f"ImageTextToText CPU load failed ({e}), trying CausalLM cpu...")
        m = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="cpu", attn_implementation=attn_impl, trust_remote_code=True, low_cpu_mem_usage=True)
        log(f"loaded via AutoModelForCausalLM ({type(m).__name__}) on CPU")
        return m

    # GPU available: use auto with offload on D:\
    offload_dir = get_offload_dir()
    Path(offload_dir).mkdir(parents=True, exist_ok=True)
    try:
        m = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype, device_map="auto", attn_implementation=attn_impl, trust_remote_code=True, low_cpu_mem_usage=True, offload_folder=offload_dir, offload_state_dict=True)
        log(f"loaded via AutoModelForImageTextToText ({type(m).__name__})")
        return m
    except Exception as e:
        log(f"ImageTextToText load failed ({e}), trying CausalLM...")
    m = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="auto", attn_implementation=attn_impl, trust_remote_code=True, low_cpu_mem_usage=True, offload_folder=offload_dir, offload_state_dict=True)
    log(f"loaded via AutoModelForCausalLM ({type(m).__name__})")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--attn", default="sdpa", help="attn_implementation")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"loading base model from {args.base_model} (dtype={args.dtype}, attn={args.attn})...")
    start = time.time()
    model = load_base_model(args.base_model, dtype, args.attn)
    log(f"base model loaded in {time.time()-start:.1f}s")

    log(f"loading LoRA adapter from {args.adapter}...")
    start = time.time()
    try:
        from safetensors import safe_open
        ap = Path(args.adapter) / "adapter_model.safetensors"
        if ap.exists():
            with safe_open(str(ap), framework="pt") as f:
                ks = list(f.keys())
                if ks:
                    mean_abs = sum(f.get_tensor(k).abs().mean().item() for k in ks) / len(ks)
                    log(f"adapter mean_abs={mean_abs:.6f} ({len(ks)} tensors) - ~0 means LoRA not trained")
    except Exception as e:
        log(f"adapter norm check skip: {e}")
    # PEFT dispatch on CPU should not need offload when base is on cpu
    try:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    except ValueError as e:
        if "offload_dir" in str(e):
            log(f"Peft offload error, retry with offload_folder on D:\\...")
            offload_dir = get_offload_dir()
            Path(offload_dir).mkdir(parents=True, exist_ok=True)
            model = PeftModel.from_pretrained(model, str(args.adapter), offload_folder=offload_dir)
        else:
            raise
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
    tok_src = args.adapter / "tokenizer" if (args.adapter / "tokenizer").exists() else (args.adapter if (args.adapter / "tokenizer.json").exists() else args.base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    # copy processor/vocab extras for Qwen3.5 multimodal (needed for vLLM)
    try:
        base_path = pathlib.Path(args.base_model)
        if base_path.is_dir():
            for name in ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "tokenizer_config.json", "generation_config.json"]:
                src = base_path / name
                dst = args.output / name
                if src.exists() and not dst.exists():
                    import shutil as _sh
                    _sh.copy2(src, dst)
                    log(f"copied {name} from base")
    except Exception as e:
        log(f"extra copy skip: {e}")
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