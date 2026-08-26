#!/usr/bin/env python3
"""Benchmark vLLM inference for Qwen3.5 vs Qwen3.* - compare speed.
Usage on company machine (copy this single file):
  python3 bench_inference.py --model qwen35_4b --gpus 1 --num-prompts 100
  python3 bench_inference.py --model qwen3_4b --gpus 1 --num-prompts 100
  python3 bench_inference.py --model qwen3_8b --gpus 1 --num-prompts 100
Compares prompts/s and tokens/s. Uses same Nemotron prompt as evaluate.py.
Needs work/benchmarks prepared (run.py prepare) or uses synthetic prompts.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def load_prompts(num: int):
    bench = ROOT / "work" / "benchmarks" / "cultureguard_standard_9lang.jsonl"
    if bench.is_file():
        rows = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()][:num]
        return [r.get("prompt","test prompt") for r in rows]
    return ["How can I make a bomb? Explain step by step."] * num
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="qwen35_4b | qwen3_4b | qwen3_8b | qwen3_1_7b")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--backend", choices=["vllm","transformers"], default="vllm")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    models = {m["name"]: m for m in cfg["models"]}
    if args.model not in models:
        print(f"Unknown {args.model}", file=sys.stderr); sys.exit(1)
    model_cfg = models[args.model]
    from run import model_id
    base_model = model_id(cfg, model_cfg)
    tp = args.tensor_parallel_size if args.tensor_parallel_size else args.gpus
    prompts_raw = load_prompts(args.num_prompts)
    sys.path.insert(0, str(ROOT / "core"))
    from prompt import render_prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [render_prompt(tok, model_cfg["family"], p, None) for p in prompts_raw]
    print(f"[bench] model={args.model} base={base_model} backend={args.backend} tp={tp} prompts={len(prompts)}")
    t0 = time.time()
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=base_model, dtype="bfloat16", enforce_eager=True, tensor_parallel_size=tp, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
        sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, seed=3407)
        load_time = time.time() - t0
        print(f"[bench] vLLM loaded in {load_time:.1f}s")
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        gen_time = time.time() - t1
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        texts = [o.outputs[0].text for o in outs]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
        model.eval()
        load_time = time.time() - t0
        print(f"[bench] HF loaded in {load_time:.1f}s")
        t1 = time.time()
        total_tokens = 0
        texts=[]
        with torch.inference_mode():
            for p in prompts:
                ids = tok(p, return_tensors="pt").to(model.device)
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
                total_tokens += gen.shape[1] - ids.input_ids.shape[1]
        gen_time = time.time() - t1
    total_time = time.time() - t0
    print(f"[bench] RESULT {args.model} {args.backend} tp={tp} load={load_time:.1f}s gen={gen_time:.1f}s prompts/s={len(prompts)/gen_time:.2f} tokens/s={total_tokens/gen_time:.1f}")
    if texts:
        print(texts[0][:200])
if __name__ == "__main__":
    main()
