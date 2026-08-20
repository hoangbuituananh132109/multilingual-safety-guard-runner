# vLLM Evaluation Guide

`core/evaluate.py` supports two generation backends for guard evaluation:

- `transformers` (default): native HF `model.generate`, batched.
- `vllm`: vLLM offline batch engine, much faster for large evals.

## Install (company B200, CUDA 13, torch 2.11.0+cu130)

Pick the vLLM version that matches your torch. Do NOT let pip upgrade torch.

```bash
# torch 2.11.0+cu130 -> vllm 0.22.0 (matches exactly)
pip install "vllm==0.22.0"

# if torch is newer (2.13.0+cu130) -> vllm 0.27.1
# pip install "vllm==0.27.1"
```

vLLM needs `ninja` for flashinfer JIT kernels:
```bash
sudo apt-get install -y ninja-build
```

Full requirements (same as requirements.txt + vllm):
```bash
pip install "vllm==0.22.0" "transformers==5.15.0" "tokenizers>=0.22.2" \
  "accelerate==1.14.0" "datasets>=4.4.1" "peft==0.19.1" \
  "bitsandbytes==0.49.1" "safetensors>=0.8.0" "huggingface-hub>=0.36.0" \
  "scikit-learn==1.9.0" "pyyaml==6.0.3"
```

## Run

Via run.py (recommended):
```bash
# vLLM backend
python run.py evaluate --model qwen3_8b --checkpoint before --backend vllm

# transformers backend (default)
python run.py evaluate --model qwen3_8b --checkpoint before
```

Or set default in config.yaml:
```yaml
evaluation:
  backend: vllm   # or transformers
```

Direct core script:
```bash
python core/evaluate.py --base-model /path/to/model --family nemotron \
  --backend vllm \
  --benchmark sea_vi=work/benchmarks/sea_safeguard_vi.jsonl \
  --output-dir runs/my_model/base/guard
```

## Notes
- First vLLM run compiles flashinfer kernels (slow, ~1-2 min). Subsequent runs reuse the cache.
- vLLM gathers all prompts per benchmark and generates in one batch -> ~15x faster than transformers on 200+ samples.
- likelihood/BPB still uses transformers (vLLM does not expose per-token NLL).
- Works with Qwen3, Qwen3.5 (multimodal text-only), Llama-3.1 via `AutoModelForCausalLM` + nemotron prompt.

## Limitation: LoRA adapters
- vLLM backend loads `args.base_model` directly and does NOT apply a LoRA adapter.
- For post-training eval (`--checkpoint after --method lora`), either:
  - merge the adapter into the base weights first, then point `--base-model` at the merged dir, or
  - keep using the transformers backend for LoRA eval.
- `before` checkpoint eval works fine with vLLM.
