# vLLM Evaluation Guide

`core/evaluate.py` supports two generation backends for guard evaluation:

- `transformers` (default): native HF `model.generate`, batched.
- `vllm`: vLLM offline batch engine, much faster for large evals.

## Install (company B200, CUDA 13, torch 2.11.0+cu130)

Pick the vLLM version that matches your torch. Do NOT let pip upgrade torch.

```bash
# torch 2.11.0+cu130 -> vllm 0.23.0
# This pins flashinfer-python/cubin to the matching 0.6.12 pair.
python3 -m pip install --upgrade --no-cache-dir "vllm==0.23.0"

# if torch is newer (2.13.0+cu130) -> vllm 0.27.1
# pip install "vllm==0.27.1"
```

Full requirements (same as requirements.txt + vllm):
```bash
pip install "vllm==0.23.0" "transformers==5.15.0" "tokenizers>=0.22.2" \
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

Evaluate only the two missing base models, sequentially:
```bash
nohup bash eval_bases.sh > eval_bases.log 2>&1 & echo $! > eval_bases.pid
tail -f eval_bases.log
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
- Stop training before eval so the trainer and vLLM do not compete for GPU memory.
- `eval_bases.sh` refuses to start while the Qwen full trainer is still running.
- vLLM gathers all prompts per benchmark and generates in one batch -> ~15x faster than transformers on 200+ samples.
- likelihood/BPB still uses transformers (vLLM does not expose per-token NLL).
- Works with Qwen3, Qwen3.5 (multimodal text-only), Llama-3.1 via `AutoModelForCausalLM` + nemotron prompt.

## LoRA adapters (after-training eval with vLLM)
- vLLM cannot load a `PeftModel` directly. For post-training eval
  (`--checkpoint after --method lora --backend vllm`), `run.py` now auto-merges
  the adapter into the base weights once (into `runs/<model>/<method>_<mode>/merged`)
  and points vLLM at the merged dir. The cache includes an adapter SHA-256, so
  finishing more epochs automatically rebuilds a stale merge.
- Manual merge for a standalone custom adapter:
  ```bash
  python merge_adapter.py --base-model <BASE> --adapter <ADAPTER_DIR> --output <OUT_DIR>
  ```
- NVIDIA's Nemotron v3 root directory already contains inference-ready full
  weights; its official inference script loads the root directly. Do not apply
  the bundled `lora_adapter/` again on top of those root weights.
- `before` checkpoint eval works fine with vLLM (no adapter).
