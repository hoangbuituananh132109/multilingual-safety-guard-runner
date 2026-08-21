#!/usr/bin/env bash
# One-shot: smoke-train Qwen3-8B, then eval base/reference/after models with
# vLLM 0.23 and emit NVIDIA-compatible reports.
# Full 5-epoch training is NOT auto-run here: run it manually with --resume so
# it continues from the copied epoch-1 checkpoint (see SETUP_DATA.md / README).
# Run from the repo root on the company machine.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ---- Model paths (override as needed) --------------------------------------
export MODEL_PATH_QWEN3_8B="${MODEL_PATH_QWEN3_8B:-/workspace/storage-shared/models/Qwen3-8B}"
export MODEL_PATH_LLAMA31_8B_INSTRUCT="${MODEL_PATH_LLAMA31_8B_INSTRUCT:-/workspace/storage-shared/models/Llama-3.1-8B-Instruct}"
export MODEL_PATH_LLAMA31_NEMOTRON="${MODEL_PATH_LLAMA31_NEMOTRON:-/workspace/storage-shared/nlp/huypq51/models/Llama-3.1-Nemotron-Safety-Guard-8B-v3}"

PY="python3"
if pgrep -f '[c]ore/train.py --config runs/qwen3_8b/lora_full/train_config.yaml' >/dev/null; then
  echo "Qwen3-8B training is still running. Stop it before eval." >&2
  exit 3
fi
VLLM_VERSION="$($PY -c 'import importlib.metadata as metadata; print(metadata.version("vllm"))')"
case "$VLLM_VERSION" in
  0.23.*) ;;
  *) echo "Expected vLLM 0.23.x, found $VLLM_VERSION" >&2; exit 2 ;;
esac

echo "===== [1/12] prepare benchmarks (standard/JB split) ====="
$PY run.py prepare

echo "===== [2/12] SMOKE train Qwen3-8B (no checkpoints) ====="
$PY run.py train --model qwen3_8b --method lora --mode smoke --no-checkpoints

echo "===== [3/12] SMOKE eval: Qwen3-8B base ====="
$PY run.py evaluate --model qwen3_8b --checkpoint before --backend vllm --sample 20

echo "===== [4/12] SMOKE eval: Llama-3.1-8B-Instruct base ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint before --backend vllm --sample 20

echo "===== [5/12] SMOKE eval: Nemotron reference ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm --sample 20

echo "===== [6/12] SMOKE eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm --sample 20

echo "===== [7/12] SMOKE eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm --sample 20

echo "===== [8/12] FULL eval: Qwen3-8B base ====="
$PY run.py evaluate --model qwen3_8b --checkpoint before --backend vllm

echo "===== [9/12] FULL eval: Llama-3.1-8B-Instruct base ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint before --backend vllm

echo "===== [10/12] FULL eval: Nemotron reference ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm

echo "===== [11/12] FULL eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm

echo "===== [12/12] FULL eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm

echo "===== REPORTS (NVIDIA-compatible) ====="
$PY run.py nvidia-report --model qwen3_8b --checkpoint before
$PY run.py nvidia-report --model llama31_8b_instruct --checkpoint before
$PY run.py nvidia-report --model llama31_nemotron --checkpoint before
$PY run.py nvidia-report --model qwen3_8b --checkpoint after --method lora --run-mode full
$PY run.py nvidia-report --model llama31_8b_instruct --checkpoint after --method lora --run-mode full

echo "===== DONE ====="
