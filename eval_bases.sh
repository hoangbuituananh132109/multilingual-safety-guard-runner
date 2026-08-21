#!/usr/bin/env bash
# Evaluate the two missing base models sequentially with vLLM 0.23.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export MODEL_PATH_QWEN3_8B="${MODEL_PATH_QWEN3_8B:-/workspace/storage-shared/models/Qwen3-8B}"
export MODEL_PATH_LLAMA31_8B_INSTRUCT="${MODEL_PATH_LLAMA31_8B_INSTRUCT:-/workspace/storage-shared/models/Llama-3.1-8B-Instruct}"

PY="python3"
if pgrep -f '[c]ore/train.py --config runs/qwen3_8b/lora_full/train_config.yaml' >/dev/null; then
  echo "Qwen3-8B training is still running. Stop it before base eval." >&2
  exit 3
fi
VLLM_VERSION="$($PY -c 'import importlib.metadata as metadata; print(metadata.version("vllm"))')"
case "$VLLM_VERSION" in
  0.23.*) ;;
  *) echo "Expected vLLM 0.23.x, found $VLLM_VERSION" >&2; exit 2 ;;
esac

echo "===== [1/6] SMOKE Qwen3-8B base ====="
$PY run.py evaluate --model qwen3_8b --checkpoint before --backend vllm --sample 20

echo "===== [2/6] SMOKE Llama-3.1-8B-Instruct base ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint before --backend vllm --sample 20

echo "===== [3/6] FULL Qwen3-8B base ====="
$PY run.py evaluate --model qwen3_8b --checkpoint before --backend vllm

echo "===== [4/6] REPORT Qwen3-8B base ====="
$PY run.py nvidia-report --model qwen3_8b --checkpoint before

echo "===== [5/6] FULL Llama-3.1-8B-Instruct base ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint before --backend vllm

echo "===== [6/6] REPORT Llama-3.1-8B-Instruct base ====="
$PY run.py nvidia-report --model llama31_8b_instruct --checkpoint before

echo "===== BASE EVAL DONE ====="
