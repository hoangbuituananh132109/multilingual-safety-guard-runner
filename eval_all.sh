#!/usr/bin/env bash
# One-shot: smoke-train Qwen3-8B (no checkpoints), then eval (smoke -> full) on
# the three priority models with vLLM, then emit NVIDIA-compatible reports,
# then FULL train Qwen3-8B for 5 epochs (only reached if all smoke steps passed).
# Run from the repo root on the company machine.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ---- Model paths (override as needed) --------------------------------------
export MODEL_PATH_QWEN3_8B="${MODEL_PATH_QWEN3_8B:-/workspace/storage-shared/models/Qwen3-8B}"
export MODEL_PATH_LLAMA31_8B_INSTRUCT="${MODEL_PATH_LLAMA31_8B_INSTRUCT:-/workspace/storage-shared/models/Llama-3.1-8B-Instruct}"
export MODEL_PATH_LLAMA31_NEMOTRON="${MODEL_PATH_LLAMA31_NEMOTRON:-/workspace/storage-shared/nlp/huypq51/models/Llama-3.1-Nemotron-Safety-Guard-8B-v3}"

PY="python3"
echo "===== [1/9] prepare benchmarks (standard/JB split) ====="
$PY run.py prepare

echo "===== [2/9] SMOKE train Qwen3-8B (no checkpoints) ====="
$PY run.py train --model qwen3_8b --method lora --mode smoke --no-checkpoints

echo "===== [3/9] SMOKE eval: Nemotron merged (before) ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm --limit 20

echo "===== [4/9] SMOKE eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm --limit 20

echo "===== [5/9] SMOKE eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm --limit 20

echo "===== [6/9] FULL eval: Nemotron merged (before) ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm

echo "===== [7/9] FULL eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm

echo "===== [8/9] FULL eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm

echo "===== REPORTS (NVIDIA-compatible) ====="
$PY run.py nvidia-report --model llama31_nemotron --checkpoint before
$PY run.py nvidia-report --model qwen3_8b --checkpoint after --method lora --run-mode full
$PY run.py nvidia-report --model llama31_8b_instruct --checkpoint after --method lora --run-mode full

echo "===== [9/9] FULL train Qwen3-8B (5 epochs, no checkpoints) ====="
$PY run.py train --model qwen3_8b --method lora --mode full --no-checkpoints

echo "===== ALL DONE ====="