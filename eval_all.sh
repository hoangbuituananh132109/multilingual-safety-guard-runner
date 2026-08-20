#!/usr/bin/env bash
# One-shot: smoke-train Qwen3-8B (no checkpoints), then eval (smoke -> full) on
# the three priority models with vLLM, then emit NVIDIA-compatible reports.
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
export FLASHINFER_DISABLE_VERSION_CHECK=1
echo "===== [1/8] prepare benchmarks (standard/JB split) ====="
$PY run.py prepare

echo "===== [2/8] SMOKE train Qwen3-8B (no checkpoints) ====="
$PY run.py train --model qwen3_8b --method lora --mode smoke --no-checkpoints

echo "===== [3/8] SMOKE eval: Nemotron merged (before) ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm --limit 20

echo "===== [4/8] SMOKE eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm --limit 20

echo "===== [5/8] SMOKE eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm --limit 20

echo "===== [6/8] FULL eval: Nemotron merged (before) ====="
$PY run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm

echo "===== [7/8] FULL eval: Qwen3-8B after ====="
$PY run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full --backend vllm

echo "===== [8/8] FULL eval: Llama-3.1-8B-Instruct after ====="
$PY run.py evaluate --model llama31_8b_instruct --checkpoint after --method lora --run-mode full --backend vllm

echo "===== REPORTS (NVIDIA-compatible) ====="
$PY run.py nvidia-report --model llama31_nemotron --checkpoint before
$PY run.py nvidia-report --model qwen3_8b --checkpoint after --method lora --run-mode full
$PY run.py nvidia-report --model llama31_8b_instruct --checkpoint after --method lora --run-mode full

echo "===== DONE ====="