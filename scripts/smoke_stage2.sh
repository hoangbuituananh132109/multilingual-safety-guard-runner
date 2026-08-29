#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH_STAGE1_MERGED:?Set MODEL_PATH_STAGE1_MERGED to the merged Stage-1 directory first}"
test -f "$MODEL_PATH_STAGE1_MERGED/config.json"
test -f "$MODEL_PATH_STAGE1_MERGED/tokenizer_config.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p logs

run_smoke() {
  local name="$1"
  local config="$2"
  echo "=== smoke: ${name} (${config}) ==="
  python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
    core/train.py --config "$config" --output-dir "runs-stage2/_smoke/${name}" \
    --max-steps 3 --skip-eval --no-checkpoints --no-final-save \
    2>&1 | tee "logs/smoke_${name}.log"
}

run_smoke vi stage2_train_qwen3_8b_vi_gemini_1epoch.yaml
run_smoke reasoning stage2_train_qwen3_8b_reasoning_only_1epoch.yaml
run_smoke full stage2_train_qwen3_8b_bundle.yaml

echo "All Stage-2 smoke runs passed."
