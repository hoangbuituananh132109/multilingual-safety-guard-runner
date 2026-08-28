#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH_STAGE1_MERGED:?Set MODEL_PATH_STAGE1_MERGED to the merged Stage-1 directory first}"
test -f "$MODEL_PATH_STAGE1_MERGED/config.json"
test -f "$MODEL_PATH_STAGE1_MERGED/tokenizer_config.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p logs

run_experiment() {
  local name="$1"
  local config="$2"
  local output_dir="$3"
  local result_file="${output_dir}/train_results.json"
  local resume_args=()

  if [[ -f "$result_file" ]]; then
    echo "[$(date -Is)] SKIP ${name}: completed (${result_file})"
    return 0
  fi
  if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
    resume_args=(--resume)
    echo "[$(date -Is)] RESUME ${name} from latest checkpoint"
  else
    echo "[$(date -Is)] START ${name}"
  fi

  python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
    core/train.py --config "$config" "${resume_args[@]}" \
    2>&1 | tee -a "logs/${name}.log"
  echo "[$(date -Is)] DONE ${name}"
}

run_experiment vi_1epoch stage2_train_qwen3_8b_vi_gemini_1epoch.yaml runs-stage2/qwen3_8b/ablation_vi_gemini_1epoch
run_experiment reasoning_1epoch stage2_train_qwen3_8b_reasoning_only_1epoch.yaml runs-stage2/qwen3_8b/ablation_reasoning_only_1epoch
run_experiment full_1epoch stage2_train_qwen3_8b_bundle.yaml runs-stage2/qwen3_8b/lora_phase2_gemini_policy
run_experiment vi_5epoch stage2_train_qwen3_8b_vi_gemini_5epoch.yaml runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch
run_experiment reasoning_5epoch stage2_train_qwen3_8b_reasoning_only_5epoch.yaml runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch
run_experiment full_5epoch stage2_train_qwen3_8b_bundle_5epoch_each_epoch.yaml runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch

echo "[$(date -Is)] ALL STAGE-2 ABLATIONS COMPLETE"
