#!/usr/bin/env bash
set -euo pipefail

# Offline sequential matrix for the two locked scaled mixtures on Qwen3
# 32B, 8B, and 4B. Every train and eval phase uses all eight visible B200s.
# Run foreground and tee the master command; this script also writes one live
# log per phase so a failed arm can be resumed without repeating completed work.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${SOURCE_STUDY_PYTHON_BIN:-python3}"
TORCHRUN_BIN="${SOURCE_STUDY_TORCHRUN_BIN:-torchrun}"
PHASE="${1:-show-config}"
MODEL_FILTER="${2:-}"
RATIO_FILTER="${3:-}"

GPU_IDS="${SOURCE_STUDY_B200_GPUS:-0,1,2,3,4,5,6,7}"
GPU_COUNT=8
MICROBATCH="${SOURCE_STUDY_B200_MICROBATCH:-2}"
GRAD_ACCUM="${SOURCE_STUDY_B200_GRAD_ACCUM:-2}"
EVAL_BATCH="${SOURCE_STUDY_B200_EVAL_BATCH:-2000}"
EVAL_MAX_MODEL_LEN="${SOURCE_STUDY_B200_EVAL_MAX_MODEL_LEN:-8192}"
EVAL_GPU_MEMORY="${SOURCE_STUDY_B200_EVAL_GPU_MEMORY:-0.967}"
BENCHMARK_BUNDLE="${SOURCE_STUDY_UNIFIED_BENCHMARK:-$ROOT/work/eval-hosted/benchmark-total/qwen3_safety_benchmark_total.jsonl}"
TRAIN_CONFIG="$ROOT/source_study_scaled_train_qwen3_generic.yaml"

MODELS=(32b 8b 4b)
RATIOS=(30_70 70_30)
LOG_ROOT="$ROOT/logs/scaled-qwen-matrix-b200"
RUN_ROOT="$ROOT/runs-source-study-scaled"
mkdir -p "$LOG_ROOT" "$RUN_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date -Is)] $*"; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

model_path() {
  case "$1" in
    32b) echo "${QWEN3_32B_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-32B}" ;;
    8b) echo "${QWEN3_8B_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-8B}" ;;
    4b) echo "${QWEN3_4B_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-4B}" ;;
    *) die "model must be 32b, 8b, or 4b: $1" ;;
  esac
}

arm_name() {
  case "$1" in
    30_70) echo "nemotron30_wildguard70_scaled" ;;
    70_30) echo "nemotron70_wildguard30_scaled" ;;
    *) die "ratio must be 30_70 or 70_30: $1" ;;
  esac
}

archive_path() {
  case "$1" in
    30_70) echo "$ROOT/zip/nemotron_wildguard_scaled_max_30_70_v2.gated.zip" ;;
    70_30) echo "$ROOT/zip/nemotron_wildguard_scaled_max_70_30_v2.gated.zip" ;;
  esac
}

archive_sha() {
  case "$1" in
    30_70) echo "7b65704a2e305d725b2da99b59eda0f89266d5d59aa5372b0ed319578305c628" ;;
    70_30) echo "61e43353485d1d0e3986507b2b4524a2f90afa5f42dc98c67575f8126e61ccf4" ;;
  esac
}

selected_models() {
  local item
  for item in "${MODELS[@]}"; do
    [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "$item" ]] && echo "$item"
  done
}

selected_ratios() {
  local item
  for item in "${RATIOS[@]}"; do
    [[ -z "$RATIO_FILTER" || "$RATIO_FILTER" == "$item" ]] && echo "$item"
  done
}

validate_filters() {
  if [[ -n "$MODEL_FILTER" ]]; then
    [[ "$MODEL_FILTER" == "32b" || "$MODEL_FILTER" == "8b" || "$MODEL_FILTER" == "4b" ]] || \
      die "model filter must be 32b, 8b, or 4b"
  fi
  if [[ -n "$RATIO_FILTER" ]]; then
    [[ "$RATIO_FILTER" == "30_70" || "$RATIO_FILTER" == "70_30" ]] || \
      die "ratio filter must be 30_70 or 70_30"
  fi
  [[ "$MICROBATCH" =~ ^[1-9][0-9]*$ && "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] || \
    die "microbatch and grad accumulation must be positive integers"
  [[ "$EVAL_BATCH" =~ ^[1-9][0-9]*$ ]] || die "eval batch must be a positive integer"
}

run_logged() {
  local logfile="$1"
  shift
  mkdir -p "$(dirname "$logfile")"
  "$@" 2>&1 | tee "$logfile"
}

require_eight_free_gpus() {
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -c \
    "import torch; n=torch.cuda.device_count(); print(f'CUDA devices visible: {n}'); raise SystemExit(0 if n == $GPU_COUNT else 1)" || \
    die "expected exactly eight selected GPUs: $GPU_IDS"
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    local index used
    while IFS=',' read -r index used; do
      index="${index//[!0-9]/}"
      used="${used//[!0-9]/}"
      if [[ -n "$index" && "$index" -le 7 && -n "$used" && "$used" -gt 2048 ]]; then
        die "GPU $index already uses ${used} MiB; refusing duplicate launch"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  fi
}

require_model() {
  local path="$1"
  [[ -d "$path" ]] || die "model directory missing: $path"
  require_file "$path/config.json"
  require_file "$path/tokenizer_config.json"
  find "$path" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . || \
    die "no model weights under $path"
}

install_ratio() {
  local ratio="$1" archive output expected actual
  archive="$(archive_path "$ratio")"
  output="$ROOT/work/source-study-scaled/$(arm_name "$ratio")"
  expected="$(archive_sha "$ratio")"
  require_file "$archive"
  actual="$(sha256sum "$archive" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "archive SHA mismatch: $archive expected=$expected actual=$actual"
  if [[ -f "$output/train.jsonl" && -f "$output/validation.jsonl" && -f "$output/manifest.json" ]]; then
    log "DATA INSTALL SKIP ratio=$ratio: existing bundle will be validated"
  else
    if [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      die "refusing to overwrite incomplete non-empty data directory: $output"
    fi
    run_logged "$LOG_ROOT/install_${ratio}.log" \
      "$PYTHON_BIN" scripts/source_study_bundle.py install --zip "$archive" --output "$output"
  fi
  run_logged "$LOG_ROOT/validate_${ratio}.log" \
    "$PYTHON_BIN" source_study_natural.py validate --data-dir "$output"
}

preflight_one() {
  local model="$1" ratio="$2" path data
  path="$(model_path "$model")"
  data="$ROOT/work/source-study-scaled/$(arm_name "$ratio")"
  require_model "$path"
  require_file "$TRAIN_CONFIG"
  require_file "$BENCHMARK_BUNDLE"
  require_file "$data/train.jsonl"
  require_file "$data/validation.jsonl"
  "$PYTHON_BIN" source_study_natural.py validate --data-dir "$data" >/dev/null
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,vllm,yaml; print("offline matrix imports: OK")'
  log "PREFLIGHT PASS model=$model ratio=$ratio data=$data"
}

paths_for() {
  local model="$1" ratio="$2" arm
  arm="$(arm_name "$ratio")"
  MATRIX_RUN_DIR="$RUN_ROOT/qwen3_${model}/${arm}_1epoch"
  MATRIX_MERGED_DIR="$RUN_ROOT/qwen3_${model}/merged/$arm"
  MATRIX_EVAL_DIR="$RUN_ROOT/qwen3_${model}/evaluations_unified/$arm"
  MATRIX_ARM_LOG="$LOG_ROOT/qwen3_${model}/$ratio"
}

smoke_one() {
  local model="$1" ratio="$2" path data smoke_dir smoke_data smoke_rows
  preflight_one "$model" "$ratio"
  require_eight_free_gpus
  path="$(model_path "$model")"
  data="$ROOT/work/source-study-scaled/$(arm_name "$ratio")"
  paths_for "$model" "$ratio"
  smoke_dir="$RUN_ROOT/qwen3_${model}/_smoke/$(arm_name "$ratio")_8gpu_2steps"
  if [[ -f "$smoke_dir/train_results.json" ]]; then
    log "TRAIN SMOKE SKIP model=$model ratio=$ratio: complete"
    return
  fi
  # A two-step DDP smoke must exercise the model, not tokenize the entire
  # 200k-400k-row training arm eight times before its first optimizer step.
  smoke_data="$smoke_dir/_smoke_data"
  smoke_rows=$((GPU_COUNT * MICROBATCH * GRAD_ACCUM * 2))
  mkdir -p "$smoke_data"
  head -n "$smoke_rows" "$data/train.jsonl" > "$smoke_data/train.jsonl"
  head -n 8 "$data/validation.jsonl" > "$smoke_data/validation.jsonl"
  [[ "$(wc -l < "$smoke_data/train.jsonl")" -eq "$smoke_rows" ]] || \
    die "smoke dataset has fewer than $smoke_rows training rows: $data"
  [[ "$(wc -l < "$smoke_data/validation.jsonl")" -eq 8 ]] || \
    die "smoke dataset has fewer than 8 validation rows: $data"
  log "TRAIN SMOKE DATA model=$model ratio=$ratio rows=$smoke_rows"
  run_logged "$MATRIX_ARM_LOG/smoke_train.log" env \
    SOURCE_STUDY_MODEL_PATH="$path" SOURCE_STUDY_SCALED_DATA="$data" SOURCE_STUDY_SCALED_RUN="$smoke_dir" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPU_COUNT" core/train.py \
    --config "$TRAIN_CONFIG" --per-device-batch-size "$MICROBATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" --train-data "$smoke_data/train.jsonl" \
    --validation-data "$smoke_data/validation.jsonl" --max-steps 2 --skip-eval --no-checkpoints --no-final-save
  require_file "$smoke_dir/train_results.json"
  log "TRAIN SMOKE PASS model=$model ratio=$ratio"
}

train_one() {
  local model="$1" ratio="$2" path data
  local -a resume=()
  preflight_one "$model" "$ratio"
  require_eight_free_gpus
  path="$(model_path "$model")"
  data="$ROOT/work/source-study-scaled/$(arm_name "$ratio")"
  paths_for "$model" "$ratio"
  if [[ -f "$MATRIX_RUN_DIR/run_complete.json" && -f "$MATRIX_RUN_DIR/final/adapter_config.json" ]]; then
    log "TRAIN SKIP model=$model ratio=$ratio: complete"
    return
  fi
  if compgen -G "$MATRIX_RUN_DIR/checkpoint-*" >/dev/null; then resume=(--resume); fi
  run_logged "$MATRIX_ARM_LOG/train.log" env \
    SOURCE_STUDY_MODEL_PATH="$path" SOURCE_STUDY_SCALED_DATA="$data" SOURCE_STUDY_SCALED_RUN="$MATRIX_RUN_DIR" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPU_COUNT" core/train.py \
    --config "$TRAIN_CONFIG" --per-device-batch-size "$MICROBATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" "${resume[@]}"
  require_file "$MATRIX_RUN_DIR/run_complete.json"
  require_file "$MATRIX_RUN_DIR/final/adapter_config.json"
  log "TRAIN COMPLETE model=$model ratio=$ratio"
}

merge_one() {
  local model="$1" ratio="$2" path
  path="$(model_path "$model")"
  paths_for "$model" "$ratio"
  require_file "$MATRIX_RUN_DIR/final/adapter_config.json"
  if [[ -f "$MATRIX_MERGED_DIR/config.json" && -f "$MATRIX_MERGED_DIR/merge_manifest.json" ]]; then
    log "MERGE SKIP model=$model ratio=$ratio: complete"
    return
  fi
  if [[ -e "$MATRIX_MERGED_DIR" ]]; then
    die "incomplete merge directory exists: $MATRIX_MERGED_DIR"
  fi
  require_eight_free_gpus
  run_logged "$MATRIX_ARM_LOG/merge.log" env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$PYTHON_BIN" merge_adapter.py --base-model "$path" --revision main \
    --adapter "$MATRIX_RUN_DIR/final" --output "$MATRIX_MERGED_DIR" --dtype bf16
  require_file "$MATRIX_MERGED_DIR/config.json"
  require_file "$MATRIX_MERGED_DIR/merge_manifest.json"
  log "MERGE COMPLETE model=$model ratio=$ratio"
}

eval_one() {
  local model="$1" ratio="$2" mode="$3" output log_file
  local -a limit_args=()
  paths_for "$model" "$ratio"
  require_model "$MATRIX_MERGED_DIR"
  require_file "$BENCHMARK_BUNDLE"
  if [[ "$mode" == "smoke" ]]; then
    output="${MATRIX_EVAL_DIR}_smoke"
    log_file="$MATRIX_ARM_LOG/eval_smoke.log"
    limit_args=(--limit-per-benchmark 8)
  else
    output="$MATRIX_EVAL_DIR"
    log_file="$MATRIX_ARM_LOG/eval_full.log"
    if [[ -f "$output/metrics.json" && -f "$output/run_manifest.json" && -f "$output/predictions.jsonl" ]]; then
      log "EVAL SKIP model=$model ratio=$ratio: complete"
      return
    fi
  fi
  require_eight_free_gpus
  run_logged "$log_file" env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$PYTHON_BIN" scripts/evaluate_qwen235b_bundle.py \
    --model "$MATRIX_MERGED_DIR" --bundle "$BENCHMARK_BUNDLE" --output-dir "$output" \
    --tensor-parallel-size "$GPU_COUNT" --gpu-memory-utilization "$EVAL_GPU_MEMORY" \
    --batch-size "$EVAL_BATCH" --max-model-len "$EVAL_MAX_MODEL_LEN" --max-new-tokens 16 \
    --thinking-mode no_think --trust-remote-code "${limit_args[@]}"
  require_file "$output/metrics.json"
  require_file "$output/run_manifest.json"
  log "EVAL COMPLETE model=$model ratio=$ratio mode=$mode output=$output"
}

show_config() {
  printf 'models=%s\n' "${MODELS[*]}"
  printf 'ratios=%s\n' "${RATIOS[*]}"
  printf 'gpu_ids=%s gpu_count=%s\n' "$GPU_IDS" "$GPU_COUNT"
  printf 'microbatch=%s grad_accum=%s effective_global_batch=%s\n' \
    "$MICROBATCH" "$GRAD_ACCUM" "$((GPU_COUNT * MICROBATCH * GRAD_ACCUM))"
  printf 'eval_batch=%s eval_max_model_len=%s eval_gpu_memory=%s\n' \
    "$EVAL_BATCH" "$EVAL_MAX_MODEL_LEN" "$EVAL_GPU_MEMORY"
  printf 'benchmark=%s\n' "$BENCHMARK_BUNDLE"
  local model ratio
  while IFS= read -r model; do printf '%s_model=%s\n' "$model" "$(model_path "$model")"; done < <(selected_models)
  while IFS= read -r ratio; do printf '%s_archive=%s\n' "$ratio" "$(archive_path "$ratio")"; done < <(selected_ratios)
}

run_matrix_phase() {
  local action="$1" model ratio
  while IFS= read -r model; do
    while IFS= read -r ratio; do
      case "$action" in
        preflight) preflight_one "$model" "$ratio" ;;
        smoke) smoke_one "$model" "$ratio" ;;
        run)
          train_one "$model" "$ratio"
          merge_one "$model" "$ratio"
          eval_one "$model" "$ratio" smoke
          eval_one "$model" "$ratio" full
          ;;
      esac
    done < <(selected_ratios)
  done < <(selected_models)
}

validate_filters
case "$PHASE" in
  show-config) show_config ;;
  install)
    while IFS= read -r ratio; do install_ratio "$ratio"; done < <(selected_ratios)
    ;;
  preflight) run_matrix_phase preflight ;;
  smoke) run_matrix_phase smoke ;;
  run) run_matrix_phase run ;;
  all)
    while IFS= read -r ratio; do install_ratio "$ratio"; done < <(selected_ratios)
    run_matrix_phase preflight
    run_matrix_phase smoke
    run_matrix_phase run
    ;;
  *)
    die "usage: bash scripts/run_scaled_qwen_matrix_b200.sh [show-config|install|preflight|smoke|run|all] [32b|8b|4b] [30_70|70_30]"
    ;;
esac

log "MATRIX COMPLETE phase=$PHASE model_filter=${MODEL_FILTER:-all} ratio_filter=${RATIO_FILTER:-all}"
