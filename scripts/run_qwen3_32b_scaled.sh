#!/usr/bin/env bash
set -euo pipefail

# Offline Qwen3-32B base evaluation and LoRA transfer of one scaled
# Nemotron/WildGuard common-contract recipe. Training is fail-closed and needs
# SOURCE_STUDY_ALLOW_32B_TRAIN=1 so an evaluation command cannot burn the node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SETTINGS_FILE="${SOURCE_STUDY_32B_SETTINGS_FILE-$ROOT/config/qwen3_32b_b200.env.sh}"
if [[ -n "$SETTINGS_FILE" ]]; then
  [[ -f "$SETTINGS_FILE" ]] || {
    echo "ERROR: settings file not found: $SETTINGS_FILE" >&2
    exit 1
  }
  # shellcheck source=/dev/null
  source "$SETTINGS_FILE"
fi

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
PHASE="${1:-preflight}"
RATIO="${SOURCE_STUDY_SCALE_RATIO:-30_70}"
export SOURCE_STUDY_32B_MODEL_PATH="${SOURCE_STUDY_32B_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-32B}"
TRAIN_GPUS="${SOURCE_STUDY_32B_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_GPU="${SOURCE_STUDY_32B_EVAL_GPU:-0}"
MERGE_GPU="${SOURCE_STUDY_32B_MERGE_GPU:-$EVAL_GPU}"
TARGET_GLOBAL_BATCH="${SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH:-32}"
MICROBATCH="${SOURCE_STUDY_32B_MICROBATCH:-2}"

[[ "$TRAIN_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "ERROR: SOURCE_STUDY_32B_TRAIN_GPUS must be comma-separated GPU ids" >&2
  exit 1
}
[[ "$EVAL_GPU" =~ ^[0-9]+$ && "$MERGE_GPU" =~ ^[0-9]+$ ]] || {
  echo "ERROR: eval/merge GPU must each be one integer id" >&2
  exit 1
}
[[ "$TARGET_GLOBAL_BATCH" =~ ^[1-9][0-9]*$ && "$MICROBATCH" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: target global batch and microbatch must be positive integers" >&2
  exit 1
}
IFS=',' read -r -a TRAIN_GPU_IDS <<< "$TRAIN_GPUS"
TRAIN_GPU_COUNT="${#TRAIN_GPU_IDS[@]}"
declare -A SEEN_TRAIN_GPUS=()
for gpu in "${TRAIN_GPU_IDS[@]}"; do
  [[ -z "${SEEN_TRAIN_GPUS[$gpu]:-}" ]] || {
    echo "ERROR: duplicate train GPU id: $gpu" >&2
    exit 1
  }
  SEEN_TRAIN_GPUS[$gpu]=1
done
GLOBAL_BATCH_DIVISOR=$((TRAIN_GPU_COUNT * MICROBATCH))
if (( TARGET_GLOBAL_BATCH % GLOBAL_BATCH_DIVISOR != 0 )); then
  echo "ERROR: cannot preserve target global batch=$TARGET_GLOBAL_BATCH with GPUs=$TRAIN_GPU_COUNT and microbatch=$MICROBATCH" >&2
  exit 1
fi
GRADIENT_ACCUMULATION=$((TARGET_GLOBAL_BATCH / GLOBAL_BATCH_DIVISOR))
export SOURCE_STUDY_32B_TRAIN_GPUS="$TRAIN_GPUS"
export SOURCE_STUDY_32B_EVAL_GPU="$EVAL_GPU"
export SOURCE_STUDY_32B_MERGE_GPU="$MERGE_GPU"
export SOURCE_STUDY_32B_GRADIENT_ACCUMULATION="$GRADIENT_ACCUMULATION"

case "$RATIO" in
  30_70)
    SELECTION="$ROOT/source_study_scaled_selection_30_70.json"
    ARM="nemotron30_wildguard70_scaled"
    ;;
  70_30)
    SELECTION="$ROOT/source_study_scaled_selection_70_30.json"
    ARM="nemotron70_wildguard30_scaled"
    ;;
  *)
    echo "ERROR: SOURCE_STUDY_SCALE_RATIO must be 30_70 or 70_30" >&2
    exit 2
    ;;
esac

if [[ -z "${SOURCE_STUDY_NEMOTRON_ROOT:-}" ]]; then
  if [[ -f "$ROOT/../offline-bundle-work/snapshots/nemotron-9lang/en/train.jsonl" ]]; then
    SOURCE_STUDY_NEMOTRON_ROOT="$ROOT/../offline-bundle-work/snapshots/nemotron-9lang"
  else
    SOURCE_STUDY_NEMOTRON_ROOT="$ROOT/input/nemotron"
  fi
fi
if [[ -z "${SOURCE_STUDY_WILDGUARD_PATH:-}" ]]; then
  if [[ -f "$ROOT/input/stage2/wildguard/wildguardtrain.jsonl" ]]; then
    SOURCE_STUDY_WILDGUARD_PATH="$ROOT/input/stage2/wildguard/wildguardtrain.jsonl"
  else
    SOURCE_STUDY_WILDGUARD_PATH="$ROOT/input/stage2/wildguard/train/wildguard_train.parquet"
  fi
fi
export SOURCE_STUDY_NEMOTRON_ROOT SOURCE_STUDY_WILDGUARD_PATH

DATA_ROOT="$ROOT/work/source-study-scaled"
DATA_DIR="$DATA_ROOT/$ARM"
RUN_ROOT="$ROOT/runs-source-study-scaled/qwen3_32b"
RUN_DIR="$RUN_ROOT/${ARM}_1epoch"
MERGED_DIR="$RUN_ROOT/merged/$ARM"
BASE_EVAL_DIR="$RUN_ROOT/evaluations/qwen3_32b_base"
TRAINED_EVAL_DIR="$RUN_ROOT/evaluations/$ARM"
LOG_ROOT="$ROOT/logs/source-study-scaled/qwen3_32b/$RATIO"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

BENCHMARK_ARGS=(
  --benchmark "cultureguard_jb=$ROOT/work/benchmarks/cultureguard_jb_9lang.jsonl"
  --benchmark "cultureguard_standard=$ROOT/work/benchmarks/cultureguard_standard_9lang.jsonl"
  --benchmark "multijail=$ROOT/work/benchmarks/multijail_4lang.jsonl"
  --benchmark "polyguard_prompts=$ROOT/work/benchmarks/polyguard_prompts_9lang.jsonl"
  --benchmark "sea_vi=$ROOT/work/benchmarks/sea_safeguard_vi.jsonl"
  --benchmark "xsafety=$ROOT/work/benchmarks/xsafety_multilingual.jsonl"
  --benchmark "xstest_en=$ROOT/work/benchmarks/xstest_en.jsonl"
  --benchmark "wildguardtest_en=$ROOT/work/benchmarks/wildguardtest_en.jsonl"
  --benchmark "sealsbench_vi=$ROOT/work/benchmarks/sealsbench_vi.jsonl"
  --benchmark "linguasafe_vi=$ROOT/work/benchmarks/linguasafe_vi.jsonl"
)
BENCHMARK_COUNTS=(13266 24993 1260 30906 1840 19600 450 3408 26644 3884)

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

LOCK_DIR="$LOG_ROOT/.runner.lock"
acquire_lock() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    local old_pid=""
    [[ ! -f "$LOCK_DIR/pid" ]] || old_pid="$(tr -cd '0-9' < "$LOCK_DIR/pid")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "Qwen3-32B phase already active as PID=$old_pid"
    fi
    local stale="${LOCK_DIR}.stale.$(date +%Y%m%d-%H%M%S)"
    mv -- "$LOCK_DIR" "$stale"
    mkdir "$LOCK_DIR"
  fi
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  printf '%s\n' "$(date -Is)" > "$LOCK_DIR/started_at"
}
cleanup() {
  local status=$?
  if [[ -d "$LOCK_DIR" ]] && [[ "$(cat "$LOCK_DIR/pid" 2>/dev/null || true)" == "$$" ]]; then
    rm -f -- "$LOCK_DIR/pid" "$LOCK_DIR/started_at"
    rmdir -- "$LOCK_DIR" 2>/dev/null || true
  fi
  exit "$status"
}
acquire_lock
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_model() {
  [[ -d "$SOURCE_STUDY_32B_MODEL_PATH" ]] || die "local Qwen3-32B missing: $SOURCE_STUDY_32B_MODEL_PATH"
  require_file "$SOURCE_STUDY_32B_MODEL_PATH/config.json"
  require_file "$SOURCE_STUDY_32B_MODEL_PATH/tokenizer_config.json"
  find "$SOURCE_STUDY_32B_MODEL_PATH" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . || \
    die "no local Qwen3-32B weights under $SOURCE_STUDY_32B_MODEL_PATH"
}

require_raw_sources() {
  local language
  for language in en ar de es fr hi ja th zh; do
    require_file "$SOURCE_STUDY_NEMOTRON_ROOT/$language/train.jsonl"
    require_file "$SOURCE_STUDY_NEMOTRON_ROOT/$language/valid.jsonl"
  done
  require_file "$SOURCE_STUDY_WILDGUARD_PATH"
}

require_benchmarks() {
  local i spec path actual index=0
  for ((i=1; i<${#BENCHMARK_ARGS[@]}; i+=2)); do
    spec="${BENCHMARK_ARGS[$i]}"
    path="${spec#*=}"
    require_file "$path"
    actual="$(wc -l < "$path")"
    actual="${actual//[!0-9]/}"
    [[ "$actual" == "${BENCHMARK_COUNTS[$index]}" ]] || \
      die "benchmark count mismatch: $path expected=${BENCHMARK_COUNTS[$index]} actual=$actual"
    index=$((index + 1))
  done
}

require_visible_gpu_count() {
  local visible="$1" expected="$2"
  CUDA_VISIBLE_DEVICES="$visible" "$PYTHON_BIN" -c "import torch; n=torch.cuda.device_count(); print(f'CUDA devices visible: {n}'); raise SystemExit(0 if n == $expected else 1)"
}

require_selected_free_gpus() {
  require_visible_gpu_count "$TRAIN_GPUS" "$TRAIN_GPU_COUNT"
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    local index used selected
    while IFS=',' read -r index used; do
      index="${index//[!0-9]/}"
      used="${used//[!0-9]/}"
      selected=0
      for gpu in "${TRAIN_GPU_IDS[@]}"; do
        [[ "$gpu" != "$index" ]] || selected=1
      done
      if [[ "$selected" == "1" && -n "$used" && "$used" -gt 2048 ]]; then
        die "GPU $index already uses ${used} MiB; refusing duplicate 32B launch"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  fi
}

show_config() {
  "$PYTHON_BIN" - <<'PY'
import json, os
print(json.dumps({
    "settings_file": os.environ.get("SOURCE_STUDY_32B_SETTINGS_FILE", "config/qwen3_32b_b200.env.sh"),
    "model_path": os.environ["SOURCE_STUDY_32B_MODEL_PATH"],
    "ratio": os.environ.get("SOURCE_STUDY_SCALE_RATIO", "30_70"),
    "train_gpus": os.environ["SOURCE_STUDY_32B_TRAIN_GPUS"],
    "train_gpu_count": len(os.environ["SOURCE_STUDY_32B_TRAIN_GPUS"].split(",")),
    "eval_gpu": os.environ["SOURCE_STUDY_32B_EVAL_GPU"],
    "merge_gpu": os.environ["SOURCE_STUDY_32B_MERGE_GPU"],
    "microbatch_per_gpu": int(os.environ.get("SOURCE_STUDY_32B_MICROBATCH", "2")),
    "gradient_accumulation_steps": int(os.environ["SOURCE_STUDY_32B_GRADIENT_ACCUMULATION"]),
    "effective_global_batch": int(os.environ.get("SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH", "32")),
}, indent=2))
PY
}

prepare_data() {
  require_raw_sources
  require_file "$SELECTION"
  require_file "$ROOT/work/benchmarks/cultureguard_standard_9lang.jsonl"
  "$PYTHON_BIN" scripts/build_scaled_nemotron_wildguard.py \
    --selection "$SELECTION" \
    --nemotron-root "$SOURCE_STUDY_NEMOTRON_ROOT" \
    --wildguard-path "$SOURCE_STUDY_WILDGUARD_PATH" \
    --benchmark-root "$ROOT/work/benchmarks" \
    --output-root "$DATA_ROOT" --seed 3407 \
    > "$LOG_ROOT/prepare_data.json"
  "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_DIR" \
    > "$LOG_ROOT/validate_data.json"
  echo "[$(date -Is)] 32B DATA READY ratio=$RATIO data=$DATA_DIR"
}

preflight() {
  require_model
  require_file "$ROOT/source_study_scaled_train_qwen3_32b.yaml"
  require_file "$ROOT/scripts/build_scaled_nemotron_wildguard.py"
  require_file "$SELECTION"
  require_benchmarks
  [[ -d "$DATA_DIR" ]] || die "scaled data missing; run prepare-data first: $DATA_DIR"
  "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_DIR" >/dev/null
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,vllm,yaml; print("offline Qwen3-32B imports: OK")'
  echo "[$(date -Is)] 32B PREFLIGHT PASSED ratio=$RATIO"
}

eval_complete() {
  [[ -f "$1/metrics.json" ]] && grep -q '"status": "complete"' "$1/progress.json" 2>/dev/null
}

eval_model() {
  local model="$1" output="$2" label="$3" sample="${4:-}"
  if [[ -z "$sample" ]] && eval_complete "$output"; then
    echo "[$(date -Is)] 32B EVAL SKIP label=$label: complete"
    return
  fi
  require_visible_gpu_count "$EVAL_GPU" 1
  local sample_args=()
  [[ -z "$sample" ]] || sample_args=(--sample "$sample")
  local target="$output"
  [[ -z "$sample" ]] || target="${output}_smoke"
  echo "[$(date -Is)] 32B EVAL START label=$label sample=${sample:-full}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" core/evaluate.py \
    --base-model "$model" --revision main --family nemotron \
    --backend vllm --tensor-parallel-size 1 \
    --gpu-memory-utilization "${SOURCE_STUDY_32B_EVAL_GPU_MEMORY:-0.90}" \
    --max-model-len "${SOURCE_STUDY_MAX_MODEL_LEN:-8192}" \
    --batch-size "${SOURCE_STUDY_32B_EVAL_BATCH:-16}" \
    --vllm-chunk-size "${SOURCE_STUDY_VLLM_CHUNK_SIZE:-512}" \
    --decoding-profile greedy --taxonomy-mode off --thinking-mode no_think \
    --max-new-tokens 128 --seed 3407 --parse-error-policy incorrect \
    "${sample_args[@]}" "${BENCHMARK_ARGS[@]}" --output-dir "$target" \
    > "$LOG_ROOT/eval_${label}_${sample:-full}.log" 2>&1
  eval_complete "$target" || die "evaluation incomplete: $target"
  echo "[$(date -Is)] 32B EVAL COMPLETE label=$label sample=${sample:-full}"
}

smoke_train() {
  preflight
  require_selected_free_gpus
  local output="$RUN_ROOT/_smoke/${ARM}_${TRAIN_GPU_COUNT}gpu_2steps"
  mkdir -p "$RUN_ROOT/_smoke"
  echo "[$(date -Is)] 32B TRAIN SMOKE START ratio=$RATIO"
  SOURCE_STUDY_SCALED_DATA="$DATA_DIR" \
  SOURCE_STUDY_32B_RUN="$output" \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_GPU_COUNT" core/train.py \
    --config source_study_scaled_train_qwen3_32b.yaml \
    --per-device-batch-size "$MICROBATCH" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --max-steps 2 --skip-eval --no-checkpoints --no-final-save \
    > "$LOG_ROOT/smoke_train.log" 2>&1
  require_file "$output/train_results.json"
  echo "[$(date -Is)] 32B TRAIN SMOKE PASS ratio=$RATIO"
}

train() {
  [[ "${SOURCE_STUDY_ALLOW_32B_TRAIN:-0}" == "1" ]] || \
    die "set SOURCE_STUDY_ALLOW_32B_TRAIN=1 after base eval and smoke pass"
  preflight
  require_selected_free_gpus
  local resume=()
  if [[ -f "$RUN_DIR/run_complete.json" && -f "$RUN_DIR/final/adapter_config.json" ]]; then
    echo "[$(date -Is)] 32B TRAIN SKIP ratio=$RATIO: complete"
    return
  fi
  if compgen -G "$RUN_DIR/checkpoint-*" >/dev/null; then resume=(--resume); fi
  echo "[$(date -Is)] 32B TRAIN START ratio=$RATIO data=$DATA_DIR"
  SOURCE_STUDY_SCALED_DATA="$DATA_DIR" \
  SOURCE_STUDY_32B_RUN="$RUN_DIR" \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_GPU_COUNT" core/train.py \
    --config source_study_scaled_train_qwen3_32b.yaml \
    --per-device-batch-size "$MICROBATCH" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    "${resume[@]}" \
    > "$LOG_ROOT/train.log" 2>&1
  require_file "$RUN_DIR/run_complete.json"
  require_file "$RUN_DIR/final/adapter_config.json"
  echo "[$(date -Is)] 32B TRAIN COMPLETE ratio=$RATIO"
}

merge_trained() {
  require_model
  require_file "$RUN_DIR/final/adapter_config.json"
  if [[ -f "$MERGED_DIR/config.json" && -f "$MERGED_DIR/merge_manifest.json" ]]; then
    echo "[$(date -Is)] 32B MERGE SKIP ratio=$RATIO: complete"
    return
  fi
  [[ ! -e "$MERGED_DIR" ]] || die "incomplete merge directory exists: $MERGED_DIR"
  require_visible_gpu_count "$MERGE_GPU" 1
  CUDA_VISIBLE_DEVICES="$MERGE_GPU" "$PYTHON_BIN" merge_adapter.py \
    --base-model "$SOURCE_STUDY_32B_MODEL_PATH" --revision main \
    --adapter "$RUN_DIR/final" --output "$MERGED_DIR" --dtype bf16 \
    > "$LOG_ROOT/merge.log" 2>&1
  require_file "$MERGED_DIR/config.json"
  require_file "$MERGED_DIR/merge_manifest.json"
  echo "[$(date -Is)] 32B MERGE COMPLETE ratio=$RATIO"
}

case "$PHASE" in
  show-config) show_config ;;
  prepare-data) prepare_data ;;
  preflight) preflight ;;
  eval-base-smoke) preflight; eval_model "$SOURCE_STUDY_32B_MODEL_PATH" "$BASE_EVAL_DIR" base 8 ;;
  eval-base) preflight; eval_model "$SOURCE_STUDY_32B_MODEL_PATH" "$BASE_EVAL_DIR" base ;;
  smoke-train) smoke_train ;;
  train) train ;;
  merge) merge_trained ;;
  eval-trained-smoke) preflight; require_file "$MERGED_DIR/config.json"; eval_model "$MERGED_DIR" "$TRAINED_EVAL_DIR" trained 8 ;;
  eval-trained) preflight; require_file "$MERGED_DIR/config.json"; eval_model "$MERGED_DIR" "$TRAINED_EVAL_DIR" trained ;;
  *) die "usage: bash scripts/run_qwen3_32b_scaled.sh [show-config|prepare-data|preflight|eval-base-smoke|eval-base|smoke-train|train|merge|eval-trained-smoke|eval-trained]" ;;
esac

if [[ "$PHASE" != "show-config" ]]; then
  echo "[$(date -Is)] 32B PIPELINE COMPLETE ratio=$RATIO phase=$PHASE"
fi
