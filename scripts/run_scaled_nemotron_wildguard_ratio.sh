#!/usr/bin/env bash
set -euo pipefail

# Build and train one max-scale Nemotron/WildGuard recipe fully offline.
# The default arm is the empirical 30:70 main recipe; set
# SOURCE_STUDY_SCALE_RATIO=70_30 for the Nemo-heavy scale comparator.

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
export SOURCE_STUDY_MODEL_PATH="${SOURCE_STUDY_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-4B}"

PYTHON_BIN="${SOURCE_STUDY_PYTHON_BIN:-python3}"
TORCHRUN_BIN="${SOURCE_STUDY_TORCHRUN_BIN:-torchrun}"
PHASE="${1:-all}"
RATIO="${SOURCE_STUDY_SCALE_RATIO:-30_70}"

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

DATA_ROOT="$ROOT/work/source-study-scaled"
DATA_DIR="$DATA_ROOT/$ARM"
RUN_ROOT="$ROOT/runs-source-study-scaled/qwen3_4b"
RUN_DIR="$RUN_ROOT/${ARM}_1epoch"
LOG_ROOT="$ROOT/logs/source-study-scaled/$RATIO"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

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

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

LOCK_DIR="$LOG_ROOT/.runner.lock"
acquire_lock() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    local old_pid=""
    [[ ! -f "$LOCK_DIR/pid" ]] || old_pid="$(tr -cd '0-9' < "$LOCK_DIR/pid")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "ratio $RATIO is already running as PID=$old_pid"
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
  [[ -d "$SOURCE_STUDY_MODEL_PATH" ]] || die "local model directory missing: $SOURCE_STUDY_MODEL_PATH"
  require_file "$SOURCE_STUDY_MODEL_PATH/config.json"
  require_file "$SOURCE_STUDY_MODEL_PATH/tokenizer_config.json"
  find "$SOURCE_STUDY_MODEL_PATH" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . || \
    die "no local model weights under $SOURCE_STUDY_MODEL_PATH"
}

require_raw_sources() {
  local language
  for language in en ar de es fr hi ja th zh; do
    require_file "$SOURCE_STUDY_NEMOTRON_ROOT/$language/train.jsonl"
    require_file "$SOURCE_STUDY_NEMOTRON_ROOT/$language/valid.jsonl"
  done
  require_file "$SOURCE_STUDY_WILDGUARD_PATH"
}

require_four_gpus() {
  "$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); print(f"CUDA devices: {n}"); raise SystemExit(0 if n >= 4 else 1)'
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    local index used
    while IFS=',' read -r index used; do
      index="${index//[!0-9]/}"
      used="${used//[!0-9]/}"
      if [[ -n "$index" && "$index" -le 3 && -n "$used" && "$used" -gt 2048 ]]; then
        die "GPU $index already uses ${used} MiB; refusing duplicate launch"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  fi
}

prepare() {
  require_raw_sources
  require_file "$ROOT/work/benchmarks/cultureguard_standard_9lang.jsonl"
  mkdir -p "$DATA_ROOT"
  "$PYTHON_BIN" scripts/build_scaled_nemotron_wildguard.py \
    --selection "$SELECTION" \
    --nemotron-root "$SOURCE_STUDY_NEMOTRON_ROOT" \
    --wildguard-path "$SOURCE_STUDY_WILDGUARD_PATH" \
    --benchmark-root "$ROOT/work/benchmarks" \
    --output-root "$DATA_ROOT" \
    --seed 3407 \
    > "$LOG_ROOT/prepare.json"
  "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_DIR" \
    > "$LOG_ROOT/validate.json"
  echo "[$(date -Is)] SCALE DATA READY ratio=$RATIO data=$DATA_DIR"
}

preflight() {
  require_model
  require_file "$ROOT/source_study_scaled_train_qwen3_4b.yaml"
  require_file "$ROOT/scripts/build_scaled_nemotron_wildguard.py"
  require_file "$SELECTION"
  require_raw_sources
  [[ -d "$DATA_DIR" ]] || die "scaled data missing; run prepare first: $DATA_DIR"
  "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_DIR" >/dev/null
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,yaml; print("offline training imports: OK")'
  echo "[$(date -Is)] SCALE PREFLIGHT PASSED ratio=$RATIO"
}

smoke() {
  preflight
  require_four_gpus
  local output="$RUN_ROOT/_smoke/${ARM}_4gpu_2steps"
  mkdir -p "$RUN_ROOT/_smoke"
  echo "[$(date -Is)] SCALE SMOKE START ratio=$RATIO"
  SOURCE_STUDY_SCALED_DATA="$DATA_DIR" \
  SOURCE_STUDY_SCALED_RUN="$output" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN_BIN" --standalone --nproc_per_node=4 core/train.py \
    --config source_study_scaled_train_qwen3_4b.yaml \
    --max-steps 2 --skip-eval --no-checkpoints --no-final-save \
    > "$LOG_ROOT/smoke.log" 2>&1
  require_file "$output/train_results.json"
  echo "[$(date -Is)] SCALE SMOKE PASS ratio=$RATIO log=$LOG_ROOT/smoke.log"
}

train() {
  preflight
  require_four_gpus
  local resume=()
  if [[ -f "$RUN_DIR/run_complete.json" && -f "$RUN_DIR/final/adapter_config.json" ]]; then
    echo "[$(date -Is)] SCALE TRAIN SKIP ratio=$RATIO: complete"
    return
  fi
  if compgen -G "$RUN_DIR/checkpoint-*" >/dev/null; then
    resume=(--resume)
    echo "[$(date -Is)] SCALE TRAIN RESUME ratio=$RATIO"
  else
    echo "[$(date -Is)] SCALE TRAIN START ratio=$RATIO data=$DATA_DIR"
  fi
  SOURCE_STUDY_SCALED_DATA="$DATA_DIR" \
  SOURCE_STUDY_SCALED_RUN="$RUN_DIR" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN_BIN" --standalone --nproc_per_node=4 core/train.py \
    --config source_study_scaled_train_qwen3_4b.yaml "${resume[@]}" \
    > "$LOG_ROOT/train.log" 2>&1
  require_file "$RUN_DIR/run_complete.json"
  require_file "$RUN_DIR/final/adapter_config.json"
  echo "[$(date -Is)] SCALE TRAIN COMPLETE ratio=$RATIO run=$RUN_DIR"
}

case "$PHASE" in
  prepare) prepare ;;
  preflight) preflight ;;
  smoke) smoke ;;
  train) train ;;
  all) prepare; preflight; smoke; train ;;
  *) die "usage: bash scripts/run_scaled_nemotron_wildguard_ratio.sh [prepare|preflight|smoke|train|all]" ;;
esac

echo "[$(date -Is)] SCALE PIPELINE COMPLETE ratio=$RATIO phase=$PHASE"
