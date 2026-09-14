#!/usr/bin/env bash
set -euo pipefail

# Offline fixed-80k Nemotron/WildGuard mixture study for Qwen3-4B.
# Three one-GPU arms run concurrently on GPUs 0, 1 and 2.

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
PHASE="${1:-all}"
SOURCE_ROOT="$ROOT/work/source-study-natural"
DATA_ROOT="$ROOT/work/source-study-mixtures"
RUN_ROOT="$ROOT/runs-source-study-mixtures/qwen3_4b"
LOG_ROOT="$ROOT/logs/source-study-mixtures"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

ARMS=(
  nemotron50_wildguard50
  nemotron70_wildguard30
  nemotron80_wildguard20
)
CONFIGS=(
  source_study_mix_train_qwen3_4b_n50_w50.yaml
  source_study_mix_train_qwen3_4b_n70_w30.yaml
  source_study_mix_train_qwen3_4b_n80_w20.yaml
)
RUN_DIRS=(
  "$RUN_ROOT/nemotron50_wildguard50_80k_1epoch"
  "$RUN_ROOT/nemotron70_wildguard30_80k_1epoch"
  "$RUN_ROOT/nemotron80_wildguard20_80k_1epoch"
)

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

LOCK_DIR="$LOG_ROOT/.runner.lock"
acquire_lock() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    local old_pid=""
    [[ ! -f "$LOCK_DIR/pid" ]] || old_pid="$(tr -cd '0-9' < "$LOCK_DIR/pid")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "phase '$PHASE' is already running as PID=$old_pid"
    fi
    local stale="${LOCK_DIR}.stale.$(date +%Y%m%d-%H%M%S)"
    mv -- "$LOCK_DIR" "$stale"
    mkdir "$LOCK_DIR"
    echo "Moved stale lock to $stale"
  fi
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  printf '%s\n' "$(date -Is)" > "$LOCK_DIR/started_at"
}

CHILD_PIDS=()
terminate_tree() {
  local parent="$1" child
  while IFS= read -r child; do
    child="${child//[!0-9]/}"
    [[ -z "$child" ]] || terminate_tree "$child"
  done < <(ps -eo pid=,ppid= | awk -v parent="$parent" '$2 == parent {print $1}')
  kill -TERM "$parent" 2>/dev/null || true
}

cleanup() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    for pid in "${CHILD_PIDS[@]:-}"; do
      [[ -z "$pid" ]] || terminate_tree "$pid"
    done
  fi
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
    die "no local model weight file under $SOURCE_STUDY_MODEL_PATH"
}

require_three_free_gpus() {
  "$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); print(f"CUDA devices: {n}"); raise SystemExit(0 if n >= 3 else 1)'
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    local index used
    while IFS=',' read -r index used; do
      index="${index//[!0-9]/}"
      used="${used//[!0-9]/}"
      if [[ -n "$index" && "$index" -le 2 && -n "$used" && "$used" -gt 2048 ]]; then
        die "GPU $index already uses ${used} MiB; refusing a duplicate launch"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  fi
}

validate_tree() {
  local root="$1" arm
  for arm in "${ARMS[@]}"; do
    "$PYTHON_BIN" source_study_natural.py validate --data-dir "$root/$arm" >/dev/null
  done
}

validate_sources() {
  "$PYTHON_BIN" source_study_natural.py validate \
    --data-dir "$SOURCE_ROOT/nemotron_v3_9lang_natural" >/dev/null
  "$PYTHON_BIN" source_study_natural.py validate \
    --data-dir "$SOURCE_ROOT/wildguardtrain_en_natural" >/dev/null
}

tree_state() {
  local root="$1" present=0 arm
  for arm in "${ARMS[@]}"; do
    [[ ! -e "$root/$arm" ]] || present=$((present + 1))
  done
  printf '%s\n' "$present"
}

prepare() {
  local full_state smoke_state
  validate_sources
  full_state="$(tree_state "$DATA_ROOT")"
  smoke_state="$(tree_state "$DATA_ROOT/_smoke")"

  if [[ "$full_state" == "0" ]]; then
    "$PYTHON_BIN" scripts/build_nemotron_wildguard_mixtures.py \
      --source-root "$SOURCE_ROOT" --output-root "$DATA_ROOT" --seed 3407 \
      > "$LOG_ROOT/prepare_full.json"
  elif [[ "$full_state" == "3" ]]; then
    validate_tree "$DATA_ROOT"
    echo "[$(date -Is)] PREPARE SKIP: full mixture tree already valid"
  else
    die "partial full mixture tree exists under $DATA_ROOT; do not overwrite it"
  fi

  if [[ "$smoke_state" == "0" ]]; then
    "$PYTHON_BIN" scripts/build_nemotron_wildguard_mixtures.py \
      --source-root "$SOURCE_ROOT" --output-root "$DATA_ROOT" --seed 3407 --smoke \
      > "$LOG_ROOT/prepare_smoke.json"
  elif [[ "$smoke_state" == "3" ]]; then
    validate_tree "$DATA_ROOT/_smoke"
    echo "[$(date -Is)] PREPARE SKIP: smoke mixture tree already valid"
  else
    die "partial smoke mixture tree exists under $DATA_ROOT/_smoke; do not overwrite it"
  fi

  validate_tree "$DATA_ROOT"
  validate_tree "$DATA_ROOT/_smoke"
  echo "[$(date -Is)] MIXTURE DATA PREPARED"
}

preflight() {
  require_model
  validate_sources
  validate_tree "$DATA_ROOT"
  validate_tree "$DATA_ROOT/_smoke"
  require_three_free_gpus
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,yaml; print("offline training imports: OK")'
  for config in "${CONFIGS[@]}"; do require_file "$ROOT/$config"; done
  echo "[$(date -Is)] MIXTURE PREFLIGHT PASSED"
}

smoke_one() {
  local gpu="$1" index="$2" arm="${ARMS[$2]}" config="${CONFIGS[$2]}"
  local output="$RUN_ROOT/_smoke/${arm}_2steps"
  echo "[$(date -Is)] SMOKE START gpu=$gpu arm=$arm"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/train.py \
    --config "$config" --output-dir "$output" \
    --train-data "$DATA_ROOT/_smoke/$arm/train.jsonl" \
    --validation-data "$DATA_ROOT/_smoke/$arm/validation.jsonl" \
    --max-steps 2 --skip-eval --no-checkpoints --no-final-save \
    > "$LOG_ROOT/smoke_${arm}.log" 2>&1
  require_file "$output/train_results.json"
  echo "[$(date -Is)] SMOKE PASS gpu=$gpu arm=$arm"
}

smoke_all() {
  require_model
  validate_tree "$DATA_ROOT/_smoke"
  require_three_free_gpus
  local failed=0
  CHILD_PIDS=()
  for index in 0 1 2; do
    smoke_one "$index" "$index" &
    CHILD_PIDS+=("$!")
  done
  for pid in "${CHILD_PIDS[@]}"; do wait "$pid" || failed=1; done
  CHILD_PIDS=()
  [[ "$failed" == "0" ]] || die "a smoke failed; inspect $LOG_ROOT/smoke_*.log"
}

train_one() {
  local gpu="$1" index="$2" arm="${ARMS[$2]}" config="${CONFIGS[$2]}" output="${RUN_DIRS[$2]}"
  local resume=()
  if [[ -f "$output/run_complete.json" && -f "$output/final/adapter_config.json" ]]; then
    echo "[$(date -Is)] TRAIN SKIP gpu=$gpu arm=$arm: complete"
    return
  fi
  if compgen -G "$output/checkpoint-*" >/dev/null; then
    resume=(--resume)
    echo "[$(date -Is)] TRAIN RESUME gpu=$gpu arm=$arm"
  else
    echo "[$(date -Is)] TRAIN START gpu=$gpu arm=$arm"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/train.py --config "$config" "${resume[@]}" \
    >> "$LOG_ROOT/train_${arm}.log" 2>&1
  require_file "$output/run_complete.json"
  require_file "$output/final/adapter_config.json"
  echo "[$(date -Is)] TRAIN COMPLETE gpu=$gpu arm=$arm"
}

train_all() {
  require_model
  validate_tree "$DATA_ROOT"
  require_three_free_gpus
  local failed=0
  CHILD_PIDS=()
  for index in 0 1 2; do
    train_one "$index" "$index" &
    CHILD_PIDS+=("$!")
  done
  for pid in "${CHILD_PIDS[@]}"; do wait "$pid" || failed=1; done
  CHILD_PIDS=()
  [[ "$failed" == "0" ]] || die "a training arm failed; inspect $LOG_ROOT/train_*.log"
}

case "$PHASE" in
  all) prepare; preflight; smoke_all; train_all ;;
  prepare) prepare ;;
  preflight) preflight ;;
  smoke) smoke_all ;;
  train) train_all ;;
  *) die "usage: bash scripts/run_nemotron_wildguard_mixtures.sh [all|prepare|preflight|smoke|train]" ;;
esac

echo "[$(date -Is)] NEMOTRON/WILDGUARD MIXTURE COMPLETE phase=$PHASE"
