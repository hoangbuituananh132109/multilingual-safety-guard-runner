#!/usr/bin/env bash
set -euo pipefail

# Offline, source-prior-preserving Qwen3-4B study.
# Four one-GPU training arms run concurrently with the same global batch (32).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export SOURCE_STUDY_MODEL_PATH="${SOURCE_STUDY_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-4B}"

PYTHON_BIN="${SOURCE_STUDY_PYTHON_BIN:-python3}"
PHASE="${1:-all}"
DATA_ROOT="$ROOT/work/source-study-natural"
RUN_ROOT="$ROOT/runs-source-study-natural/qwen3_4b"
MERGED_ROOT="$RUN_ROOT/merged"
EVAL_ROOT="$RUN_ROOT/evaluations"
LOG_ROOT="$ROOT/logs/source-study-natural"
mkdir -p "$RUN_ROOT" "$MERGED_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

ARMS=(
  nemotron_v3_9lang_natural
  wildguardtrain_en_natural
  sea_cultural_vi_natural
  sea_cultural_bilingual_id50_natural
)
CONFIGS=(
  source_study_natural_train_qwen3_4b_nemotron_v3.yaml
  source_study_natural_train_qwen3_4b_wildguard_en.yaml
  source_study_natural_train_qwen3_4b_sea_vi.yaml
  source_study_natural_train_qwen3_4b_sea_bilingual.yaml
)
RUN_DIRS=(
  "$RUN_ROOT/nemotron_v3_9lang_natural_80k_1epoch"
  "$RUN_ROOT/wildguardtrain_en_natural_80k_1epoch"
  "$RUN_ROOT/sea_cultural_vi_natural_80k_1epoch"
  "$RUN_ROOT/sea_cultural_bilingual_id50_natural_1epoch"
)

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

require_four_free_gpus() {
  "$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); print(f"CUDA devices: {n}"); raise SystemExit(0 if n >= 4 else 1)'
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    while IFS= read -r used; do
      used="${used//[!0-9]/}"
      [[ -z "$used" || "$used" -le 2048 ]] || die "a GPU already uses ${used} MiB; refusing a duplicate launch"
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  fi
}

require_benchmarks() {
  local spec path actual benchmark_index=0
  for ((i=1; i<${#BENCHMARK_ARGS[@]}; i+=2)); do
    spec="${BENCHMARK_ARGS[$i]}"
    path="${spec#*=}"
    require_file "$path"
    actual="$(wc -l < "$path")"
    actual="${actual//[!0-9]/}"
    [[ "$actual" == "${BENCHMARK_COUNTS[$benchmark_index]}" ]] || \
      die "benchmark count mismatch: $path expected=${BENCHMARK_COUNTS[$benchmark_index]} actual=$actual"
    benchmark_index=$((benchmark_index + 1))
  done
}

require_data() {
  local arm
  for arm in "${ARMS[@]}"; do
    "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_ROOT/$arm" >/dev/null
  done
}

require_smoke_data() {
  local arm
  for arm in "${ARMS[@]}"; do
    "$PYTHON_BIN" source_study_natural.py validate --data-dir "$DATA_ROOT/_smoke/$arm" >/dev/null
  done
}

preflight() {
  require_model
  require_data
  require_smoke_data
  require_benchmarks
  require_four_free_gpus
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,sklearn,torch,transformers,vllm,yaml; print("offline training/eval imports: OK")'
  "$PYTHON_BIN" -m unittest tests.test_source_study_natural_data tests.test_source_study_data
  "$PYTHON_BIN" scripts/smoke_source_study_natural_contract.py \
    --model "$SOURCE_STUDY_MODEL_PATH" \
    --data-root "$DATA_ROOT/_smoke" \
    --output "$LOG_ROOT/contract_smoke.json"
  echo "[$(date -Is)] NATURAL SOURCE STUDY PREFLIGHT PASSED"
}

smoke_one() {
  local gpu="$1" index="$2" arm="${ARMS[$2]}" config="${CONFIGS[$2]}"
  local output="$RUN_ROOT/_smoke/${arm}_2steps"
  echo "[$(date -Is)] GPU SMOKE START gpu=$gpu arm=$arm"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/train.py \
    --config "$config" --output-dir "$output" \
    --train-data "$DATA_ROOT/_smoke/$arm/train.jsonl" \
    --validation-data "$DATA_ROOT/_smoke/$arm/validation.jsonl" \
    --max-steps 2 --skip-eval --no-checkpoints --no-final-save \
    > "$LOG_ROOT/smoke_${arm}.log" 2>&1
  require_file "$output/train_results.json"
  echo "[$(date -Is)] GPU SMOKE PASS gpu=$gpu arm=$arm"
}

gpu_smoke() {
  require_model
  require_smoke_data
  require_four_free_gpus
  local failed=0
  CHILD_PIDS=()
  for i in 0 1 2 3; do
    smoke_one "$i" "$i" &
    CHILD_PIDS+=("$!")
  done
  for pid in "${CHILD_PIDS[@]}"; do wait "$pid" || failed=1; done
  CHILD_PIDS=()
  [[ "$failed" == 0 ]] || die "a smoke failed; inspect $LOG_ROOT/smoke_*.log"
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
  require_data
  require_four_free_gpus
  local failed=0
  CHILD_PIDS=()
  for i in 0 1 2 3; do
    train_one "$i" "$i" &
    CHILD_PIDS+=("$!")
  done
  for pid in "${CHILD_PIDS[@]}"; do wait "$pid" || failed=1; done
  CHILD_PIDS=()
  [[ "$failed" == 0 ]] || die "a training arm failed; inspect $LOG_ROOT/train_*.log"
}

merge_one() {
  local index="$1" arm="${ARMS[$1]}" adapter="${RUN_DIRS[$1]}/final" output="$MERGED_ROOT/${ARMS[$1]}"
  if [[ -f "$output/config.json" && -f "$output/tokenizer_config.json" && -f "$output/merge_manifest.json" ]]; then
    echo "[$(date -Is)] MERGE SKIP arm=$arm: complete"
    return
  fi
  [[ ! -e "$output" ]] || die "incomplete merge directory exists; move it aside first: $output"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" merge_adapter.py \
    --base-model "$SOURCE_STUDY_MODEL_PATH" --revision main \
    --adapter "$adapter" --output "$output" --dtype bf16 \
    > "$LOG_ROOT/merge_${arm}.log" 2>&1
  require_file "$output/config.json"
  require_file "$output/tokenizer_config.json"
  require_file "$output/merge_manifest.json"
  echo "[$(date -Is)] MERGE COMPLETE arm=$arm"
}

merge_all() {
  require_model
  for i in 0 1 2 3; do merge_one "$i"; done
}

eval_complete() {
  [[ -f "$1/metrics.json" ]] && grep -q '"status": "complete"' "$1/progress.json" 2>/dev/null
}

eval_one() {
  local gpu="$1" name="$2" model="$3" family="$4"
  local output="$EVAL_ROOT/$name" max_new=128
  [[ "$family" != "sea_guard" ]] || max_new=16
  if [[ "${SOURCE_STUDY_FORCE_EVAL:-0}" != "1" ]] && eval_complete "$output"; then
    echo "[$(date -Is)] EVAL SKIP gpu=$gpu model=$name: complete"
    return
  fi
  local sample=()
  [[ -z "${SOURCE_STUDY_EVAL_SAMPLE:-}" ]] || sample=(--sample "$SOURCE_STUDY_EVAL_SAMPLE")
  echo "[$(date -Is)] EVAL START gpu=$gpu model=$name family=$family"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/evaluate.py \
    --base-model "$model" --revision main --family "$family" \
    --backend vllm --tensor-parallel-size 1 \
    --gpu-memory-utilization "${SOURCE_STUDY_EVAL_GPU_MEMORY:-0.97}" \
    --max-model-len "${SOURCE_STUDY_MAX_MODEL_LEN:-8192}" \
    --batch-size 8 --vllm-chunk-size "${SOURCE_STUDY_VLLM_CHUNK_SIZE:-512}" \
    --decoding-profile greedy --taxonomy-mode off --thinking-mode no_think \
    --max-new-tokens "$max_new" --seed 3407 --parse-error-policy incorrect \
    "${sample[@]}" "${BENCHMARK_ARGS[@]}" --output-dir "$output" \
    > "$LOG_ROOT/eval_${name}.log" 2>&1
  eval_complete "$output" || die "evaluation incomplete: $name"
  echo "[$(date -Is)] EVAL COMPLETE gpu=$gpu model=$name"
}

eval_queue() {
  local gpu="$1" queue="$2"
  case "$queue" in
    0)
      eval_one "$gpu" qwen3_4b_base_nemotron_prompt "$SOURCE_STUDY_MODEL_PATH" nemotron
      eval_one "$gpu" nemotron_v3_9lang_natural "$MERGED_ROOT/nemotron_v3_9lang_natural" nemotron
      ;;
    1)
      eval_one "$gpu" qwen3_4b_base_sea_prompt "$SOURCE_STUDY_MODEL_PATH" sea_guard
      eval_one "$gpu" wildguardtrain_en_natural "$MERGED_ROOT/wildguardtrain_en_natural" nemotron
      ;;
    2) eval_one "$gpu" sea_cultural_vi_natural "$MERGED_ROOT/sea_cultural_vi_natural" sea_guard ;;
    3) eval_one "$gpu" sea_cultural_bilingual_id50_natural "$MERGED_ROOT/sea_cultural_bilingual_id50_natural" sea_guard ;;
    *) die "invalid eval queue: $queue" ;;
  esac
}

eval_all() {
  require_data
  require_benchmarks
  require_four_free_gpus
  merge_all
  local failed=0
  CHILD_PIDS=()
  for gpu in 0 1 2 3; do
    eval_queue "$gpu" "$gpu" > "$LOG_ROOT/eval_worker_gpu${gpu}.log" 2>&1 &
    CHILD_PIDS+=("$!")
  done
  for pid in "${CHILD_PIDS[@]}"; do wait "$pid" || failed=1; done
  CHILD_PIDS=()
  [[ "$failed" == 0 ]] || die "an eval queue failed; inspect $LOG_ROOT/eval_worker_gpu*.log"
  "$PYTHON_BIN" scripts/summarize_source_study_natural.py --eval-root "$EVAL_ROOT"
}

case "$PHASE" in
  all) preflight; gpu_smoke; train_all; eval_all ;;
  preflight) preflight ;;
  smoke) gpu_smoke ;;
  train) train_all ;;
  merge) merge_all ;;
  eval-smoke) SOURCE_STUDY_EVAL_SAMPLE="${SOURCE_STUDY_EVAL_SAMPLE:-8}" eval_all ;;
  eval) eval_all ;;
  *) die "usage: bash scripts/run_source_study_natural.sh [all|preflight|smoke|train|merge|eval-smoke|eval]" ;;
esac

echo "[$(date -Is)] NATURAL SOURCE STUDY COMPLETE phase=$PHASE"
