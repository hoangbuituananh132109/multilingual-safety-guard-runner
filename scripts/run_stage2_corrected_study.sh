#!/usr/bin/env bash
set -euo pipefail

# One command for the corrected Stage-2 replacement study:
# data build -> three 1-step GPU smokes -> three sequential 1-epoch LoRA runs
# on all four GPUs -> merge -> four full-benchmark prompt modes on four GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${STAGE2_PYTHON_BIN:-python3}"
PHASE="${1:-all}"
STAGE1="${MODEL_PATH_STAGE1_MERGED:-$ROOT/runs/qwen3_8b/lora_full/merged}"
export MODEL_PATH_STAGE1_MERGED="$STAGE1"

DATA_VI="$ROOT/work/stage2/corrected_vi_gemini"
DATA_REASONING="$ROOT/work/stage2/corrected_reasoning"
DATA_FULL="$ROOT/work/stage2/corrected_full"
BUNDLE_DIR="${STAGE2_BUNDLE_DIR:-$ROOT/zip}"
RUN_VI="$ROOT/runs-stage2/qwen3_8b/corrected_vi_gemini_1epoch"
RUN_REASONING="$ROOT/runs-stage2/qwen3_8b/corrected_reasoning_1epoch"
RUN_FULL="$ROOT/runs-stage2/qwen3_8b/corrected_full_1epoch"
MERGED_ROOT="$ROOT/runs-stage2/qwen3_8b/corrected_merged"
EVAL_ROOT="${STAGE2_EVAL_ROOT:-$ROOT/runs-stage2/qwen3_8b/corrected_study/evaluations}"
EVAL_SMOKE_ROOT="${STAGE2_EVAL_SMOKE_ROOT:-$ROOT/runs-stage2/qwen3_8b/corrected_study/eval-smoke}"
LOG_ROOT="${STAGE2_LOG_ROOT:-$ROOT/logs/stage2-corrected}"
mkdir -p "$LOG_ROOT" "$MERGED_ROOT" "$EVAL_ROOT" "$EVAL_SMOKE_ROOT"

case "$PHASE" in
  all|bundle-all|install|preflight|data|smoke|train|eval|--eval-worker) ;;
  *)
    echo "Usage: bash scripts/run_stage2_corrected_study.sh [all|bundle-all|install|preflight|data|smoke|train|eval]" >&2
    exit 2
    ;;
esac

BENCHMARK_ARGS=(
  --benchmark "cultureguard_jb=$ROOT/work/benchmarks/cultureguard_jb_9lang.jsonl"
  --benchmark "cultureguard_standard=$ROOT/work/benchmarks/cultureguard_standard_9lang.jsonl"
  --benchmark "multijail=$ROOT/work/benchmarks/multijail_4lang.jsonl"
  --benchmark "polyguard_prompts=$ROOT/work/benchmarks/polyguard_prompts_9lang.jsonl"
  --benchmark "sea_vi=$ROOT/work/benchmarks/sea_safeguard_vi.jsonl"
  --benchmark "xsafety=$ROOT/work/benchmarks/xsafety_multilingual.jsonl"
)

BENCHMARK_COUNTS=(
  "$ROOT/work/benchmarks/cultureguard_jb_9lang.jsonl" 13266
  "$ROOT/work/benchmarks/cultureguard_standard_9lang.jsonl" 24993
  "$ROOT/work/benchmarks/multijail_4lang.jsonl" 1260
  "$ROOT/work/benchmarks/polyguard_prompts_9lang.jsonl" 30906
  "$ROOT/work/benchmarks/sea_safeguard_vi.jsonl" 1840
  "$ROOT/work/benchmarks/xsafety_multilingual.jsonl" 19600
)

require_file() {
  test -f "$1" || { echo "Missing required file: $1" >&2; exit 1; }
}

require_benchmarks() {
  local spec path expected actual
  for ((index=1; index<${#BENCHMARK_ARGS[@]}; index+=2)); do
    spec="${BENCHMARK_ARGS[$index]}"
    path="${spec#*=}"
    require_file "$path"
  done
  # Counts are part of the established evaluation contract.  Hashes can differ
  # after harmless JSONL serialization, but a row-count mismatch means this is
  # not the benchmark suite used by the baseline and must fail before GPU work.
  for ((index=0; index<${#BENCHMARK_COUNTS[@]}; index+=2)); do
    path="${BENCHMARK_COUNTS[$index]}"
    expected="${BENCHMARK_COUNTS[$((index + 1))]}"
    actual="$(wc -l < "$path")"
    actual="${actual//[!0-9]/}"
    if [[ "$actual" != "$expected" ]]; then
      echo "Benchmark row-count mismatch: $path expected=$expected actual=$actual" >&2
      exit 1
    fi
  done
}

dataset_ready() {
  local directory="$1"
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if p.get("full_ready") and p.get("training_ready") and not p.get("blockers") else 1)' "$directory/manifest.json" 2>/dev/null
}

smoke_dataset_clean() {
  local directory="$1"
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if not p.get("source_blockers") and not p.get("integrity_blockers") else 1)' "$directory/manifest.json"
}

build_arm() {
  local name="$1"
  local output="$2"
  local smoke_count="$3"
  shift 3
  local smoke_args=()
  if [[ "$smoke_count" != "0" ]]; then
    smoke_args=(--smoke-per-source "$smoke_count")
  elif [[ "${STAGE2_FORCE_DATA:-0}" != "1" ]] && dataset_ready "$output"; then
    echo "[$(date -Is)] DATA SKIP $name: ready manifest exists"
    return 0
  fi
  echo "[$(date -Is)] DATA BUILD $name -> $output"
  "$PYTHON_BIN" stage2.py build \
    --config stage2_config.yaml \
    --translation-source gemini \
    --integrity-policy quarantine \
    --output-dir "$output" \
    "${smoke_args[@]}" \
    "$@"
  "$PYTHON_BIN" stage2.py validate --data-dir "$output"
  if [[ "$smoke_count" == "0" ]]; then
    dataset_ready "$output"
  else
    smoke_dataset_clean "$output"
  fi
}

build_all_data() {
  require_benchmarks
  build_arm vi-smoke "$ROOT/work/stage2/_smoke_corrected_vi" 8 \
    --exclude-source v3 --exclude-source wildguard --exclude-source reasoning --exclude-source nemotron35
  build_arm reasoning-smoke "$ROOT/work/stage2/_smoke_corrected_reasoning" 8 \
    --exclude-source v3 --exclude-source vi --exclude-source wildguard --exclude-source nemotron35
  build_arm full-smoke "$ROOT/work/stage2/_smoke_corrected_full" 8
  build_arm vi "$DATA_VI" 0 \
    --exclude-source v3 --exclude-source wildguard --exclude-source reasoning --exclude-source nemotron35
  build_arm reasoning "$DATA_REASONING" 0 \
    --exclude-source v3 --exclude-source vi --exclude-source wildguard --exclude-source nemotron35
  build_arm full "$DATA_FULL" 0
}

install_bundle() {
  local name="$1"
  local archive="$2"
  local output="$3"
  if [[ "${STAGE2_FORCE_DATA:-0}" != "1" ]] && dataset_ready "$output"; then
    echo "[$(date -Is)] INSTALL SKIP $name: ready manifest exists"
    return 0
  fi
  require_file "$archive"
  if [[ -e "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to replace non-empty dataset directory: $output" >&2
    exit 1
  fi
  echo "[$(date -Is)] INSTALL $name <- $archive"
  "$PYTHON_BIN" scripts/unpack_stage2_bundle.py \
    --zip "$archive" --output-dir "$output" --flatten-data
  "$PYTHON_BIN" stage2.py validate --data-dir "$output"
  dataset_ready "$output"
}

install_bundle_data() {
  install_bundle corrected_vi \
    "$BUNDLE_DIR/stage2_corrected_vi_gemini_1e.zip" "$DATA_VI"
  install_bundle corrected_reasoning \
    "$BUNDLE_DIR/stage2_corrected_reasoning_1e.zip" "$DATA_REASONING"
  install_bundle corrected_full \
    "$BUNDLE_DIR/stage2_corrected_full_1e.zip" "$DATA_FULL"
}

require_four_free_gpus() {
  require_file "$STAGE1/config.json"
  require_file "$STAGE1/tokenizer_config.json"
  "$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); print(f"CUDA devices: {n}"); raise SystemExit(0 if n >= 4 else 1)'
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${STAGE2_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    local used
    while IFS= read -r used; do
      used="${used//[!0-9]/}"
      if [[ -n "$used" && "$used" -gt 2048 ]]; then
        echo "A GPU already uses ${used} MiB. Stop the old job, or set STAGE2_ALLOW_BUSY_GPUS=1 after checking it manually." >&2
        exit 1
      fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  fi
}

preflight_ready() {
  require_benchmarks
  dataset_ready "$DATA_VI"
  dataset_ready "$DATA_REASONING"
  dataset_ready "$DATA_FULL"
  require_four_free_gpus
  if ! "$PYTHON_BIN" core/train.py --help 2>&1 | grep -q -- "--output-dir"; then
    echo "core/train.py is stale: missing --output-dir (required for isolated smoke outputs)." >&2
    echo "Update core/train.py from branch no-dataset commit c9fb666 or newer before using any GPU." >&2
    exit 1
  fi
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,vllm,yaml; print("Python training/eval imports: OK")'
  echo "[$(date -Is)] PREFLIGHT PASSED"
}

run_gpu_smoke() {
  local name="$1"
  local config="$2"
  local smoke_id="${STAGE2_SMOKE_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  local output="$ROOT/runs-stage2/_smoke_corrected/${smoke_id}_${name}"
  echo "[$(date -Is)] GPU SMOKE $name (1 optimizer step)"
  CUDA_VISIBLE_DEVICES="${STAGE2_TRAIN_GPUS:-0,1,2,3}" \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
    core/train.py --config "$config" --output-dir "$output" \
    --max-steps 1 --skip-eval --no-checkpoints --no-final-save \
    2>&1 | tee "$LOG_ROOT/smoke_${name}.log"
  require_file "$output/train_results.json"
}

smoke_all_training() {
  dataset_ready "$DATA_VI"
  dataset_ready "$DATA_REASONING"
  dataset_ready "$DATA_FULL"
  require_four_free_gpus
  run_gpu_smoke corrected_vi stage2_train_qwen3_8b_corrected_vi_1epoch.yaml
  run_gpu_smoke corrected_reasoning stage2_train_qwen3_8b_corrected_reasoning_1epoch.yaml
  run_gpu_smoke corrected_full stage2_train_qwen3_8b_corrected_full_1epoch.yaml
}

run_training() {
  local name="$1"
  local config="$2"
  local output="$3"
  local resume_args=()
  if [[ -f "$output/run_complete.json" && -f "$output/final/adapter_config.json" ]]; then
    echo "[$(date -Is)] TRAIN SKIP $name: complete final adapter exists"
    return 0
  fi
  if compgen -G "$output/checkpoint-*" >/dev/null; then
    resume_args=(--resume)
    echo "[$(date -Is)] TRAIN RESUME $name"
  else
    echo "[$(date -Is)] TRAIN START $name"
  fi
  CUDA_VISIBLE_DEVICES="${STAGE2_TRAIN_GPUS:-0,1,2,3}" \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
    core/train.py --config "$config" "${resume_args[@]}" \
    2>&1 | tee -a "$LOG_ROOT/train_${name}.log"
  require_file "$output/run_complete.json"
  require_file "$output/final/adapter_config.json"
  echo "[$(date -Is)] TRAIN DONE $name"
}

train_all() {
  dataset_ready "$DATA_VI"
  dataset_ready "$DATA_REASONING"
  dataset_ready "$DATA_FULL"
  require_four_free_gpus
  run_training corrected_vi stage2_train_qwen3_8b_corrected_vi_1epoch.yaml "$RUN_VI"
  run_training corrected_reasoning stage2_train_qwen3_8b_corrected_reasoning_1epoch.yaml "$RUN_REASONING"
  run_training corrected_full stage2_train_qwen3_8b_corrected_full_1epoch.yaml "$RUN_FULL"
}

ensure_merged() {
  local name="$1"
  local adapter="$2"
  local output="$3"
  require_file "$adapter/adapter_config.json"
  if [[ -f "$output/config.json" && -f "$output/tokenizer_config.json" && -f "$output/merge_manifest.json" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if p.get("adapter") == sys.argv[2] else 1)' "$output/merge_manifest.json" "$adapter" || {
      echo "Existing merge has a different adapter: $output" >&2
      exit 1
    }
    echo "[$(date -Is)] MERGE SKIP $name: verified existing output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "Incomplete merge directory already exists; move it aside before retrying: $output" >&2
    exit 1
  fi
  echo "[$(date -Is)] MERGE START $name on GPU 0"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" merge_adapter.py \
    --base-model "$STAGE1" --adapter "$adapter" --output "$output" --dtype bf16 \
    2>&1 | tee "$LOG_ROOT/merge_${name}.log"
  require_file "$output/config.json"
  require_file "$output/tokenizer_config.json"
  require_file "$output/merge_manifest.json"
}

merge_all() {
  ensure_merged corrected_vi "$RUN_VI/final" "$MERGED_ROOT/corrected_vi_1e"
  ensure_merged corrected_reasoning "$RUN_REASONING/final" "$MERGED_ROOT/corrected_reasoning_1e"
  ensure_merged corrected_full "$RUN_FULL/final" "$MERGED_ROOT/corrected_full_1e"
}

eval_complete() {
  local output="$1"
  [[ -f "$output/metrics.json" && -f "$output/progress.json" ]] && \
    grep -q '"status": "complete"' "$output/progress.json"
}

run_eval_cell() {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local taxonomy="$4"
  local thinking="$5"
  local thinking_tag="$thinking"
  [[ "$thinking_tag" == "no_think" ]] && thinking_tag="nothink"
  local tag="tax_${taxonomy}_${thinking_tag}"
  local output="$EVAL_ROOT/$name/$tag"
  local log="$LOG_ROOT/eval_${name}_${tag}.log"
  local max_new_tokens=128
  local sample_args=()
  [[ "$thinking" == "think" ]] && max_new_tokens="${STAGE2_THINK_MAX_NEW_TOKENS:-512}"
  if [[ -n "${STAGE2_EVAL_SAMPLE:-}" ]]; then
    sample_args=(--sample "$STAGE2_EVAL_SAMPLE")
  fi
  if [[ "${STAGE2_FORCE_EVAL:-0}" != "1" ]] && eval_complete "$output"; then
    echo "[$(date -Is)] EVAL SKIP GPU=$gpu $name $tag: complete"
    return 0
  fi
  echo "[$(date -Is)] EVAL START GPU=$gpu $name $tag"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/evaluate.py \
    --base-model "$model" \
    --family nemotron \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${STAGE2_EVAL_GPU_MEMORY:-0.88}" \
    --batch-size 4 \
    --vllm-chunk-size "${STAGE2_VLLM_CHUNK_SIZE:-256}" \
    --decoding-profile greedy \
    --taxonomy-mode "$taxonomy" \
    --thinking-mode "$thinking" \
    --max-new-tokens "$max_new_tokens" \
    --seed 3407 \
    --parse-error-policy incorrect \
    "${sample_args[@]}" \
    "${BENCHMARK_ARGS[@]}" \
    --output-dir "$output" \
    2>&1 | tee "$log"
  require_file "$output/metrics.json"
  grep -q '"status": "complete"' "$output/progress.json"
  "$PYTHON_BIN" nvidia_report.py --metrics "$output/metrics.json" --out "$output/nvidia_report.json" \
    > "$output/nvidia_report.stdout.json"
  echo "[$(date -Is)] EVAL DONE GPU=$gpu $name $tag"
}

eval_worker() {
  local gpu="$1"
  case "$gpu" in
    0)
      run_eval_cell 0 corrected_vi_1e "$MERGED_ROOT/corrected_vi_1e" on no_think
      run_eval_cell 0 corrected_vi_1e "$MERGED_ROOT/corrected_vi_1e" off no_think
      run_eval_cell 0 corrected_vi_1e "$MERGED_ROOT/corrected_vi_1e" on think
      ;;
    1)
      run_eval_cell 1 corrected_reasoning_1e "$MERGED_ROOT/corrected_reasoning_1e" on no_think
      run_eval_cell 1 corrected_reasoning_1e "$MERGED_ROOT/corrected_reasoning_1e" off no_think
      run_eval_cell 1 corrected_reasoning_1e "$MERGED_ROOT/corrected_reasoning_1e" on think
      ;;
    2)
      run_eval_cell 2 corrected_full_1e "$MERGED_ROOT/corrected_full_1e" on no_think
      run_eval_cell 2 corrected_full_1e "$MERGED_ROOT/corrected_full_1e" off no_think
      run_eval_cell 2 corrected_full_1e "$MERGED_ROOT/corrected_full_1e" on think
      ;;
    3)
      run_eval_cell 3 corrected_vi_1e "$MERGED_ROOT/corrected_vi_1e" off think
      run_eval_cell 3 corrected_reasoning_1e "$MERGED_ROOT/corrected_reasoning_1e" off think
      run_eval_cell 3 corrected_full_1e "$MERGED_ROOT/corrected_full_1e" off think
      ;;
    *) echo "Invalid eval worker GPU: $gpu" >&2; exit 2 ;;
  esac
}

eval_all() {
  require_benchmarks
  require_four_free_gpus
  merge_all
  local pids=()
  local failed=0
  # Exercise model loading, all prompt modes, all benchmark adapters and report
  # generation first. Smoke outputs live apart from full-study metrics.
  echo "[$(date -Is)] EVAL SMOKE MATRIX START (3 models x 4 modes, sample=8/benchmark)"
  trap 'for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
  for gpu in 0 1 2 3; do
    STAGE2_EVAL_SAMPLE=8 \
      STAGE2_EVAL_ROOT="$EVAL_SMOKE_ROOT" \
      bash "$0" --eval-worker "$gpu" \
      > "$LOG_ROOT/eval_smoke_worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == "0" ]] || {
    echo "At least one eval smoke worker failed; inspect $LOG_ROOT/eval_smoke_worker_gpu*.log" >&2
    exit 1
  }
  echo "[$(date -Is)] EVAL SMOKE MATRIX PASSED"

  pids=()
  failed=0
  echo "[$(date -Is)] FULL EVAL MATRIX START"
  for gpu in 0 1 2 3; do
    bash "$0" --eval-worker "$gpu" > "$LOG_ROOT/eval_worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  trap - INT TERM
  [[ "$failed" == "0" ]] || { echo "At least one eval worker failed; inspect $LOG_ROOT/eval_worker_gpu*.log" >&2; exit 1; }
  "$PYTHON_BIN" scripts/summarize_stage2_study.py \
    --eval-root "$EVAL_ROOT" \
    --output-prefix "$ROOT/runs-stage2/qwen3_8b/corrected_study/results"
  "$PYTHON_BIN" scripts/summarize_stage2_behavior.py \
    --eval-root "$EVAL_ROOT" \
    --output-prefix "$ROOT/runs-stage2/qwen3_8b/corrected_study/behavior"
}

if [[ "$PHASE" == "--eval-worker" ]]; then
  eval_worker "${2:?missing GPU for eval worker}"
  exit 0
fi

case "$PHASE" in
  all)
    build_all_data
    preflight_ready
    smoke_all_training
    train_all
    eval_all
    ;;
  bundle-all)
    # Company bundle mode must not spend hours training only to discover that
    # the full evaluation suite was never copied into the fresh checkout.
    require_benchmarks
    install_bundle_data
    preflight_ready
    smoke_all_training
    train_all
    eval_all
    ;;
  install) install_bundle_data ;;
  preflight) preflight_ready ;;
  data) build_all_data ;;
  smoke) smoke_all_training ;;
  train) train_all ;;
  eval) eval_all ;;
esac

echo "[$(date -Is)] STAGE-2 CORRECTED STUDY COMPLETE phase=$PHASE"
