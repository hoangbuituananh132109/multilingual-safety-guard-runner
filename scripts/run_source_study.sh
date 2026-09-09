#!/usr/bin/env bash
set -euo pipefail

# Fair Qwen3-4B source screen: preflight -> 2-step smokes -> three 1E runs
# -> merges -> four independent one-GPU evaluations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${SOURCE_STUDY_PYTHON_BIN:-python3}"
PHASE="${1:-all}"
MODEL="Qwen/Qwen3-4B"
MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
RUN_ROOT="$ROOT/runs-source-study/qwen3_4b"
MERGED_ROOT="$RUN_ROOT/merged"
EVAL_ROOT="$RUN_ROOT/evaluations"
LOG_ROOT="$ROOT/logs/source-study"
mkdir -p "$RUN_ROOT" "$MERGED_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

ARMS=(nemotron_v3_9lang wildguardtrain_en sea_cultural_vi)
CONFIGS=(
  source_study_train_qwen3_4b_nemotron_v3.yaml
  source_study_train_qwen3_4b_wildguard_en.yaml
  source_study_train_qwen3_4b_sea_vi.yaml
)
RUN_DIRS=(
  "$RUN_ROOT/nemotron_v3_9lang_80k_1epoch"
  "$RUN_ROOT/wildguardtrain_en_80k_1epoch"
  "$RUN_ROOT/sea_cultural_vi_80k_1epoch"
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
)
BENCHMARK_COUNTS=(13266 24993 1260 30906 1840 19600 450 3408 26644)

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

require_four_free_gpus() {
  "$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); print(f"CUDA devices: {n}"); raise SystemExit(0 if n >= 4 else 1)'
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SOURCE_STUDY_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    while IFS= read -r used; do
      used="${used//[!0-9]/}"
      [[ -z "$used" || "$used" -le 2048 ]] || die "GPU already uses ${used} MiB; stop old jobs or explicitly set SOURCE_STUDY_ALLOW_BUSY_GPUS=1"
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  fi
}

require_data_and_benchmarks() {
  local arm spec path actual benchmark_index=0
  for arm in "${ARMS[@]}"; do
    "$PYTHON_BIN" source_study.py validate --data-dir "$ROOT/work/source-study/$arm" >/dev/null
  done
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

preflight() {
  require_data_and_benchmarks
  require_four_free_gpus
  "$PYTHON_BIN" -c 'import accelerate,datasets,peft,torch,transformers,vllm,yaml; print("training/eval imports: OK")'
  "$PYTHON_BIN" scripts/smoke_source_study_contract.py --sample-per-arm 64 >/dev/null
  echo "[$(date -Is)] PREFLIGHT PASSED"
}

gpu_smoke() {
  require_data_and_benchmarks
  require_four_free_gpus
  local config name output
  for i in 0 1 2; do
    config="${CONFIGS[$i]}"; name="${ARMS[$i]}"
    output="$RUN_ROOT/_smoke/${name}_2steps"
    echo "[$(date -Is)] GPU SMOKE $name: exactly 2 optimizer steps"
    CUDA_VISIBLE_DEVICES="${SOURCE_STUDY_TRAIN_GPUS:-0,1,2,3}" \
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
      core/train.py --config "$config" --output-dir "$output" \
      --max-steps 2 --skip-eval --no-checkpoints --no-final-save \
      2>&1 | tee "$LOG_ROOT/smoke_${name}.log"
    require_file "$output/train_results.json"
  done
  echo "[$(date -Is)] ALL GPU SMOKES PASSED"
}

train_one() {
  local name="$1" config="$2" output="$3"
  local resume=()
  if [[ -f "$output/run_complete.json" && -f "$output/final/adapter_config.json" ]]; then
    echo "[$(date -Is)] TRAIN SKIP $name: complete"
    return
  fi
  if compgen -G "$output/checkpoint-*" >/dev/null; then
    resume=(--resume)
    echo "[$(date -Is)] TRAIN RESUME $name"
  else
    echo "[$(date -Is)] TRAIN START $name"
  fi
  CUDA_VISIBLE_DEVICES="${SOURCE_STUDY_TRAIN_GPUS:-0,1,2,3}" \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
    core/train.py --config "$config" "${resume[@]}" \
    2>&1 | tee -a "$LOG_ROOT/train_${name}.log"
  require_file "$output/run_complete.json"
  require_file "$output/final/adapter_config.json"
}

train_all() {
  require_data_and_benchmarks
  require_four_free_gpus
  for i in 0 1 2; do train_one "${ARMS[$i]}" "${CONFIGS[$i]}" "${RUN_DIRS[$i]}"; done
}

merge_one() {
  local name="$1" adapter="$2" output="$3"
  if [[ -f "$output/config.json" && -f "$output/tokenizer_config.json" && -f "$output/merge_manifest.json" ]]; then
    echo "[$(date -Is)] MERGE SKIP $name: complete"
    return
  fi
  [[ ! -e "$output" ]] || die "incomplete merge directory exists: $output"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" merge_adapter.py \
    --base-model "$MODEL" --revision "$MODEL_REVISION" \
    --adapter "$adapter" --output "$output" --dtype bf16 \
    2>&1 | tee "$LOG_ROOT/merge_${name}.log"
  require_file "$output/config.json"
  require_file "$output/merge_manifest.json"
}

merge_all() {
  for i in 0 1 2; do merge_one "${ARMS[$i]}" "${RUN_DIRS[$i]}/final" "$MERGED_ROOT/${ARMS[$i]}"; done
}

eval_complete() { [[ -f "$1/metrics.json" ]] && grep -q '"status": "complete"' "$1/progress.json" 2>/dev/null; }

eval_one() {
  local gpu="$1" name="$2" model="$3" revision="$4"
  local output="$EVAL_ROOT/$name"
  if [[ "${SOURCE_STUDY_FORCE_EVAL:-0}" != "1" ]] && eval_complete "$output"; then
    echo "[$(date -Is)] EVAL SKIP $name: complete"
    return
  fi
  local sample=()
  [[ -z "${SOURCE_STUDY_EVAL_SAMPLE:-}" ]] || sample=(--sample "$SOURCE_STUDY_EVAL_SAMPLE")
  echo "[$(date -Is)] EVAL START GPU=$gpu model=$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" core/evaluate.py \
    --base-model "$model" --revision "$revision" --family nemotron \
    --backend vllm --tensor-parallel-size 1 \
    --gpu-memory-utilization "${SOURCE_STUDY_EVAL_GPU_MEMORY:-0.88}" \
    --batch-size 8 --vllm-chunk-size "${SOURCE_STUDY_VLLM_CHUNK_SIZE:-512}" \
    --decoding-profile greedy --taxonomy-mode off --thinking-mode no_think \
    --max-new-tokens 128 --seed 3407 --parse-error-policy incorrect \
    "${sample[@]}" "${BENCHMARK_ARGS[@]}" --output-dir "$output" \
    2>&1 | tee "$LOG_ROOT/eval_${name}.log"
  eval_complete "$output" || die "evaluation incomplete: $name"
}

eval_worker() {
  case "$1" in
    0) eval_one 0 qwen3_4b_base "$MODEL" "$MODEL_REVISION" ;;
    1) eval_one 1 nemotron_v3_9lang "$MERGED_ROOT/nemotron_v3_9lang" main ;;
    2) eval_one 2 wildguardtrain_en "$MERGED_ROOT/wildguardtrain_en" main ;;
    3) eval_one 3 sea_cultural_vi "$MERGED_ROOT/sea_cultural_vi" main ;;
    *) die "eval worker must be GPU 0,1,2,3" ;;
  esac
}

eval_all() {
  require_data_and_benchmarks
  require_four_free_gpus
  merge_all
  local pids=() failed=0
  trap 'for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
  for gpu in 0 1 2 3; do
    bash "$0" --eval-worker "$gpu" > "$LOG_ROOT/eval_worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  trap - INT TERM
  [[ "$failed" == 0 ]] || die "an eval worker failed; inspect $LOG_ROOT/eval_worker_gpu*.log"
  "$PYTHON_BIN" scripts/summarize_source_study.py --eval-root "$EVAL_ROOT"
}

if [[ "$PHASE" == "--eval-worker" ]]; then eval_worker "${2:?missing GPU}"; exit 0; fi
case "$PHASE" in
  all) preflight; gpu_smoke; train_all; merge_all; eval_all ;;
  study) train_all; merge_all; eval_all ;;
  preflight) preflight ;;
  smoke) gpu_smoke ;;
  train) train_all ;;
  merge) merge_all ;;
  eval-smoke) SOURCE_STUDY_EVAL_SAMPLE="${SOURCE_STUDY_EVAL_SAMPLE:-8}" eval_all ;;
  eval) eval_all ;;
  *) die "usage: bash scripts/run_source_study.sh [all|study|preflight|smoke|train|merge|eval-smoke|eval]" ;;
esac

echo "[$(date -Is)] SOURCE STUDY COMPLETE phase=$PHASE"
