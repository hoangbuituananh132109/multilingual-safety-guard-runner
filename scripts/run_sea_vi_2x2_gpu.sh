#!/usr/bin/env bash
set -euo pipefail

# Run the SEA-VI taxonomy x thinking 2x2 matrix on one assigned GPU.
# Launch this script once per GPU (0, 1, 2, 3). Each GPU has a fixed model
# queue, and every cell is run sequentially to avoid two vLLM engines sharing
# one A30.

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-3]$ ]]; then
  echo "Usage: bash scripts/run_sea_vi_2x2_gpu.sh <GPU: 0|1|2|3>" >&2
  exit 2
fi

GPU="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SEA_FILE="${SEA_FILE:-$ROOT/work/benchmarks/sea_safeguard_vi.jsonl}"
STAGE1="${MODEL_PATH_STAGE1_MERGED:-$ROOT/runs/qwen3_8b/lora_full/merged}"
QWEN3_BASE="${QWEN3_BASE_MODEL:-Qwen/Qwen3-8B}"
OUTPUT_ROOT="${SEA_OUTPUT_ROOT:-$ROOT/runs-stage2/eval-sea-2x2}"
LOG_ROOT="${SEA_LOG_ROOT:-$ROOT/logs/sea-vi-2x2}"
MODE_FILTER="${SEA_MODES:-all}"
MAX_NEW_TOKENS="${SEA_MAX_NEW_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${SEA_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_CHUNK_SIZE="${SEA_VLLM_CHUNK_SIZE:-100}"
FORCE="${SEA_FORCE:-0}"

case "$MODE_FILTER" in
  all|think|no_think) ;;
  *)
    echo "SEA_MODES must be all, think, or no_think (got: $MODE_FILTER)" >&2
    exit 2
    ;;
esac

test -f "$SEA_FILE" || { echo "Missing SEA-VI benchmark: $SEA_FILE" >&2; exit 1; }
test -f "$STAGE1/config.json" || { echo "Missing Stage-1 merged model: $STAGE1/config.json" >&2; exit 1; }
test -f "$STAGE1/tokenizer_config.json" || { echo "Missing Stage-1 tokenizer: $STAGE1/tokenizer_config.json" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

select_adapter() {
  local preferred="$1"
  local fallback="$2"
  if [[ -f "$preferred/adapter_config.json" ]]; then
    printf '%s\n' "$preferred"
  elif [[ -f "$fallback/adapter_config.json" ]]; then
    printf '%s\n' "$fallback"
  else
    echo "Neither adapter exists:" >&2
    echo "  $preferred" >&2
    echo "  $fallback" >&2
    return 1
  fi
}

ensure_merged() {
  local adapter="$1"
  local output="$2"
  if [[ -f "$output/config.json" && -f "$output/tokenizer_config.json" ]]; then
    echo "[$(date -Is)] MERGE SKIP: $output already exists"
    return 0
  fi
  echo "[$(date -Is)] MERGE START: $adapter -> $output on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python3 merge_adapter.py \
    --base-model "$STAGE1" \
    --adapter "$adapter" \
    --output "$output" \
    --dtype bf16
  test -f "$output/config.json"
  test -f "$output/tokenizer_config.json"
  echo "[$(date -Is)] MERGE DONE: $output"
}

run_cell() {
  local name="$1"
  local model="$2"
  local taxonomy_mode="$3"
  local thinking_mode="$4"
  local thinking_tag="$thinking_mode"
  if [[ "$thinking_tag" == "no_think" ]]; then
    thinking_tag="nothink"
  fi
  local tag="tax_${taxonomy_mode}_${thinking_tag}"
  if [[ -n "${SEA_SAMPLE:-}" ]]; then
    tag="${tag}_sample${SEA_SAMPLE}"
  fi
  local output="$OUTPUT_ROOT/$name/$tag"
  local log="$LOG_ROOT/${name}_${tag}.log"

  if [[ "$FORCE" != "1" && -f "$output/metrics.json" ]] && \
     grep -q '"status": "complete"' "$output/progress.json" 2>/dev/null; then
    echo "[$(date -Is)] SKIP GPU=$GPU model=$name cell=$tag (complete metrics exist)"
    return 0
  fi

  local sample_args=()
  if [[ -n "${SEA_SAMPLE:-}" ]]; then
    sample_args=(--sample "$SEA_SAMPLE")
  fi

  echo "[$(date -Is)] START GPU=$GPU model=$name cell=$tag"
  CUDA_VISIBLE_DEVICES="$GPU" python3 core/evaluate.py \
    --base-model "$model" \
    --family nemotron \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --batch-size 4 \
    --vllm-chunk-size "$VLLM_CHUNK_SIZE" \
    --decoding-profile greedy \
    --taxonomy-mode "$taxonomy_mode" \
    --thinking-mode "$thinking_mode" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed 3407 \
    --benchmark "sea_vi=$SEA_FILE" \
    --output-dir "$output" \
    "${sample_args[@]}" \
    2>&1 | tee "$log"
  test -f "$output/metrics.json"
  # core/evaluate.py writes the terminal state as "complete".
  grep -q '"status": "complete"' "$output/progress.json"
  echo "[$(date -Is)] DONE GPU=$GPU model=$name cell=$tag"
}

run_matrix() {
  local name="$1"
  local model="$2"
  case "$MODE_FILTER" in
    all)
      run_cell "$name" "$model" on no_think
      run_cell "$name" "$model" off no_think
      run_cell "$name" "$model" on think
      run_cell "$name" "$model" off think
      ;;
    no_think)
      run_cell "$name" "$model" on no_think
      run_cell "$name" "$model" off no_think
      ;;
    think)
      run_cell "$name" "$model" on think
      run_cell "$name" "$model" off think
      ;;
  esac
}

run_stage2_model() {
  local name="$1"
  local preferred_adapter="$2"
  local fallback_adapter="$3"
  local merged="$4"
  local adapter
  adapter="$(select_adapter "$preferred_adapter" "$fallback_adapter")"
  ensure_merged "$adapter" "$merged"
  run_matrix "$name" "$merged"
}

run_stage2_nothink_model() {
  local name="$1"
  local preferred_adapter="$2"
  local fallback_adapter="$3"
  local merged="$4"
  local adapter
  if [[ "$MODE_FILTER" == "think" ]]; then
    echo "[$(date -Is)] SKIP model=$name: VI-only models were requested for no-think only"
    return 0
  fi
  adapter="$(select_adapter "$preferred_adapter" "$fallback_adapter")"
  ensure_merged "$adapter" "$merged"
  run_cell "$name" "$merged" on no_think
  run_cell "$name" "$merged" off no_think
}

echo "[$(date -Is)] GPU $GPU queue starting; modes=$MODE_FILTER max_new_tokens=$MAX_NEW_TOKENS"

case "$GPU" in
  0)
    run_matrix qwen3_base "$QWEN3_BASE"
    run_stage2_model \
      full_1e \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy/final" \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch/checkpoint-7305" \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch/merged_epoch1"
    ;;
  1)
    run_matrix qwen3_v3_5e "$STAGE1"
    run_stage2_model \
      full_5e \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch/final" \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch/checkpoint-36525" \
      "$ROOT/runs-stage2/qwen3_8b/lora_phase2_gemini_policy_5epoch/merged"
    ;;
  2)
    run_stage2_model \
      reasoning_1e \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_1epoch/final" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch/checkpoint-1109" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch/merged_epoch1"
    run_stage2_nothink_model \
      vi_1e \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_1epoch/final" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch/checkpoint-1583" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch/merged_epoch1"
    ;;
  3)
    run_stage2_model \
      reasoning_5e \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch/final" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch/checkpoint-5545" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_reasoning_only_5epoch/merged"
    run_stage2_nothink_model \
      vi_5e \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch/final" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch/checkpoint-7915" \
      "$ROOT/runs-stage2/qwen3_8b/ablation_vi_gemini_5epoch/merged"
    ;;
esac

echo "[$(date -Is)] GPU $GPU queue complete"
