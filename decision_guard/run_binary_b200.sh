#!/usr/bin/env bash
# One independent Qwen3-4B experiment, foreground and offline. Reuse the same
# paths on restart: resume checkpoints/eval parts, never restart a completed run.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARM="${1:-}"
PHASE="${2:-run}"
case "$ARM" in binary36|binary18) ;; *)
  echo 'Usage: bash decision_guard/run_binary_b200.sh binary36|binary18 [run|eval]'; exit 2;; esac
case "$PHASE" in run|eval) ;; *) echo 'Use run or eval'; exit 2;; esac
export DG_MICRO_BATCH="${DG_MICRO_BATCH:-4}"
export DG_SAVE_EVERY="${DG_SAVE_EVERY:-2000}"
export DG_DATA_DIR="${DG_DATA_DIR:-work/source-study-scaled/nemotron70_wildguard30_scaled}"
# Stable v2 output is isolated from old SFT and v1 decision-head artifacts.
# Separate cache/log/lock roots also allow binary36 and binary18 on two nodes.
export DG_RUN_ROOT="${DG_RUN_ROOT:-runs-decision-guard/qwen3_4b_70_30_binary_v2/$ARM}"
export DG_CACHE_ROOT="${DG_CACHE_ROOT:-work/decision-guard/binary-v2/$ARM}"
export DG_LOG_ROOT="${DG_LOG_ROOT:-logs/decision-guard/binary-v2/$ARM}"
export DG_BENCHMARK="${DG_BENCHMARK:-/workspace/storage-shared/nlp/anhhbt/multilingual-safety-guard-runner-b200-full-data-study/work/eval-hosted/benchmark-total/qwen3_safety_benchmark_total.jsonl}"
exec bash "$ROOT/decision_guard/run.sh" "$PHASE" "$ARM"
