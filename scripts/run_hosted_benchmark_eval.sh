#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SETTINGS_FILE="${HOSTED_EVAL_SETTINGS_FILE-$ROOT/config/hosted_235b_eval.env.sh}"
if [[ -n "$SETTINGS_FILE" ]]; then
  [[ -f "$SETTINGS_FILE" ]] || {
    echo "ERROR: settings file not found: $SETTINGS_FILE" >&2
    exit 1
  }
  # shellcheck source=/dev/null
  source "$SETTINGS_FILE"
fi

PHASE="${1:-show-config}"
PYTHON_BIN="${HOSTED_EVAL_PYTHON_BIN:-python3}"
ENDPOINT="${HOSTED_MODEL_ENDPOINT:-}"
MODEL="${HOSTED_MODEL_NAME:-}"
ARCHIVE="${HOSTED_BENCHMARK_ZIP:-zip/qwen3_safety_benchmark_total_v2.zip}"
EXTRACT_DIR="${HOSTED_BENCHMARK_EXTRACT_DIR:-work/eval-hosted/benchmark-total}"
OUTPUT_ROOT="${HOSTED_EVAL_OUTPUT_ROOT:-runs/hosted-qwen3-235b}"
CONCURRENCY="${HOSTED_EVAL_CONCURRENCY:-8}"
MAX_TOKENS="${HOSTED_EVAL_MAX_TOKENS:-16}"
TIMEOUT="${HOSTED_EVAL_TIMEOUT:-120}"
RETRIES="${HOSTED_EVAL_RETRIES:-2}"
PROGRESS_EVERY="${HOSTED_EVAL_PROGRESS_EVERY:-100}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "HOSTED_EVAL_CONCURRENCY must be positive"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "HOSTED_EVAL_MAX_TOKENS must be positive"
[[ "$RETRIES" =~ ^[0-9]+$ ]] || die "HOSTED_EVAL_RETRIES must be non-negative"
[[ -f "$ARCHIVE" ]] || [[ "$PHASE" == "show-config" ]] || die "missing local benchmark ZIP: $ARCHIVE"

export HOSTED_MODEL_ENDPOINT="$ENDPOINT"
export HOSTED_MODEL_NAME="$MODEL"
export HOSTED_BENCHMARK_ZIP="$ARCHIVE"
export HOSTED_BENCHMARK_EXTRACT_DIR="$EXTRACT_DIR"
export HOSTED_EVAL_OUTPUT_ROOT="$OUTPUT_ROOT"
export HOSTED_EVAL_CONCURRENCY="$CONCURRENCY"

show_config() {
  "$PYTHON_BIN" - <<'PY'
import json, os
print(json.dumps({
    "settings_file": os.environ.get("HOSTED_EVAL_SETTINGS_FILE", "config/hosted_235b_eval.env.sh"),
    "endpoint": os.environ.get("HOSTED_MODEL_ENDPOINT", ""),
    "model": os.environ.get("HOSTED_MODEL_NAME", ""),
    "api_key_set": bool(os.environ.get("HOSTED_MODEL_API_KEY")),
    "benchmark_zip": os.environ.get("HOSTED_BENCHMARK_ZIP", ""),
    "extract_dir": os.environ.get("HOSTED_BENCHMARK_EXTRACT_DIR", ""),
    "output_root": os.environ.get("HOSTED_EVAL_OUTPUT_ROOT", ""),
    "concurrency": int(os.environ.get("HOSTED_EVAL_CONCURRENCY", "8")),
}, indent=2))
PY
}

run_eval() {
  local output="$1"
  shift
  mkdir -p "$output" "$(dirname "$output")"
  "$PYTHON_BIN" scripts/evaluate_hosted_benchmark.py \
    --zip "$ARCHIVE" \
    --extract-dir "$EXTRACT_DIR" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --output-dir "$output" \
    --max-tokens "$MAX_TOKENS" \
    --timeout "$TIMEOUT" \
    --retries "$RETRIES" \
    --concurrency "$CONCURRENCY" \
    --progress-every "$PROGRESS_EVERY" \
    "$@"
}

case "$PHASE" in
  show-config)
    show_config
    ;;
  dry-run)
    mkdir -p "$OUTPUT_ROOT/dry-run"
    "$PYTHON_BIN" scripts/evaluate_hosted_benchmark.py \
      --zip "$ARCHIVE" --extract-dir "$EXTRACT_DIR" \
      --output-dir "$OUTPUT_ROOT/dry-run" \
      --model "$MODEL" --max-tokens "$MAX_TOKENS" \
      --limit-per-benchmark 1 --dry-run
    ;;
  smoke)
    [[ -n "$ENDPOINT" && "$ENDPOINT" != *"DIEN_DIA_CHI_MODEL"* ]] || die "fill HOSTED_MODEL_ENDPOINT first"
    run_eval "$OUTPUT_ROOT/smoke" --limit-per-benchmark "${HOSTED_EVAL_SMOKE_PER_BENCHMARK:-2}"
    ;;
  full)
    [[ -n "$ENDPOINT" && "$ENDPOINT" != *"DIEN_DIA_CHI_MODEL"* ]] || die "fill HOSTED_MODEL_ENDPOINT first"
    run_eval "$OUTPUT_ROOT/full"
    ;;
  *)
    die "usage: bash scripts/run_hosted_benchmark_eval.sh [show-config|dry-run|smoke|full]"
    ;;
esac
