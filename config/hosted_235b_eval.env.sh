# Edit this one file to evaluate an already-hosted OpenAI-compatible model.
# HOSTED_MODEL_ENDPOINT accepts either http://HOST:PORT/v1 or the full
# http://HOST:PORT/v1/chat/completions URL.

export HOSTED_MODEL_ENDPOINT="${HOSTED_MODEL_ENDPOINT:-http://DIEN_DIA_CHI_MODEL:8000/v1}"
export HOSTED_MODEL_NAME="${HOSTED_MODEL_NAME:-Qwen3-235B-A22B}"

# Leave empty when the internal vLLM server does not require a bearer token.
export HOSTED_MODEL_API_KEY="${HOSTED_MODEL_API_KEY:-}"

export HOSTED_BENCHMARK_ZIP="${HOSTED_BENCHMARK_ZIP:-zip/qwen3_safety_benchmark_total_v2.zip}"
export HOSTED_BENCHMARK_EXTRACT_DIR="${HOSTED_BENCHMARK_EXTRACT_DIR:-work/eval-hosted/benchmark-total}"
export HOSTED_EVAL_OUTPUT_ROOT="${HOSTED_EVAL_OUTPUT_ROOT:-runs/hosted-qwen3-235b}"

# Start conservatively. Raise only after the smoke passes and the server has
# spare request capacity.
export HOSTED_EVAL_CONCURRENCY="${HOSTED_EVAL_CONCURRENCY:-8}"
export HOSTED_EVAL_MAX_TOKENS="${HOSTED_EVAL_MAX_TOKENS:-16}"
export HOSTED_EVAL_TIMEOUT="${HOSTED_EVAL_TIMEOUT:-120}"
export HOSTED_EVAL_RETRIES="${HOSTED_EVAL_RETRIES:-2}"
export HOSTED_EVAL_PROGRESS_EVERY="${HOSTED_EVAL_PROGRESS_EVERY:-100}"
