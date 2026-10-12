#!/usr/bin/env bash
# Foreground runner: every phase is visible and logged. Never nohup/download/install.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/decision_guard/settings.env.sh"
PHASE="${1:-}"
ARM_FILTER="${2:-all}"
case "$PHASE" in install|preflight|prepare|smoke|train|all) ;; *)
  echo 'Usage: bash decision_guard/run.sh install|preflight|prepare|smoke|train|all [arm|all]'; exit 2;; esac
mkdir -p "$DG_LOG_ROOT" "$DG_RUN_ROOT"
command -v "$DG_PYTHON" >/dev/null
command -v flock >/dev/null || { echo 'ERROR: flock missing in fixed image'; exit 1; }
# Lock held for this entire runner; a second copy fails before starting any model process.
exec 9>"$DG_RUN_ROOT/.matrix.lock"
flock -n 9 || { echo 'ERROR: a decision-guard runner is already active in this run root'; exit 1; }
STAMP="$(date +%Y%m%d-%H%M%S)"
exec > >(tee -a "$DG_LOG_ROOT/${PHASE}_${STAMP}.master.log") 2>&1
printf '[%s] phase=%s arms=%s GPUs=%s\n' "$(date -Is)" "$PHASE" "$ARM_FILTER" "$CUDA_VISIBLE_DEVICES"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
(( DG_GLOBAL_BATCH > 0 && DG_MICRO_BATCH > 0 && GPU_COUNT >= 1 && DG_GLOBAL_BATCH % (DG_MICRO_BATCH * GPU_COUNT) == 0 )) || {
  echo 'ERROR: global batch must be divisible by micro batch times GPU count'; exit 1;
}
read -r -a ARMS <<< "$DG_ARMS"
if [[ "$ARM_FILTER" != all ]]; then ARMS=("$ARM_FILTER"); fi
"$DG_PYTHON" -c 'import sys; from decision_guard.train import validate_arm; [validate_arm(x) for x in sys.argv[1:]]' "${ARMS[@]}"

install_data() {
  test -f "$DG_DATA_ZIP" || { echo "ERROR: manually copy data ZIP to $DG_DATA_ZIP"; exit 1; }
  "$DG_PYTHON" -m decision_guard.data install --zip "$DG_DATA_ZIP" --data-dir "$DG_DATA_DIR"
  "$DG_PYTHON" -m decision_guard.data audit --data-dir "$DG_DATA_DIR"
}
preflight() {
  "$DG_PYTHON" -m decision_guard.preflight --model "$DG_MODEL_PATH" --output "$DG_CACHE_ROOT/model_receipt.json"
  "$DG_PYTHON" -m decision_guard.data audit --data-dir "$DG_DATA_DIR"
}
prepare_cache() {
  local mode="$1"
  local extra=()
  if [[ "$mode" == smoke ]]; then extra=(--smoke); fi
  "$DG_PYTHON" -m decision_guard.data prepare --data-dir "$DG_DATA_DIR" \
    --model "$DG_MODEL_PATH" --output "$DG_CACHE_ROOT/$mode" --max-length "$DG_MAX_LENGTH" "${extra[@]}"
}
run_arms() {
  local mode="$1" arm output
  local extra=()
  test -f "$DG_CACHE_ROOT/model_receipt.json" || { echo 'ERROR: run preflight first'; exit 1; }
  prepare_cache "$mode"
  for arm in "${ARMS[@]}"; do
    output="$DG_RUN_ROOT/$arm"
    if [[ "$mode" == smoke ]]; then output="$DG_RUN_ROOT/_smoke/$arm"; fi
    if [[ "$mode" == smoke ]]; then extra=(--smoke); else
      extra=(--smoke-certificate "$DG_RUN_ROOT/_smoke/$arm/smoke_pass.json")
      test -f "${extra[1]}" || { echo "ERROR: smoke $arm first"; exit 1; }
    fi
    mkdir -p "$DG_LOG_ROOT/$arm"
    printf '[%s] START %s %s\n' "$(date -Is)" "$mode" "$arm"
    "$DG_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
      -m decision_guard.train --model "$DG_MODEL_PATH" --cache "$DG_CACHE_ROOT/$mode" \
      --model-receipt "$DG_CACHE_ROOT/model_receipt.json" --arm "$arm" --output "$output" \
      --global-batch "$DG_GLOBAL_BATCH" --micro-batch "$DG_MICRO_BATCH" --lr "$DG_LR" \
      --seed "$DG_SEED" --save-every "$DG_SAVE_EVERY" "${extra[@]}" \
      2>&1 | tee -a "$DG_LOG_ROOT/$arm/${mode}_${STAMP}.log"
    printf '[%s] PASS %s %s\n' "$(date -Is)" "$mode" "$arm"
  done
}
case "$PHASE" in
  install) install_data ;;
  preflight) preflight ;;
  prepare) prepare_cache full ;;
  smoke) run_arms smoke ;;
  train) run_arms full ;;
  all) install_data; preflight; run_arms smoke; run_arms full ;;
esac
printf '[%s] COMPLETE phase=%s\n' "$(date -Is)" "$PHASE"
