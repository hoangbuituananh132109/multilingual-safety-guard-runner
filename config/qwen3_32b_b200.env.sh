# Edit this one file before running scripts/run_qwen3_32b_scaled.sh.
# Explicit environment variables supplied by the caller take precedence.

export SOURCE_STUDY_32B_MODEL_PATH="${SOURCE_STUDY_32B_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-32B}"
export SOURCE_STUDY_SCALE_RATIO="${SOURCE_STUDY_SCALE_RATIO:-30_70}"

# Physical GPU ids. Any unique subset is accepted when the target global batch
# is divisible by GPU count * microbatch. Examples: 0,1,2,3,4,5,6,7 or 2,4,6,7.
export SOURCE_STUDY_32B_TRAIN_GPUS="${SOURCE_STUDY_32B_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
export SOURCE_STUDY_32B_EVAL_GPU="${SOURCE_STUDY_32B_EVAL_GPU:-0}"
export SOURCE_STUDY_32B_MERGE_GPU="${SOURCE_STUDY_32B_MERGE_GPU:-${SOURCE_STUDY_32B_EVAL_GPU:-0}}"

# Runner derives gradient accumulation so this global batch remains fixed.
export SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH="${SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH:-32}"
export SOURCE_STUDY_32B_MICROBATCH="${SOURCE_STUDY_32B_MICROBATCH:-2}"

export SOURCE_STUDY_32B_EVAL_GPU_MEMORY="${SOURCE_STUDY_32B_EVAL_GPU_MEMORY:-0.90}"
export SOURCE_STUDY_32B_EVAL_BATCH="${SOURCE_STUDY_32B_EVAL_BATCH:-16}"
export SOURCE_STUDY_VLLM_CHUNK_SIZE="${SOURCE_STUDY_VLLM_CHUNK_SIZE:-512}"
export SOURCE_STUDY_MAX_MODEL_LEN="${SOURCE_STUDY_MAX_MODEL_LEN:-8192}"
