#!/usr/bin/env bash
# Set local model paths for the company B200 cluster.
# Source this file before running any run.py command:
#   source ml_env.sh
# Env keys follow config.yaml model names uppercased: MODEL_PATH_<NAME>.

export MODEL_PATH_QWEN3_0_6B=/workspace/storage-shared/models/Qwen3-0.6B
export MODEL_PATH_QWEN3_1_7B=/workspace/storage-shared/models/Qwen3-1.7B
export MODEL_PATH_QWEN3_4B=/workspace/storage-shared/models/Qwen3-4B
export MODEL_PATH_QWEN3_8B=/workspace/storage-shared/models/Qwen3-8B

# config name 'llama31_8b_instruct' -> MODEL_PATH_LLAMA31_8B_INSTRUCT
export MODEL_PATH_LLAMA31_8B_INSTRUCT=/workspace/storage-shared/models/Llama-3.1-8B-Instruct
# config name 'llama31_8b' (duplicate entry) -> also point at the same model
export MODEL_PATH_LLAMA31_8B=/workspace/storage-shared/models/Llama-3.1-8B-Instruct

export MODEL_PATH_LLAMA31_NEMOTRON=/workspace/storage-shared/nlp/huypq51/models/Llama-3.1-Nemotron-Safety-Guard-8B-v3

# Qwen3.5 family
export MODEL_PATH_QWEN35_4B=/workspace/storage-shared/models/Qwen3.5-4B-Base
export MODEL_PATH_QWEN35_2B=/workspace/storage-shared/models/Qwen3.5-2B

echo 'ml_env loaded'
