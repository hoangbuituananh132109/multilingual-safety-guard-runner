#!/usr/bin/env bash
# Edit this ONE file inside nvcr.io/nvidia/nemo:26.08. No network/install commands.
export DG_IMAGE='nvcr.io/nvidia/nemo:26.08'
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export DG_MODEL_PATH='/workspace/storage-shared/models/Qwen3-4B'
export DG_DATA_ZIP='zip/nemotron_wildguard_scaled_max_70_30_v2.gated.zip'
export DG_DATA_DIR='work/decision-guard/data-70-30'
export DG_CACHE_ROOT='work/decision-guard/tokens-70-30-v1'
export DG_RUN_ROOT='runs-decision-guard/qwen3_4b_70_30_v1'
export DG_LOG_ROOT='logs/decision-guard/qwen3_4b_70_30_v1'
export DG_PYTHON='python3'
# All seven arms start from the same pretrained checkpoint. One epoch each.
export DG_ARMS='json36 binary36 pointer36 binary27 pointer27 binary18 pointer18'
# Global batch stays fixed when switching between 1/2/4/8 GPUs; accumulation is automatic.
export DG_GLOBAL_BATCH=32
export DG_MICRO_BATCH=1
export DG_MAX_LENGTH=2048
export DG_LR=1e-5
export DG_SEED=3407
export DG_SAVE_EVERY=250
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
