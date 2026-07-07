#!/bin/bash

# UD-VLA LIBERO Evaluation Script (Background)
# Usage: bash scripts/libero/run_eval_libero_background.sh [task_suite_name] [gpu_id]

# Default parameters
TASK_SUITE=${1:-libero_10}
GPU_ID=${2:-2}
STEPS=72

# Paths
PROJECT_ROOT="/mnt/storage/users/liangyiwen/projects/UD-VLA"
ROBOVLMS_ROOT="${PROJECT_ROOT}/reference/RoboVLMs"
EMU_HUB="${PROJECT_ROOT}/pretrain/UD-VLA_LIBERO"
VQ_HUB="${PROJECT_ROOT}/pretrain/Emu3-Stage1"
VISION_HUB="${PROJECT_ROOT}/pretrain/Emu3-VisionTokenizer"
ACTION_TOKENIZER="${PROJECT_ROOT}/pretrain/fast_libero_all_t10_s50"
CACHE_ROOT="./cache/libero_eval"

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${ROBOVLMS_ROOT}/logs/libero_eval_${TASK_SUITE}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${ROBOVLMS_ROOT}/logs"

cd ${ROBOVLMS_ROOT}

echo "=========================================="
echo "UD-VLA LIBERO Evaluation"
echo "=========================================="
echo "Task Suite: ${TASK_SUITE}"
echo "GPU ID: ${GPU_ID}"
echo "Diffusion Steps: ${STEPS}"
echo "Log File: ${LOG_FILE}"
echo "Started at: $(date)"
echo "=========================================="

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun \
    --nproc_per_node=1 \
    --master_addr="localhost" \
    --master_port=29500 \
    eval/libero/evaluate_libero_emu.py \
    --emu_hub ${EMU_HUB} \
    --vq_hub ${VQ_HUB} \
    --vision_hub ${VISION_HUB} \
    --task_suite_name ${TASK_SUITE} \
    --dis_i2a \
    --steps ${STEPS} \
    --cache_root ${CACHE_ROOT} \
    --action_tokenizer ${ACTION_TOKENIZER} \
    2>&1 | tee ${LOG_FILE}

echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to: ${CACHE_ROOT}/eval/"
echo "=========================================="
