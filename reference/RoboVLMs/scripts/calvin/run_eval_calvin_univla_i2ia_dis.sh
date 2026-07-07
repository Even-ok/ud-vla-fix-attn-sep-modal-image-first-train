#!/bin/bash

# Run
# export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN          # 启用 NCCL 调试信息
export NCCL_P2P_DISABLE=1        # 禁用 P2P 直连通信
export NCCL_IB_DISABLE=1         # 关闭 InfiniBand
export NCCL_SHM_DISABLE=1        # 禁用共享内存
export NCCL_LAUNCH_MODE=PARALLEL # 避免 NCCL 死锁
unset DISPLAY
export PYOPENGL_PLATFORM=egl
# export EGL_DEVICE_ID=0
# emu_hub=$1
emu_hub="/mnt/mnt/storage/projects/UD-VLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS64_24k_I2IA_mi_0514/checkpoint-24000"
PROJECT_ROOT="/mnt/mnt/storage/projects/UD-VLA"
VQ_HUB="${PROJECT_ROOT}/pretrain/Emu3-Stage1"
VISION_HUB="${PROJECT_ROOT}/pretrain/Emu3-VisionTokenizer"
# emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS64_24k_I2IA_mi_0905/checkpoint-24000" 

export PYTHONPATH=/mnt/mnt/storage/projects/UD-VLA/reference/RoboVLMs:/mnt/mnt/storage/projects/UD-VLA/reference/Emu3:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6099 eval/calvin/evaluate_ddp-emu.py \
--emu_hub $emu_hub \
--vq_hub ${VQ_HUB} \
--vision_hub ${VISION_HUB} \
--dataset_path /mnt/mnt/storage/datasets/calvin/dataset/task_ABCD_D \
--dis_i2a \
--steps 72 \
--CACHE_ROOT  /mnt/mnt/storage/projects/UD-VLA/calvin_eval_our_train/24000step \
> /mnt/mnt/storage/projects/UD-VLA/logs/calvin_eval/abcd_d_dis_72steps_0519_24000step.log 2>&1 
# --debug \
