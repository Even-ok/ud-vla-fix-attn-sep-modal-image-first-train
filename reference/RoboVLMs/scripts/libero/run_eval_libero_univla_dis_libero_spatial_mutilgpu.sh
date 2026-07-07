#!/bin/bash
# module load cuda/12.1
# export CUDA_VISIBLE_DEVICES=1
# export ARNOLD_WORKER_GPU=1
# export ARNOLD_WORKER_NUM=1
# export ARNOLD_ID=0
# export RANK=0

# export OMP_NUM_THREADS=16


ckpt_dir="/mnt/mnt/storage/projects/UD-VLA/logs/libero/train/UDVLA_LIBERO_SPATIAL_VIDEO_BS196_64k_10_action_continued_4gpu/checkpoint-2000"
steps=72
TIMESTAMP=$(date +%Y%m%d%H%M)
# GPUS_PER_NODE=$ARNOLD_WORKER_GPU

export CUDA_VISIBLE_DEVICES=4,5
torchrun --nproc_per_node=2 --master_addr="localhost" --master_port=29881 eval/libero/evaluate_libero_emu_ddp.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_spatial \
--dis_i2a \
--steps $steps \
--cache_root /mnt/mnt/storage/projects/UD-VLA/reference/RoboVLMs/cache/libero_eval_ours/spatial/eval/${TIMESTAMP} \
--action_tokenizer /mnt/mnt/storage/projects/UD-VLA/pretrain/fast_libero_all_t10_s50 \
> ./logs/libero_eval_udvla_dis_libero_spatial_${steps}steps_0516_2000_continue.log 2>&1 
# --debug
