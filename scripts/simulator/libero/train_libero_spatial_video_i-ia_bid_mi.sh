# module load cuda/12.1
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_P2P_DISABLE=1  # 禁用 P2P 通信
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_PORT=24888
MASTER_ADDR=127.0.0.1
# MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-24888}
NGPUS=4

# DATAPATH='/share/user/iperror/data/univla/processed_data/meta/libero_all_norm.pkl'
DATAPATH='/mnt/mnt/storage/projects/UD-VLA/processed_data/meta/libero_spatial_fix_normall_infin.pkl'
ACTION_TOKENIZER_PATH="/mnt/mnt/storage/projects/UD-VLA/pretrain/fast_libero_all_t10_s50"
EXP_NAME="UDVLA_LIBERO_SPATIAL_VIDEO_BS196_64k_10_action_continued_4gpu_0525" # 

export PYTHONPATH=$(pwd)

torchrun \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path /mnt/mnt/storage/projects/UD-VLA/pretrain/UD-VLA_LIBERO \
    --model_config_path /mnt/mnt/storage/projects/UD-VLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3.json \
    --output_dir "logs/libero/train/"${EXP_NAME} \
    --learning_rate 5.254031264536109e-06 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 20000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 0 \
    --per_device_train_batch_size 4 \
    --frames 2 \
    --action_frames 10 \
    --max_position_embeddings 1650 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action False \
    --actions True \
    --actions_format "fast" \
    --use_gripper True \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --with_i_ia True \
    --mask_image True \
    --use_blockwise_attn_mask True \
    --attn_type "None" \
    --max_action_len 70 \
    --report_to wandb \
    > ./logs/libero/spatial/train_libero_with_i_ia_fast_libero_spatial_all_0525_continue_adam8bit.log 2>&1
