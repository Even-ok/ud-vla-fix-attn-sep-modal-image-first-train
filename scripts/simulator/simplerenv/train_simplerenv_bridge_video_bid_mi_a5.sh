# module load cuda/12.1
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-24889}
NGPUS=4

DATAPATH='./processed_data/meta/simplerenv_bridge_trainval_200pix.pkl'
ACTION_TOKENIZER_PATH="./ori_pretrain/fast_bridge_a5_s50_norm_ours_0826new"
EXP_NAME="UNIVLA_SIMPLERENV_BRIDGE_VIDEO_40k_a5_0831"

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path ./pretrained_models/WORLD_MODEL_POSTTRAIN \
    --model_config_path ./configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 2.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 40000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 500 \
    --per_device_train_batch_size 6 \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 1550 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 4000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action False \
    --actions True \
    --actions_format "fast" \
    --use_gripper False \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --with_i_ia True \
    --mask_image True \
    --use_blockwise_attn_mask True \
    --attn_type "None" \
    --max_action_len 40 \
    --fixbe True \
    --report_to wandb \
    > ./logs/simplerenv/train/train_simplerenv_bridge_video_with_i_ia_mi_a5_0831_fixbe.log 2>&1
