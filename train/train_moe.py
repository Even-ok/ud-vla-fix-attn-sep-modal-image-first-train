import os
import os.path as osp
import shutil
import torch
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
sys.path.append("./reference/Emu3")
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig
from transformers import AutoModel,Trainer
from datasets import Emu3WorldModelDataset,Emu3SFTDatasetI2IA,Emu3SFTDataset,Emu3SFTDatasetI2IA_mi
from torch.utils.data import WeightedRandomSampler, DataLoader

logger = logging.getLogger(__name__)


class RobustDeepSpeedTrainer(tf.Trainer):
    """
    Trainer that skips DeepSpeed optimizer/zero state saves (no global_stepXXXX dirs).
    Only model weights and scheduler state are saved.
    """
    _LOCAL_DS_CACHE = "/root/udvla_ckpt"
    
    def log(self, logs: dict) -> None:
        """
        Override log to include loss ratio metrics from the model.
        Loss ratio is computed in the model's forward pass and stored in _last_loss_ratio.
        """
        # Get loss ratio from model if available
        if hasattr(self.model, '_last_loss_ratio') and self.model._last_loss_ratio is not None:
            logs.update(self.model._last_loss_ratio)
            self.model._last_loss_ratio = None  # Clear after logging
            
        super().log(logs)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override to:
        1. Skip saves for steps that are neither:
           - a multiple of save_steps, nor in force_kept_steps
        2. After saving, clean up old checkpoints but always keep force_kept_steps.
        """
        steps_taken = self.state.global_step

        # Determine if this step should be saved
        save_strategy = self.args.save_strategy.value if hasattr(self.args.save_strategy, 'value') else self.args.save_strategy
        save_steps = self.args.save_steps
        force_kept_steps = self.args.force_kept_steps if self.args.force_kept_steps else []

        should_save = False
        if save_strategy == "steps" and save_steps is not None:
            is_periodic = (steps_taken % save_steps == 0)
            is_forced = (steps_taken in force_kept_steps)
            should_save = is_periodic or is_forced
        elif save_strategy == "epoch":
            should_save = True
        else:
            should_save = True

        if not should_save:
            return

        # Do the actual save via parent
        super()._save_checkpoint(model, trial, metrics)

        # Cleanup old checkpoints, but always keep force_kept_steps
        if save_strategy == "steps" and save_steps is not None and force_kept_steps:
            self._cleanup_old_checkpoints_except_kept(save_steps, force_kept_steps)

    def _cleanup_old_checkpoints_except_kept(self, save_steps, force_kept_steps):
        """
        Remove old periodic checkpoints, but preserve the force_kept_steps.
        Only runs on local process zero.
        """
        if not self.is_local_process_zero():
            return

        output_dir = self.args.output_dir
        if not os.path.isdir(output_dir):
            return

        force_kept = set(force_kept_steps)
        kept_dirs = []
        periodic_dirs = []

        for d in os.scandir(output_dir):
            if not (d.is_dir() and d.name.startswith("checkpoint-")):
                continue
            try:
                step_num = int(d.name.split("checkpoint-")[-1])
            except ValueError:
                continue

            if step_num in force_kept:
                kept_dirs.append(d.path)
            elif step_num % save_steps != 0:
                # Not a periodic save (e.g., intermediate checkpoint) - safe to delete
                periodic_dirs.append((step_num, d.path))

        # Delete non-force-kept, non-periodic dirs
        for _, path in periodic_dirs:
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"[CLEANUP] Removed non-periodic checkpoint: {path}")

    def _save_optimizer_and_scheduler(self, output_dir):
        logger.info("[DEEPSPEED SAVE] Skipping optimizer/zero state save")


class WeightedSamplerTrainer(RobustDeepSpeedTrainer):
    def get_train_dataloader(self):
        # Assuming train_dataset has a sample_weights attribute
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    with_cot: bool = field(default=False)
    with_i_ia: bool = field(default=False)
    mask_image: bool = field(default=False)
    use_clean_image: bool = field(default=False)
    max_action_len: int = field(default=70)
    fixbe: bool = field(default=True)
    use_schedule: bool = field(default=True)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)
    use_blockwise_attn_mask: bool = field(default=False)
    use_bidirectional_attn_mask: bool = field(default=False)
    force_kept_steps: List[int] = field(default_factory=list)
    # optim: str = field(default="adamw_bnb_8bit")


def load_model(model_args, model_config, training_args):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
        return Emu3MoE(config=model_config)
    else:
        return Emu3MoE.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None,
            # use_blockwise_attn_mask=training_args.use_blockwise_attn_mask,
        )

def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
        # return Emu3SFTDataset(data_args, tokenizer=tokenizer)
    # elif data_args.real_robot:
    #     return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    # elif data_args.with_cot:
    #     return Emu3CoTDataset(data_args, tokenizer=tokenizer)
    elif data_args.mask_image:
        return Emu3SFTDatasetI2IA_mi(data_args, tokenizer=tokenizer)
    elif data_args.with_i_ia:
        return Emu3SFTDatasetI2IA(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    # 自动划分 90% train, 10% val
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def validate_checkpoint(ckpt_path):
    """Check if a checkpoint directory is valid (has complete optimizer states)."""
    global_step_dir = None
    for d in pathlib.Path(ckpt_path).iterdir():
        if d.is_dir() and d.name.startswith("global_step"):
            global_step_dir = d
            break

    if global_step_dir is None:
        return False

    # Check if all optimizer state files exist and have similar sizes (for multi-rank)
    optim_files = list(global_step_dir.glob("bf16_zero_pp_rank_*_optim_states.pt"))
    if not optim_files:
        return True  # No optimizer states, might be a model-only checkpoint

    # Check file sizes are reasonable (within 10% of each other for ZeRO)
    sizes = [f.stat().st_size for f in optim_files]
    if len(sizes) > 1:
        avg_size = sum(sizes) / len(sizes)
        for s in sizes:
            if abs(s - avg_size) / avg_size > 0.5:  # More than 50% difference = likely corrupt
                return False
    return True


def train_with_checkpoint_retry(trainer, max_retries=3, retry_delay=60, checkpoint_dirs=None):
    """
    Train with checkpoint save retry mechanism.
    Handles checkpoint save failures by retrying or falling back to previous checkpoint.

    Args:
        trainer: The HuggingFace Trainer instance
        max_retries: Maximum number of save retries before giving up
        retry_delay: Seconds to wait before retrying
        checkpoint_dirs: List of existing checkpoint directories for fallback
    """
    def get_step(d):
        try:
            return int(str(d).split("checkpoint-")[-1])
        except:
            return 0

    # Filter and validate checkpoints
    valid_checkpoints = []
    if checkpoint_dirs:
        for ckpt in checkpoint_dirs:
            if validate_checkpoint(ckpt):
                valid_checkpoints.append(ckpt)
                logger.info(f"[CHECKPOINT] Valid: {ckpt}")
            else:
                logger.warning(f"[CHECKPOINT] Invalid/corrupt: {ckpt} - will skip")

        valid_checkpoints = sorted(valid_checkpoints, key=get_step, reverse=True)
        logger.info(f"[CHECKPOINT RETRY] Found {len(valid_checkpoints)} valid checkpoints")

    last_checkpoint = valid_checkpoints[0] if valid_checkpoints else None
    if last_checkpoint:
        logger.info(f"[CHECKPOINT RETRY] Will resume from: {last_checkpoint}")

    attempt = 0
    while attempt < max_retries:
        try:
            logger.info(f"[TRAIN] Starting training (attempt {attempt + 1}/{max_retries})")
            trainer.train(resume_from_checkpoint=last_checkpoint)
            logger.info("[TRAIN] Training completed successfully!")
            return True
        except RuntimeError as e:
            error_msg = str(e)
            attempt += 1

            # Check if it's a checkpoint save error
            is_checkpoint_error = any(keyword in error_msg for keyword in [
                "PytorchStreamWriter failed",
                "inline_container.cc",
                "unexpected pos",
                "file write failed",
                "zipfile",
                "torch.save"
            ])

            if is_checkpoint_error:
                logger.error(f"[CHECKPOINT SAVE ERROR] Attempt {attempt}/{max_retries} failed")
                logger.error(f"[CHECKPOINT SAVE ERROR] {error_msg[:300]}")

                # Report potentially corrupted checkpoint
                trainer_state = pathlib.Path(trainer.args.output_dir) / "trainer_state.json"
                if trainer_state.exists():
                    try:
                        import json
                        with open(trainer_state) as f:
                            state = json.load(f)
                        last_ckpt = state.get("last_checkpoint", "")
                        if last_ckpt:
                            logger.warning(f"[CORRUPTED CHECKPOINT] {last_ckpt} may be corrupted due to save failure")
                            logger.warning(f"[CORRUPTED CHECKPOINT] Please manually delete it before resuming")
                    except Exception:
                        pass

                if attempt < max_retries:
                    logger.info(f"[RETRY] Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

                    # Re-validate checkpoints after potential cleanup
                    if checkpoint_dirs:
                        valid_checkpoints = []
                        for ckpt in checkpoint_dirs:
                            if validate_checkpoint(ckpt):
                                valid_checkpoints.append(ckpt)
                        valid_checkpoints = sorted(valid_checkpoints, key=get_step, reverse=True)
                        last_checkpoint = valid_checkpoints[0] if valid_checkpoints else None
                        logger.info(f"[RETRY] Resuming from: {last_checkpoint}")
                else:
                    logger.warning("[RETRY] Max retries reached, attempting fallback save...")
                    try:
                        trainer.save_state()
                        trainer.save_model(trainer.args.output_dir)
                        logger.info("[FALLBACK] Model saved via fallback mechanism")
                    except Exception as save_err:
                        logger.error(f"[FALLBACK SAVE FAILED] {save_err}")
                    return False
            else:
                # Not a checkpoint error, re-raise
                logger.error(f"[TRAIN ERROR] Non-checkpoint error: {error_msg[:200]}")
                raise
        except Exception as e:
            logger.error(f"[TRAIN ERROR] Unexpected error: {str(e)[:200]}")
            raise

    return False


def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")
    print("training_args.attn_type:", training_args.attn_type)
    # Load model configuration and tokenizer
    model_config = Emu3MoEConfig.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    model_config.use_blockwise_attn_mask = training_args.use_blockwise_attn_mask
    model_config.use_bidirectional_attn_mask = training_args.use_bidirectional_attn_mask
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )
    print(f"[DEBUG] training_args.max_position_embeddings: {training_args.max_position_embeddings}")
    print(f"[DEBUG] tokenizer.model_max_length: {tokenizer.model_max_length}")
    # if tokenizer.mask_token is None and data_args.mask_image:

    #     # tokenizer.mask_token = "<|extra_202|>"
    #     # mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    #     # print("mask_token_id:", mask_token_id)
    #     tokenizer.mask_token_id = 151848
    # print("tokenizer.mask_token:", tokenizer.mask_token)


    # Initialize model
    model = load_model(model_args, model_config, training_args)

    # Initialize dataset
    train_dataset = get_dataset(data_args, tokenizer)
    # print("data_args.mask_image:", data_args.mask_image)
    # print("train_dataset[0]:", train_dataset[0])

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    else:
        # Setup Trainer
        trainer = RobustDeepSpeedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,  # Pass tokenizer to trainer
        )

    # Check if resuming from checkpoint
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        print(f"[RESUME CHECK] Output dir: {training_args.output_dir}")
        print(f"[RESUME CHECK] Found checkpoints: {checkpoint_dirs}")
        print(f"[RESUME CHECK] Will RESUME from checkpoint")
    else:
        print(f"[RESUME CHECK] No checkpoint found, starting from scratch")

    # Train with checkpoint retry mechanism
    train_with_checkpoint_retry(
        trainer,
        max_retries=3,
        retry_delay=60,
        checkpoint_dirs=checkpoint_dirs if checkpoint_dirs else None
    )

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)
    print("train_dataset.max_token_len:", train_dataset.max_token_len)

if __name__ == "__main__":
    train()
