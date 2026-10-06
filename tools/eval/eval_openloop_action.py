#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open-loop action evaluation for the Emu3 VLA (masked-action in-fill test).

Motivation
----------
During SimplerEnv inference the model emits eoa after only ~14 FAST tokens
(training distribution: median 25, p5=20, P(len<=14) ~ 0.2%). This script runs
a controlled *in-domain* experiment on the training set to discriminate:

  (a) undertraining / objective-vs-decode mismatch:
        even in-domain, with clean gold image tokens, the model stops the
        action chunk early (pred_len still ~14 at 100% mask);
  (b) inference scaffold / domain gap:
        in-domain completions are healthy (pred_len back to ~20-28) -> the
        problem lives in the sim-side scaffold (joint image+action denoise,
        open-ended slot budget, etc.) or in sim observations.

Method
------
Random training samples are rebuilt EXACTLY as in training
(``Emu3SFTDatasetI2IA_mi``), except:
  * both image token blocks stay clean gold (never masked, labels -100);
  * inside the action block, k = ratio * (N_fast + 1) slots are replaced by
    the mask token (N FAST slots + the eoa slot; boa is protected), and one
    teacher-forced forward pass fills them;
  * the same chunk (same start_idx, same permutation) is reused across all
    mask ratios -> paired comparison with nested masks.

Metrics per ratio: FAST-token acc / CE / top-5 / gold-prob on masked slots,
eoa behaviour (premature-eoa rate on FAST slots, eoa-slot acc, no-eoa rate),
out-of-range argmax rate, predicted chunk-length distribution (THE diagnostic
vs. the ~14 tokens seen in simulation), and action reconstruction
(argmax -> FAST decode -> MSE / first-step MAE vs. gold actions).

Optional ``--denoise-steps N``: iterative discrete-diffusion decoding at 100%
mask, replicating the inference scaffold's ``denoise()`` but on a 40-slot
action region only (images stay gold), to isolate the decode-procedure effect.

Usage
-----
# full run (user executes when a GPU is free):
CUDA_VISIBLE_DEVICES=<gpu> python tools/eval/eval_openloop_action.py \
    --checkpoint logs/UNIVLA_SIMPLERENV_BRIDGE_VIDEO_40k_a5_1004_save_4k/checkpoint-16000 \
    --n-samples 100 --ratios 20,40,60,80,100 --seed 42 --batch-size 4

# token-layout selftest (CPU, no model):
python tools/eval/eval_openloop_action.py --selftest

# GPU smoke test:
... --n-samples 4 --ratios 100

# iterative-decode comparison at 100% mask:
... --ratios 100 --denoise-steps 72 --denoise-samples 20
"""

import os
import sys

# this file lives at <repo>/tools/eval/, so three dirnames up = repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_imports():
    # train_moe.py / datasets.py carry CWD-relative sys.path hacks, and
    # `datasets` must resolve to train/datasets.py (not the HF package).
    os.chdir(REPO_ROOT)
    for p in (REPO_ROOT,
              os.path.join(REPO_ROOT, "train"),
              os.path.join(REPO_ROOT, "reference", "Emu3")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_setup_imports()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_moe  # noqa: E402  (DataArguments; also wires up the emu3 imports)
from datasets import Emu3SFTDatasetI2IA_mi  # noqa: E402
from emu3.mllm import Emu3MoE, Emu3MoEConfig, Emu3Tokenizer  # noqa: E402
from emu3.sampling import get_mask_chedule, mask_by_random_topk  # noqa: E402

# ---------------------------------------------------------------------------
# constants (token ids verified against training/inference code)
# ---------------------------------------------------------------------------
BOA_ID = 151844          # <|action_begin>           (protected, never masked)
EOA_ID = 151845          # <|action_end>            (maskable like FAST slots)
PAD_ID = 151643          # <|endoftext|> pad token
MASK_ID = 151848         # mask token used by training
MODEL_MAX_LEN = 1550     # --max_position_embeddings of the training run

DEFAULT_CKPT = "logs/UNIVLA_SIMPLERENV_BRIDGE_VIDEO_40k_a5_1004_save_4k/checkpoint-16000"
FALLBACK_TOKENIZER_PATH = "./pretrained_models/WORLD_MODEL_POSTTRAIN"

HIST_BINS = [(0, 5), (6, 10), (11, 14), (15, 18), (19, 22), (23, 26), (27, 30), (31, 10 ** 9)]


def bin_label(n):
    for lo, hi in HIST_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10 ** 9 else '+'}"
    return "?"


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class OpenLoopActionDataset(Emu3SFTDatasetI2IA_mi):
    """Training dataset with a deterministic, ratio-controlled action mask.

    Everything (prompt, gold images, chunk selection, FAST encoding, padding)
    is inherited untouched from the training dataset, so the token layout is
    byte-identical to training.  Only the masking is replaced:

      * image branch  -> tokens unchanged, labels all -100 (clean gold image);
      * action branch -> k of the (N_fast + 1) maskable slots (FAST slots +
        the eoa slot; boa protected) become MASK_ID, labels carry the gold id.

    ``__getitem__`` returns ``(sample, extras)`` where extras holds the gold
    actions / gold vocab ids / masked slots needed for scoring.
    """

    def __init__(self, args, tokenizer, eval_seed=42):
        super().__init__(args, tokenizer=tokenizer)
        self.eval_seed = eval_seed
        self.eval_ratio = 1.0
        self._mask_rng = random.Random(0)
        self._gold_actions = None      # (5, 7) normalized, stashed per sample
        self._gold_vocab_ids = None    # FAST tokens in *vocab* id space
        self._last_extras = None

    # -- stashes ------------------------------------------------------------
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None):
        tensor, selected_actions = super().random_frames_to_tensor(
            img_list, T, action_prompt=action_prompt)
        self._gold_actions = np.asarray(selected_actions, dtype=np.float64)
        return tensor, selected_actions

    def wrap_action_sequence(self, action_ids):
        # action_ids are already in vocab-id space (last_vocab_idx - fast_id)
        self._gold_vocab_ids = [int(v) for v in action_ids]
        return super().wrap_action_sequence(action_ids)

    # -- masking --------------------------------------------------------------
    def mask_or_random_replace_tokens(self, image_tokens, mask_id, mask_schedule,
                                      is_action=False, fix_be=True, max_action_len=70,
                                      mask_ratios_list=None, use_schedule=True,
                                      return_timestep=False, timesteps=None):
        if not is_action:
            # clean gold image: tokens untouched, nothing to predict
            labels = torch.full_like(image_tokens, -100)
            ts = timesteps if timesteps is not None else torch.rand(1)
            if return_timestep:
                return image_tokens, labels, ts
            return image_tokens, labels

        # ---- action branch -------------------------------------------------
        gold_block = image_tokens                     # [boa, *N FAST, eoa] (pre-pad)
        seq_len = int(gold_block.shape[0])
        n_fast = seq_len - 2
        assert int(gold_block[0].item()) == BOA_ID
        assert int(gold_block[-1].item()) == EOA_ID

        maskable = n_fast + 1                         # FAST slots + eoa slot
        k = int(round(self.eval_ratio * maskable))
        k = max(1, min(k, maskable))
        perm = list(range(maskable))
        self._mask_rng.shuffle(perm)                  # ratio-independent -> nested
        masked_slots = sorted(perm[:k])

        input_ids = gold_block.clone()
        labels = torch.full_like(gold_block, -100)
        for slot in masked_slots:
            pos = 1 + slot                            # boa at 0
            labels[pos] = gold_block[pos]
            input_ids[pos] = mask_id

        pad = max_action_len - seq_len
        if pad > 0:                                   # same padding as training
            input_ids = torch.cat([input_ids,
                                   torch.full((pad,), mask_id, dtype=input_ids.dtype)])
            labels = torch.cat([labels,
                                torch.full((pad,), -100, dtype=labels.dtype)])

        self._last_extras = {
            "index": None,                            # filled by __getitem__
            "gold_actions": self._gold_actions,
            "gold_vocab_ids": list(self._gold_vocab_ids),
            "n_fast": n_fast,
            "masked_slots": list(masked_slots),
            "k": k,
        }

        ts = timesteps if timesteps is not None else torch.rand(1)
        if return_timestep:
            return input_ids, labels, ts
        return input_ids, labels

    def __getitem__(self, index):
        # ratio-independent seeding -> every ratio sees the identical chunk
        h = int(hashlib.md5(f"{self.eval_seed}:{index}".encode("utf-8")).hexdigest()[:8], 16)
        random.seed(h)
        np.random.seed(h % (2 ** 31))
        torch.manual_seed(h)
        self._mask_rng = random.Random(h)
        sample = super().__getitem__(index)
        extras = dict(self._last_extras)
        extras["index"] = index
        return sample, extras


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_data_args(args):
    """DataArguments mirroring train_simplerenv_bridge_video_bid_mi_a5.sh."""
    return train_moe.DataArguments(
        data_path=args.data_path,
        actions=True,
        actions_format="fast",
        action_tokenizer_path=args.fast_path,
        video_format="interleave",
        frames=args.frames,
        action_frames=args.action_frames,
        random_frame_sampling=True,
        raw_image=False,
        post_training=False,
        mask_image=True,
        use_clean_image=False,
        max_action_len=args.max_action_len,
        fixbe=True,
        use_schedule=True,
        with_i_ia=True,
        use_gripper=False,
        null_prompt_prob=0.15,
        apply_loss_on_only_action=False,
        apply_loss_on_only_vision=False,
    )


def load_tokenizer(path):
    try:
        return Emu3Tokenizer.from_pretrained(
            path, model_max_length=MODEL_MAX_LEN, padding_side="right", use_fast=False)
    except Exception as e:  # checkpoints sometimes lack tokenizer files
        print(f"[warn] loading tokenizer from {path} failed ({e}); "
              f"falling back to {FALLBACK_TOKENIZER_PATH}")
        return Emu3Tokenizer.from_pretrained(
            FALLBACK_TOKENIZER_PATH, model_max_length=MODEL_MAX_LEN,
            padding_side="right", use_fast=False)


def load_model(path, device):
    # NOTE: the checkpoint config stores _attn_implementation=flash_attention_2,
    # but the blockwise 4D bias must NOT go through FA2 (training used SDPA via
    # --attn_type None) -> force "sdpa".
    config = Emu3MoEConfig.from_pretrained(path)
    config.use_blockwise_attn_mask = True
    config.use_bidirectional_attn_mask = getattr(config, "use_bidirectional_attn_mask", False)
    config._attn_implementation = "sdpa"
    model = Emu3MoE.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.eval().to(device)
    print(f"[model] loaded {path} | dtype={next(model.parameters()).dtype} | "
          f"blockwise_attn={config.use_blockwise_attn_mask} | attn=sdpa")
    return model


def forward_hidden(model, input_ids, attention_mask):
    """Return final hidden states without materializing full-vocab logits."""
    out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


# ---------------------------------------------------------------------------
# one-shot (single forward) evaluation
# ---------------------------------------------------------------------------
def decode_actions(fast_proc, vocab_ids):
    """vocab-space ids -> (5, 7) action chunk via FAST decode."""
    fast_ids = [LAST_VOCAB_IDX[0] - int(v) for v in vocab_ids]
    return fast_proc.decode([fast_ids], time_horizon=5, action_dim=7)[0]


LAST_VOCAB_IDX = [151642]  # set from tokenizer at runtime (pad_token_id - 1)


def walk_pred_chunk(pred_by_slot, gold_vocab_ids, n_fast, fast_lo, fast_hi):
    """Mimic inference extraction: FAST tokens before the FIRST eoa."""
    seq, first_eoa_slot, invalid = [], None, False
    for slot in range(n_fast):
        vid = pred_by_slot.get(slot, int(gold_vocab_ids[slot]))
        if vid == EOA_ID:
            first_eoa_slot = slot
            break
        seq.append(vid)
        if not (fast_lo <= vid <= fast_hi):
            invalid = True
    return seq, first_eoa_slot, invalid


def evaluate_one_shot(model, ds, indices, ratios, args, fast_proc, fast_lo, fast_hi, out_dir):
    jsonl_path = os.path.join(out_dir, "per_sample.jsonl")
    results, gold_recon_mse = {}, {}
    t0 = time.time()

    with open(jsonl_path, "w") as jf:
        for pct in ratios:
            ds.eval_ratio = pct / 100.0
            n_done = 0
            n_masked_slots_total = 0
            # slot-level accumulators
            n_fast_slots = n_fast_correct = n_fast_top5 = 0
            ce_sum = gold_p_sum = 0.0
            n_eoa_on_fast = n_fast_oob = 0
            n_eoa_masked = n_eoa_correct = 0
            # sample-level accumulators
            n_premature = n_no_eoa = n_invalid = 0
            pred_lens, abs_dlens, mses, mae0s = [], [], [], []
            hist = Counter()

            for b0 in range(0, len(indices), args.batch_size):
                batch_idx = indices[b0:b0 + args.batch_size]
                samples, extras_list = zip(*(ds[i] for i in batch_idx))
                input_ids = torch.stack([s["input_ids"] for s in samples]).to(args.device)
                labels = torch.stack([s["labels"] for s in samples]).to(args.device)
                attn = torch.stack([s["attention_mask"] for s in samples]).to(args.device)

                hidden = forward_hidden(model, input_ids, attn)
                mask_pos = labels != -100
                b_ix, p_ix = mask_pos.nonzero(as_tuple=True)
                logits_sel = model.lm_head(hidden[b_ix, p_ix - 1]).float()
                preds = logits_sel.argmax(-1)
                golds = labels[b_ix, p_ix]
                probs = torch.softmax(logits_sel, -1)
                gold_p = probs.gather(-1, golds[:, None]).squeeze(-1)
                top5_hit = (logits_sel.topk(5, -1).indices == golds[:, None]).any(-1)
                ce = F.cross_entropy(logits_sel, golds, reduction="none")

                pred_ids = preds.tolist()
                gold_ids = golds.tolist()
                gold_p_l = gold_p.tolist()
                top5_l = top5_hit.tolist()
                ce_l = ce.tolist()

                row = 0
                for bi, idx in enumerate(batch_idx):
                    ex = extras_list[bi]
                    n_fast, slots = ex["n_fast"], ex["masked_slots"]
                    k = int(mask_pos[bi].sum().item())
                    assert k == len(slots)
                    # positions must be boa+1+slot, ascending (order == slot order)
                    boa_pos = int((input_ids[bi] == BOA_ID).nonzero()[0])
                    exp_pos = [boa_pos + 1 + s for s in slots]
                    assert p_ix[row:row + k].tolist() == exp_pos

                    p_rows = pred_ids[row:row + k]
                    pred_by_slot = dict(zip(slots, p_rows))

                    # gold reconstruction sanity (once per index)
                    if idx not in gold_recon_mse:
                        g_act = decode_actions(fast_proc, ex["gold_vocab_ids"])
                        gold_recon_mse[idx] = float(np.mean((g_act - ex["gold_actions"]) ** 2))

                    # FAST-slot / eoa-slot statistics
                    for j, slot in enumerate(slots):
                        if slot < n_fast:
                            n_fast_slots += 1
                            n_fast_correct += int(p_rows[j] == gold_ids[row + j])
                            n_fast_top5 += int(top5_l[row + j])
                            ce_sum += ce_l[row + j]
                            gold_p_sum += gold_p_l[row + j]
                            n_eoa_on_fast += int(p_rows[j] == EOA_ID)
                            n_fast_oob += int(p_rows[j] != EOA_ID and
                                              not (fast_lo <= p_rows[j] <= fast_hi))
                        else:  # the eoa slot itself
                            n_eoa_masked += 1
                            n_eoa_correct += int(p_rows[j] == EOA_ID)

                    # chunk-length walk + reconstruction
                    seq, first_eoa_slot, invalid = walk_pred_chunk(
                        pred_by_slot, ex["gold_vocab_ids"], n_fast, fast_lo, fast_hi)
                    pred_len = len(seq)
                    pred_lens.append(pred_len)
                    abs_dlens.append(abs(pred_len - n_fast))
                    hist[bin_label(pred_len)] += 1
                    if any(pred_by_slot.get(s) == EOA_ID for s in slots if s < n_fast):
                        n_premature += 1
                    if n_fast in slots and pred_by_slot.get(n_fast) != EOA_ID:
                        n_no_eoa += 1

                    if invalid:
                        n_invalid += 1
                    else:
                        act = decode_actions(fast_proc, seq)
                        gold = ex["gold_actions"]
                        mses.append(float(np.mean((act - gold) ** 2)))
                        mae0s.append(float(np.mean(np.abs(act[0] - gold[0]))))

                    jf.write(json.dumps({
                        "ratio_pct": pct, "index": idx, "n_fast": n_fast,
                        "masked_slots": slots, "pred_len": pred_len,
                        "first_eoa_slot": first_eoa_slot, "invalid": invalid,
                        "mse": mses[-1] if mses and not invalid else None,
                    }) + "\n")
                    n_masked_slots_total += k
                    row += k
                    n_done += 1

            n = n_done
            mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
            pctl = lambda xs, q: float(np.percentile(xs, q)) if xs else float("nan")
            eoa_slot = {"n_masked": n_eoa_masked,
                        "acc": n_eoa_correct / n_eoa_masked if n_eoa_masked else None,
                        "no_eoa_rate": n_no_eoa / n_eoa_masked if n_eoa_masked else None}
            results[pct] = {
                "ratio_pct": pct, "n_samples": n,
                "mean_masked_slots": n_masked_slots_total / max(n, 1),
                "fast_slot": {
                    "n": n_fast_slots,
                    "acc": n_fast_correct / max(n_fast_slots, 1),
                    "top5_acc": n_fast_top5 / max(n_fast_slots, 1),
                    "ce_nats": ce_sum / max(n_fast_slots, 1),
                    "gold_prob": gold_p_sum / max(n_fast_slots, 1),
                    "argmax_eoa_rate": n_eoa_on_fast / max(n_fast_slots, 1),
                    "oob_argmax_rate": n_fast_oob / max(n_fast_slots, 1),
                },
                "eoa_slot": eoa_slot,
                "sample": {
                    "premature_eoa_rate": n_premature / n,
                    "pred_len": {
                        "mean": mean(pred_lens), "median": pctl(pred_lens, 50),
                        "p5": pctl(pred_lens, 5), "p95": pctl(pred_lens, 95),
                        "min": int(min(pred_lens)), "max": int(max(pred_lens)),
                        "frac_in_20_28": float(np.mean([(20 <= L <= 28) for L in pred_lens])),
                        "hist": {b: hist[b] for b in sorted(hist)},
                    },
                    "mean_abs_dlen": mean(abs_dlens),
                },
                "recon": {
                    "mse_mean": mean(mses), "mse_median": pctl(mses, 50),
                    "mae0_mean": mean(mae0s),
                    "invalid_rate": n_invalid / n, "n_decoded": len(mses),
                },
            }
            print(f"[ratio {pct:>3}%] done ({n} samples, "
                  f"{time.time() - t0:.0f}s cumulative)")

    return results, gold_recon_mse


# ---------------------------------------------------------------------------
# iterative denoising at 100% mask (mirrors the inference scaffold)
# ---------------------------------------------------------------------------
def denoise_step(model, input_ids, input_ids_gen, generator, ratio, mask_token_id,
                 noise_schedule, seq_len, temperature, allowed, action_len):
    """Verbatim adaptation of model_wrapper.py::denoise for a 40-slot region."""
    # re-mask everything after the first sampled eoa (scaffold behaviour)
    eoa_positions = torch.where(input_ids_gen == EOA_ID)[1]
    if len(eoa_positions) > 0:
        eoa_index = int(eoa_positions[0].item())
        input_ids[:, -seq_len + eoa_index + 1:] = mask_token_id

    hidden = forward_hidden(model, input_ids, None)   # inference passes attn=None
    logits = model.lm_head(hidden[:, -(seq_len + 1):-1, :]).float()

    # constrain the action region to the valid vocabulary (FAST + eoa)
    constraint = torch.full_like(logits, float("-inf"))
    constraint[:, -action_len:, :] = 0.0
    constraint[:, :, allowed] = 0.0
    logits = logits + constraint

    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))
    sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0] \
        .view(*logits.shape[:-1])

    unknown_map = input_ids_gen == mask_token_id
    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
    selected_probs = torch.where(unknown_map, selected_probs,
                                 torch.finfo(selected_probs.dtype).max)
    mask_len = (seq_len * mask_ratio).floor().unsqueeze(0)
    mask_len = torch.max(torch.tensor([1], device=logits.device),
                         torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len))
    temperature = temperature * (1.0 - ratio)
    masking = mask_by_random_topk(mask_len, selected_probs, temperature,
                                  generator=generator)

    input_ids[:, -seq_len:] = torch.where(masking, mask_token_id, sampled_ids)
    input_ids_gen = torch.where(masking, mask_token_id, sampled_ids)
    return input_ids, input_ids_gen, temperature, sampled_ids


def evaluate_denoise(model, ds, indices, args, fast_proc, fast_lo, fast_hi):
    ds.eval_ratio = 1.0  # 100% mask: all FAST slots + eoa slot masked, boa kept
    schedule = get_mask_chedule("cosine")
    allowed = torch.tensor(list(range(fast_lo, fast_hi + 1)) + [EOA_ID],
                           dtype=torch.long, device=args.device)

    n_done = 0
    pred_lens, mses, mae0s = [], [], []
    hist = Counter()
    for idx in indices[:args.denoise_samples]:
        sample, ex = ds[idx]
        ids = sample["input_ids"].unsqueeze(0).to(args.device)
        boa_pos = int((ids[0] == BOA_ID).nonzero()[0])
        L = boa_pos + args.max_action_len
        input_ids = ids[:, :L].clone()
        gen = input_ids.clone()
        assert bool((gen[0, boa_pos + 1:] == MASK_ID).all()), \
            "denoise mode requires a 100%-masked action block"

        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)
        temperature = 1.0
        for step in range(args.denoise_steps):
            ratio = 1.0 * (step + 1) / args.denoise_steps
            input_ids, gen, temperature, _ = denoise_step(
                model, input_ids, gen, generator, ratio, MASK_ID, schedule,
                seq_len=args.max_action_len, temperature=temperature,
                allowed=allowed, action_len=args.max_action_len)

        region = gen[0, boa_pos + 1:].tolist()
        seq = []
        for vid in region:
            if vid == EOA_ID:
                break
            seq.append(vid)
        pred_len = len(seq)
        pred_lens.append(pred_len)
        hist[bin_label(pred_len)] += 1
        act = decode_actions(fast_proc, seq)
        gold = ex["gold_actions"]
        mses.append(float(np.mean((act - gold) ** 2)))
        mae0s.append(float(np.mean(np.abs(act[0] - gold[0]))))
        n_done += 1
        if n_done % 5 == 0:
            print(f"[denoise] {n_done}/{min(len(indices), args.denoise_samples)}")

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    pctl = lambda xs, q: float(np.percentile(xs, q)) if xs else float("nan")
    return {
        "steps": args.denoise_steps, "n_samples": n_done,
        "pred_len": {
            "mean": mean(pred_lens), "median": pctl(pred_lens, 50),
            "p5": pctl(pred_lens, 5), "p95": pctl(pred_lens, 95),
            "min": int(min(pred_lens)), "max": int(max(pred_lens)),
            "frac_in_20_28": float(np.mean([(20 <= x <= 28) for x in pred_lens])),
            "hist": {b: hist[b] for b in sorted(hist)},
        },
        "recon": {"mse_mean": mean(mses), "mae0_mean": mean(mae0s)},
    }


# ---------------------------------------------------------------------------
# selftest (CPU, no model)
# ---------------------------------------------------------------------------
def run_selftest(args):
    print("[selftest] building dataset (CPU, no model) ...")
    tok = load_tokenizer(args.checkpoint)
    LAST_VOCAB_IDX[0] = tok.pad_token_id - 1
    fast_probe = None
    ds = OpenLoopActionDataset(build_data_args(args), tokenizer=tok,
                               eval_seed=args.seed)
    fast_proc = ds.action_tokenizer
    fast_vs = int(fast_proc.vocab_size)
    fast_lo, fast_hi = LAST_VOCAB_IDX[0] - fast_vs + 1, LAST_VOCAB_IDX[0]

    idxs = [0, len(ds) // 2, len(ds) - 1]
    n_checked = 0
    for idx in idxs:
        per_ratio = {}
        for pct in (20, 100):
            ds.eval_ratio = pct / 100.0
            sample, ex = ds[idx]
            ids = sample["input_ids"]
            labels = sample["labels"]
            att = sample["attention_mask"]

            boa_positions = (ids == BOA_ID).nonzero()
            assert boa_positions.numel() == 1, "boa must appear exactly once"
            bp = int(boa_positions[0])
            assert int(ids[bp]) == BOA_ID and int(labels[bp]) == -100

            n_fast, slots = ex["n_fast"], ex["masked_slots"]
            maskable = n_fast + 1
            exp_k = max(1, min(maskable, int(round(pct / 100.0 * maskable))))
            assert len(slots) == exp_k, (len(slots), exp_k)
            assert len(ex["gold_vocab_ids"]) == n_fast
            assert all(fast_lo <= v <= fast_hi for v in ex["gold_vocab_ids"])
            assert ex["gold_actions"].shape == (5, 7)

            block_end = bp + args.max_action_len  # block is max_action_len long INCL. boa
            assert block_end <= ids.shape[0]
            for slot in range(maskable):
                pos = bp + 1 + slot
                gold = EOA_ID if slot == n_fast else ex["gold_vocab_ids"][slot]
                if slot in slots:
                    assert int(ids[pos]) == MASK_ID
                    assert int(labels[pos]) == gold
                else:
                    assert int(ids[pos]) == gold
                    assert int(labels[pos]) == -100
            # padding after the block
            assert bool((ids[block_end:] == PAD_ID).all())
            assert bool((att[block_end:] == 0).all())
            assert bool((att[:block_end] == 1).all())
            per_ratio[pct] = ex
            n_checked += 1

        # same chunk across ratios + nested masks
        assert per_ratio[20]["gold_vocab_ids"] == per_ratio[100]["gold_vocab_ids"]
        assert np.allclose(per_ratio[20]["gold_actions"], per_ratio[100]["gold_actions"])
        assert set(per_ratio[20]["masked_slots"]) <= set(per_ratio[100]["masked_slots"])
        print(f"[selftest] index {idx}: n_fast={per_ratio[20]['n_fast']} "
              f"k(20%)={len(per_ratio[20]['masked_slots'])} "
              f"k(100%)={len(per_ratio[100]['masked_slots'])} OK")

    print(f"[selftest] {n_checked} samples checked; FAST valid range "
          f"[{fast_lo}, {fast_hi}] (vocab_size={fast_vs})")
    print("SELFTEST PASSED")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--data-path",
                   default="./processed_data/meta/simplerenv_bridge_trainval_200pix.pkl")
    p.add_argument("--fast-path",
                   default="./ori_pretrain/fast_bridge_a5_s50_norm_ours_0826new")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--ratios", type=str, default="20,40,60,80,100",
                   help="comma-separated mask ratios in percent")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-action-len", type=int, default=40)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--action-frames", type=int, default=5)
    p.add_argument("--denoise-steps", type=int, default=0,
                   help=">0 enables iterative decoding at 100%% mask (e.g. 72)")
    p.add_argument("--denoise-samples", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def default_out_dir(ckpt):
    parts = os.path.normpath(ckpt).split(os.sep)
    name = "_".join(parts[-2:]) if parts[-1].startswith("checkpoint") else parts[-1]
    return os.path.join(REPO_ROOT, "tools", "eval", "results", "openloop_" + name)


def fmt(x, spec=".4f"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    if isinstance(x, float):
        if 0 < abs(x) < 1e-3:
            return f"{x:.1e}"
        return format(x, spec)
    return str(x)


def print_table(results, gold_recon_mse):
    hdr = (f"{'ratio':>6} {'k':>5} {'acc':>6} {'top5':>6} {'CE':>6} {'goldp':>6} "
           f"{'eoaAcc':>6} {'eoaFAST':>7} {'noEoa':>6} {'OOB':>6} "
           f"{'lenMed':>6} {'p5':>4} {'p95':>4} {'%20-28':>6} {'|dLen|':>6} "
           f"{'MSE':>9} {'MAE@0':>9} {'inv':>5}")
    print("\n" + "=" * len(hdr))
    print("Open-loop masked-action eval (one forward pass, clean gold images)")
    print(hdr)
    print("-" * len(hdr))
    for pct, r in sorted(results.items()):
        fs, eo, sp, rc = r["fast_slot"], r["eoa_slot"], r["sample"], r["recon"]
        pl = sp["pred_len"]
        print(f"{pct:>5}% {r['fast_slot']['n'] / max(r['n_samples'], 1):>5.1f} "
              f"{fmt(fs['acc']):>6} {fmt(fs['top5_acc']):>6} "
              f"{fmt(fs['ce_nats']):>6} {fmt(fs['gold_prob']):>6} "
              f"{fmt(eo['acc']):>6} {fmt(fs['argmax_eoa_rate']):>7} "
              f"{fmt(eo['no_eoa_rate']):>6} {fmt(fs['oob_argmax_rate']):>6} "
              f"{fmt(pl['median'], '.1f'):>6} {fmt(pl['p5'], '.0f'):>4} "
              f"{fmt(pl['p95'], '.0f'):>4} {fmt(pl['frac_in_20_28']):>6} "
              f"{fmt(sp['mean_abs_dlen'], '.1f'):>6} "
              f"{fmt(rc['mse_mean'], '.2e'):>9} {fmt(rc['mae0_mean'], '.2e'):>9} "
              f"{fmt(rc['invalid_rate']):>5}")
    if gold_recon_mse:
        g = float(np.mean(list(gold_recon_mse.values())))
        print("-" * len(hdr))
        print(f"gold reconstruction sanity (gold FAST ids -> decode): "
              f"MSE = {g:.2e} over {len(gold_recon_mse)} samples "
              f"(expected ~1e-8..1e-4)")


def main():
    args = parse_args()
    ratios = [int(r) for r in args.ratios.split(",") if r.strip()]

    if args.selftest:
        run_selftest(args)
        return

    assert torch.cuda.is_available(), "eval requires a GPU (selftest runs on CPU)"
    device = args.device
    torch.set_grad_enabled(False)

    out_dir = args.out_dir or default_out_dir(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] checkpoint = {args.checkpoint}")
    print(f"[eval] output dir = {out_dir}")

    tokenizer = load_tokenizer(args.checkpoint)
    LAST_VOCAB_IDX[0] = tokenizer.pad_token_id - 1

    model = load_model(args.checkpoint, device)
    ds = OpenLoopActionDataset(build_data_args(args), tokenizer=tokenizer,
                               eval_seed=args.seed)
    fast_proc = ds.action_tokenizer
    fast_vs = int(fast_proc.vocab_size)
    fast_lo, fast_hi = LAST_VOCAB_IDX[0] - fast_vs + 1, LAST_VOCAB_IDX[0]
    print(f"[eval] dataset = {len(ds)} scenes | FAST vocab ids in "
          f"[{fast_lo}, {fast_hi}] | ratios = {ratios}")

    n = min(args.n_samples, len(ds))
    indices = random.Random(args.seed).sample(range(len(ds)), n)

    results, gold_recon_mse = evaluate_one_shot(
        model, ds, indices, ratios, args, fast_proc, fast_lo, fast_hi, out_dir)
    print_table(results, gold_recon_mse)

    denoise_res = None
    if args.denoise_steps > 0:
        print(f"\n[denoise] iterative decoding at 100% mask, "
              f"{args.denoise_steps} steps on {min(n, args.denoise_samples)} samples ...")
        denoise_res = evaluate_denoise(model, ds, indices, args,
                                       fast_proc, fast_lo, fast_hi)
        pl = denoise_res["pred_len"]
        print(f"[denoise] pred_len median={pl['median']:.1f} "
              f"[p5={pl['p5']:.0f}, p95={pl['p95']:.0f}] "
              f"MSE={denoise_res['recon']['mse_mean']:.2e} "
              f"hist={pl['hist']}")

    # ---- takeaway line -----------------------------------------------------
    diag_ratio = max(results)
    med = results[diag_ratio]["sample"]["pred_len"]["median"]
    print("\n" + "=" * 72)
    print(f"DIAGNOSTIC (ratio {diag_ratio}% row, pred_len median = {med:.1f}):")
    print("  training-set gold chunk length: median 25, p5=20, [12, 32]")
    print("  SimplerEnv inference currently stops at ~14 tokens")
    if med >= 20:
        print("  -> in-domain completions are healthy: the early-eoa behaviour is")
        print("     NOT intrinsic to the model; investigate the inference scaffold")
        print("     (joint image+action denoise, slot budget) or sim domain gap.")
    else:
        print("  -> the model reproduces the early stop IN-DOMAIN: training")
        print("     objective vs. decode-procedure mismatch / undertraining.")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "meta": {"checkpoint": args.checkpoint, "data_path": args.data_path,
                     "fast_path": args.fast_path, "n_samples": n,
                     "ratios_pct": ratios, "seed": args.seed,
                     "batch_size": args.batch_size,
                     "max_action_len": args.max_action_len,
                     "frames": args.frames, "action_frames": args.action_frames,
                     "denoise_steps": args.denoise_steps,
                     "denoise_samples": args.denoise_samples,
                     "fast_vocab_size": fast_vs,
                     "fast_valid_vocab_range": [fast_lo, fast_hi]},
            "gold_reconstruction": {
                "mse_mean": float(np.mean(list(gold_recon_mse.values())))
                if gold_recon_mse else None,
                "per_index": gold_recon_mse},
            "one_shot": {str(k): v for k, v in results.items()},
            "denoise_100": denoise_res,
        }, f, indent=2)
    print(f"[done] results.json + per_sample.jsonl written to {out_dir}")


if __name__ == "__main__":
    main()
