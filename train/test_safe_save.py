# -*- coding: utf-8 -*-
"""
Tests for RobustDeepSpeedTrainer._save: temp-dir save -> copy -> delete flow.

Background: saving the big safetensors shards directly into the final output dir
failed with "Device or resource busy (os error 16)". The fix routes every model
save through _TEMP_SAVE_ROOT first, copies size-verified files into the final
dir, then deletes the temp copy. These tests verify the real code path.

Run from anywhere with the training env (the script chdirs to the repo root
itself, which train_moe.py relies on for "./reference/Emu3"):

    /opt/conda/envs/udvla/bin/python train/test_safe_save.py

Use --shrink to cut the big-shard tests from 4 GiB (real checkpoint shard size)
down to 1 GiB for a faster run.
"""
import os
import shutil
import sys
import time
import traceback
from uuid import uuid4

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)  # train_moe.py imports rely on CWD == repo root
sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))  # train/datasets.py, as in real runs

import train_moe  # noqa: E402  (heavy: torch/transformers/emu3)
from train_moe import RobustDeepSpeedTrainer  # noqa: E402

SHRINK = "--shrink" in sys.argv
BIG_ELEMENTS = 2**29 * 2 if SHRINK else 2**31  # bf16 -> 1 GiB shrunk, 4 GiB otherwise
COPY_TEST_BYTES = 256 * 2**20  # 256 MB

TMP_ROOT = RobustDeepSpeedTrainer._TEMP_SAVE_ROOT
assert os.path.isdir(TMP_ROOT), f"temp save root does not exist: {TMP_ROOT}"

# The training script saves to the relative "logs/<exp>" dir, and repo-root ./logs
# is a symlink to /workspace/simpler_train_logs, which is bosfs2 (FUSE object
# storage) - a different filesystem from /share (NFS) where the temp save goes.
REAL_LOGS_DIR = "/workspace/simpler_train_logs"


def make_probe_root():
    root = os.path.join(TMP_ROOT, f"_safe_save_test_{uuid4().hex[:8]}")
    os.makedirs(root, exist_ok=True)
    return root


def no_copying_leftovers(final_dir):
    return [f for f in os.listdir(final_dir) if ".copying_" in f]


def temp_dirs_left():
    return [d for d in os.listdir(TMP_ROOT) if ".tmp_" in d]


def test_copy_file_checked_basic(root):
    """A plain copy lands at dst with the exact size, no .copying_ leftovers."""
    final_dir = os.path.join(root, "final_basic")
    os.makedirs(final_dir, exist_ok=True)
    src = os.path.join(root, "src.bin")
    with open(src, "wb") as f:
        f.write(os.urandom(1024) * (COPY_TEST_BYTES // 1024))

    t0 = time.time()
    RobustDeepSpeedTrainer._copy_file_checked(src, os.path.join(final_dir, "dst.bin"))
    dt = time.time() - t0

    dst = os.path.join(final_dir, "dst.bin")
    assert os.path.getsize(dst) == COPY_TEST_BYTES, "copied file has wrong size"
    assert not no_copying_leftovers(final_dir), f"leftover .copying_ files: {no_copying_leftovers(final_dir)}"
    print(f"    copy {COPY_TEST_BYTES / 2**20:.0f} MiB in {dt:.1f}s ({COPY_TEST_BYTES / 2**20 / max(dt, 1e-9):.0f} MiB/s)")


def test_copy_file_checked_failure_cleanup(root):
    """A failed / truncated copy raises and never leaves .copying_ or dst behind."""
    final_dir = os.path.join(root, "final_fail")
    os.makedirs(final_dir, exist_ok=True)
    src = os.path.join(root, "src_fail.bin")
    with open(src, "wb") as f:
        f.write(b"x" * (10 * 2**20))

    # 1) copyfile blows up mid-copy -> exception, no leftovers
    orig_copyfile = shutil.copyfile

    def boom(src_, dst_):
        with open(dst_, "wb"):
            pass
        raise IOError("simulated mid-copy failure")

    shutil.copyfile = boom
    try:
        RobustDeepSpeedTrainer._copy_file_checked(src, os.path.join(final_dir, "dst1.bin"))
        raise AssertionError("expected IOError from simulated copy failure")
    except IOError:
        pass
    finally:
        shutil.copyfile = orig_copyfile
    assert not os.path.exists(os.path.join(final_dir, "dst1.bin")), "dst written despite failure"
    assert not no_copying_leftovers(final_dir), f"leftover .copying_: {no_copying_leftovers(final_dir)}"

    # 2) copyfile truncates the data -> size check must catch it, no leftovers
    def truncate(src_, dst_):
        with open(src_, "rb") as f:
            head = f.read(1024)
        with open(dst_, "wb") as g:
            g.write(head)

    shutil.copyfile = truncate
    try:
        RobustDeepSpeedTrainer._copy_file_checked(src, os.path.join(final_dir, "dst2.bin"))
        raise AssertionError("expected IOError from size mismatch")
    except IOError as e:
        assert "size mismatch" in str(e), f"unexpected error: {e}"
    finally:
        shutil.copyfile = orig_copyfile
    assert not os.path.exists(os.path.join(final_dir, "dst2.bin")), "dst written despite size mismatch"
    assert not no_copying_leftovers(final_dir), f"leftover .copying_: {no_copying_leftovers(final_dir)}"


def make_trainer(output_dir):
    from transformers import BertConfig, BertModel, TrainingArguments

    cfg = BertConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = BertModel(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        use_cpu=True,
        report_to=[],
        save_safetensors=True,
        disable_tqdm=True,
        log_level="error",
    )
    return RobustDeepSpeedTrainer(model=model, args=args, train_dataset=None, tokenizer=None), model


def test_save_through_trainer(root):
    """The real trainer._save writes into temp, copies into final, deletes temp."""
    import torch
    from transformers import BertModel

    final_dir = os.path.join(root, "final_ckpt")
    trainer, model = make_trainer(os.path.join(root, "args_output_dir"))

    trainer._save(final_dir)

    expected = ["model.safetensors", "config.json", "training_args.bin"]
    for name in expected:
        assert os.path.isfile(os.path.join(final_dir, name)), f"missing {name} in final dir"
    assert not no_copying_leftovers(final_dir), f"leftover .copying_: {no_copying_leftovers(final_dir)}"
    left = temp_dirs_left()
    assert not left, f"temp dirs not deleted: {left}"

    # The copied checkpoint must actually load and match the saved weights.
    reloaded = BertModel.from_pretrained(final_dir)
    key = "encoder.layer.0.attention.self.query.weight"
    assert torch.equal(model.state_dict()[key], reloaded.state_dict()[key]), "reloaded weight mismatch"
    print("    checkpoint reload OK, weights match")


def test_save_big_shard(root):
    """
    The real failure case: a multi-GiB safetensors shard (Emu3 shards are ~4 GiB)
    serialized into the temp dir and copied into the final dir.
    """
    import torch

    size_gib = BIG_ELEMENTS * 2 / 2**30
    final_dir = os.path.join(root, "final_big_ckpt")
    trainer, _ = make_trainer(os.path.join(root, "args_output_dir_big"))

    big = torch.zeros(BIG_ELEMENTS, dtype=torch.bfloat16)
    state_dict = {"big_weight": big}

    t0 = time.time()
    trainer._save(final_dir, state_dict=state_dict)  # serializes into temp, then copies
    dt = time.time() - t0

    shard = os.path.join(final_dir, "model.safetensors")
    assert os.path.isfile(shard), "big shard missing in final dir"
    assert os.path.getsize(shard) >= BIG_ELEMENTS * 2, "big shard too small"
    assert not no_copying_leftovers(final_dir), f"leftover .copying_: {no_copying_leftovers(final_dir)}"
    left = temp_dirs_left()
    assert not left, f"temp dirs not deleted: {left}"

    # Read the shard back through safetensors to prove it is not a partial file.
    from safetensors.torch import load_file

    loaded = load_file(shard)
    assert loaded["big_weight"].shape == (BIG_ELEMENTS,), "reloaded tensor shape mismatch"
    print(f"    {size_gib:.0f} GiB shard save+copy in {dt:.1f}s ({size_gib / max(dt, 1e-9):.0f} GiB/s)")
    return big


def test_save_to_real_target(root):
    """
    End-to-end check of the real training flow: save through the relative
    "logs/<exp>/checkpoint-N" path exactly as the training script's --output_dir
    does. That path resolves through the logs symlink onto bosfs
    (/workspace/simpler_train_logs), so this proves the big shard really lands
    in the user's target dir and that the atomic os.replace rename works there.
    """
    final_dir = os.path.join("logs", f"_safe_save_test_exp_{uuid4().hex[:8]}", "checkpoint-999")
    exp = final_dir.split(os.sep)[1]
    trainer, _ = make_trainer(os.path.join("logs", exp, "args_dir"))

    big = torch.zeros(BIG_ELEMENTS, dtype=torch.bfloat16)
    trainer._save(final_dir, state_dict={"big_weight": big})

    resolved = os.path.join(REAL_LOGS_DIR, exp, "checkpoint-999")
    shard = os.path.join(resolved, "model.safetensors")
    assert os.path.isfile(shard), f"shard missing at real target: {resolved}"
    assert os.path.getsize(shard) >= BIG_ELEMENTS * 2, "big shard too small"
    assert not no_copying_leftovers(resolved), f"leftover .copying_: {no_copying_leftovers(resolved)}"
    left = temp_dirs_left()
    assert not left, f"temp dirs not deleted: {left}"

    # Read the shard back through safetensors to prove it is not a partial file.
    from safetensors.torch import load_file

    loaded = load_file(shard)
    assert loaded["big_weight"].shape == (BIG_ELEMENTS,), "reloaded tensor shape mismatch"
    print(f"    {BIG_ELEMENTS * 2 / 2**30:.0f} GiB shard landed in {resolved} (bosfs target)")
    return exp


def probe_old_direct_path(exp, big):
    """
    Diagnostic only (does not affect pass/fail): reproduce the old behavior by
    serializing the same big shard directly into the final dir on bosfs, the way
    the run that crashed at step 4000 did. EBUSY here confirms the root cause.
    """
    from safetensors.torch import save_file

    direct = os.path.join(REAL_LOGS_DIR, exp, "direct", "model.safetensors")
    os.makedirs(os.path.dirname(direct), exist_ok=True)
    try:
        t0 = time.time()
        save_file({"big_weight": big}, direct)
        print(f"    [diagnostic] direct save into bosfs final dir OK in {time.time() - t0:.1f}s (EBUSY not reproduced this time)")
    except Exception as e:
        print(f"    [diagnostic] direct save into bosfs final dir FAILED: {type(e).__name__}: {e}")


def main():
    print(f"temp save root: {TMP_ROOT}")
    print(f"real target (repo ./logs symlink): {os.path.realpath(os.path.join(_REPO_ROOT, 'logs'))}")
    print(f"free space on {TMP_ROOT}: {shutil.disk_usage(TMP_ROOT).free / 2**30:.1f} GiB")
    print(f"big shard size: {BIG_ELEMENTS * 2 / 2**30:.0f} GiB (pass --shrink for 1 GiB)")
    root = make_probe_root()

    tests = [
        ("copy_file_checked_basic", test_copy_file_checked_basic),
        ("copy_file_checked_failure_cleanup", test_copy_file_checked_failure_cleanup),
        ("save_through_trainer", test_save_through_trainer),
        ("save_big_shard", test_save_big_shard),
        ("save_to_real_target_bosfs", test_save_to_real_target),
    ]
    results = []
    exp = None
    failed = False
    try:
        for name, fn in tests:
            print(f"[TEST] {name}")
            try:
                ret = fn(root)
                if name == "save_to_real_target_bosfs":
                    exp = ret  # the probe reuses this dir, cleaned up below
                results.append((name, True, ""))
            except Exception:
                traceback.print_exc()
                results.append((name, False, "see traceback above"))
                failed = True

        if exp is not None:
            # Only worth probing the old path if the new flow works.
            print("[PROBE] old direct-save path on bosfs (diagnostic)")
            try:
                big = torch.zeros(BIG_ELEMENTS, dtype=torch.bfloat16)
                probe_old_direct_path(exp, big)
            except Exception:
                traceback.print_exc()
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if exp is not None:
            shutil.rmtree(os.path.join(REAL_LOGS_DIR, exp), ignore_errors=True)
        for d in list(temp_dirs_left()):
            shutil.rmtree(os.path.join(TMP_ROOT, d), ignore_errors=True)

    print("\n===== RESULTS =====")
    for name, ok, msg in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + msg) if msg else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
