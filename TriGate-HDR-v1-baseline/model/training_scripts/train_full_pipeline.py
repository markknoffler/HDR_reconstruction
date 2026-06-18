"""
Run Stage-1 -> Stage-2 -> Stage-3 sequentially on the same data split.

Stage-3 automatically uses best.pt from Stage-1 and Stage-2 checkpoint dirs.
After each stage, exports val_export_count random validation images.
"""

import argparse
import os
import subprocess
import sys


def _base_paths():
    root = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR"
    return {
        "repo": os.path.join(root, "TriGate-HDR"),
        "ldr": os.path.join(root, "SingleHDR_training_data/HDR-Real/LDR_in"),
        "hdr": os.path.join(root, "SingleHDR_training_data/HDR-Real/HDR_gt"),
        "sam": os.path.join(root, "SingleHDR_training_data/segmented_masks"),
        "ckpt_root": os.path.join(root, "TriGate-HDR/experiments"),
    }


def _run(cmd):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    defaults = _base_paths()
    parser = argparse.ArgumentParser(description="Full TriGate progressive training pipeline.")
    parser.add_argument("--ldr_dir", type=str, default=defaults["ldr"])
    parser.add_argument("--hdr_dir", type=str, default=defaults["hdr"])
    parser.add_argument("--sam_mask_dir", type=str, default=defaults["sam"])
    parser.add_argument("--experiment_name", type=str, default="smoke_packet0")
    parser.add_argument("--ckpt_root", type=str, default=defaults["ckpt_root"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--subset_fraction", type=float, default=0.2)
    parser.add_argument("--subset_packet", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_export_count", type=int, default=10)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_test:
        if args.max_train_samples <= 0:
            args.max_train_samples = 6
        if args.max_val_samples <= 0:
            args.max_val_samples = 4
        if args.max_dim <= 0:
            args.max_dim = 256
        args.val_export_count = min(args.val_export_count, args.max_val_samples)

    exp_dir = os.path.join(args.ckpt_root, args.experiment_name)
    s1_dir = os.path.join(exp_dir, "stage1")
    s2_dir = os.path.join(exp_dir, "stage2")
    s3_dir = os.path.join(exp_dir, "stage3")

    py = sys.executable
    common = [
        "--ldr_dir",
        args.ldr_dir,
        "--hdr_dir",
        args.hdr_dir,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--max_dim",
        str(args.max_dim),
        "--subset_fraction",
        str(args.subset_fraction),
        "--subset_packet",
        str(args.subset_packet),
        "--val_ratio",
        str(args.val_ratio),
        "--split_seed",
        str(args.split_seed),
        "--val_export_count",
        str(args.val_export_count),
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_val_samples",
        str(args.max_val_samples),
    ]
    if args.amp:
        common.append("--amp")
    if args.continue_train:
        common.append("--continue_train")
    if args.smoke_test:
        common.append("--smoke_test")

    os.chdir(defaults["repo"])
    env = os.environ.copy()
    env["PYTHONPATH"] = defaults["repo"]

    if not args.skip_stage1:
        cmd = [
            py,
            "-m",
            "model.training_scripts.train_stage1_dual_diffusion",
            "--checkpoint_dir",
            s1_dir,
            "--sam_mask_dir",
            args.sam_mask_dir,
            "--val_export_dir",
            os.path.join(s1_dir, "val_exports"),
            *common,
        ]
        _run(cmd)

    if not args.skip_stage2:
        cmd = [
            py,
            "-m",
            "model.training_scripts.train_stage2_crf_recovery",
            "--checkpoint_dir",
            s2_dir,
            "--val_export_dir",
            os.path.join(s2_dir, "val_exports"),
            *common,
        ]
        _run(cmd)

    if not args.skip_stage3:
        cmd = [
            py,
            "-m",
            "model.training_scripts.train_stage3_seaming_gan",
            "--checkpoint_dir",
            s3_dir,
            "--sam_mask_dir",
            args.sam_mask_dir,
            "--stage1_ckpt_dir",
            s1_dir,
            "--stage2_ckpt_dir",
            s2_dir,
            "--val_export_dir",
            os.path.join(s3_dir, "val_exports"),
            *common,
        ]
        _run(cmd)

    print(f"\nDone. Experiment artifacts: {exp_dir}")
    print(f"  Stage1 exports: {os.path.join(s1_dir, 'val_exports')}")
    print(f"  Stage2 exports: {os.path.join(s2_dir, 'val_exports')}")
    print(f"  Stage3 exports: {os.path.join(s3_dir, 'val_exports')} (full pipeline)")


if __name__ == "__main__":
    main()
