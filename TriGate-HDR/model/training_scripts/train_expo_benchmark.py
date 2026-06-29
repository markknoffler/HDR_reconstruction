#!/usr/bin/env python3
"""
Train TriGate-HDR on an ExpoCM paper benchmark with full Table 1 metrics.

ExpoCM protocol (expoCM_final_followup.pdf §4.1):
  - Metrics: PSNR/SSIM in linear, μ-law, PU21; MS-SSIM; HDR-VDP-2/3; LPIPS; ΔE2000
  - Splits: 80/20 train/val (seed 42), log train / val / val_full each epoch
  - Training target: 500k iters, batch 4, 256² crops (use max_dim=256; batch 1–2 on 20GB GPU)
  - Inference eval: native resolution (512² HDR-REAL/EYE, 256² AIM2025)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expo_benchmarks.registry import get_dataset_spec, list_datasets  # noqa: E402


def _arch_tag(arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> str:
    parts = ["trigate"]
    if arch_v2:
        parts.append("v2")
    if use_rso or use_lr_cfp:
        parts.append("gpure")
    return "_".join(parts)


def _default_checkpoint_dir(slug: str, arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> str:
    tag = _arch_tag(arch_v2, use_rso, use_lr_cfp)
    return f"experiments/{tag}_{slug}"


def _metrics_csv_name(slug: str, arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> str:
    tag = _arch_tag(arch_v2, use_rso, use_lr_cfp)
    return f"{tag}_{slug}_benchmark_metrics.csv"


def _training_profile(spec, n_train: int) -> dict:
    """ExpoCM-aligned defaults; scaled for 20GB GPU."""
    slug = spec.slug
    # ExpoCM: 500k iters, batch 4
    iters = 500_000
    batch = 4
    steps_per_epoch = max(1, math.ceil((n_train * 0.8) / batch))
    epochs = max(1, math.ceil(iters / steps_per_epoch))

    if slug == "hdr_eye":
        return dict(batch_size=1, epochs=min(epochs, 300), max_dim=spec.max_dim, val_ratio=0.2)
    if slug == "aim2025":
        return dict(batch_size=2, epochs=epochs, max_dim=256, val_ratio=0.2)
    return dict(batch_size=1, epochs=epochs, max_dim=256, val_ratio=spec.val_ratio)


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage-2 on an ExpoCM benchmark with full metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--warm_start_from", type=str, default="")
    parser.add_argument("--arch_v2", action="store_true")
    parser.add_argument("--use_rso", action="store_true")
    parser.add_argument("--use_lr_cfp", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--no_benchmark_metrics", action="store_true",
                        help="Disable full Table-1 metric CSV (PSNR-l/μ/PU, MS-SSIM, LPIPS, ΔE2000).")
    args, extra = parser.parse_known_args()

    spec = get_dataset_spec(args.dataset).resolve_paths()
    if not spec.is_ready():
        print(f"Dataset not ready: {spec.ldr_dir.parent}")
        print(f"Run: python scripts/download_expo_datasets.py --dataset {args.dataset}")
        sys.exit(1)

    n_pairs = len(list(spec.ldr_dir.glob("*")))
    profile = _training_profile(spec, n_pairs)
    batch_size = args.batch_size or profile["batch_size"]
    epochs = args.epochs or profile["epochs"]
    max_dim = args.max_dim or profile["max_dim"]
    ckpt = args.checkpoint_dir or _default_checkpoint_dir(
        spec.slug, args.arch_v2, args.use_rso, args.use_lr_cfp
    )
    metrics_csv = _metrics_csv_name(spec.slug, args.arch_v2, args.use_rso, args.use_lr_cfp)
    arch_tag = _arch_tag(args.arch_v2, args.use_rso, args.use_lr_cfp)

    t = spec.expo_targets
    print(f"\n=== TriGate benchmark run: {arch_tag} on {spec.name} ===")
    print(f"  Pairs: {n_pairs}  train_max_dim={max_dim}  epochs≈{epochs}")
    print(f"  Metrics file: {ckpt}/{metrics_csv}")
    print(f"  ExpoCM reference — PSNR-μ={t.psnr_mu:.2f} SSIM-μ={t.ssim_mu:.4f}\n")

    cmd = [
        sys.executable, "-u", "-m", "model.training_scripts.train_stage2_crf_recovery",
        "--ldr_dir", str(spec.ldr_dir),
        "--hdr_dir", str(spec.hdr_dir),
        "--checkpoint_dir", ckpt,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--max_dim", str(max_dim),
        "--val_ratio", str(spec.val_ratio),
        "--split_seed", str(spec.split_seed),
        "--inference_timesteps", "25",
        "--inference_loss_weight", "0",
        "--freeze_vae_after_warmup",
        "--vae_warmup_epochs", "0",
        "--early_stop_patience", "20",
        "--full_val_every", "1",
    ]
    if not args.no_benchmark_metrics:
        cmd.extend(["--benchmark_metrics", "--benchmark_metrics_csv", metrics_csv])
    if args.warm_start_from:
        cmd.extend(["--warm_start_from", args.warm_start_from])
    elif Path("experiments/stage2_lorcd_v2_arch/best.pt").is_file():
        cmd.extend(["--warm_start_from", "experiments/stage2_lorcd_v2_arch/best.pt"])
    if args.arch_v2:
        cmd.append("--arch_v2")
    if args.use_rso:
        cmd.append("--use_rso")
    if args.use_lr_cfp:
        cmd.append("--use_lr_cfp")
    if args.amp:
        cmd.append("--amp")
    cmd.extend(extra)

    if args.dry_run:
        print(" ".join(cmd))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("HDRVDP_OCTAVE_BIN", os.path.expanduser("~/anaconda3/envs/trigate-hdrvdp/bin/octave"))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
