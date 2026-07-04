#!/usr/bin/env python3
"""
Print a copy-paste train_stage2_crf_recovery command for an ExpoCM benchmark dataset.

Does not train — only emits the exact shell command with absolute paths.

Usage:
  cd TriGate-HDR
  python scripts/emit_expo_train_command.py --dataset hdr_real
  python scripts/emit_expo_train_command.py --dataset hdr_eye --continue_train
  python scripts/emit_expo_train_command.py --all
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIGATE = ROOT / "TriGate-HDR"
sys.path.insert(0, str(TRIGATE))

from expo_benchmarks.registry import get_dataset_spec, list_datasets  # noqa: E402


def _arch_tag(arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> str:
    parts = ["trigate"]
    if arch_v2:
        parts.append("v2")
    if use_rso or use_lr_cfp:
        parts.append("gpure")
    return "_".join(parts)


def _default_ckpt(slug: str, arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> Path:
    return TRIGATE / "experiments" / f"{_arch_tag(arch_v2, use_rso, use_lr_cfp)}_{slug}"


def _metrics_csv(slug: str, arch_v2: bool, use_rso: bool, use_lr_cfp: bool) -> str:
    return f"{_arch_tag(arch_v2, use_rso, use_lr_cfp)}_{slug}_benchmark_metrics.csv"


def _epochs_for_dataset(slug: str, n_pairs: int, batch_size: int) -> int:
    """ExpoCM §4.1: 500k iterations, batch 4 — scaled for dataset size."""
    iters = 500_000
    batch = 4 if slug not in ("hdr_eye",) else 1
    steps = max(1, math.ceil((n_pairs * 0.8) / batch))
    epochs = max(1, math.ceil(iters / steps))
    if slug == "hdr_eye":
        return min(epochs, 300)
    return epochs


def emit_command(
    slug: str,
    *,
    gpu: int = 0,
    arch_v2: bool = True,
    use_rso: bool = True,
    use_lr_cfp: bool = True,
    amp: bool = True,
    batch_size: int = 0,
    max_dim: int = 0,
    epochs: int = 0,
    continue_train: bool = False,
    warm_start_from: str = "",
    checkpoint_dir: str = "",
) -> str:
    spec = get_dataset_spec(slug).resolve_paths()
    if not spec.is_ready():
        raise SystemExit(
            f"Dataset not ready: {spec.ldr_dir.parent}\n"
            f"Run: cd TriGate-HDR && python scripts/download_expo_datasets.py --dataset {slug}"
        )

    n_pairs = len(list(spec.ldr_dir.glob("*")))
    bs = batch_size or (1 if slug in ("hdr_real", "hdr_real_full", "hdr_eye") else 2)
    md = max_dim or (256 if slug == "aim2025" else spec.max_dim)
    ep = epochs or _epochs_for_dataset(slug, n_pairs, bs)
    ckpt = Path(checkpoint_dir) if checkpoint_dir else _default_ckpt(slug, arch_v2, use_rso, use_lr_cfp)
    metrics_csv = _metrics_csv(slug, arch_v2, use_rso, use_lr_cfp)

    home = Path.home()
    ldr = spec.ldr_dir
    hdr = spec.hdr_dir

    lines = [
        f"CUDA_VISIBLE_DEVICES={gpu} python -u -m model.training_scripts.train_stage2_crf_recovery \\",
        f'    --ldr_dir "{ldr}" \\',
        f'    --hdr_dir "{hdr}" \\',
        f'    --checkpoint_dir "{ckpt}" \\',
    ]
    if arch_v2:
        lines.append("    --arch_v2 \\")
    if use_rso:
        lines.append("    --use_rso \\")
    if use_lr_cfp:
        lines.append("    --use_lr_cfp \\")
    if amp:
        lines.append("    --amp \\")
    lines.extend([
        f"    --batch_size {bs} --max_dim {md} --num_workers 2 \\",
        "    --cold_lr 1e-5 --ema_decay 0 --vae_warmup_epochs 0 --freeze_vae_after_warmup \\",
        f"    --val_ratio {spec.val_ratio} --split_seed {spec.split_seed} --skip_trial_validation \\",
        "    --full_val_every 5 --train_eval_samples 50 --val_eval_samples 50 \\",
        "    --early_stop_patience 8 --inference_timesteps 25 \\",
        "    --inference_loss_weight 0 \\",
        "    --benchmark_metrics \\",
        f'    --benchmark_metrics_csv "{metrics_csv}" \\',
        f"    --epochs {ep} \\",
    ])
    if warm_start_from:
        lines.append(f'    --warm_start_from "{warm_start_from}"')
    elif continue_train:
        lines.append("    --continue_train")
    else:
        # drop trailing backslash on last line
        lines[-1] = lines[-1].rstrip(" \\")

    preamble = (
        f"# ExpoCM benchmark: {spec.name} ({slug}) — {n_pairs} pairs, "
        f"ExpoCM target PSNR-μ={spec.expo_targets.psnr_mu:.2f}\n"
        f"cd {TRIGATE}\n"
        f"export PYTHONPATH=\"$(pwd)\" CUDA_VISIBLE_DEVICES={gpu}\n"
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f"conda activate deeplearning\n"
    )
    return preamble + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Emit TriGate ExpoCM training shell commands.")
    parser.add_argument("--dataset", default="hdr_real", choices=list_datasets())
    parser.add_argument("--all", action="store_true", help="Print commands for hdr_real, hdr_eye, aim2025, hdr_synth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch_v2", action="store_true", default=True)
    parser.add_argument("--no_arch_v2", action="store_false", dest="arch_v2")
    parser.add_argument("--use_rso", action="store_true", default=True)
    parser.add_argument("--no_rso", action="store_false", dest="use_rso")
    parser.add_argument("--use_lr_cfp", action="store_true", default=True)
    parser.add_argument("--no_lr_cfp", action="store_false", dest="use_lr_cfp")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--warm_start_from", default="")
    parser.add_argument("--checkpoint_dir", default="")
    args = parser.parse_args()

    slugs = ["hdr_real", "hdr_eye", "hdr_synth", "aim2025", "hdr_real_full"] if args.all else [args.dataset]
    for i, slug in enumerate(slugs):
        if i:
            print("\n" + "=" * 72 + "\n")
        try:
            print(emit_command(
                slug,
                gpu=args.gpu,
                arch_v2=args.arch_v2,
                use_rso=args.use_rso,
                use_lr_cfp=args.use_lr_cfp,
                amp=args.amp,
                batch_size=args.batch_size,
                max_dim=args.max_dim,
                epochs=args.epochs,
                continue_train=args.continue_train,
                warm_start_from=args.warm_start_from,
                checkpoint_dir=args.checkpoint_dir,
            ))
        except SystemExit as exc:
            print(f"# SKIP {slug}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
