#!/usr/bin/env python3
"""
Recompute ExpoCM benchmark CSV rows using the current metric code.

Use this after fixing HDR-VDP-3 / PU21 metrics — do NOT hand-edit CSV values.

Examples:
  cd TriGate-HDR
  python scripts/recompute_benchmark_csv.py \\
      --dataset hdr_real_full \\
      --checkpoint experiments/trigate_v2_gpure_hdr_real_full/best.pt \\
      --output experiments/trigate_v2_gpure_hdr_real_full/trigate_v2_gpure_hdr_real_full_benchmark_metrics.csv

  # Recompute one row per saved epoch_*.pt in an experiment dir:
  python scripts/recompute_benchmark_csv.py \\
      --dataset hdr_real_full \\
      --checkpoint_dir experiments/trigate_v2_gpure_hdr_real_full \\
      --all_epoch_checkpoints
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expo_benchmarks.registry import get_dataset_spec  # noqa: E402
from model.training_scripts.evaluate_expo_benchmark import (  # noqa: E402
    _load_stage2_predictor,
    evaluate_split,
)
from model.training_scripts.common_training import HDRVDPMetrics, sanitize_data_path  # noqa: E402
from model.training_scripts.data_loader import TriGateHDRDataset  # noqa: E402
from model.training_scripts.dataset_splits import compute_split_indices  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
import torch  # noqa: E402


def _epoch_from_path(path: str) -> int:
    m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "epoch", "split", "psnr_mu", "ssim_mu", "psnr_pu", "ssim_pu",
        "psnr_l", "ssim_l", "ms_ssim", "hdrvdp2", "hdrvdp3", "lpips", "delta_e2000",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _eval_checkpoint(
    ckpt_path: Path,
    spec,
    device: torch.device,
    arch_v2: bool,
    use_rso: bool,
    use_lr_cfp: bool,
    hdrvdp_fast_proxy: bool,
    splits: list[str],
) -> list[dict]:
    max_dim = spec.max_dim
    ds = TriGateHDRDataset(
        sanitize_data_path(str(spec.ldr_dir)),
        sanitize_data_path(str(spec.hdr_dir)),
        mode="train",
        max_dim=max_dim if max_dim > 0 else 0,
    )
    train_idx, val_idx, _ = compute_split_indices(len(ds), val_ratio=spec.val_ratio, split_seed=spec.split_seed)
    split_map = {"train": train_idx, "val": val_idx, "val-full": val_idx}

    predict_fn = _load_stage2_predictor(str(ckpt_path), device, arch_v2, use_rso, use_lr_cfp)
    hdrvdp = HDRVDPMetrics(use_real_hdrvdp=not hdrvdp_fast_proxy)

    epoch = _epoch_from_path(str(ckpt_path))
    if epoch < 0:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0

    rows = []
    for split_name in splits:
        indices = split_map.get(split_name, val_idx)
        loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=0)
        m = evaluate_split(loader, device, predict_fn, hdrvdp, desc=f"{ckpt_path.name} {split_name}")
        row = {"epoch": epoch, "split": split_name.replace("val-full", "val-full")}
        if split_name == "val-full":
            row["split"] = "val-full"
        for k, v in m.as_dict().items():
            row[k] = f"{v:.6f}"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Recompute benchmark metrics CSV with current metric code.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="", help="Single .pt file (e.g. best.pt)")
    parser.add_argument("--checkpoint_dir", default="", help="Directory with epoch_*.pt files")
    parser.add_argument("--all_epoch_checkpoints", action="store_true")
    parser.add_argument("--output", default="", help="Output CSV path")
    parser.add_argument("--splits", default="train,val,val-full")
    parser.add_argument("--arch_v2", action="store_true")
    parser.add_argument("--use_rso", action="store_true")
    parser.add_argument("--use_lr_cfp", action="store_true")
    parser.add_argument("--hdrvdp_fast_proxy", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset).resolve_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    ckpts: list[Path] = []
    if args.checkpoint:
        ckpts = [Path(args.checkpoint)]
    elif args.checkpoint_dir and args.all_epoch_checkpoints:
        ckpts = sorted(
            Path(p) for p in glob.glob(os.path.join(args.checkpoint_dir, "epoch_*.pt"))
        )
        ckpts = sorted(ckpts, key=lambda p: _epoch_from_path(str(p)))
    else:
        parser.error("Provide --checkpoint or (--checkpoint_dir and --all_epoch_checkpoints)")

    out = Path(args.output) if args.output else (
        Path(args.checkpoint_dir) / f"trigate_v2_gpure_{args.dataset}_benchmark_metrics.csv"
        if args.checkpoint_dir
        else Path(ckpts[0]).parent / f"trigate_v2_gpure_{args.dataset}_benchmark_metrics.csv"
    )

    if args.backup and out.is_file():
        bak = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, bak)
        print(f"Backed up existing CSV -> {bak}")

    all_rows: list[dict] = []
    for ckpt in ckpts:
        print(f"\n=== Recomputing {ckpt} ===")
        all_rows.extend(
            _eval_checkpoint(
                ckpt, spec, device, args.arch_v2, args.use_rso, args.use_lr_cfp,
                args.hdrvdp_fast_proxy, splits,
            )
        )

    _write_rows(out, all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
