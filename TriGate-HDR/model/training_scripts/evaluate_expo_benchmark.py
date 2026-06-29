"""ExpoCM benchmark evaluation on train / val / test splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expo_benchmarks.registry import get_dataset_spec, list_datasets  # noqa: E402
from model.decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion  # noqa: E402
from model.metrics.expo_metrics import (  # noqa: E402
    ExpoMetricVector,
    average_expo_metrics,
    compute_expo_metrics_pair,
    format_expo_table_row,
)
from model.training_scripts.common_training import HDRVDPMetrics, sanitize_data_path  # noqa: E402
from model.training_scripts.data_loader import TriGateHDRDataset  # noqa: E402
from model.training_scripts.dataset_splits import compute_split_indices  # noqa: E402
from model.training_scripts.val_export import make_stage2_predictor  # noqa: E402


def _load_stage2_predictor(checkpoint: str, device: torch.device, arch_v2: bool, use_rso: bool, use_lr_cfp: bool):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args_ckpt = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    base_ch = int(args_ckpt.get("base_ch", 96 if arch_v2 else 64))
    latent_ch = int(args_ckpt.get("latent_ch", 8 if arch_v2 else 4))
    vae_base_ch = int(args_ckpt.get("vae_base_ch", 48 if arch_v2 else 32))
    model = ColdHDRDiffusion(
        base_ch=base_ch,
        latent_ch=latent_ch,
        vae_base_ch=vae_base_ch,
        use_pixel_refiner=bool(args_ckpt.get("use_pixel_refiner", arch_v2)),
        use_lr_cfp=use_lr_cfp,
        use_rso=use_rso,
    ).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return make_stage2_predictor(model)


@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    device: torch.device,
    predict_fn: Callable,
    hdrvdp: HDRVDPMetrics,
    desc: str = "Eval",
) -> ExpoMetricVector:
    vectors = []
    for batch in tqdm(loader, desc=desc):
        ldr = batch["ldr_image"].to(device)
        gt = batch["hdr_image"].to(device)
        pred = predict_fn(batch, ldr, gt, device).float()
        for i in range(pred.shape[0]):
            vectors.append(compute_expo_metrics_pair(pred[i], gt[i], hdrvdp))
    return average_expo_metrics(vectors)


def main():
    parser = argparse.ArgumentParser(description="ExpoCM Table 1 metrics on train/val/test splits.")
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--val_ratio", type=float, default=-1.0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_dim", type=int, default=0, help="0 = native resolution")
    parser.add_argument("--arch_v2", action="store_true")
    parser.add_argument("--use_rso", action="store_true")
    parser.add_argument("--use_lr_cfp", action="store_true")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--hdrvdp_fast_proxy", action="store_true")
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset).resolve_paths()
    val_ratio = spec.val_ratio if args.val_ratio < 0 else args.val_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_dim = args.max_dim or spec.max_dim

    ds = TriGateHDRDataset(
        sanitize_data_path(str(spec.ldr_dir)),
        sanitize_data_path(str(spec.hdr_dir)),
        mode="train",
        max_dim=max_dim if max_dim > 0 else 0,
    )
    train_idx, val_idx, _ = compute_split_indices(len(ds), val_ratio=val_ratio, split_seed=args.split_seed)
    splits = {}
    if args.split in ("train", "all"):
        splits["train"] = train_idx
    if args.split in ("val", "all"):
        splits["val"] = val_idx
    if args.split == "test":
        splits["test"] = val_idx  # official test when no separate test manifest

    predict_fn = _load_stage2_predictor(
        args.checkpoint, device, args.arch_v2, args.use_rso, args.use_lr_cfp
    )
    hdrvdp = HDRVDPMetrics(use_real_hdrvdp=not args.hdrvdp_fast_proxy)

    results = {}
    for name, indices in splits.items():
        if not indices:
            continue
        loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=0)
        m = evaluate_split(loader, device, predict_fn, hdrvdp, desc=f"{name} ({len(indices)} imgs)")
        results[name] = m.as_dict()
        print(f"\n[{spec.name} / {name}] {format_expo_table_row(m)}")

    t = spec.expo_targets
    if "val" in results or "test" in results:
        key = "val" if "val" in results else "test"
        r = results[key]
        print(
            f"\nExpoCM 'Ours' target — PSNR-μ={t.psnr_mu:.2f} SSIM-μ={t.ssim_mu:.4f} "
            f"HDR-VDP-2={t.hdrvdp2:.2f} ΔE00={t.delta_e2000:.2f}"
        )
        print(f"Your {key} — PSNR-μ={r['psnr_mu']:.2f} SSIM-μ={r['ssim_mu']:.4f}")

    out = args.output_json or os.path.join(
        os.path.dirname(args.checkpoint), f"trigate_eval_{args.dataset}.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "checkpoint": args.checkpoint, "splits": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
