"""
Run Stage-1 LDR->HDR inference on random dataset images and save outputs.
"""

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
import torch
from torch.amp import autocast

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from .common_training import load_checkpoint, mu_tonemap, save_hdr_image, save_ldr_image_01, sanitize_hdr_tensor
from .data_loader import TriGateHDRDataset


def _save_tonemapped_preview(hdr_tensor, path, batch_idx=0):
    """Save a viewable PNG from HDR tensor in [-1, 1]."""
    img = hdr_tensor.detach().float().cpu()
    if img.dim() == 4:
        img = img[batch_idx]
    rgb = (img + 1.0) * 0.5
    rgb = rgb.permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    tm = mu_tonemap(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)).squeeze(0)
    tm = tm.permute(1, 2, 0).numpy()
    tm = np.clip(tm, 0.0, 1.0)
    bgr = cv2.cvtColor((tm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, bgr)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Stage-1 LDR to HDR inference on random samples.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to epoch_*.pt or best.pt")
    parser.add_argument("--ldr_dir", type=str, required=True)
    parser.add_argument("--hdr_dir", type=str, required=True)
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_dim", type=int, default=0, help="0 = native resolution")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = Stage1TriEncoderDiffusionSystem().to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.jpg")))
    if not ldr_files:
        ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.png")))
    if not ldr_files:
        raise FileNotFoundError(f"No LDR images in {args.ldr_dir}")

    rng = random.Random(args.seed)
    picks = rng.sample(ldr_files, min(args.num_samples, len(ldr_files)))

    ds = TriGateHDRDataset(
        args.ldr_dir,
        args.hdr_dir,
        mode="infer",
        sam_mask_dir=args.sam_mask_dir,
        max_sam_classes=args.max_sam_classes,
        max_dim=args.max_dim,
    )
    name_to_idx = {os.path.basename(ds.pairs[i][0]): i for i in range(len(ds))}

    for i, ldr_path in enumerate(picks):
        name = os.path.basename(ldr_path)
        stem = os.path.splitext(name)[0]
        if name not in name_to_idx:
            print(f"[WARN] skip (no HDR pair): {name}")
            continue
        batch = ds[name_to_idx[name]]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr_gt = batch["hdr_image"].unsqueeze(0).to(device)
        segmap = batch.get("segmap", batch["ldr_image"])
        if torch.is_tensor(segmap):
            segmap = segmap.unsqueeze(0).to(device)
        else:
            segmap = ldr

        t = torch.zeros((1,), device=device, dtype=torch.long)
        with autocast("cuda", enabled=args.amp and device.type == "cuda"):
            pred, _, _, _ = model(ldr, t, segmap=segmap)
        pred = sanitize_hdr_tensor(pred)

        out_stem = os.path.join(args.output_dir, stem)
        save_ldr_image_01(ldr, 0, f"{out_stem}_input_ldr.png")
        save_hdr_image(pred, 0, f"{out_stem}_pred_hdr.hdr")
        save_hdr_image(hdr_gt, 0, f"{out_stem}_gt_hdr.hdr")
        _save_tonemapped_preview(pred, f"{out_stem}_pred_tonemap.png")
        _save_tonemapped_preview(hdr_gt, f"{out_stem}_gt_tonemap.png")
        shutil.copy2(ldr_path, f"{out_stem}_input_original.jpg")
        print(f"  saved: {stem}")

    print(f"\nOutputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
