"""
Frozen Stable Diffusion 2.1 img2img baseline: LDR in -> image out (no TriGate training).

Use this to verify Hugging Face weights load and to compare against GT tonemaps
before fine-tuning SD with TriGate encoders/losses.
"""

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
import torch

from ..decoders.stable_diffusion_stage1_decoder import FrozenStableDiffusionStage1
from .common_training import mu_tonemap, save_hdr_image, save_ldr_image_01, sanitize_hdr_tensor
from .data_loader import TriGateHDRDataset


def _save_tonemapped_preview(hdr_tensor, path, batch_idx=0):
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
    parser = argparse.ArgumentParser(description="Frozen SD 2.1 img2img LDR baseline.")
    parser.add_argument("--ldr_dir", type=str, required=True)
    parser.add_argument("--hdr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--strength", type=float, default=0.55, help="img2img strength (0-1)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_side", type=int, default=768, help="Max side sent to SD (VRAM)")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Stable Diffusion from: {args.model_id}")
    print(f"Device: {device}  dtype: {args.dtype}  (all weights frozen)")
    model = FrozenStableDiffusionStage1.from_pretrained(
        model_id=args.model_id,
        device=device,
        torch_dtype=args.dtype,
        cache_dir=args.cache_dir or None,
        local_files_only=args.local_files_only,
        enable_cpu_offload=args.cpu_offload,
    )

    ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.jpg")))
    if not ldr_files:
        ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.png")))
    if not ldr_files:
        raise FileNotFoundError(f"No LDR images in {args.ldr_dir}")

    rng = random.Random(args.seed)
    picks = rng.sample(ldr_files, min(args.num_samples, len(ldr_files)))

    ds = TriGateHDRDataset(args.ldr_dir, args.hdr_dir, mode="infer", max_dim=args.max_dim)
    name_to_idx = {os.path.basename(ds.pairs[i][0]): i for i in range(len(ds))}

    prompt = args.prompt or None
    negative_prompt = args.negative_prompt or None

    for ldr_path in picks:
        name = os.path.basename(ldr_path)
        stem = os.path.splitext(name)[0]
        if name not in name_to_idx:
            print(f"[WARN] skip (no HDR pair): {name}")
            continue

        batch = ds[name_to_idx[name]]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr_gt = batch["hdr_image"].unsqueeze(0)

        pred = model(
            ldr,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            max_side=args.max_side,
            output_range="trigate",
        )
        pred = sanitize_hdr_tensor(pred)

        out_stem = os.path.join(args.output_dir, stem)
        save_ldr_image_01(ldr.cpu(), 0, f"{out_stem}_input_ldr.png")
        save_hdr_image(pred, 0, f"{out_stem}_pred_sd_proxy.hdr")
        save_hdr_image(hdr_gt, 0, f"{out_stem}_gt_hdr.hdr")
        _save_tonemapped_preview(pred, f"{out_stem}_pred_tonemap.png")
        _save_tonemapped_preview(hdr_gt, f"{out_stem}_gt_tonemap.png")
        shutil.copy2(ldr_path, f"{out_stem}_input_original.jpg")
        print(f"  saved: {stem}")

    print(f"\nOutputs: {os.path.abspath(args.output_dir)}")
    print(
        "Note: SD baseline outputs sRGB-like img2img results, not fine-tuned linear HDR. "
        "Use for plumbing/qualitative comparison only."
    )


if __name__ == "__main__":
    main()
