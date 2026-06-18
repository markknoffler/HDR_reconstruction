"""
Frozen diffusion baseline: LDR in -> image out (no TriGate training).

Default: InstructPix2Pix (trained on input image + text instruction -> edited image).
Legacy: SD 2.1 text-only img2img via --pipeline legacy_img2img.
"""

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
import torch

from ..decoders.stable_diffusion_instruct_pix2pix_decoder import (
    DEFAULT_HDR_INSTRUCTION,
    DEFAULT_NEGATIVE_PROMPT,
    FrozenInstructPix2PixStage1,
)
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
    parser = argparse.ArgumentParser(
        description="Frozen diffusion LDR baseline (InstructPix2Pix default)."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="instruct_pix2pix",
        choices=["instruct_pix2pix", "legacy_img2img"],
        help="instruct_pix2pix: native image+text edit model (recommended). "
        "legacy_img2img: SD2.1 text+img2img only.",
    )
    parser.add_argument("--ldr_dir", type=str, required=True)
    parser.add_argument("--hdr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="", help="HF model id; pipeline-specific default if empty")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Use float32 if outputs are white/NaN (fp16 VAE decode issue).",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Text instruction scale")
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="InstructPix2Pix: how strongly to follow input LDR (1.0-2.0)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.55,
        help="legacy_img2img only: noise strength on LDR latents",
    )
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--prompt", type=str, default="", help="Edit instruction; default HDR instruction")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    prompt = args.prompt if args.prompt else None
    negative_prompt = args.negative_prompt if args.negative_prompt else None

    if args.pipeline == "instruct_pix2pix":
        model_id = args.model_id or "timbrooks/instruct-pix2pix"
        print(f"Loading InstructPix2Pix (image + text native): {model_id}")
        print(f"  image_guidance_scale={args.image_guidance_scale}  guidance_scale={args.guidance_scale}")
        model = FrozenInstructPix2PixStage1.from_pretrained(
            model_id=model_id,
            device=device,
            torch_dtype=args.dtype,
            cache_dir=args.cache_dir or None,
            local_files_only=args.local_files_only,
            enable_cpu_offload=args.cpu_offload,
        )
        if prompt is None:
            prompt = DEFAULT_HDR_INSTRUCTION
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
    else:
        model_id = args.model_id or "stabilityai/stable-diffusion-2-1-base"
        print(f"Loading legacy SD img2img (text-primary): {model_id}")
        model = FrozenStableDiffusionStage1.from_pretrained(
            model_id=model_id,
            device=device,
            torch_dtype=args.dtype,
            cache_dir=args.cache_dir or None,
            local_files_only=args.local_files_only,
            enable_cpu_offload=args.cpu_offload,
        )

    print(f"Device: {device}  dtype: {args.dtype}  (frozen weights)")

    ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.jpg")))
    if not ldr_files:
        ldr_files = sorted(glob.glob(os.path.join(args.ldr_dir, "*.png")))
    if not ldr_files:
        raise FileNotFoundError(f"No LDR images in {args.ldr_dir}")

    rng = random.Random(args.seed)
    picks = rng.sample(ldr_files, min(args.num_samples, len(ldr_files)))

    ds = TriGateHDRDataset(args.ldr_dir, args.hdr_dir, mode="infer", max_dim=args.max_dim)
    name_to_idx = {os.path.basename(ds.pairs[i][0]): i for i in range(len(ds))}

    for ldr_path in picks:
        name = os.path.basename(ldr_path)
        stem = os.path.splitext(name)[0]
        if name not in name_to_idx:
            print(f"[WARN] skip (no HDR pair): {name}")
            continue

        batch = ds[name_to_idx[name]]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr_gt = batch["hdr_image"].unsqueeze(0)

        if args.pipeline == "instruct_pix2pix":
            pred = model(
                ldr,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                max_side=args.max_side,
                output_range="trigate",
            )
        else:
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
        # Save raw [0,1] RGB before TriGate [-1,1] mapping (debug white/NaN issues).
        pred_01 = ((pred.float().clamp(-1, 1) + 1.0) * 0.5).cpu()
        out_stem = os.path.join(args.output_dir, stem)
        save_ldr_image_01(ldr.cpu(), 0, f"{out_stem}_input_ldr.png")
        save_ldr_image_01(pred_01, 0, f"{out_stem}_pred_rgb_01.png")

        pred = sanitize_hdr_tensor(pred)
        save_hdr_image(pred, 0, f"{out_stem}_pred_sd_proxy.hdr")
        save_hdr_image(hdr_gt, 0, f"{out_stem}_gt_hdr.hdr")
        _save_tonemapped_preview(pred, f"{out_stem}_pred_tonemap.png")
        _save_tonemapped_preview(hdr_gt, f"{out_stem}_gt_tonemap.png")
        shutil.copy2(ldr_path, f"{out_stem}_input_original.jpg")
        print(f"  saved: {stem}")

    print(f"\nOutputs: {os.path.abspath(args.output_dir)}")
    print(
        "Note: frozen edit diffusion outputs sRGB-like proxies until fine-tuned on HDR pairs "
        "with TriGate encoders + diffusion/novelty losses."
    )


if __name__ == "__main__":
    main()
