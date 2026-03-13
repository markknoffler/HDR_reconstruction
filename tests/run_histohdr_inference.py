#!/usr/bin/env python3
"""
HistoHDRnet inference script: loads LDR images from tests/ldr_images,
processes them with the pretrained HistoHDRNet model (checkpoint_epoch_200),
and saves HDR outputs to tests/hdr_output_histohdr.
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch

# Add HistoHDRnet to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HISTOHDR_PATH = os.path.join(PROJECT_ROOT, "HistoHDRnet")
sys.path.insert(0, HISTOHDR_PATH)

from model import HistoHDRNet

# Default paths
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "HistoHDRnet", "checkpoints", "checkpoint_epoch_200.pth"
)
DEFAULT_LDR_DIR = os.path.join(SCRIPT_DIR, "ldr_images")
DEFAULT_HDR_DIR = os.path.join(SCRIPT_DIR, "hdr_output_histohdr")


def load_ldr_image(path, size=None):
    """Load LDR image [0, 1], RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    if size is not None:
        img = cv2.resize(img, (size, size))
    return img


def histogram_equalization(img):
    """Histogram equalization on luminance (Y channel) - same as HistoHDRnet training."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_eq.astype(np.float32) / 255.0


def save_hdr_image(img_np, path):
    """Save HDR image in Radiance .hdr format using imageio."""
    import imageio
    img_np = np.clip(img_np, 0.0, None).astype(np.float32)
    imageio.imwrite(path, img_np)


def run_inference(checkpoint_path, ldr_dir, hdr_dir, image_size, device):
    """Run HistoHDRNet inference on all LDR images in ldr_dir."""
    os.makedirs(hdr_dir, exist_ok=True)

    # Resolve checkpoint
    ckpt = checkpoint_path
    if not os.path.isfile(ckpt):
        for alt in [
            os.path.join(PROJECT_ROOT, "HistoHDRnet", "checkpoints", "checkpoint_epoch_200.pth"),
            os.path.join(PROJECT_ROOT, "HistoHDRnet", "checkpoints", "best_model.pth"),
        ]:
            if os.path.isfile(alt):
                ckpt = alt
                break
    if not os.path.isfile(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    # Collect LDR images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    ldr_files = []
    for ext in exts:
        ldr_files.extend(glob.glob(os.path.join(ldr_dir, ext)))
    ldr_files = sorted(set(ldr_files))

    if not ldr_files:
        print(f"No LDR images found in {ldr_dir}")
        sys.exit(1)

    # HistoHDRNet uses ResNet50; minimum 224x224
    min_size = 224
    if image_size is not None and image_size < min_size:
        image_size = min_size

    print(f"Found {len(ldr_files)} LDR images")
    print(f"Checkpoint: {ckpt}")
    print(f"Output dir: {hdr_dir}")

    # Load model
    model = HistoHDRNet(pretrained=True)

    try:
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    for ldr_path in ldr_files:
        base = os.path.splitext(os.path.basename(ldr_path))[0]
        out_path = os.path.join(hdr_dir, f"{base}.hdr")

        ldr_gt = load_ldr_image(ldr_path, size=image_size)
        ldr_his = histogram_equalization(ldr_gt)

        ldr_gt_t = torch.from_numpy(ldr_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
        ldr_his_t = torch.from_numpy(ldr_his).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            hdr_pred = model(ldr_gt_t, ldr_his_t)

        # Model outputs Tanh [-1, 1]; denormalize to [0, 1]
        hdr_np = hdr_pred[0].cpu().numpy().transpose(1, 2, 0)
        hdr_np = hdr_np * 0.5 + 0.5
        hdr_np = np.clip(hdr_np, 0.0, None).astype(np.float32)

        save_hdr_image(hdr_np, out_path)
        print(f"  {base} -> {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="HistoHDRNet inference on LDR images")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ldr_dir", type=str, default=DEFAULT_LDR_DIR)
    parser.add_argument("--hdr_dir", type=str, default=DEFAULT_HDR_DIR)
    parser.add_argument("--image_size", type=int, default=512, help="0 = keep original (min 224 for ResNet)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    size = args.image_size if args.image_size > 0 else None
    run_inference(
        checkpoint_path=args.checkpoint,
        ldr_dir=args.ldr_dir,
        hdr_dir=args.hdr_dir,
        image_size=size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
