#!/usr/bin/env python3
"""
ARThdrNet inference script: loads LDR images from tests/ldr_images,
processes them with the pretrained model (checkpoint_epoch_200.pth),
and saves HDR outputs to tests/hdr_output.
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch

# Add ARThdrNet to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTHDRNET_PATH = os.path.join(PROJECT_ROOT, "ARThdrNet")
sys.path.insert(0, ARTHDRNET_PATH)

from model import ArtHDRNet

# Default paths (relative to project root)
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "ARThdrNet", "checkpoints", "checkpoint_epoch_200.pth"
)
DEFAULT_LDR_DIR = os.path.join(SCRIPT_DIR, "ldr_images")
DEFAULT_HDR_DIR = os.path.join(SCRIPT_DIR, "hdr_output")


def load_ldr_image(path, size=None):
    """Load LDR image and preprocess for model input [0, 1], RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    if size is not None:
        img = cv2.resize(img, (size, size))
    return img


def save_hdr_image(img_np, path):
    """Save HDR image in Radiance .hdr format using imageio."""
    import imageio
    img_np = np.clip(img_np, 0.0, None).astype(np.float32)
    # Use extension-based format (avoids format= kwarg issues across imageio versions)
    imageio.imwrite(path, img_np)


def run_inference(checkpoint_path, ldr_dir, hdr_dir, image_size, device, use_epoch_200=True):
    """Run ARThdrNet inference on all LDR images in ldr_dir."""
    os.makedirs(hdr_dir, exist_ok=True)

    # Resolve checkpoint path
    ckpt = checkpoint_path
    if use_epoch_200 and not os.path.isfile(ckpt):
        alt = os.path.join(PROJECT_ROOT, "ARThdrNet", "checkpoints", "checkpoint_epoch_200.pth")
        if os.path.isfile(alt):
            ckpt = alt
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
        print("Supported: .jpg, .jpeg, .png, .bmp")
        sys.exit(1)

    print(f"Found {len(ldr_files)} LDR images")
    print(f"Checkpoint: {ckpt}")
    print(f"Output dir: {hdr_dir}")

    # Load model
    model = ArtHDRNet(in_channels=3, base_channels=64, num_iterations=4)

    # Load checkpoint (PyTorch 2.6+ compatible: weights_only=False for full checkpoint)
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

        ldr = load_ldr_image(ldr_path, size=image_size)
        ldr_t = torch.from_numpy(ldr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            hdr_outputs = model(ldr_t)
            hdr_pred = hdr_outputs[-1]

        # Model outputs Tanh [-1, 1]; denormalize to [0, 1] for HDR
        hdr_np = hdr_pred[0].cpu().numpy().transpose(1, 2, 0)
        hdr_np = hdr_np * 0.5 + 0.5
        hdr_np = np.clip(hdr_np, 0.0, None).astype(np.float32)

        save_hdr_image(hdr_np, out_path)
        print(f"  {base} -> {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="ARThdrNet inference on LDR images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint (default: ARThdrNet/checkpoints/checkpoint_epoch_200.pth)",
    )
    parser.add_argument(
        "--ldr_dir",
        type=str,
        default=DEFAULT_LDR_DIR,
        help=f"Directory with LDR images (default: {DEFAULT_LDR_DIR})",
    )
    parser.add_argument(
        "--hdr_dir",
        type=str,
        default=DEFAULT_HDR_DIR,
        help=f"Directory for HDR outputs (default: {DEFAULT_HDR_DIR})",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize LDR images to this size (default: 512, 0 = keep original)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
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
