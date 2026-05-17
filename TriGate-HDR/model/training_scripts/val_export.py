"""
Per-epoch validation matching ARThdrNet/m_training.py validate_model().
"""

import os
import random
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from .common_training import compute_psnr_ssim, save_hdr_image, save_ldr_image_01


def pick_val_export_indices(val_indices: List[int], count: int, seed: int) -> List[int]:
    if not val_indices:
        return []
    rng = random.Random(seed)
    pool = list(val_indices)
    rng.shuffle(pool)
    return sorted(pool[: min(count, len(pool))])


@torch.no_grad()
def validate_model_mtraining(
    val_loader,
    device,
    epoch: int,
    hdrvdp_calculator,
    predict_hdr: Callable,
    validation_root: str,
    save_samples: bool = True,
    max_samples: int = 10,
    amp: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Mirror ARThdrNet/m_training.py validate_model:
    - full val split metrics (PSNR-mu, SSIM, HDR-VDP-2/3)
    - optional sample dump: validation_results/epoch_{epoch}/ldr_*.png, pred_hdr_*.hdr, gt_hdr_*.hdr
    """
    total_psnr = 0.0
    total_ssim = 0.0
    total_hdrvdp2 = 0.0
    total_hdrvdp3 = 0.0
    num_samples = 0

    sample_dir = os.path.join(validation_root, f"epoch_{epoch}")
    if save_samples:
        os.makedirs(sample_dir, exist_ok=True)
    sample_count = 0

    for batch in tqdm(val_loader, desc="Validation"):
        input_ldr = batch["ldr_image"].to(device)
        ground_truth = batch["hdr_image"].to(device)

        with autocast("cuda", enabled=amp and device.type == "cuda"):
            hdr_pred = predict_hdr(batch, input_ldr, ground_truth, device)

        if save_samples and sample_count < max_samples:
            for i in range(min(hdr_pred.shape[0], max_samples - sample_count)):
                ldr_path = os.path.join(sample_dir, f"ldr_{sample_count}.png")
                pred_path = os.path.join(sample_dir, f"pred_hdr_{sample_count}.hdr")
                gt_path = os.path.join(sample_dir, f"gt_hdr_{sample_count}.hdr")
                save_ldr_image_01(input_ldr, i, ldr_path)
                save_hdr_image(hdr_pred, i, pred_path)
                save_hdr_image(ground_truth, i, gt_path)
                sample_count += 1

        for i in range(hdr_pred.shape[0]):
            pred_img = hdr_pred[i]
            gt_img = ground_truth[i]
            psnr, ssim = compute_psnr_ssim(pred_img, gt_img)
            total_psnr += psnr
            total_ssim += ssim
            total_hdrvdp2 += hdrvdp_calculator.compute_hdrvdp2(pred_img, gt_img)
            total_hdrvdp3 += hdrvdp_calculator.compute_hdrvdp3(pred_img, gt_img)
            num_samples += 1

    if num_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_psnr / num_samples,
        total_ssim / num_samples,
        total_hdrvdp2 / num_samples,
        total_hdrvdp3 / num_samples,
    )


def _build_composited_input(stage2_hdr, stage1_hdr, gate):
    clip_mask = (1.0 - gate).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    seam_band = torch.maximum(seam_band, clip_mask)
    return composed, seam_band


def make_stage1_predictor(model):
    def predict(batch, input_ldr, ground_truth, device):
        segmap = batch.get("segmap", input_ldr)
        if not torch.is_tensor(segmap):
            segmap = input_ldr
        else:
            segmap = segmap.to(device)
        t = torch.zeros((input_ldr.shape[0],), device=device, dtype=torch.long)
        pred, _, _, _ = model(input_ldr, t, segmap=segmap)
        return pred

    return predict


def make_stage2_predictor(model):
    def predict(batch, input_ldr, ground_truth, device):
        return model.restore_hdr(input_ldr)

    return predict


def make_stage3_predictor(stage1, stage2, generator):
    def predict(batch, input_ldr, ground_truth, device):
        gate = batch["gate"].to(device)
        segmap = batch.get("segmap", input_ldr)
        if not torch.is_tensor(segmap):
            segmap = input_ldr
        else:
            segmap = segmap.to(device)
        t = torch.zeros((input_ldr.shape[0],), device=device, dtype=torch.long)
        gen_clip, _, _, _ = stage1(input_ldr, t, segmap=segmap)
        stage2_hdr = stage2.restore_hdr(input_ldr)
        composed, seam_mask = _build_composited_input(stage2_hdr, gen_clip, gate)
        return generator(composed, gen_clip, seam_mask)

    return predict
