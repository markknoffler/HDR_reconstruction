"""
Per-epoch validation matching ARThdrNet/m_training.py validate_model().
"""

import os
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Subset
from tqdm import tqdm

from .common_training import (
    _finite_metric,
    compute_psnr_ssim,
    mu_tonemap,
    sanitize_hdr_tensor,
    save_hdr_image,
    save_ldr_image_01,
)


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
        # Metrics: FHDR/test.py — no sanitize/clamp before PSNR-μ / SSIM
        hdr_pred = hdr_pred.float()
        ground_truth = ground_truth.float()

        if save_samples and sample_count < max_samples:
            for i in range(min(hdr_pred.shape[0], max_samples - sample_count)):
                ldr_path = os.path.join(sample_dir, f"ldr_{sample_count}.png")
                pred_path = os.path.join(sample_dir, f"pred_hdr_{sample_count}.hdr")
                gt_path = os.path.join(sample_dir, f"gt_hdr_{sample_count}.hdr")
                save_ldr_image_01(input_ldr, i, ldr_path)
                save_hdr_image(hdr_pred, i, pred_path)
                save_hdr_image(ground_truth, i, gt_path)
                sample_count += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for i in range(hdr_pred.shape[0]):
            pred_img = hdr_pred[i]
            gt_img = ground_truth[i]
            psnr, ssim = compute_psnr_ssim(pred_img, gt_img)
            h2 = hdrvdp_calculator.compute_hdrvdp2(pred_img, gt_img)
            h3 = hdrvdp_calculator.compute_hdrvdp3(pred_img, gt_img)
            total_psnr += psnr
            total_ssim += ssim
            total_hdrvdp2 += _finite_metric(h2)
            total_hdrvdp3 += _finite_metric(h3)
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


def _save_tonemapped_preview(hdr_tensor: torch.Tensor, path: str, batch_idx: int = 0) -> None:
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
def export_final_test_samples(
    full_dataset,
    val_indices: List[int],
    device,
    predict_hdr: Callable,
    output_dir: str,
    count: int = 5,
    seed: int = 123,
    amp: bool = False,
) -> None:
    """
    After training completes: pick random validation indices, run LDR->HDR, save previews.
    """
    picks = pick_val_export_indices(val_indices, count, seed)
    if not picks:
        print("[WARN] No validation indices for final test export.")
        return

    os.makedirs(output_dir, exist_ok=True)
    val_subset = Subset(full_dataset, picks)
    print(f"Final test export: {len(picks)} images -> {output_dir}")

    for i, idx in enumerate(picks):
        batch = full_dataset[idx]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr_gt = batch["hdr_image"].unsqueeze(0).to(device)
        stem = f"test_{i:02d}_idx{idx}"

        with autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = predict_hdr(batch, ldr, hdr_gt, device)
        pred = sanitize_hdr_tensor(pred)
        hdr_gt = sanitize_hdr_tensor(hdr_gt)

        save_ldr_image_01(ldr, 0, os.path.join(output_dir, f"{stem}_input_ldr.png"))
        save_hdr_image(pred, 0, os.path.join(output_dir, f"{stem}_pred_hdr.hdr"))
        save_hdr_image(hdr_gt, 0, os.path.join(output_dir, f"{stem}_gt_hdr.hdr"))
        _save_tonemapped_preview(pred, os.path.join(output_dir, f"{stem}_pred_tonemap.png"))
        _save_tonemapped_preview(hdr_gt, os.path.join(output_dir, f"{stem}_gt_tonemap.png"))
        print(f"  saved {stem}")


def make_stage1_instruct_predictor(model, num_inference_steps: int = 25):
    """Validation predictor for TrainableTriGateInstructPix2PixStage1."""

    def predict(batch, input_ldr, ground_truth, device):
        segmap = batch.get("segmap", input_ldr)
        if not torch.is_tensor(segmap):
            segmap = input_ldr
        else:
            segmap = segmap.to(device)
        return model.restore_hdr(
            input_ldr,
            segmap=segmap,
            num_inference_steps=num_inference_steps,
        )

    return predict


def make_stage2_predictor(model):
    def predict(batch, input_ldr, ground_truth, device):
        gate = batch.get("gate")
        if gate is not None:
            gate = gate.to(device)
        return model.restore_hdr(input_ldr, gate=gate)

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
        stage2_hdr = stage2.restore_hdr(input_ldr, gate=gate)
        composed, seam_mask = _build_composited_input(stage2_hdr, gen_clip, gate)
        return generator(composed, gen_clip, seam_mask)

    return predict
