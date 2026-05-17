"""Validation metrics and exporting tonemapped preview images."""

import os
import random
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from .common_training import compute_psnr_ssim


def _tensor_to_ldr_preview(t: torch.Tensor) -> np.ndarray:
    """CHW tensor in approx [-1,1] -> uint8 RGB."""
    arr = t.detach().cpu().float()
    arr = (arr + 1.0) / 2.0
    arr = arr.clamp(0.0, 1.0)
    arr = arr.permute(1, 2, 0).numpy()
    return (arr * 255.0).astype(np.uint8)


def _save_rgb(path: str, rgb_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))


def pick_val_export_indices(val_indices: List[int], count: int, seed: int) -> List[int]:
    if not val_indices:
        return []
    rng = random.Random(seed)
    pool = list(val_indices)
    rng.shuffle(pool)
    return sorted(pool[: min(count, len(pool))])


@torch.no_grad()
def validate_stage1(model, val_loader, device, amp: bool = False):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for batch in val_loader:
        ldr = batch["ldr_image"].to(device)
        hdr = batch["hdr_image"].to(device)
        segmap = batch.get("segmap", ldr).to(device)
        t = torch.zeros((ldr.shape[0],), device=device, dtype=torch.long)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            pred, _, _, _ = model(ldr, t, segmap=segmap)
        for i in range(pred.shape[0]):
            psnr, ssim = compute_psnr_ssim(pred[i], hdr[i])
            total_psnr += psnr
            total_ssim += ssim
            n += 1
    if n == 0:
        return 0.0, 0.0
    return total_psnr / n, total_ssim / n


@torch.no_grad()
def validate_stage2(model, val_loader, device, amp: bool = False):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for batch in val_loader:
        ldr = batch["ldr_image"].to(device)
        hdr = batch["hdr_image"].to(device)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model.restore_hdr(ldr)
        for i in range(pred.shape[0]):
            psnr, ssim = compute_psnr_ssim(pred[i], hdr[i])
            total_psnr += psnr
            total_ssim += ssim
            n += 1
    if n == 0:
        return 0.0, 0.0
    return total_psnr / n, total_ssim / n


def _build_composited_input(stage2_hdr, stage1_hdr, gate):
    clip_mask = (1.0 - gate).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    seam_band = torch.maximum(seam_band, clip_mask)
    return composed, seam_band


@torch.no_grad()
def validate_stage3(stage1, stage2, generator, val_loader, device, amp: bool = False):
    stage1.eval()
    stage2.eval()
    generator.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for batch in val_loader:
        ldr = batch["ldr_image"].to(device)
        hdr = batch["hdr_image"].to(device)
        gate = batch["gate"].to(device)
        segmap = batch.get("segmap", ldr).to(device)
        t = torch.zeros((ldr.shape[0],), device=device, dtype=torch.long)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            gen_clip, _, _, _ = stage1(ldr, t, segmap=segmap)
            stage2_hdr = stage2.restore_hdr(ldr)
            composed, seam_mask = _build_composited_input(stage2_hdr, gen_clip, gate)
            fake = generator(composed, gen_clip, seam_mask)
        for i in range(fake.shape[0]):
            psnr, ssim = compute_psnr_ssim(fake[i], hdr[i])
            total_psnr += psnr
            total_ssim += ssim
            n += 1
    if n == 0:
        return 0.0, 0.0
    return total_psnr / n, total_ssim / n


@torch.no_grad()
def export_stage1_samples(model, full_dataset, export_indices, out_dir, device, amp: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for global_i in tqdm(export_indices, desc="Export Stage1"):
        batch = full_dataset[global_i]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr = batch["hdr_image"].unsqueeze(0).to(device)
        segmap = batch["segmap"].unsqueeze(0).to(device)
        stem = os.path.splitext(batch["ldr_path"])[0]
        t = torch.zeros((1,), device=device, dtype=torch.long)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            pred, _, _, _ = model(ldr, t, segmap=segmap)
        _save_rgb(os.path.join(out_dir, f"{stem}_pred.png"), _tensor_to_ldr_preview(pred[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_gt.png"), _tensor_to_ldr_preview(hdr[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_ldr.png"), _tensor_to_ldr_preview(ldr[0]))


@torch.no_grad()
def export_stage2_samples(model, full_dataset, export_indices, out_dir, device, amp: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for global_i in tqdm(export_indices, desc="Export Stage2"):
        batch = full_dataset[global_i]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr = batch["hdr_image"].unsqueeze(0).to(device)
        stem = os.path.splitext(batch["ldr_path"])[0]
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model.restore_hdr(ldr)
        _save_rgb(os.path.join(out_dir, f"{stem}_pred.png"), _tensor_to_ldr_preview(pred[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_gt.png"), _tensor_to_ldr_preview(hdr[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_ldr.png"), _tensor_to_ldr_preview(ldr[0]))


@torch.no_grad()
def export_stage3_samples(stage1, stage2, generator, full_dataset, export_indices, out_dir, device, amp: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    stage1.eval()
    stage2.eval()
    generator.eval()
    for global_i in tqdm(export_indices, desc="Export Stage3 (full pipeline)"):
        batch = full_dataset[global_i]
        ldr = batch["ldr_image"].unsqueeze(0).to(device)
        hdr = batch["hdr_image"].unsqueeze(0).to(device)
        gate = batch["gate"].unsqueeze(0).to(device)
        segmap = batch["segmap"].unsqueeze(0).to(device)
        stem = os.path.splitext(batch["ldr_path"])[0]
        t = torch.zeros((1,), device=device, dtype=torch.long)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            gen_clip, _, _, _ = stage1(ldr, t, segmap=segmap)
            stage2_hdr = stage2.restore_hdr(ldr)
            composed, seam_mask = _build_composited_input(stage2_hdr, gen_clip, gate)
            fake = generator(composed, gen_clip, seam_mask)
        _save_rgb(os.path.join(out_dir, f"{stem}_final.png"), _tensor_to_ldr_preview(fake[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_gt.png"), _tensor_to_ldr_preview(hdr[0]))
        _save_rgb(os.path.join(out_dir, f"{stem}_ldr.png"), _tensor_to_ldr_preview(ldr[0]))
