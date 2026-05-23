"""Shared helpers for Stable Diffusion decoders."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from PIL import Image


def require_diffusers():
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Stable Diffusion support requires: pip install diffusers transformers accelerate safetensors\n"
            "See model/decoders/README_STABLE_DIFFUSION.md"
        ) from exc


def freeze_module(module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def tensor_bchw_to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    """Convert BCHW float tensor in [0, 1] to PIL RGB."""
    if images.dim() != 4:
        raise ValueError(f"Expected BCHW, got shape {tuple(images.shape)}")
    out: List[Image.Image] = []
    x = images.detach().float().cpu().clamp(0.0, 1.0)
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def pil_list_to_tensor_bchw(images: List[Image.Image], device, dtype) -> torch.Tensor:
    """PIL RGB -> BCHW float [0, 1]."""
    batches = []
    for im in images:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        batches.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(batches, dim=0).to(device=device, dtype=dtype)


def round_size_to_multiple(size: int, multiple: int = 8) -> int:
    return max(multiple, int(round(size / multiple)) * multiple)


def resize_ldr_for_sd(ldr_bchw: torch.Tensor, max_side: int = 768) -> torch.Tensor:
    """Resize LDR to SD-friendly resolution (multiples of 8, cap max side)."""
    import torch.nn.functional as F

    _, _, h, w = ldr_bchw.shape
    scale = 1.0
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
    new_h = round_size_to_multiple(int(round(h * scale)))
    new_w = round_size_to_multiple(int(round(w * scale)))
    if new_h == h and new_w == w:
        return ldr_bchw
    return F.interpolate(ldr_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)


def sd_output_to_trigate_hdr_range(pred_01: torch.Tensor) -> torch.Tensor:
    """Map SD RGB [0,1] to TriGate HDR tensor convention [-1, 1] for saving/metrics."""
    return (2.0 * pred_01.clamp(0.0, 1.0) - 1.0).clamp(-1.0, 1.0)
