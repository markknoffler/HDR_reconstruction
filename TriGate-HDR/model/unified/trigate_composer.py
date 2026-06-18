"""
Differentiable TriGate composer: gate-partitioned HDR fusion and seam-band construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_seam_band(clip_mask: torch.Tensor) -> torch.Tensor:
    """Morphological seam band from clip mask (matches legacy val_export behavior)."""
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    return torch.maximum(seam_band, clip_mask)


def build_composited_input(
    stage2_hdr: torch.Tensor,
    stage1_hdr: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compose Path-C (cold) and Path-G (generative) using TriGate clip mask.

    gate: 1 = well-exposed (trust cold path), 0 = clipped (use generative path).
    """
    if gate.dim() == 3:
        gate = gate.unsqueeze(1)
    clip_mask = (1.0 - gate.float()).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    seam_band = build_seam_band(clip_mask)
    return composed, seam_band


@dataclass
class ComposeOutputs:
    composed: torch.Tensor
    seam_mask: torch.Tensor
    clip_mask: torch.Tensor
    x_cold: torch.Tensor
    x_gen: torch.Tensor


class TriGateComposer(nn.Module):
    """
    Differentiable composition layer with optional soft seam blending during training.
    """

    def __init__(self, soft_seam_gamma: float = 0.0):
        super().__init__()
        self.soft_seam_gamma = float(soft_seam_gamma)

    def forward(
        self,
        x_cold: torch.Tensor,
        x_gen: torch.Tensor,
        gate: torch.Tensor,
        x_seam: torch.Tensor | None = None,
        training_soft_blend: bool = False,
    ) -> ComposeOutputs:
        composed, seam_mask = build_composited_input(x_cold, x_gen, gate)
        if gate.dim() == 3:
            gate = gate.unsqueeze(1)
        clip_mask = (1.0 - gate.float()).clamp(0.0, 1.0)

        if x_seam is not None and training_soft_blend and self.soft_seam_gamma > 0.0:
            gamma = self.soft_seam_gamma
            composed = composed * (1.0 - gamma * seam_mask) + x_seam * (gamma * seam_mask)

        return ComposeOutputs(
            composed=composed,
            seam_mask=seam_mask,
            clip_mask=clip_mask,
            x_cold=x_cold,
            x_gen=x_gen,
        )
