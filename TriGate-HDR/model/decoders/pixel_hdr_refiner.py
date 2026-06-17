"""
Pixel-space HDR refinement head for Stage 2.

Cold diffusion + VAE decode produce a coarse HDR; this network learns a
residual correction conditioned on the input LDR, recovering high-frequency
detail that the /8 latent bottleneck smears.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PixelHDRRefiner(nn.Module):
    """
    Residual pixel refiner: concat(LDR, coarse_HDR) -> delta_HDR.

    Operates in [-1, 1] HDR space (same as VAE decode output).
    """

    def __init__(self, base_ch: int = 48, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(6, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(base_ch) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1),
        )

    def forward(self, ldr_hdr: torch.Tensor, hdr_coarse: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ldr_hdr, hdr_coarse], dim=1)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return (hdr_coarse + self.head(h)).clamp(-1.0, 1.0)


def sobel_gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 on image gradients — encourages edge/texture fidelity (SSIM-related)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    pred_gx = F.conv2d(pred, kx, padding=1, groups=3)
    pred_gy = F.conv2d(pred, ky, padding=1, groups=3)
    gt_gx = F.conv2d(gt, kx, padding=1, groups=3)
    gt_gy = F.conv2d(gt, ky, padding=1, groups=3)
    return F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)
