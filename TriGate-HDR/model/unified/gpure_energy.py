"""
Gate-Partitioned Unified Radiance Energy (GPURE) loss terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class GPUREEnergyConfig:
    lambda_rad: float = 1.0
    lambda_cold: float = 1.0
    lambda_gen: float = 1.0
    lambda_bracket: float = 0.5
    lambda_seam: float = 0.25
    lambda_outside_lock: float = 0.5


def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def mu_law(x: torch.Tensor, mu: float = 5000.0) -> torch.Tensor:
    x01 = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    return torch.log1p(mu * x01) / torch.log1p(torch.tensor(mu, device=x.device, dtype=x.dtype))


def bracket_consistency_loss(
    x_gen: torch.Tensor,
    x_cold: torch.Tensor,
    seam_mask: torch.Tensor,
) -> torch.Tensor:
    """ECC: exposure-consistent coupling on seam band only."""
    if seam_mask.dim() == 3:
        seam_mask = seam_mask.unsqueeze(1)
    m = seam_mask.float().clamp(0.0, 1.0)
    mu_g = mu_law(x_gen)
    mu_c = mu_law(x_cold)
    l1 = (m * (mu_g - mu_c).abs()).sum() / m.sum().clamp_min(1.0)
    grad_g = _sobel_grad(x_gen)
    grad_c = _sobel_grad(x_cold)
    grad = (m * (grad_g - grad_c).abs()).sum() / m.sum().clamp_min(1.0)
    return l1 + grad


def seam_smoothness_loss(x: torch.Tensor, seam_mask: torch.Tensor) -> torch.Tensor:
    if seam_mask.dim() == 3:
        seam_mask = seam_mask.unsqueeze(1)
    m = seam_mask.float()
    grad = _sobel_grad(x)
    return (m * grad).mean()


def masked_gen_loss(x_gen: torch.Tensor, x_gt: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
    """Generative path supervision only in clipped regions."""
    if clip_mask.dim() == 3:
        clip_mask = clip_mask.unsqueeze(1)
    m = clip_mask.float().clamp(0.0, 1.0)
    err = F.l1_loss(x_gen, x_gt, reduction="none")
    return (m * err).sum() / m.sum().clamp_min(1.0)


def radiance_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_pred, x_gt)


def outside_lock_loss(x_out: torch.Tensor, x_comp: torch.Tensor, seam_mask: torch.Tensor) -> torch.Tensor:
    if seam_mask.dim() == 3:
        seam_mask = seam_mask.unsqueeze(1)
    outside = (1.0 - seam_mask.float()).clamp(0.0, 1.0)
    return (outside * (x_out - x_comp).abs()).mean()


@dataclass
class GPURELossParts:
    total: torch.Tensor
    rad: torch.Tensor
    cold: torch.Tensor
    gen: torch.Tensor
    bracket: torch.Tensor
    seam: torch.Tensor
    outside: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))


def compute_gpure_energy(
    x_final: torch.Tensor,
    x_gt: torch.Tensor,
    x_gen: torch.Tensor,
    x_cold: torch.Tensor,
    seam_mask: torch.Tensor,
    clip_mask: torch.Tensor,
    cold_loss: torch.Tensor | None = None,
    gen_loss: torch.Tensor | None = None,
    x_seam: torch.Tensor | None = None,
    x_comp: torch.Tensor | None = None,
    cfg: GPUREEnergyConfig | None = None,
) -> GPURELossParts:
    """
    Unified GPURE energy combining radiance, cold, generative, bracket, and seam terms.
    """
    cfg = cfg or GPUREEnergyConfig()

    l_rad = radiance_loss(x_final, x_gt)
    l_cold = cold_loss if cold_loss is not None else torch.tensor(0.0, device=x_final.device)
    l_gen = gen_loss if gen_loss is not None else masked_gen_loss(x_gen, x_gt, clip_mask)
    l_bracket = bracket_consistency_loss(x_gen, x_cold, seam_mask)
    l_seam = seam_smoothness_loss(x_final, seam_mask)

    l_outside = torch.tensor(0.0, device=x_final.device)
    if x_seam is not None and x_comp is not None:
        l_outside = outside_lock_loss(x_seam, x_comp, seam_mask)

    total = (
        cfg.lambda_rad * l_rad
        + cfg.lambda_cold * l_cold
        + cfg.lambda_gen * l_gen
        + cfg.lambda_bracket * l_bracket
        + cfg.lambda_seam * l_seam
        + cfg.lambda_outside_lock * l_outside
    )

    return GPURELossParts(
        total=total,
        rad=l_rad,
        cold=l_cold,
        gen=l_gen,
        bracket=l_bracket,
        seam=l_seam,
        outside=l_outside,
    )
