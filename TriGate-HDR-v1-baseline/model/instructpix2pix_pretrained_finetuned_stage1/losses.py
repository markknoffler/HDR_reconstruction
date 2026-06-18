"""Stage-1 InstructPix2Pix fine-tune loss composition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..losses.codebook_losses import kl_codebook_loss
from ..losses.stage_composite_losses import stage1_loss


def diffusion_epsilon_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise.float(), target_noise.float())


def predict_x0_from_noise(
    scheduler,
    noisy_latents: torch.Tensor,
    pred_noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """DDPM x0 estimate from predicted noise (for novelty decode)."""
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta = 1.0 - alpha
    return (noisy_latents - beta.sqrt() * pred_noise) / alpha.sqrt().clamp(min=1e-8)


def novelty_losses(
    pred_hdr: torch.Tensor,
    target_hdr: torch.Tensor,
    gate: torch.Tensor,
    class_probs,
    sam_class_masks,
    aux,
    kl_weight: float = 0.01,
):
    loss, parts = stage1_loss(
        pred_hdr,
        target_hdr,
        gate,
        class_probs=class_probs,
        class_masks=sam_class_masks,
    )
    kl = kl_codebook_loss(aux["mus"], aux["logvars"])
    total = loss + kl_weight * kl
    parts["kl"] = kl
    parts["novelty_total"] = total
    return total, parts


def curriculum_novelty_weight(
    epoch: int,
    diffusion_only_epochs: int,
    ramp_epochs: int,
    max_weight: float,
) -> float:
    if epoch <= diffusion_only_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_weight
    t = min(1.0, float(epoch - diffusion_only_epochs) / float(ramp_epochs))
    return max_weight * t
