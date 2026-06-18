"""
Log-Radiance Cold Forward Process (LR-CFP).

Optically calibrated corruption: blend in log-radiance coordinates before latent
expansion decomposition, preserving spatial support while anchoring to LDR.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OpticalColdForward(nn.Module):
    """
    Maps HDR tensors in [-1, 1] to log-radiance features and supports LR-CFP blending.

    ell(x) = log(1 + k * max(radiance(x), 0))
    ell_t = (1 - alpha_t) * ell(x_hdr) + alpha_t * ell(x_ldr)
    """

    def __init__(self, num_channels: int = 3, init_k: float = 5000.0, log_scale: float = 10.0):
        super().__init__()
        self.log_scale = float(log_scale)
        # Per-channel learnable tone scale (positive via softplus).
        self._k_raw = nn.Parameter(torch.full((1, num_channels, 1, 1), float(init_k)).log())

    @property
    def k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._k_raw) + 1e-6

    @staticmethod
    def _to_radiance(x: torch.Tensor) -> torch.Tensor:
        """Map model HDR [-1, 1] to non-negative radiance proxy."""
        return ((x + 1.0) * 0.5).clamp(min=0.0)

    def encode_log_radiance(self, x: torch.Tensor) -> torch.Tensor:
        """ell(x) normalized for VAE encoding."""
        radiance = self._to_radiance(x)
        log_r = torch.log1p(self.k * radiance)
        return (log_r / self.log_scale).clamp(-1.0, 1.0)

    def decode_log_radiance(self, log_norm: torch.Tensor) -> torch.Tensor:
        """Approximate inverse: log_norm -> radiance proxy in [-1, 1]."""
        log_r = log_norm * self.log_scale
        radiance = (torch.expm1(log_r) / self.k).clamp(min=0.0, max=1.0)
        return radiance * 2.0 - 1.0

    def cold_blend_log(
        self,
        x_hdr: torch.Tensor,
        x_ldr_hdr: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        LR-CFP pixel-space blend (optional auxiliary path).

        alpha_t: (B, 1, 1, 1) or broadcastable corruption level in [0, 1].
        """
        ell_hdr = self.encode_log_radiance(x_hdr)
        ell_ldr = self.encode_log_radiance(x_ldr_hdr)
        ell_t = (1.0 - alpha_t) * ell_hdr + alpha_t * ell_ldr
        return self.decode_log_radiance(ell_t)

    def prepare_vae_input(self, x: torch.Tensor) -> torch.Tensor:
        """When LR-CFP is enabled, VAE sees log-radiance-normalized tensors."""
        return self.encode_log_radiance(x)
