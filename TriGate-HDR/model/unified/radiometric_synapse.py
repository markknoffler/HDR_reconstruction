"""
Radiometric Synapse Operators (RSO).

Domain-specific fusion replacing generic wx+b / conv gates at skip connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraResponseFn(nn.Module):
    """Learnable per-channel mu-law / Hable-style response Phi_k(h)."""

    def __init__(self, channels: int, init_k: float = 5000.0):
        super().__init__()
        self._k_raw = nn.Parameter(torch.full((1, channels, 1, 1), float(init_k)).log())

    @property
    def k(self) -> torch.Tensor:
        return F.softplus(self._k_raw) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.abs()
        return torch.sign(x) * torch.log1p(self.k * mag) / (torch.log1p(self.k) + 1e-6)


class SeamJacobian(nn.Module):
    """Time-modulated depthwise spatial Jacobian Psi_t(h_c, h_a)."""

    def __init__(self, channels: int, time_dim: int | None = None):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.time_proj = nn.Linear(time_dim, channels) if time_dim else None
        self.mix = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, cold: torch.Tensor, anchor: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.mix(torch.cat([self.dw(cold), self.dw(anchor)], dim=1))
        if self.time_proj is not None and time_emb is not None:
            t_bias = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t_bias
        return torch.tanh(h)


class RSOCell(nn.Module):
    """
    RSO(h_c, h_a, tau, t) = h_c + sigma(tau) * [exp(beta * log-ratio) * Phi_k(h_a) * Psi_t(h_c, h_a)]
    """

    def __init__(
        self,
        cold_ch: int,
        anchor_ch: int,
        time_dim: int | None = None,
        beta_init: float = 0.5,
    ):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.phi = CameraResponseFn(cold_ch)
        self.psi = SeamJacobian(cold_ch, time_dim=time_dim)
        self.anchor_proj = nn.Conv2d(anchor_ch, cold_ch, 1) if anchor_ch != cold_ch else nn.Identity()
        self.trust_gate = nn.Conv2d(1, cold_ch, 1)

    def forward(
        self,
        cold: torch.Tensor,
        anchor: torch.Tensor,
        trust: torch.Tensor,
        time_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if trust.shape[-2:] != cold.shape[-2:]:
            trust = F.interpolate(trust, size=cold.shape[-2:], mode="bilinear", align_corners=False)
        trust = trust.clamp(0.0, 1.0)

        anchor_p = self.anchor_proj(anchor)
        log_ratio = torch.log1p(cold.abs()) - torch.log1p(anchor_p.abs())
        gain = torch.exp(self.beta.clamp(-2.0, 2.0) * log_ratio)
        phi_a = self.phi(anchor_p)
        psi = self.psi(cold, anchor_p, time_emb=time_emb)

        injection = gain * phi_a * psi
        sigma_tau = torch.sigmoid(self.trust_gate(trust))
        return cold + (1.0 - sigma_tau) * injection + sigma_tau * anchor_p


class RSOStem(nn.Module):
    """Multi-stream stem fusion for Stage-3 (7-ch input -> base_ch)."""

    def __init__(self, in_ch: int = 7, out_ch: int = 64):
        super().__init__()
        self.base = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cold_slice = nn.Conv2d(3, out_ch, 1)
        self.gen_slice = nn.Conv2d(3, out_ch, 1)
        self.mask_slice = nn.Conv2d(1, out_ch, 1)
        self.rso = RSOCell(out_ch, out_ch)

    def forward(
        self,
        base_hdr: torch.Tensor,
        gen_clip: torch.Tensor,
        seam_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([base_hdr, gen_clip, seam_mask], dim=1)
        h = self.base(x)
        cold = self.cold_slice(base_hdr)
        anchor = self.gen_slice(gen_clip) + self.mask_slice(seam_mask)
        trust = 1.0 - seam_mask.clamp(0.0, 1.0)
        return self.rso(h, anchor, trust)
