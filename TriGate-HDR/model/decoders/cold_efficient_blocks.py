"""
Building blocks for ColdEfficient-LORCD: dual-stream latent UNet with RGCF fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch) if time_dim else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class RGCFBlock(nn.Module):
    """
    Trust-gated radiance fusion (conv cross-gate, O(HW) memory).
    tau ~ 1: lock to anchor; tau ~ 0: inject anchor context into cold stream.
    Full spatial attention was removed — it OOMs at 64x64 latent on 12GB GPUs.
    """

    def __init__(self, cold_ch: int, anchor_ch: int, heads: int = 4):  # noqa: ARG002
        super().__init__()
        mid = max(cold_ch, anchor_ch)
        self.cross_gate = nn.Sequential(
            nn.Conv2d(cold_ch + anchor_ch, mid, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, cold_ch, 1),
            nn.Sigmoid(),
        )
        self.cross_proj = nn.Conv2d(anchor_ch, cold_ch, 1)
        self.anchor_proj = nn.Conv2d(anchor_ch, cold_ch, 1)

    def forward(self, cold: torch.Tensor, anchor: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
        if trust.shape[-2:] != cold.shape[-2:]:
            trust = F.interpolate(trust, size=cold.shape[-2:], mode="bilinear", align_corners=False)
        trust = trust.clamp(0, 1)
        gate = self.cross_gate(torch.cat([cold, anchor], dim=1))
        cross = self.cross_proj(anchor) * gate
        anchor_p = self.anchor_proj(anchor)
        return cold + (1.0 - trust) * cross + trust * anchor_p


class ColdEfficientLatentUNet(nn.Module):
    """
    Dual-stream latent UNet: anchor from z_ldr, cold from concat(z_t, z_ldr).
    Predicts z_exp_0 (expansion latent).
    """

    def __init__(self, latent_ch: int = 4, base_ch: int = 64, num_levels: int = 4):
        super().__init__()
        self.latent_ch = latent_ch
        self.num_levels = num_levels
        time_dim = base_ch * 4
        self.time_mlp = TimeMLP(base_ch, time_dim)

        cold_in = latent_ch * 2
        self.cold_stem = nn.Conv2d(cold_in, base_ch, 3, padding=1)
        self.anchor_stem = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        ch_mult = [1, 2, 4, 8]
        self.cold_down = nn.ModuleList()
        self.anchor_down = nn.ModuleList()
        self.cold_pools = nn.ModuleList()
        self.anchor_pools = nn.ModuleList()
        self.lateral_projs = nn.ModuleList()
        self.rgcf_blocks = nn.ModuleList()

        in_c = base_ch
        for i, mult in enumerate(ch_mult):
            out_c = base_ch * mult
            self.cold_down.append(ResBlock(in_c, out_c, time_dim))
            self.anchor_down.append(ResBlock(in_c, out_c, time_dim=None))
            self.lateral_projs.append(nn.Conv2d(out_c, out_c, 1))
            self.rgcf_blocks.append(RGCFBlock(out_c, out_c))
            if i < len(ch_mult) - 1:
                self.cold_pools.append(nn.Conv2d(out_c, out_c, 4, 2, 1))
                self.anchor_pools.append(nn.Conv2d(out_c, out_c, 4, 2, 1))
            in_c = out_c

        self.mid = ResBlock(in_c, in_c, time_dim)
        self.mid_rgcf = RGCFBlock(in_c, in_c)

        self.cold_up = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for i in range(len(ch_mult) - 1):
            out_c = base_ch * ch_mult[-(i + 2)]
            skip_c = out_c
            self.cold_up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                    ResBlock(out_c + skip_c, out_c, time_dim),
                )
            )
            self.up_projs.append(RGCFBlock(out_c, skip_c))
            in_c = out_c

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, latent_ch, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_ldr: torch.Tensor,
        t: torch.Tensor,
        trust: torch.Tensor,
        return_features: bool = False,
    ):
        time_emb = self.time_mlp(t)
        cold = self.cold_stem(torch.cat([z_t, z_ldr], dim=1))
        anchor = self.anchor_stem(z_ldr)

        cold_skips = []
        anchor_skips = []
        cold_feats = []

        for i, (cd, ad) in enumerate(zip(self.cold_down, self.anchor_down)):
            cold = cd(cold, time_emb)
            anchor = ad(anchor)
            fused = self.rgcf_blocks[i](cold, anchor, trust)
            cold_skips.append(fused)
            anchor_skips.append(anchor)
            cold_feats.append(fused)
            if i < len(self.cold_pools):
                cold = self.cold_pools[i](fused)
                anchor = self.anchor_pools[i](anchor)
                cold = cold + self.lateral_projs[i](anchor)

        cold = self.mid(cold, time_emb)
        cold = self.mid_rgcf(cold, anchor_skips[-1], trust)

        for i, up in enumerate(self.cold_up):
            cold = up[0](cold)
            skip = cold_skips[-(i + 2)]
            anc = anchor_skips[-(i + 2)]
            cold = self.up_projs[i](cold, anc, trust)
            cold = up[1](torch.cat([cold, skip], dim=1), time_emb)

        z_exp_pred = self.head(cold)
        if return_features:
            return z_exp_pred, cold_feats
        return z_exp_pred
