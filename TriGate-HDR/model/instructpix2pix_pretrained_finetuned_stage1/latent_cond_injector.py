"""Fuse TriGate encoder streams into InstructPix2Pix image latents (4-ch)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import HorizontalTriStreamFusion
from .tri_encoder_bundle import TriEncoderBundle


class SinusoidalTimestepEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LatentCondInjector(nn.Module):
    """
    Adds TriGate fused features to native InstructPix2Pix image_latents before UNet concat.
    """

    def __init__(self, latent_ch: int = 4, base_ch: int = 64, mat_ch: int = 523, struct_ch: int = 256):
        super().__init__()
        self.encoders = TriEncoderBundle(base_ch=base_ch)
        t_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmb(64),
            nn.Linear(64, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )
        sem_ch = base_ch * 8
        mask_ch = base_ch * 8
        self.fusion = HorizontalTriStreamFusion(
            target_ch=latent_ch,
            in_mat=mat_ch,
            in_struct=struct_ch,
            in_sem=sem_ch,
            in_mask=mask_ch,
            t_dim=t_dim,
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        ldr: torch.Tensor,
        image_latents: torch.Tensor,
        timesteps: torch.Tensor,
        segmap=None,
        return_aux: bool = False,
    ):
        dev = image_latents.device
        timesteps = timesteps.to(device=dev)
        t_emb = self.time_mlp(timesteps)
        mat_feat, struct_feat, gate, sem_feats, mask_feats, class_probs, aux = self.encoders(ldr, segmap)
        delta = self.fusion(image_latents, t_emb, mat_feat, struct_feat, sem_feats[3], mask_feats[3])
        out = image_latents + self.res_scale * delta
        if return_aux:
            return out, gate, class_probs, aux
        return out
