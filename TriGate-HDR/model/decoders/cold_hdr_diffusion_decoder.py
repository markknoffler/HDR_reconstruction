"""
ColdEfficient-LORCD: Latent Orthogonal Radiance Cold Diffusion for Stage 2.

Expansion-only cold corruption in latent space with dual-stream trust-gated UNet.
Trained from scratch (mini VAE + latent denoiser, no pretrained foundation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cold_efficient_blocks import ColdEfficientLatentUNet
from .mini_hdr_vae import MiniHDRVAE


class ColdHDRDiffusion(nn.Module):
    """
    Cold diffusion with LDR latent as anchor; cold corruption on expansion latent only.

    z_exp_t = (1 - alpha_t) * z_exp_0
    z_t = z_lift + z_exp_t
    """

    def __init__(
        self,
        model=None,
        vae=None,
        timesteps: int = 100,
        base_ch: int = 64,
        latent_ch: int = 4,
        vae_kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.vae = vae if vae is not None else MiniHDRVAE(latent_ch=latent_ch, kl_weight=vae_kl_weight)
        self.model = model if model is not None else ColdEfficientLatentUNet(
            latent_ch=latent_ch, base_ch=base_ch
        )
        self.timesteps = int(timesteps)
        self.inference_timesteps = int(timesteps)
        self.latent_ch = latent_ch
        alphas = torch.linspace(0.0, 1.0, self.timesteps)
        self.register_buffer("alphas", alphas)

    @staticmethod
    def ldr_to_hdr_space(ldr: torch.Tensor) -> torch.Tensor:
        if ldr.min() >= -1.05 and ldr.max() <= 1.05:
            return ldr
        return ldr * 2.0 - 1.0

    def cold_forward_exp(self, z_exp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alphas[t].view(-1, 1, 1, 1)
        return (1.0 - a) * z_exp

    def cold_forward_exp_at_level(self, z_exp: torch.Tensor, t: torch.Tensor, level: int) -> torch.Tensor:
        """Scale-adaptive cold schedule: coarser levels corrupt faster."""
        num_levels = 4
        a = self.alphas[t].view(-1, 1, 1, 1)
        scale = min(1.0, float(2 ** (level - (num_levels - 1))))
        a_level = torch.clamp(a * scale, 0.0, 1.0)
        return (1.0 - a_level) * z_exp

    def _encode_pair(self, hdr: torch.Tensor, ldr_hdr: torch.Tensor):
        z_hdr, mu_h, lv_h = self.vae.encode(hdr, sample=True)
        z_ldr, mu_l, lv_l = self.vae.encode(ldr_hdr, sample=True)
        return z_hdr, z_ldr, mu_h, lv_h, mu_l, lv_l

    def _decompose(self, z_hdr: torch.Tensor, z_ldr: torch.Tensor):
        z_lift = self.vae.mln(z_ldr)
        z_exp = z_hdr - z_lift
        return z_lift, z_exp

    def forward(
        self,
        hdr: torch.Tensor,
        ldr: torch.Tensor,
        gate: torch.Tensor | None = None,
        vae_only: bool = False,
    ):
        """
        Training forward. gate: (B,1,H,W) trust mask (1 = well-exposed, lock expansion).
        vae_only: if True, only compute VAE reconstruction loss (warmup phase).
        """
        ldr_hdr = self.ldr_to_hdr_space(ldr)
        b = hdr.shape[0]
        device = hdr.device

        z_hdr, mu_h, lv_h = self.vae.encode(hdr, sample=True)
        z_ldr, mu_l, lv_l = self.vae.encode(ldr_hdr, sample=True)
        recon_hdr = self.vae.decode(z_hdr)
        recon_ldr = self.vae.decode(z_ldr)
        vae_loss_h, vae_parts_h = self.vae.vae_loss(hdr, recon_hdr, mu_h, lv_h)
        vae_loss_l, _ = self.vae.vae_loss(ldr_hdr, recon_ldr, mu_l, lv_l)
        vae_loss = vae_loss_h + 0.5 * vae_loss_l

        if vae_only:
            return recon_hdr, {
                "loss": vae_loss,
                "vae_loss": vae_loss,
                "recon_loss": vae_parts_h["recon_loss"],
                "kl_loss": vae_parts_h["kl_loss"],
                "hdr_loss": torch.tensor(0.0, device=device),
                "cold_loss": torch.tensor(0.0, device=device),
                "exp_loss": torch.tensor(0.0, device=device),
                "trust_loss": torch.tensor(0.0, device=device),
                "ms_cold_loss": torch.tensor(0.0, device=device),
                "mono_loss": torch.tensor(0.0, device=device),
                "t": torch.tensor(0.0, device=device),
            }

        z_lift, z_exp_0 = self._decompose(z_hdr, z_ldr)

        if gate is None:
            trust = torch.ones(b, 1, hdr.shape[2], hdr.shape[3], device=device)
        else:
            trust = gate.float()
            if trust.dim() == 3:
                trust = trust.unsqueeze(1)

        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        z_exp_t = self.cold_forward_exp(z_exp_0, t)
        z_t = z_lift + z_exp_t

        z_exp_pred, cold_feats = self.model(z_t, z_ldr, t, trust, return_features=True)

        z_hdr_pred = z_lift + z_exp_pred
        hdr_pred = self.vae.decode(z_hdr_pred)

        hdr_loss = F.l1_loss(hdr, hdr_pred)
        exp_loss = F.l1_loss(z_exp_0, z_exp_pred)
        z_exp_t_pred = self.cold_forward_exp(z_exp_pred, t)
        cold_loss = F.l1_loss(z_exp_t, z_exp_t_pred)

        trust_ds = F.interpolate(trust, size=z_exp_pred.shape[-2:], mode="bilinear", align_corners=False)
        trust_loss = (trust_ds * z_exp_pred.abs()).mean()

        ms_cold_loss = torch.tensor(0.0, device=device)
        for level, feat in enumerate(cold_feats[:3]):
            target = F.interpolate(z_exp_t, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            pred_level = self.cold_forward_exp_at_level(z_exp_pred, t, level + 1)
            pred_level = F.interpolate(pred_level, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            ms_cold_loss = ms_cold_loss + F.l1_loss(pred_level, target)

        mono_loss = self.vae.mln.monotonicity_penalty(z_ldr)

        loss = hdr_loss + cold_loss + exp_loss + trust_loss + 0.25 * ms_cold_loss + 0.01 * mono_loss + 0.1 * vae_loss

        return hdr_pred, {
            "loss": loss,
            "hdr_loss": hdr_loss,
            "cold_loss": cold_loss,
            "exp_loss": exp_loss,
            "trust_loss": trust_loss,
            "ms_cold_loss": ms_cold_loss,
            "mono_loss": mono_loss,
            "vae_loss": vae_loss,
            "recon_loss": vae_parts_h["recon_loss"],
            "kl_loss": vae_parts_h["kl_loss"],
            "t": t.float().mean(),
        }

    @torch.no_grad()
    def restore_hdr(self, ldr: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse cold chain on expansion latent; decode to HDR."""
        self.eval()
        ldr_hdr = self.ldr_to_hdr_space(ldr)
        b = ldr_hdr.shape[0]
        device = ldr_hdr.device

        z_ldr, _, _ = self.vae.encode(ldr_hdr, sample=False)
        z_lift = self.vae.mln(z_ldr)

        if gate is None:
            trust = torch.ones(b, 1, ldr_hdr.shape[2], ldr_hdr.shape[3], device=device)
        else:
            trust = gate.float()
            if trust.dim() == 3:
                trust = trust.unsqueeze(1)

        z_exp = torch.zeros_like(z_ldr)
        n_steps = max(1, int(self.inference_timesteps))
        step_ids = torch.linspace(self.timesteps - 1, 0, n_steps, device=device).long().tolist()
        for idx, t_val in enumerate(step_ids):
            t_batch = torch.full((b,), int(t_val), device=device, dtype=torch.long)
            z_t = z_lift + z_exp
            z_exp_hat_0 = self.model(z_t, z_ldr, t_batch, trust)
            cold_at_t = self.cold_forward_exp(z_exp_hat_0, t_batch)
            if idx < len(step_ids) - 1:
                t_prev_val = int(step_ids[idx + 1])
                t_prev = torch.full((b,), t_prev_val, device=device, dtype=torch.long)
                cold_at_prev = self.cold_forward_exp(z_exp_hat_0, t_prev)
                z_exp = z_exp - cold_at_t + cold_at_prev
            else:
                z_exp = z_exp_hat_0

        z_out = z_lift + z_exp
        return self.vae.decode(z_out).clamp(-1.0, 1.0)


# Backward-compatible aliases (legacy imports / docs)
ColdDiffusionUNet = ColdEfficientLatentUNet
HDR_UNet = ColdEfficientLatentUNet
