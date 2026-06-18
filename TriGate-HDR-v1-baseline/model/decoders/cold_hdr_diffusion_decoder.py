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
from .pixel_hdr_refiner import PixelHDRRefiner, sobel_gradient_loss


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
        vae_base_ch: int = 32,
        vae_kl_weight: float = 1e-4,
        use_pixel_refiner: bool = False,
        refiner_base_ch: int = 48,
        refiner_blocks: int = 6,
    ):
        super().__init__()
        self.vae = vae if vae is not None else MiniHDRVAE(
            latent_ch=latent_ch, base_ch=vae_base_ch, kl_weight=vae_kl_weight
        )
        self.model = model if model is not None else ColdEfficientLatentUNet(
            latent_ch=latent_ch, base_ch=base_ch
        )
        self.timesteps = int(timesteps)
        self.inference_timesteps = int(timesteps)
        self.latent_ch = latent_ch
        self.vae_frozen = False
        self.use_pixel_refiner = bool(use_pixel_refiner)
        self.pixel_refiner = (
            PixelHDRRefiner(base_ch=refiner_base_ch, num_blocks=refiner_blocks)
            if self.use_pixel_refiner
            else None
        )
        alphas = torch.linspace(0.0, 1.0, self.timesteps)
        self.register_buffer("alphas", alphas)

    def set_vae_trainable(self, trainable: bool) -> None:
        """Freeze VAE/MLN after warmup so cold UNet cannot hide behind a drifting encoder."""
        self.vae_frozen = not trainable
        for p in self.vae.parameters():
            p.requires_grad = trainable

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Bias toward high t (z_t ~ z_lift) — matches inference start state."""
        u = torch.rand(batch_size, device=device)
        t = (u.pow(0.5) * float(self.timesteps - 1)).long()
        return t.clamp(0, self.timesteps - 1)

    def _timestep_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Higher weight at high corruption (large alpha) for inference-critical steps."""
        a = self.alphas[t].view(-1, 1, 1, 1)
        return 0.25 + 0.75 * a

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

    def _apply_pixel_refiner(self, ldr_hdr: torch.Tensor, hdr_coarse: torch.Tensor) -> torch.Tensor:
        if self.pixel_refiner is None:
            return hdr_coarse
        return self.pixel_refiner(ldr_hdr, hdr_coarse)

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

        encode_sample = not self.vae_frozen
        z_hdr, mu_h, lv_h = self.vae.encode(hdr, sample=encode_sample)
        z_ldr, mu_l, lv_l = self.vae.encode(ldr_hdr, sample=encode_sample)
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
                "anchor_exp": torch.tensor(0.0, device=device),
                "anchor_hdr": torch.tensor(0.0, device=device),
                "hf_loss": torch.tensor(0.0, device=device),
                "t": torch.tensor(0.0, device=device),
            }

        z_lift, z_exp_0 = self._decompose(z_hdr, z_ldr)

        if gate is None:
            trust = torch.ones(b, 1, hdr.shape[2], hdr.shape[3], device=device)
        else:
            trust = gate.float()
            if trust.dim() == 3:
                trust = trust.unsqueeze(1)

        t = self.sample_timesteps(b, device)
        z_exp_t = self.cold_forward_exp(z_exp_0, t)
        z_t = z_lift + z_exp_t

        z_exp_pred, cold_feats = self.model(z_t, z_ldr, t, trust, return_features=True)

        z_hdr_pred = z_lift + z_exp_pred
        hdr_pred = torch.nan_to_num(self.vae.decode(z_hdr_pred), nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0, 1.0
        )
        hdr_out = self._apply_pixel_refiner(ldr_hdr, hdr_pred)

        t_w = self._timestep_weight(t)
        hdr_loss = (F.l1_loss(hdr, hdr_out, reduction="none").mean(dim=(1, 2, 3)) * t_w.squeeze()).mean()
        hf_loss = sobel_gradient_loss(hdr_out, hdr)
        exp_loss = (
            F.l1_loss(z_exp_0, z_exp_pred, reduction="none").mean(dim=(1, 2, 3)) * t_w.squeeze()
        ).mean()
        z_exp_t_pred = self.cold_forward_exp(z_exp_pred, t)
        cold_loss = (
            F.l1_loss(z_exp_t, z_exp_t_pred, reduction="none").mean(dim=(1, 2, 3)) * t_w.squeeze()
        ).mean()

        trust_ds = F.interpolate(trust, size=z_exp_pred.shape[-2:], mode="bilinear", align_corners=False)
        trust_loss = (trust_ds * z_exp_pred.abs()).mean()

        ms_cold_loss = torch.tensor(0.0, device=device)
        for level, feat in enumerate(cold_feats[:3]):
            target = F.interpolate(z_exp_t, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            pred_level = self.cold_forward_exp_at_level(z_exp_pred, t, level + 1)
            pred_level = F.interpolate(pred_level, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            ms_cold_loss = ms_cold_loss + F.l1_loss(pred_level, target)

        mono_loss = self.vae.mln.monotonicity_penalty(z_ldr)

        # Matches restore_hdr start: z_exp=0, t=T-1, z_t=z_lift.
        t_max = self.timesteps - 1
        anchor_mask = (t == t_max).float()
        anchor_denom = anchor_mask.sum().clamp_min(1.0)
        anchor_exp = (
            F.l1_loss(z_exp_0, z_exp_pred, reduction="none").mean(dim=(1, 2, 3)) * anchor_mask
        ).sum() / anchor_denom
        anchor_hdr = (
            F.l1_loss(hdr, hdr_out, reduction="none").mean(dim=(1, 2, 3)) * anchor_mask
        ).sum() / anchor_denom

        vae_term = vae_loss if not self.vae_frozen else torch.tensor(0.0, device=device)
        loss = hdr_loss + cold_loss + exp_loss + trust_loss + 0.25 * ms_cold_loss + 0.01 * mono_loss + 0.1 * vae_term

        return hdr_out, {
            "loss": loss,
            "hdr_loss": hdr_loss,
            "cold_loss": cold_loss,
            "exp_loss": exp_loss,
            "trust_loss": trust_loss,
            "ms_cold_loss": ms_cold_loss,
            "mono_loss": mono_loss,
            "anchor_exp": anchor_exp,
            "anchor_hdr": anchor_hdr,
            "hf_loss": hf_loss,
            "vae_loss": vae_loss,
            "recon_loss": vae_parts_h["recon_loss"],
            "kl_loss": vae_parts_h["kl_loss"],
            "t": t.float().mean(),
        }

    def _restore_hdr_impl(
        self,
        ldr: torch.Tensor,
        gate: torch.Tensor | None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Reverse cold chain (differentiable when called under grad enabled)."""
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

        n_steps = max(1, int(n_steps if n_steps is not None else self.inference_timesteps))
        if n_steps == 1:
            t_batch = torch.full((b,), self.timesteps - 1, device=device, dtype=torch.long)
            z_exp_hat_0 = self.model(z_lift, z_ldr, t_batch, trust)
            z_out = z_lift + z_exp_hat_0
            hdr_coarse = self.vae.decode(z_out).clamp(-1.0, 1.0)
            return self._apply_pixel_refiner(ldr_hdr, hdr_coarse)

        z_exp = torch.zeros_like(z_ldr)
        step_ids = torch.linspace(self.timesteps - 1, 0, n_steps, device=device).long().tolist()
        for idx, t_val in enumerate(step_ids):
            t_batch = torch.full((b,), int(t_val), device=device, dtype=torch.long)
            z_t = z_lift + z_exp
            z_exp_hat_0 = self.model(z_t, z_ldr, t_batch, trust)
            if idx < len(step_ids) - 1:
                t_prev_val = int(step_ids[idx + 1])
                t_prev = torch.full((b,), t_prev_val, device=device, dtype=torch.long)
                z_exp = self.cold_forward_exp(z_exp_hat_0, t_prev)
            else:
                z_exp = z_exp_hat_0

        z_out = z_lift + z_exp
        hdr_coarse = self.vae.decode(z_out).clamp(-1.0, 1.0)
        return self._apply_pixel_refiner(ldr_hdr, hdr_coarse)

    @torch.no_grad()
    def restore_hdr(self, ldr: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse cold chain on expansion latent; decode to HDR (eval/inference)."""
        was_training = self.training
        self.eval()
        try:
            return self._restore_hdr_impl(ldr, gate)
        finally:
            self.train(was_training)

    def restore_hdr_train(
        self,
        ldr: torch.Tensor,
        gate: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Differentiable restore for training-time alignment with validation."""
        return self._restore_hdr_impl(ldr, gate, n_steps=n_steps)


# Backward-compatible aliases (legacy imports / docs)
ColdDiffusionUNet = ColdEfficientLatentUNet
HDR_UNet = ColdEfficientLatentUNet
