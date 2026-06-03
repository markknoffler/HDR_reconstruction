"""
Lightweight HDR VAE for ColdEfficient-LORCD Stage 2 (train from scratch, /8 spatial).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniHDREncoder(nn.Module):
    def __init__(self, in_ch: int = 3, latent_ch: int = 4, base_ch: int = 32, down_levels: int = 3):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(down_levels)]
        layers = []
        cin = in_ch
        for cout in chs:
            layers.extend(
                [
                    nn.Conv2d(cin, cout, 3, padding=1),
                    nn.GroupNorm(min(8, cout), cout),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1),
                    nn.GroupNorm(min(8, cout), cout),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(cout, cout, 4, 2, 1),
                ]
            )
            cin = cout
        self.body = nn.Sequential(*layers)
        self.out_ch = chs[-1]
        self.mu = nn.Conv2d(self.out_ch, latent_ch, 1)
        self.logvar = nn.Conv2d(self.out_ch, latent_ch, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class MiniHDRDecoder(nn.Module):
    def __init__(self, out_ch: int = 3, latent_ch: int = 4, base_ch: int = 32, up_levels: int = 3):
        super().__init__()
        deepest = base_ch * (2 ** (up_levels - 1))
        self.in_proj = nn.Conv2d(latent_ch, deepest, 3, padding=1)
        blocks = []
        ch = deepest
        for _ in range(up_levels):
            next_ch = max(base_ch, ch // 2)
            blocks.extend(
                [
                    nn.ConvTranspose2d(ch, next_ch, 4, 2, 1),
                    nn.GroupNorm(min(8, next_ch), next_ch),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(next_ch, next_ch, 3, padding=1),
                    nn.GroupNorm(min(8, next_ch), next_ch),
                    nn.SiLU(inplace=True),
                ]
            )
            ch = next_ch
        self.body = nn.Sequential(*blocks)
        self.out = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_ch, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        h = self.body(h)
        return self.out(h)


class MonoLiftLatent(nn.Module):
    """
    Maps z_ldr -> z_lift with soft monotonicity on a luminance proxy channel.
    """

    def __init__(self, latent_ch: int = 4):
        super().__init__()
        mid = max(16, latent_ch * 4)
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, latent_ch, 3, padding=1),
        )
        self.luma_weights = nn.Parameter(torch.ones(latent_ch) / max(latent_ch, 1))

    def forward(self, z_ldr: torch.Tensor) -> torch.Tensor:
        return z_ldr + self.net(z_ldr)

    def monotonicity_penalty(self, z_ldr: torch.Tensor) -> torch.Tensor:
        """Penalize non-monotonic response along sorted latent intensity bins."""
        w = F.softmax(self.luma_weights, dim=0)
        proxy = (z_ldr * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        z_lift = self.forward(z_ldr)
        lift_proxy = (z_lift * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        diff = lift_proxy[:, :, :, 1:] - lift_proxy[:, :, :, :-1]
        proxy_diff = proxy[:, :, :, 1:] - proxy[:, :, :, :-1]
        # lift should increase when proxy increases
        violation = F.relu(-diff * torch.sign(proxy_diff + 1e-6))
        return violation.mean()


class MiniHDRVAE(nn.Module):
    def __init__(self, latent_ch: int = 4, base_ch: int = 32, kl_weight: float = 1e-4):
        super().__init__()
        self.latent_ch = latent_ch
        self.kl_weight = kl_weight
        self.encoder = MiniHDREncoder(latent_ch=latent_ch, base_ch=base_ch)
        self.decoder = MiniHDRDecoder(latent_ch=latent_ch, base_ch=base_ch)
        self.mln = MonoLiftLatent(latent_ch=latent_ch)

    def encode(self, x: torch.Tensor, sample: bool = True):
        mu, logvar = self.encoder(x)
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x, sample=True)
        recon = self.decode(z)
        return recon, z, mu, logvar

    def vae_loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recon_loss = F.l1_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_weight * kl, {"recon_loss": recon_loss, "kl_loss": kl}
