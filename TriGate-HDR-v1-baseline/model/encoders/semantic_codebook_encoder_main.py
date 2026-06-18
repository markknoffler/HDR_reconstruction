import torch
import torch.nn as nn


class semantic_map_encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(semantic_map_encoder, self).__init__()
        self.block1 = self._make_block(in_channels, base_channels)
        self.block2 = self._make_block(base_channels, base_channels * 2)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4)
        self.block4 = self._make_block(base_channels * 4, base_channels * 8)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, seg_map):
        s1 = self.block1(seg_map)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        return s1, s2, s3, s4


class semantic_codebook_encoder(nn.Module):
    def __init__(self, in_channels=3, seg_channels=3, base_channels=64, latent_dim=128):
        super(semantic_codebook_encoder, self).__init__()
        self.block1 = self._make_block(in_channels, base_channels)
        self.block2 = self._make_block(base_channels, base_channels * 2)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4)
        self.block4 = self._make_block(base_channels * 4, base_channels * 8)
        self.semantic_map_encoder = semantic_map_encoder(seg_channels, base_channels)
        self.mu_heads = nn.ModuleList(
            [nn.Conv2d(base_channels * (2 ** i), latent_dim, kernel_size=1) for i in range(4)]
        )
        self.logvar_heads = nn.ModuleList(
            [nn.Conv2d(base_channels * (2 ** i), latent_dim, kernel_size=1) for i in range(4)]
        )

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, ldr_input, segmap=None):
        if segmap is None:
            segmap = ldr_input

        c1 = self.block1(ldr_input)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        s1, s2, s3, s4 = self.semantic_map_encoder(segmap)

        f1 = c1 + s1
        f2 = c2 + s2
        f3 = c3 + s3
        f4 = c4 + s4
        feats = [f1, f2, f3, f4]

        mus = [head(feat) for head, feat in zip(self.mu_heads, feats)]
        logvars = [head(feat).clamp(-8.0, 8.0) for head, feat in zip(self.logvar_heads, feats)]
        latents = [self._reparam(mu, logvar) for mu, logvar in zip(mus, logvars)]
        return feats, latents, mus, logvars

