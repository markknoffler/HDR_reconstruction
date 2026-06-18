import torch
import torch.nn as nn

from .generator import SeamingGenerator
from .discriminator import SeamingDiscriminator


class SeamingGANSystem(nn.Module):
    def __init__(self, use_rso_stem: bool = False):
        super().__init__()
        self.generator = SeamingGenerator(use_rso_stem=use_rso_stem)
        self.discriminator = SeamingDiscriminator()

    def forward(self, base_hdr_x, generated_clip_hdr, seam_mask):
        fake = self.generator(base_hdr_x, generated_clip_hdr, seam_mask)
        fake_g, fake_s = self.discriminator(fake, seam_mask)
        return fake, fake_g, fake_s

    @staticmethod
    def d_hinge_loss(real_logits, fake_logits):
        return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()

    @staticmethod
    def g_hinge_loss(fake_logits):
        return -fake_logits.mean()

