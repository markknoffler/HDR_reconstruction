import torch
import torch.nn as nn


def _disc_head(in_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 1, 3, 1, 1),
    )


class SeamingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_head = _disc_head(3)
        self.seam_head = _disc_head(4)

    def forward(self, image, seam_mask):
        global_score = self.global_head(image)
        seam_score = self.seam_head(torch.cat([image, seam_mask], dim=1))
        return global_score, seam_score

