import torch
import torch.nn as nn


def _disc_head(in_ch):

    return nn.Sequential(
        nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
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
