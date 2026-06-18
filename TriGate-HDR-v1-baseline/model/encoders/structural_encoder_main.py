import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(x):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    b = mag.shape[0]
    max_val = mag.view(b, -1).max(dim=1).values.view(b, 1, 1, 1).clamp(min=1e-6)
    return mag / max_val


def gamma_invert(x, gamma=0.45):
    return (1.0 - x.clamp(0.0, 1.0).pow(gamma)).pow(1.0 / gamma)


def get_sat_mask(ldr, lo=0.90, hi=0.98):
    max_val = ldr.max(dim=1, keepdim=True).values
    return ((max_val - lo) / (hi - lo)).clamp(0.0, 1.0)


class structural_encoder(nn.Module):
    def __init__(self, embed_dim=256, use_depth=True):
        super(structural_encoder, self).__init__()
        self.use_depth = use_depth
        in_ch = 5 + (1 if use_depth else 0)
        self.block1 = self._make_block(in_ch, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, embed_dim)
        self.gate_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, ldr, depth=None):
        ldr_gray = ldr.mean(dim=1, keepdim=True)
        ldr_inv = gamma_invert(ldr_gray)
        sat_mask = get_sat_mask(ldr)
        edge_orig = sobel_edges(ldr_gray) * (1.0 - sat_mask)
        edge_inv = sobel_edges(ldr_inv)
        features = [edge_orig, edge_inv, sat_mask, ldr_gray, ldr_inv]
        if self.use_depth and depth is not None:
            features.append(sobel_edges(depth))
        x = torch.cat(features, dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        struct_feat = x
        gate = self.sigmoid(self.gate_conv(x)) * (1.0 - sat_mask)
        return struct_feat, gate

