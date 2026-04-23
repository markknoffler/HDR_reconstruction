import torch
import torch.nn as nn
import torch.nn.functional as F


class SeamGatedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.mask_gate = nn.Conv2d(1, ch, 1)
        self.attn_q = nn.Conv2d(ch, ch, 1)
        self.attn_k = nn.Conv2d(ch, ch, 1)
        self.attn_v = nn.Conv2d(ch, ch, 1)

    def forward(self, x, seam_mask):
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        gate = torch.sigmoid(self.mask_gate(seam_mask))
        q = self.attn_q(h)
        k = self.attn_k(h)
        v = self.attn_v(h)
        b, c, hsz, wsz = q.shape
        q = q.view(b, c, -1).transpose(1, 2)
        k = k.view(b, c, -1)
        a = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
        v = v.view(b, c, -1).transpose(1, 2)
        attn = torch.bmm(a, v).transpose(1, 2).view(b, c, hsz, wsz)
        h = self.conv2(h + attn)
        return x + gate * h


class SeamingGenerator(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.stem = nn.Conv2d(7, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList([SeamGatedBlock(base_ch) for _ in range(6)])
        self.tail = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(base_ch, 3, 1))

    def forward(self, base_hdr_x, generated_clip_hdr, seam_mask):
        x = torch.cat([base_hdr_x, generated_clip_hdr, seam_mask], dim=1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, seam_mask)
        residual = self.tail(x)
        return base_hdr_x + seam_mask * residual

