import torch
import torch.nn as nn
import torch.nn.functional as F


class material_encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_grad=True, use_hist=True, use_gram=True):
        super(material_encoder, self).__init__()
        self.use_grad = use_grad
        self.use_hist = use_hist
        self.use_gram = use_gram

        self.block1 = self._make_block(in_channels, base_channels)
        self.block2 = self._make_block(base_channels, base_channels * 2)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4)
        self.block4 = self._make_block(base_channels * 4, base_channels * 8)

        if use_grad:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)

        if use_hist or use_gram:
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        if use_gram:
            self.gram_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        cnn_feat = self.block1(x)
        cnn_feat = self.block2(cnn_feat)
        cnn_feat = self.block3(cnn_feat)
        cnn_feat = self.block4(cnn_feat)

        handcrafted = []
        if self.use_grad:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            grad_x = F.conv2d(gray, self.sobel_x, padding=1)
            grad_y = F.conv2d(gray, self.sobel_y, padding=1)
            handcrafted.extend([grad_x, grad_y])

        if self.use_hist:
            handcrafted.append(self.avg_pool(x))

        if self.use_gram:
            mu = self.avg_pool(x)
            rr = x[:, 0:1] * x[:, 0:1]
            gg = x[:, 1:2] * x[:, 1:2]
            bb = x[:, 2:3] * x[:, 2:3]
            rg = x[:, 0:1] * x[:, 1:2]
            rb = x[:, 0:1] * x[:, 2:3]
            gb = x[:, 1:2] * x[:, 2:3]

            E_rr = self.gram_pool(rr)
            E_gg = self.gram_pool(gg)
            E_bb = self.gram_pool(bb)
            E_rg = self.gram_pool(rg)
            E_rb = self.gram_pool(rb)
            E_gb = self.gram_pool(gb)

            cov_rr = E_rr - mu[:, 0:1] * mu[:, 0:1]
            cov_gg = E_gg - mu[:, 1:2] * mu[:, 1:2]
            cov_bb = E_bb - mu[:, 2:3] * mu[:, 2:3]
            cov_rg = E_rg - mu[:, 0:1] * mu[:, 1:2]
            cov_rb = E_rb - mu[:, 0:1] * mu[:, 2:3]
            cov_gb = E_gb - mu[:, 1:2] * mu[:, 2:3]
            handcrafted.append(torch.cat([cov_rr, cov_gg, cov_bb, cov_rg, cov_rb, cov_gb], dim=1))

        return torch.cat([cnn_feat] + handcrafted, dim=1) if handcrafted else cnn_feat

