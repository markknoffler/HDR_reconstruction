import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.relu(h))
        return h + self.shortcut(x)


class HDR_UNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_ch = base_ch * 4
        self.input_layer = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down1 = SimpleResBlock(base_ch, base_ch * 2, t_ch)
        self.pool1 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.down2 = SimpleResBlock(base_ch * 2, base_ch * 4, t_ch)
        self.pool2 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.down3 = SimpleResBlock(base_ch * 4, base_ch * 8, t_ch)
        self.pool3 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)
        self.mid_block = SimpleResBlock(base_ch * 8, base_ch * 8, t_ch)
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1)
        self.up_block1 = SimpleResBlock(base_ch * 12, base_ch * 4, t_ch)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.up_block2 = SimpleResBlock(base_ch * 6, base_ch * 2, t_ch)
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.up_block3 = SimpleResBlock(base_ch * 3, base_ch, t_ch)
        self.final_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1),
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.input_layer(x)
        s1 = self.down1(x, time_emb)
        x = self.pool1(s1)
        s2 = self.down2(x, time_emb)
        x = self.pool2(s2)
        s3 = self.down3(x, time_emb)
        x = self.pool3(s3)
        x = self.mid_block(x, time_emb)
        x = self.up_block1(torch.cat([self.up1(x), s3], dim=1), time_emb)
        x = self.up_block2(torch.cat([self.up2(x), s2], dim=1), time_emb)
        x = self.up_block3(torch.cat([self.up3(x), s1], dim=1), time_emb)
        return self.final_head(x)


class ColdHDRLuminanceDiffusion(nn.Module):
    def __init__(self, model, timesteps=100, beta_start=1e-4, beta_end=2e-2, white_threshold=0.98, luminance_eps=1e-8):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.white_threshold = white_threshold
        self.luminance_eps = luminance_eps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_ab", torch.sqrt(ab))
        self.register_buffer("sqrt_1m_ab", torch.sqrt(1.0 - ab))

    def luminance(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def scale_rgb_by_luminance(self, rgb_ref, lum_target):
        lum_ref = self.luminance(rgb_ref)
        return rgb_ref * (lum_target / (lum_ref + self.luminance_eps))

    def forward_diffusion(self, x_hdr, t):
        y0 = self.luminance(x_hdr)
        eps = torch.randn_like(y0)
        y_t = self.sqrt_ab[t].view(-1, 1, 1, 1) * y0 + self.sqrt_1m_ab[t].view(-1, 1, 1, 1) * eps
        return self.scale_rgb_by_luminance(x_hdr, y_t)

