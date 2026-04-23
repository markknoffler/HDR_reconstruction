import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.cold_hdr_luminance_diffusion_decoder import SinusoidalPosEmb, SimpleResBlock

class TriEncoderGate(nn.Module):
    def __init__(self, target_ch, in_mat, in_struct, in_sem, t_dim):
        super().__init__()
        self.mat_proj = nn.Conv2d(in_mat, target_ch, 1)
        self.struct_proj = nn.Conv2d(in_struct, target_ch, 1)
        self.sem_proj = nn.Conv2d(in_sem, target_ch, 1)
        self.time_fc = nn.Linear(t_dim, target_ch * 3)
        self.out = nn.Conv2d(target_ch * 3, target_ch, 1)

    def forward(self, x, t_emb, mat, struct, sem):
        b, c, h, w = x.shape
        mat = F.interpolate(self.mat_proj(mat), size=(h, w), mode="bilinear", align_corners=False)
        struct = F.interpolate(self.struct_proj(struct), size=(h, w), mode="bilinear", align_corners=False)
        sem = F.interpolate(self.sem_proj(sem), size=(h, w), mode="bilinear", align_corners=False)
        gates = torch.sigmoid(self.time_fc(t_emb)).view(b, 3, c, 1, 1)
        fused = torch.cat([gates[:, 0] * mat, gates[:, 1] * struct, gates[:, 2] * sem], dim=1)
        return x + self.out(fused)


class GroundedHDRUNet(nn.Module):
    def __init__(self, base_ch=64, mat_ch=523, struct_ch=256, sem_ch=(64, 128, 256, 512)):
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
        self.mid = SimpleResBlock(base_ch * 8, base_ch * 8, t_ch)
        self.g1 = TriEncoderGate(base_ch * 2, mat_ch, struct_ch, sem_ch[0], t_ch)
        self.g2 = TriEncoderGate(base_ch * 4, mat_ch, struct_ch, sem_ch[1], t_ch)
        self.g3 = TriEncoderGate(base_ch * 8, mat_ch, struct_ch, sem_ch[2], t_ch)
        self.gm = TriEncoderGate(base_ch * 8, mat_ch, struct_ch, sem_ch[3], t_ch)
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1)
        self.ub1 = SimpleResBlock(base_ch * 12, base_ch * 4, t_ch)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.ub2 = SimpleResBlock(base_ch * 6, base_ch * 2, t_ch)
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.ub3 = SimpleResBlock(base_ch * 3, base_ch, t_ch)
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1),
        )

    def forward(self, x, t, mat, struct, sem_feats):
        t_emb = self.time_mlp(t)
        x = self.input_layer(x)
        s1 = self.g1(self.down1(x, t_emb), t_emb, mat, struct, sem_feats[0])
        x = self.pool1(s1)
        s2 = self.g2(self.down2(x, t_emb), t_emb, mat, struct, sem_feats[1])
        x = self.pool2(s2)
        s3 = self.g3(self.down3(x, t_emb), t_emb, mat, struct, sem_feats[2])
        x = self.pool3(s3)
        x = self.gm(self.mid(x, t_emb), t_emb, mat, struct, sem_feats[3])
        x = self.ub1(torch.cat([self.up1(x), s3], dim=1), t_emb)
        x = self.ub2(torch.cat([self.up2(x), s2], dim=1), t_emb)
        x = self.ub3(torch.cat([self.up3(x), s1], dim=1), t_emb)
        return self.final(x)

