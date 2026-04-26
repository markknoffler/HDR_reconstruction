import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.cold_hdr_luminance_diffusion_decoder import SinusoidalPosEmb, SimpleResBlock
from ..encoders.material_encoder import material_encoder
from ..encoders.structural_encoder_main import structural_encoder
from ..encoders.semantic_codebook_encoder_main import semantic_codebook_encoder


class HorizontalTriStreamFusion(nn.Module):
    def __init__(self, target_ch, in_mat, in_struct, in_sem, t_dim):
        super().__init__()
        self.mat_proj = nn.Conv2d(in_mat, target_ch, 1)
        self.struct_proj = nn.Conv2d(in_struct, target_ch, 1)
        self.sem_proj = nn.Conv2d(in_sem, target_ch, 1)
        self.q_mat = nn.Conv2d(target_ch, target_ch, 1)
        self.k_struct = nn.Conv2d(target_ch, target_ch, 1)
        self.v_struct = nn.Conv2d(target_ch, target_ch, 1)
        self.k_sem = nn.Conv2d(target_ch, target_ch, 1)
        self.v_sem = nn.Conv2d(target_ch, target_ch, 1)
        self.time_fc = nn.Linear(t_dim, target_ch * 3)
        self.mix = nn.Conv2d(target_ch * 3, target_ch, 1)
        self.out = nn.Conv2d(target_ch, target_ch, 1)

    def forward(self, x, t_emb, mat, struct, sem):
        b, c, h, w = x.shape
        mat = F.interpolate(self.mat_proj(mat), size=(h, w), mode="bilinear", align_corners=False)
        struct = F.interpolate(self.struct_proj(struct), size=(h, w), mode="bilinear", align_corners=False)
        sem = F.interpolate(self.sem_proj(sem), size=(h, w), mode="bilinear", align_corners=False)

        q = self.q_mat(mat).view(b, c, -1).transpose(1, 2)
        ks = self.k_struct(struct).view(b, c, -1)
        vs = self.v_struct(struct).view(b, c, -1).transpose(1, 2)
        km = self.k_sem(sem).view(b, c, -1)
        vm = self.v_sem(sem).view(b, c, -1).transpose(1, 2)

        attn_s = torch.softmax(torch.bmm(q, ks) / (c ** 0.5), dim=-1)
        attn_m = torch.softmax(torch.bmm(q, km) / (c ** 0.5), dim=-1)
        mix_struct = torch.bmm(attn_s, vs).transpose(1, 2).reshape(b, c, h, w)
        mix_sem = torch.bmm(attn_m, vm).transpose(1, 2).reshape(b, c, h, w)

        gates = torch.sigmoid(self.time_fc(t_emb)).view(b, 3, c, 1, 1)
        fused = torch.cat([gates[:, 0] * mat, gates[:, 1] * mix_struct, gates[:, 2] * mix_sem], dim=1)
        return x + self.out(self.mix(fused))


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
        self.g1 = HorizontalTriStreamFusion(base_ch * 2, mat_ch, struct_ch, sem_ch[0], t_ch)
        self.g2 = HorizontalTriStreamFusion(base_ch * 4, mat_ch, struct_ch, sem_ch[1], t_ch)
        self.g3 = HorizontalTriStreamFusion(base_ch * 8, mat_ch, struct_ch, sem_ch[2], t_ch)
        self.gm = HorizontalTriStreamFusion(base_ch * 8, mat_ch, struct_ch, sem_ch[3], t_ch)
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


class Stage1TriEncoderDiffusionSystem(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.material_encoder = material_encoder()
        self.structural_encoder = structural_encoder(embed_dim=256, use_depth=False)
        self.semantic_encoder = semantic_codebook_encoder(in_channels=3, seg_channels=3, base_channels=base_ch)
        self.decoder = GroundedHDRUNet(base_ch=base_ch)

    def forward(self, ldr, t, segmap=None):
        if segmap is None:
            segmap = ldr
        mat_feat = self.material_encoder(ldr)
        struct_feat, gate = self.structural_encoder(ldr)
        sem_feats, sem_latents, mus, logvars = self.semantic_encoder(ldr, segmap=segmap)
        pred = self.decoder(ldr, t, mat_feat, struct_feat, sem_feats)
        class_probs = torch.softmax(sem_latents[0], dim=1)
        return pred, gate, class_probs, {"mus": mus, "logvars": logvars}

