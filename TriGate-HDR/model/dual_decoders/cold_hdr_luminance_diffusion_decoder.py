import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.cold_hdr_luminance_diffusion_decoder import SinusoidalPosEmb, SimpleResBlock
from ..encoders.material_encoder import material_encoder
from ..encoders.structural_encoder_main import structural_encoder
from ..encoders.semantic_codebook_encoder_main import semantic_codebook_encoder


class RuntimeMaskPredictor(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, out_ch, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class SegMaskEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        self.s2_down = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)
        self.s2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1))
        self.s3_down = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)
        self.s3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1))
        self.s4_down = nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)
        self.s4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1))

    def forward(self, mask):
        m1 = self.s1(mask)
        m2 = self.s2(self.s2_down(m1))
        m3 = self.s3(self.s3_down(m2))
        m4 = self.s4(self.s4_down(m3))
        return [m1, m2, m3, m4]


class HorizontalTriStreamFusion(nn.Module):
    def __init__(self, target_ch, in_mat, in_struct, in_sem, in_mask, t_dim, attn_max_tokens=1024):
        super().__init__()
        self.mat_proj = nn.Conv2d(in_mat, target_ch, 1)
        self.struct_proj = nn.Conv2d(in_struct, target_ch, 1)
        self.sem_proj = nn.Conv2d(in_sem, target_ch, 1)
        self.mask_proj = nn.Conv2d(in_mask, target_ch, 1)
        self.q_mat = nn.Conv2d(target_ch, target_ch, 1)
        self.k_struct = nn.Conv2d(target_ch, target_ch, 1)
        self.v_struct = nn.Conv2d(target_ch, target_ch, 1)
        self.k_sem = nn.Conv2d(target_ch, target_ch, 1)
        self.v_sem = nn.Conv2d(target_ch, target_ch, 1)
        self.time_fc = nn.Linear(t_dim, target_ch * 4)
        self.mix = nn.Conv2d(target_ch * 4, target_ch, 1)
        self.out = nn.Conv2d(target_ch, target_ch, 1)
        self.attn_max_tokens = attn_max_tokens

    def forward(self, x, t_emb, mat, struct, sem, mask):
        b, c, h, w = x.shape
        mat = F.interpolate(self.mat_proj(mat), size=(h, w), mode="bilinear", align_corners=False)
        struct = F.interpolate(self.struct_proj(struct), size=(h, w), mode="bilinear", align_corners=False)
        sem = F.interpolate(self.sem_proj(sem), size=(h, w), mode="bilinear", align_corners=False)
        mask = F.interpolate(self.mask_proj(mask), size=(h, w), mode="bilinear", align_corners=False)

        # Full token-token attention is O((HW)^2) and can OOM on 12GB GPUs.
        # Compute attention on an adaptive low-res grid, then upsample.
        token_limit = max(1, int(self.attn_max_tokens))
        pool_h = h
        pool_w = w
        if h * w > token_limit:
            scale = (token_limit / float(h * w)) ** 0.5
            pool_h = max(1, int(round(h * scale)))
            pool_w = max(1, int(round(w * scale)))

        mat_a = F.interpolate(mat, size=(pool_h, pool_w), mode="bilinear", align_corners=False)
        struct_a = F.interpolate(struct, size=(pool_h, pool_w), mode="bilinear", align_corners=False)
        sem_a = F.interpolate(sem, size=(pool_h, pool_w), mode="bilinear", align_corners=False)

        q = self.q_mat(mat_a).view(b, c, -1).transpose(1, 2)
        ks = self.k_struct(struct_a).view(b, c, -1)
        vs = self.v_struct(struct_a).view(b, c, -1).transpose(1, 2)
        km = self.k_sem(sem_a).view(b, c, -1)
        vm = self.v_sem(sem_a).view(b, c, -1).transpose(1, 2)

        attn_s = torch.softmax(torch.bmm(q, ks) / (c ** 0.5), dim=-1)
        attn_m = torch.softmax(torch.bmm(q, km) / (c ** 0.5), dim=-1)
        mix_struct = torch.bmm(attn_s, vs).transpose(1, 2).reshape(b, c, pool_h, pool_w)
        mix_sem = torch.bmm(attn_m, vm).transpose(1, 2).reshape(b, c, pool_h, pool_w)
        if pool_h != h or pool_w != w:
            mix_struct = F.interpolate(mix_struct, size=(h, w), mode="bilinear", align_corners=False)
            mix_sem = F.interpolate(mix_sem, size=(h, w), mode="bilinear", align_corners=False)

        gates = torch.sigmoid(self.time_fc(t_emb)).view(b, 4, c, 1, 1)
        fused = torch.cat([gates[:, 0] * mat, gates[:, 1] * mix_struct, gates[:, 2] * mix_sem, gates[:, 3] * mask], dim=1)
        return x + self.out(self.mix(fused))


class GroundedHDRUNet(nn.Module):
    def __init__(self, base_ch=64, mat_ch=523, struct_ch=256, sem_ch=(64, 128, 256, 512), mask_ch=(64, 128, 256, 512)):
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
        self.g1 = HorizontalTriStreamFusion(base_ch * 2, mat_ch, struct_ch, sem_ch[0], mask_ch[0], t_ch)
        self.g2 = HorizontalTriStreamFusion(base_ch * 4, mat_ch, struct_ch, sem_ch[1], mask_ch[1], t_ch)
        self.g3 = HorizontalTriStreamFusion(base_ch * 8, mat_ch, struct_ch, sem_ch[2], mask_ch[2], t_ch)
        self.gm = HorizontalTriStreamFusion(base_ch * 8, mat_ch, struct_ch, sem_ch[3], mask_ch[3], t_ch)
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

    def forward(self, x, t, mat, struct, sem_feats, mask_feats):
        t_emb = self.time_mlp(t)
        x = self.input_layer(x)
        s1 = self.g1(self.down1(x, t_emb), t_emb, mat, struct, sem_feats[0], mask_feats[0])
        x = self.pool1(s1)
        s2 = self.g2(self.down2(x, t_emb), t_emb, mat, struct, sem_feats[1], mask_feats[1])
        x = self.pool2(s2)
        s3 = self.g3(self.down3(x, t_emb), t_emb, mat, struct, sem_feats[2], mask_feats[2])
        x = self.pool3(s3)
        x = self.gm(self.mid(x, t_emb), t_emb, mat, struct, sem_feats[3], mask_feats[3])
        x = self.ub1(torch.cat([self.up1(x), s3], dim=1), t_emb)
        x = self.ub2(torch.cat([self.up2(x), s2], dim=1), t_emb)
        x = self.ub3(torch.cat([self.up3(x), s1], dim=1), t_emb)
        return self.final(x)


class Stage1TriEncoderDiffusionSystem(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.runtime_mask_predictor = RuntimeMaskPredictor(in_ch=3, out_ch=3, base=32)
        self.mask_encoder = SegMaskEncoder(in_ch=3, base_ch=base_ch)
        self.material_encoder = material_encoder()
        self.structural_encoder = structural_encoder(embed_dim=256, use_depth=False)
        self.semantic_encoder = semantic_codebook_encoder(in_channels=3, seg_channels=3, base_channels=base_ch)
        self.decoder = GroundedHDRUNet(base_ch=base_ch)

    def forward(self, ldr, t, segmap=None):
        segmap_pred = self.runtime_mask_predictor(ldr)
        if segmap is None:
            segmap = segmap_pred
        else:
            segmap = 0.5 * segmap + 0.5 * segmap_pred
        mask_feats = self.mask_encoder(segmap)
        mat_feat = self.material_encoder(ldr)
        struct_feat, gate = self.structural_encoder(ldr)
        sem_feats, sem_latents, mus, logvars = self.semantic_encoder(ldr, segmap=segmap)
        pred = self.decoder(ldr, t, mat_feat, struct_feat, sem_feats, mask_feats)
        class_probs = torch.softmax(sem_latents[0], dim=1)
        return pred, gate, class_probs, {"mus": mus, "logvars": logvars, "segmap_used": segmap, "segmap_pred": segmap_pred}

