"""TriGate Stage-1 encoder streams (material, structural, semantic, mask)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import RuntimeMaskPredictor, SegMaskEncoder
from ..encoders.material_encoder import material_encoder
from ..encoders.structural_encoder_main import structural_encoder
from ..encoders.semantic_codebook_encoder_main import semantic_codebook_encoder


class TriEncoderBundle(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.runtime_mask_predictor = RuntimeMaskPredictor(in_ch=3, out_ch=3, base=32)
        self.mask_encoder = SegMaskEncoder(in_ch=3, base_ch=base_ch)
        self.material_encoder = material_encoder()
        self.structural_encoder = structural_encoder(embed_dim=256, use_depth=False)
        self.semantic_encoder = semantic_codebook_encoder(
            in_channels=3, seg_channels=3, base_channels=base_ch
        )

    def forward(self, ldr: torch.Tensor, segmap=None):
        segmap_pred = self.runtime_mask_predictor(ldr)
        if segmap is None:
            segmap_used = segmap_pred
        else:
            segmap_used = 0.5 * segmap + 0.5 * segmap_pred

        mask_feats = self.mask_encoder(segmap_used)
        mat_feat = self.material_encoder(ldr)
        struct_feat, gate = self.structural_encoder(ldr)
        sem_feats, sem_latents, mus, logvars = self.semantic_encoder(ldr, segmap=segmap_used)
        class_probs = torch.softmax(sem_latents[0], dim=1)

        aux = {
            "mus": mus,
            "logvars": logvars,
            "segmap_used": segmap_used,
            "segmap_pred": segmap_pred,
        }
        return mat_feat, struct_feat, gate, sem_feats, mask_feats, class_probs, aux
