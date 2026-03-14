
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """
    Residual conv block: two (Conv → InstanceNorm → ReLU) layers
    with optional strided downsampling on the first conv.
    InstanceNorm is used (not BatchNorm) to preserve per-image material contrast.
    """
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        # Match dimensions for residual addition
        if in_ch != out_ch or downsample:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class SoftSemanticInjection(nn.Module):
    """
    At a given spatial scale, inject soft semantic class guidance into CNN features.

    Process:
      1. Bilinearly downsample full-res one-hot map to the current feature scale.
         (Bilinear interp on one-hot produces soft boundary probabilities —
          boundary pixels get fractional values like [0.5, 0.5], which is
          physically correct: at that coarse scale, boundary pixels ARE ambiguous.)
      2. Project the num_classes channels → feat_dim via a learned 1×1 conv.
      3. Add the projection to the feature map (additive soft guidance).
         The model learns how strongly to trust this hint via the 1×1 weights.
    """
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(num_classes, feat_dim, 1, bias=False),
            nn.InstanceNorm2d(feat_dim, affine=True),
        )

    def forward(self, feat: torch.Tensor, onehot_full: torch.Tensor) -> torch.Tensor:
        # feat:        (B, feat_dim, Hs, Ws)   — current scale
        # onehot_full: (B, num_classes, H, W)  — full resolution one-hot
        Hs, Ws = feat.shape[2], feat.shape[3]
        onehot_scaled = F.interpolate(
            onehot_full, size=(Hs, Ws), mode='bilinear', align_corners=False
        )                                      # (B, num_classes, Hs, Ws)
        return feat + self.proj(onehot_scaled)  # soft additive injection


class VariationalCodebook(nn.Module):
    """
    Per-class variational prior: each class c has a learned (mu_c, log_sigma_c)
    pair in embed_dim space.

    During TRAINING: sample  z = mu + sigma * eps   (reparameterization trick)
                              → the decoder must be robust to the natural
                                luminance variability of each class.
    During INFERENCE: return mu  (deterministic, no sampling noise).

    KL loss term  KL( N(mu, sigma) || N(0,1) )  is returned and added to total
    loss to regularise the latent space.
    """
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.mu        = nn.Embedding(num_classes, embed_dim)
        self.log_sigma = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.mu.weight,        mean=0.0, std=0.02)
        nn.init.constant_(self.log_sigma.weight, val=0.0)   # sigma = exp(0) = 1

    def forward(self, class_ids: torch.Tensor):
        mu        = self.mu(class_ids)         # (num_classes, embed_dim)
        log_sigma = self.log_sigma(class_ids)  # (num_classes, embed_dim)
        if self.training:
            z = mu + log_sigma.exp() * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, log_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Main Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCodebookEncoder(nn.Module):
    """
    Semantic Codebook Encoder for TriGate-HDR.

    Inputs
    ------
    ldr    : (B, 3, H, W)  LDR image, float in [0,1]
    semseg : (B, H, W)     integer class IDs (long tensor, 0 … num_classes-1)

    Output
    ------
    sem_embed : (B, embed_dim, H, W)
        Per-pixel semantic luminance prior. Each location holds a rich vector
        encoding: what HDR luminance distribution does this object class have,
        given the global lighting of this specific image?
        → Injected into the decoder at every layer via Cross-Attention.

    kl_loss : scalar tensor
        KL divergence from the variational codebook. Add to total training loss
        as  loss_total += lambda_kl * kl_loss.

    Architecture overview
    ---------------------
    LDR image
        │
        ├─[ResConvBlock × N scales, with SoftSemanticInjection at each scale]
        │       ↑ at each scale the one-hot is bilinearly downsampled and
        │         added as soft guidance — model learns to attend to classes
        │
        └─► final feature map  (B, C_final, H/S, W/S)
                │
                ├─[Global Average Pool → MLP]  → global_ctx  (B, embed_dim)
                │   "what is the overall lighting of this scene?"
                │
                └─[Masked Class Pool → Linear]  → class_pool  (B, N, embed_dim)
                    "for each class, what do its pixels look like in this image?"

    VariationalCodebook  →  codebook_embed  (B, N, embed_dim)
        "what does this class statistically look like across all training data?"

    Fuse all three → FusionMLP → class_embeddings  (B, N, embed_dim)

    Spatial projection: multiply class_embeddings by full-res one-hot
        → sem_embed  (B, embed_dim, H, W)
    """

    def __init__(
        self,
        num_classes:    int        = 20,
        embed_dim:      int        = 256,
        feat_channels:  list[int]  = [64, 128, 256, 256],
        global_ctx_dim: int        = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim

        # ── 1. Multi-scale CNN + soft semantic injection at every scale ───────
        self.enc_blocks  = nn.ModuleList()
        self.sem_injects = nn.ModuleList()
        in_ch = 3
        for i, out_ch in enumerate(feat_channels):
            self.enc_blocks.append(
                ResConvBlock(in_ch, out_ch, downsample=(i > 0))
            )
            self.sem_injects.append(
                SoftSemanticInjection(num_classes, out_ch)
            )
            in_ch = out_ch
        final_ch = feat_channels[-1]

        # ── 2. Global context: scene-level lighting summary ───────────────────
        self.global_mlp = nn.Sequential(
            nn.Linear(final_ch, global_ctx_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_ctx_dim, embed_dim),
        )

        # ── 3. Masked class pooling projection ────────────────────────────────
        self.class_pool_proj = nn.Linear(final_ch, embed_dim)

        # ── 4. Variational codebook ───────────────────────────────────────────
        self.codebook = VariationalCodebook(num_classes, embed_dim)

        # ── 5. Fusion MLP: [class_pool || global_ctx || codebook] → embed_dim ─
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ── 6. Final spatial 1×1 conv ─────────────────────────────────────────
        self.spatial_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_onehot(self, semseg: torch.Tensor) -> torch.Tensor:
        """
        semseg : (B, H, W) long
        returns: (B, num_classes, H, W) float
        """
        B, H, W = semseg.shape
        onehot = torch.zeros(B, self.num_classes, H, W,
                             device=semseg.device, dtype=torch.float32)
        onehot.scatter_(1, semseg.unsqueeze(1).clamp(0, self.num_classes - 1), 1.0)
        return onehot

    def _masked_class_pool(
        self,
        feat:        torch.Tensor,   # (B, C, Hf, Wf) — final scale
        onehot_full: torch.Tensor,   # (B, num_classes, H, W) — full res
    ) -> torch.Tensor:               # (B, num_classes, C)
        """
        For every class c, compute a soft-attention-weighted average of CNN
        features over all pixels belonging to c in this image.

        The attention weights are normalised so they sum to 1 over all pixels
        for each class → equivalent to a soft masked global average pool.
        Classes absent from this image get a near-zero vector (mask sum ≈ 0).
        """
        B, C, Hf, Wf = feat.shape
        onehot_s = F.interpolate(
            onehot_full, size=(Hf, Wf), mode='bilinear', align_corners=False
        )                                                  # (B, N, Hf, Wf)
        mask     = onehot_s.view(B, self.num_classes, -1)  # (B, N, Hf*Wf)
        mask_n   = mask / (mask.sum(-1, keepdim=True) + 1e-6)  # normalise
        feat_f   = feat.view(B, C, -1).permute(0, 2, 1)    # (B, Hf*Wf, C)
        return torch.bmm(mask_n, feat_f)                   # (B, N, C)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, ldr: torch.Tensor, semseg: torch.Tensor):
        B, _, H, W = ldr.shape

        # ── Build full-resolution one-hot ─────────────────────────────────────
        onehot_full = self._build_onehot(semseg)           # (B, N, H, W)

        # ── Multi-scale CNN with soft semantic injection at each scale ─────────
        x = ldr
        for block, injector in zip(self.enc_blocks, self.sem_injects):
            x = block(x)                                   # conv + optional ↓
            x = injector(x, onehot_full)                   # soft semantic hint

        # x : (B, final_ch, H/S, W/S)

        # ── Global context (scene-level lighting) ─────────────────────────────
        global_ctx = self.global_mlp(x.mean(dim=[2, 3]))  # (B, embed_dim)

        # ── Per-class masked pooling from CNN features ─────────────────────────
        class_pool = self._masked_class_pool(x, onehot_full)   # (B, N, final_ch)
        class_pool = self.class_pool_proj(class_pool)           # (B, N, embed_dim)

        # ── Variational codebook ───────────────────────────────────────────────
        class_ids      = torch.arange(self.num_classes, device=ldr.device)
        cb_embed, mu, log_sigma = self.codebook(class_ids)     # (N, embed_dim)
        cb_embed = cb_embed.unsqueeze(0).expand(B, -1, -1)     # (B, N, embed_dim)
        mu       = mu.unsqueeze(0).expand(B, -1, -1)
        log_sigma= log_sigma.unsqueeze(0).expand(B, -1, -1)

        # ── Fuse: class_pool + global_ctx_per_class + codebook ─────────────────
        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, self.num_classes, -1)
        fused          = torch.cat([class_pool, global_ctx_exp, cb_embed], dim=-1)
        class_emb      = self.fusion_mlp(fused)                # (B, N, embed_dim)

        # ── Spatial projection: every pixel gets its class embedding ───────────
        # (B, embed_dim, N) × (B, N, H*W)  →  (B, embed_dim, H*W)
        class_emb_t  = class_emb.permute(0, 2, 1)             # (B, embed_dim, N)
        onehot_flat  = onehot_full.view(B, self.num_classes, H * W)
        sem_embed    = torch.bmm(class_emb_t, onehot_flat)    # (B, embed_dim, H*W)
        sem_embed    = sem_embed.view(B, self.embed_dim, H, W)
        sem_embed    = self.spatial_proj(sem_embed)            # (B, embed_dim, H, W)

        # ── KL divergence loss ─────────────────────────────────────────────────
        # KL( N(mu, sigma) || N(0,1) ) = -0.5 * sum(1 + 2*log_sigma - mu² - sigma²)
        kl_loss = -0.5 * (
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        ).mean() #remove the kl loss from here and add also remove the main functin 1

        return sem_embed, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Shape validation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, N_cls = 2, 256, 256, 20

    ldr    = torch.rand(B, 3, H, W)
    semseg = torch.randint(0, N_cls, (B, H, W))

    model = SemanticCodebookEncoder(
        num_classes   = N_cls,
        embed_dim     = 256,
        feat_channels = [64, 128, 256, 256],
    )
    model.train()

    sem_embed, kl_loss = model(ldr, semseg)

    print(f"Input LDR shape    : {ldr.shape}")
    print(f"Input semseg shape : {semseg.shape}")
    print(f"sem_embed shape    : {sem_embed.shape}")
    print(f"kl_loss            : {kl_loss.item():.6f}")
    assert sem_embed.shape == (B, 256, H, W), "Output shape mismatch!"
    print("\nAll shape assertions passed.")

    # Verify gradients flow back to the codebook
    loss = sem_embed.mean() + kl_loss
    loss.backward()
    print(f"Codebook mu grad norm    : {model.codebook.mu.weight.grad.norm():.4f}")
    print(f"Codebook sigma grad norm : {model.codebook.log_sigma.weight.grad.norm():.4f}")
    print("Gradient flow verified through full encoder.")
