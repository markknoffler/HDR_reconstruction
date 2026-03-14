import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Handcrafted helpers — no learned parameters, no training data needed
# ─────────────────────────────────────────────────────────────────────────────

def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Fixed Sobel edge detector.
    x       : (B, 1, H, W)  grayscale, float in [0,1]
    returns : (B, 1, H, W)  edge magnitude, normalised per-image to [0,1]
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx  = F.conv2d(x, kx, padding=1)
    gy  = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    # per-image normalisation → values in [0,1]
    B   = mag.shape[0]
    mx  = mag.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-6)
    return mag / mx


def gamma_invert(x: torch.Tensor, gamma: float = 0.45) -> torch.Tensor:
    """
    Gamma-stretch + invert: (1 - x^gamma)^(1/gamma)

    Near-saturation pixels (e.g. 0.99) → become 0.01 in the inverted domain.
    Their edges — invisible in the original domain — are now amplified.
    x : (B, 1, H, W) in [0,1]
    """
    return (1.0 - x.clamp(0.0, 1.0).pow(gamma)).pow(1.0 / gamma)


def soft_saturation_mask(ldr: torch.Tensor,
                         lo: float = 0.90,
                         hi: float = 0.98) -> torch.Tensor:
    """
    Soft saturation mask: ramps from 0 → 1 as max-channel value goes lo → hi.
    Smoother than a hard threshold, better gradient flow.
    ldr     : (B, 3, H, W)
    returns : (B, 1, H, W) in [0,1]  (1 = clipped, 0 = valid)
    """
    max_val = ldr.max(dim=1, keepdim=True).values   # (B, 1, H, W)
    return ((max_val - lo) / (hi - lo)).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# StructuralEncoder
# ─────────────────────────────────────────────────────────────────────────────

class StructuralEncoder(nn.Module):
    """
    Dual-domain structural gate encoder for TriGate-HDR.

    Produces a spatial gate  struct_gate ∈ [0,1]  of shape (B,1,H,W):
        ≈ 1.0  →  real edge / fine detail — decoder must preserve exactly
        ≈ 0.0  →  clipped / flat region  — decoder generates freely

    Two-domain edge detection
    ─────────────────────────
    Original domain  (edge_orig) : standard Sobel on grayscale LDR.
                                   Strong in mid-tones, blind near saturation.
    Inverted domain  (edge_inv)  : Sobel on gamma-inverted LDR.
                                   Amplifies near-saturation boundary edges
                                   that are invisible to edge_orig.
    Together they give complete structural coverage across the full
    dynamic range without any additional labelled dataset.

    Inputs
    ──────
    ldr   : (B, 3, H, W)  LDR image, float in [0,1]
    depth : (B, 1, H, W)  depth map (optional — from GTA G-Buffer if available)

    Output
    ──────
    struct_gate : (B, 1, H, W)  in [0,1]
    """

    def __init__(self, use_depth: bool = True):
        super().__init__()
        self.use_depth = use_depth

        # 5 handcrafted channels: edge_orig, edge_inv, sat_mask, ldr_gray, ldr_inv
        # +1 if depth is available
        in_ch = 5 + (1 if use_depth else 0)

        self.cnn = nn.Sequential(
            # Layer 1 — broad context
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            # Layer 2 — finer edge/saturation relationships
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # Layer 3 — compress back
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            # Layer 4 — 1×1 scalar gate + sigmoid
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ldr:   torch.Tensor,
        depth: torch.Tensor = None,
    ) -> torch.Tensor:

        # ── Handcrafted features (deterministic, no learned params) ───────────
        ldr_gray = ldr.mean(dim=1, keepdim=True)            # (B,1,H,W)
        ldr_inv  = gamma_invert(ldr_gray, gamma=0.45)       # (B,1,H,W)
        sat_mask = soft_saturation_mask(ldr)                 # (B,1,H,W) in [0,1]

        edge_orig = sobel_magnitude(ldr_gray)               # (B,1,H,W)
        edge_inv  = sobel_magnitude(ldr_inv)                # (B,1,H,W)

        # Suppress edge_orig inside clipped regions (data is meaningless there).
        # edge_inv is kept as-is — it fires at near-saturation boundaries
        # which is exactly the detail we want to recover.
        edge_orig = edge_orig * (1.0 - sat_mask)

        # ── Assemble CNN input ────────────────────────────────────────────────
        features = [edge_orig, edge_inv, sat_mask, ldr_gray, ldr_inv]

        if self.use_depth and depth is not None:
            edge_depth = sobel_magnitude(depth)             # (B,1,H,W)
            features.append(edge_depth)

        x = torch.cat(features, dim=1)                     # (B, in_ch, H,W)

        # ── Learned gate ──────────────────────────────────────────────────────
        gate = self.cnn(x)                                  # (B,1,H,W)

        # Hard override: force gate → 0 inside fully clipped regions.
        # The CNN cannot override physics: clipped pixels carry no information.
        gate = gate * (1.0 - sat_mask)

        return gate                                         # (B,1,H,W) in [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# Shape test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 256, 256

    ldr   = torch.rand(B, 3, H, W)
    depth = torch.rand(B, 1, H, W)

    # ── with depth ────────────────────────────────────────────────────────────
    enc   = StructuralEncoder(use_depth=True)
    enc.train()
    gate  = enc(ldr, depth)
    print(f"Input  ldr shape   : {ldr.shape}")
    print(f"Input  depth shape : {depth.shape}")
    print(f"Output gate shape  : {gate.shape}")
    assert gate.shape == (B, 1, H, W)
    assert gate.min() >= 0.0 and gate.max() <= 1.0

    # ── without depth ─────────────────────────────────────────────────────────
    enc2  = StructuralEncoder(use_depth=False)
    gate2 = enc2(ldr)
    assert gate2.shape == (B, 1, H, W)

    # ── gradient flow ─────────────────────────────────────────────────────────
    loss = gate.mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    print(f"Trainable param groups with grad : {len(grads)}")
    print(f"Gate value range : [{gate.min().item():.4f}, {gate.max().item():.4f}]")
    print("All assertions passed.")
