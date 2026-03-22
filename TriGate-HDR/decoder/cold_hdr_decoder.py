# ==============================================================================
# LDR to HDR RECONSTRUCTION DECODER (COLD DIFFUSION)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    Encodes the 'timestep' (e.g., how much clipping/noise we have) 
    into a vector that the neural network can understand.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleResBlock(nn.Module):
    """
    A simple building block for our UNet. 
    It takes an image (or features), processes it, and adds the original input back (skip connection).
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        
        if self.time_mlp and time_emb is not None:
            # We "inject" the time information here
            time_shift = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_shift
            
        h = self.relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class HDR_UNet(nn.Module):
    """
    THE CLEANING MACHINE (Denoising Decoder).
    This network takes an LDR image and tries to guess the original HDR image.
    """
    def __init__(self, base_ch=64):
        super().__init__()
        
        # 1. TIME ENCODING
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 4)
        )
        t_ch = base_ch * 4

        # 2. FEATURE INJECTION POINTS
        # Material features (523 channels) are injected right at the start
        self.input_layer = nn.Conv2d(3 + 523, base_ch, 3, padding=1)

        # 3. DOWNSAMPLING (Encoding)
        # We go from High Resolution -> Low Resolution to understand the overall picture
        self.down1 = SimpleResBlock(base_ch, base_ch * 2, t_ch)    # H
        self.pool1 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1) # H -> H/2
        
        self.down2 = SimpleResBlock(base_ch * 2, base_ch * 4, t_ch) # H/2
        self.pool2 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)   # H/2 -> H/4
        
        self.down3 = SimpleResBlock(base_ch * 4, base_ch * 8, t_ch) # H/4
        self.pool3 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)   # H/4 -> H/8

        # 4. BOTTLENECK (The Deepest Layer)
        # We inject the highest semantic features here
        self.mid_block = SimpleResBlock(base_ch * 8, base_ch * 8, t_ch)
        self.sem_bot_proj = nn.Conv2d(512, base_ch * 8, 1) # Project 512 sem ch to match bottleneck

        # 5. UPSAMPLING (Decoding)
        # We go from Low Resolution -> High Resolution to rebuild the HDR details
        # We also inject Semantic features at each step
        
        # Step 1: H/8 -> H/4
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1)
        # Concat: upsampled (base_ch*4) + skip from down3 (base_ch*8) + sem_feat (256)
        self.up_block1 = SimpleResBlock(base_ch * 4 + base_ch * 8 + 256, base_ch * 4, t_ch)
        
        # Step 2: H/4 -> H/2
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        # Concat: upsampled (base_ch*2) + skip from down2 (base_ch*4) + sem_feat (128)
        self.up_block2 = SimpleResBlock(base_ch * 2 + base_ch * 4 + 128, base_ch * 2, t_ch)
        
        # Step 3: H/2 -> H
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        # Concat: upsampled (base_ch) + skip from down1 (base_ch*2) + sem_feat (64)
        self.up_block3 = SimpleResBlock(base_ch + base_ch * 2 + 64, base_ch, t_ch)

        # 6. FINAL RESTORATION
        # We inject the Structural features at the very end to fix saturated areas
        self.struct_proj = nn.Conv2d(256, base_ch, 1)
        self.final_head  = nn.Sequential(
            nn.Conv2d(base_ch + base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1)
        )

    def forward(self, x, t, mat_feat=None, struct_feat=None, sem_feats=None):
        """
        x: LDR image (B, 3, H, W)
        t: timestep (B,)
        mat_feat: Material features (B, 523, H, W)
        struct_feat: (B, 256, H, W) -> Structural guidance
        sem_feats: [H, H/2, H/4, H/8] -> Semantic features
        """
        time_emb = self.time_mlp(t)
        
        # A. INJECT MATERIAL
        # Why: High-res information like gradients helps guide the shapes/edges.
        if mat_feat is not None:
             x = torch.cat([x, mat_feat], dim=1)
        else:
             # Dummy if missing
             dummy = torch.zeros(x.shape[0], 523, x.shape[2], x.shape[3], device=x.device)
             x = torch.cat([x, dummy], dim=1)
        
        x = self.input_layer(x)
        
        # B. DOWNWARD PATH (Saving shortcuts for later)
        s1 = self.down1(x, time_emb)
        x = self.pool1(s1)
        
        s2 = self.down2(x, time_emb)
        x = self.pool2(s2)
        
        s3 = self.down3(x, time_emb)
        x = self.pool3(s3)
        
        # C. BOTTLENECK (Injecting deep semantic context)
        # Why: Semantic knowledge (e.g. skin vs sky) helps the model guess the brightness.
        x = self.mid_block(x, time_emb)
        if sem_feats is not None:
            bh, bw = x.shape[2], x.shape[3]
            x = x + F.interpolate(self.sem_bot_proj(sem_feats[3]), size=(bh, bw), mode='bilinear')

        # D. UPWARD PATH (Rebuilding the image using semantic clues)
        # Step 1: H/8 -> H/4
        x = self.up1(x)
        if sem_feats is not None:
            x = torch.cat([x, s3, sem_feats[2]], dim=1) # Concat everything
        x = self.up_block1(x, time_emb)
        
        # Step 2: H/4 -> H/2
        x = self.up2(x)
        if sem_feats is not None:
            x = torch.cat([x, s2, sem_feats[1]], dim=1)
        x = self.up_block2(x, time_emb)
        
        # Step 3: H/2 -> H
        x = self.up3(x)
        if sem_feats is not None:
            x = torch.cat([x, s1, sem_feats[0]], dim=1)
        x = self.up_block3(x, time_emb)
        
        # E. FINAL REFINEMENT (Injecting Structural Guidance)
        # Why: Structural features tell the model exactly where the pixels were clipped.
        if struct_feat is not None:
            s_proj = self.struct_proj(struct_feat[0]) # index 0 is feat
            x = torch.cat([x, s_proj], dim=1)
        else:
            dummy_s = torch.zeros(x.shape[0], 64, x.shape[2], x.shape[3], device=x.device)
            x = torch.cat([x, dummy_s], dim=1)
            
        return self.final_head(x)

class ColdHDRDiffusion(nn.Module):
    """
    THE RULES OF THE GAME.
    This class handles the "Forward" (Noising) and "Reverse" (Denoising).
    """
    def __init__(self, model, timesteps=100):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Clipping Schedule: B(t)
        # We start with High Threshold (HDR) and move to 1.0 (LDR)
        self.register_buffer('thresholds', torch.linspace(10.0, 1.0, timesteps))

    def move_toward_ldr(self, x_hdr, t):
        """
        THE NOISING PART.
        Instead of adding random noise, we 'clip' the HDR step-by-step.
        """
        b_t = self.thresholds[t].view(-1, 1, 1, 1)
        return torch.min(x_hdr.clamp(min=0.0), b_t)

    def forward(self, x_hdr, mat, struct, sem):
        """
        TRAINING STEP.
        1. Pick a random step (amount of clipping).
        2. Create the clipped image.
        3. Ask the UNet to restore it.
        4. Measure the error.
        """
        b = x_hdr.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_hdr.device).long()
        
        x_clipped = self.move_toward_ldr(x_hdr, t)
        x_predicted = self.model(x_clipped, t, mat, struct, sem)
        
        # Loss 1: Compare predicted HDR with target HDR
        hdr_loss = F.l1_loss(x_hdr, x_predicted)
        
        # Loss 2: Compare predicted LDR version with input clipped version
        # (This is the LDR specific loss step-by-step)
        x_pred_clipped = self.move_toward_ldr(x_predicted, t)
        ldr_loss = F.l1_loss(x_clipped, x_pred_clipped)
        
        return hdr_loss + ldr_loss

    @torch.no_grad()
    def restore_hdr(self, x_ldr, mat, struct, sem):
        """
        THE DECODING PART (Algorithm 2).
        Restores HDR from LDR by iterating backwards.
        """
        batch_size = x_ldr.shape[0]
        device = x_ldr.device
        
        img = x_ldr # Start from LDR
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict the final x0 (Clean HDR) from current xt
            x0_guess = self.model(img, t_batch, mat, struct, sem)
            
            if t > 0:
                # Math from the paper: x_{t-1} = x_t - D(x0_guess, t) + D(x0_guess, t-1)
                d_t = self.move_toward_ldr(x0_guess, t)
                d_t_prev = self.move_toward_ldr(x0_guess, t - 1)
                
                img = img - d_t + d_t_prev
            else:
                img = x0_guess
                
        return img
