import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=none):
        super().__init__()
        self.time_mlp = nn.linear(time_emb_dim, out_ch) if time_emb_dim else none
        
        self.conv1 = nn.conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.groupnorm(1, out_ch)
        self.relu  = nn.leakyrelu(0.2, inplace=true)
        self.conv2 = nn.conv2d(out_ch, out_ch, 3, padding=1)
        
        self.shortcut = nn.conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.identity()

    def forward(self, x, time_emb=none):
        h = self.conv1(x)
        h = self.norm1(h)
        
        if self.time_mlp and time_emb is not none:
            # we "inject" the time information here
            time_shift = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_shift
            
        h = self.relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class HDR_UNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 4)
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
        self.up_block1 = SimpleResBlock(base_ch*4 + base_ch*8, base_ch*4, t_ch)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.up_block2 = SimpleResBlock(base_ch*2 + base_ch*4, base_ch*2, t_ch) 

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.up_block3 = SimpleResBlock(base_ch + base_ch*2, base_ch, t_ch)      

        self.final_head = nn.Sequential(                                        
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1)
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)

        x = self.input_layer(x)

        s1 = self.down1(x, time_emb)
        x  = self.pool1(s1)

        s2 = self.down2(x, time_emb)
        x  = self.pool2(s2)

        s3 = self.down3(x, time_emb)
        x  = self.pool3(s3)

        x = self.mid_block(x, time_emb)

        x = self.up1(x)
        x = torch.cat([x, s3], dim=1)
        x = self.up_block1(x, time_emb)

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up_block2(x, time_emb)

        x = self.up3(x)
        x = torch.cat([x, s1], dim=1)
        x = self.up_block3(x, time_emb)

        return self.final_head(x)

#add this part in the training script

class ColdHDRDiffusion(nn.Module):
    def __init__(self, model, timesteps=100):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        self.register_buffer('thresholds', torch.linspace(10.0, 1.0, timesteps))

    def move_toward_ldr(self, x_hdr, t):
        b_t = self.thresholds[t].view(-1, 1, 1, 1)
        return torch.min(x_hdr.clamp(min=0.0), b_t)

    def forward(self, x_hdr):                          
        b = x_hdr.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_hdr.device).long()
        
        x_clipped   = self.move_toward_ldr(x_hdr, t)
        x_predicted = self.model(x_clipped, t)         

        hdr_loss = F.l1_loss(x_hdr, x_predicted)

        x_pred_clipped = self.move_toward_ldr(x_predicted, t)
        ldr_loss = F.l1_loss(x_clipped, x_pred_clipped)
        
        return hdr_loss + ldr_loss

    @torch.no_grad()
    def restore_hdr(self, x_ldr):                   
        batch_size = x_ldr.shape[0]
        device     = x_ldr.device
        
        img = x_ldr
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            x0_guess = self.model(img, t_batch)

            if t > 0:
                d_t      = self.move_toward_ldr(x0_guess, t)
                d_t_prev = self.move_toward_ldr(x0_guess, t - 1)
                img = img - d_t + d_t_prev
            else:
                img = x0_guess
                
        return img

