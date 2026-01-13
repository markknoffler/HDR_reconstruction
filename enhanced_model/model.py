import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,      
            groups=in_channels,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  
        f2 = self.layer2(f1) 
        f3 = self.layer3(f2) 
        f4 = self.layer4(f3) 
        return [f1, f2, f3, f4]

class SeparateEncodersHDR(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ResNetEncoder(pretrained=False)
        self.enc2 = ResNetEncoder(pretrained=False)
        self.enc3 = ResNetEncoder(pretrained=False)
        self.enc4 = ResNetEncoder(pretrained=False)
        self.enc5 = ResNetEncoder(pretrained=False)
        self.enc6 = ResNetEncoder(pretrained=False)

        self.enc1.load_state_dict(torch.load("encoder_type1.pth", map_location="cpu"))
        self.enc2.load_state_dict(torch.load("encoder_type2.pth", map_location="cpu"))
        self.enc3.load_state_dict(torch.load("encoder_type3.pth", map_location="cpu"))
        self.enc4.load_state_dict(torch.load("encoder_type4.pth", map_location="cpu"))
        self.enc5.load_state_dict(torch.load("encoder_type5.pth", map_location="cpu"))
        self.enc6.load_state_dict(torch.load("encoder_type6.pth", map_location="cpu"))


    def forward(self, imgs):
        x1, x2, x3, x4, x5, x6 = imgs

        f1_ = self.enc1(x1)
        f2_ = self.enc2(x2)
        f3_ = self.enc3(x3)
        f4_ = self.enc4(x4)
        f5_ = self.enc5(x5)
        f6_ = self.enc6(x6)

        f1_deep = f1_[-1]
        f2_deep = f2_[-1]
        f3_deep = f3_[-1]
        f4_deep = f4_[-1]
        f5_deep = f5_[-1]
        f6_deep = f6_[-1]

        return f1_deep, f2_deep, f3_deep, f4_deep, f5_deep, f6_deep

class DAB_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca_fc = nn.Linear(channels, channels)
        self.pa_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        B,C,H,W = x.size()
        ca = F.adaptive_avg_pool2d(x, output_size=1)
        ca = ca.view(B, C)
        ca = self.ca_fc(ca)
        ca = torch.sigmoid(ca).view(B, C, 1, 1)
        CA = x * ca

        s_avg = x.mean(dim=1, keepdim=True)
        s_score = torch.sigmoid(s_avg)
        SA = x * s_score

        p_score = torch.sigmoid(self.pa_conv(x))
        PA = x * p_score
        
        out = CA + SA + PA
        return out

class PFF_block_3(nn.Module):
    def __init__(self, ch_pre, ch_curr, ch_nex, dab_block_cls):
        super().__init__()
        self.down_dw = DepthwiseConv2d(ch_pre, kernel_size=2, stride=2)
        self.down_1x1 = nn.Conv2d(ch_pre, ch_curr, kernel_size=1, stride =1, padding=0)

        self.up_1x1 = nn.Conv2d(ch_nex, ch_curr, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        concat_channels =  ch_curr * 2 
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, pre, curr, next):
        down = self.down_dw(pre)
        down = self.down_1x1(down)

        up = self.up_1x1(next)
        up = self.upsample(up)

        mul_low = down * curr
        mul_high = up * curr

        concat = torch.cat([mul_low, mul_high], dim=1)

        att = self.dab(concat)

        x = self.out_1x1(att)
        x = self.act(x)

        return x

class PFF_block_pre(nn.Module):
    def __init__(self, ch_curr, ch_next):
        super().__init__()
        self.up_1x1 = nn.Conv2d(ch_next, ch_curr, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        concat_channels = ch_curr * 2
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, curr, next):
        up = self.up_1x1(next)
        up = self.upsample(up)

        mul = up * curr

        concat = torch.cat([curr, mul], dim=1)

        att = self.dab(concat)
        x = self.out_1x1(att)
        x = self.act(x)

        return x

class PFF_block_next(nn.Module):
    def __init__(self, ch_curr, ch_prev):
        super().__init__()
        self.down_dw = DepthwiseConv2d(ch_prev, kernel_size=2, stride=2)
        self.down_1x1 = nn.Conv2d(ch_prev, ch_curr, kernel_size=1, stride=1, padding=0)

        concat_channels = 2 * ch_curr
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, curr, prev):
        down = self.down_dw(prev)
        down = self.down_1x1(down)

        mul = down * curr

        concat = torch.cat([curr, mul], dim=1)

        att = self.dab(concat)
        x = self.out_1x1(att)
        x = self.act(x)

        return x

#class ReconstructionUnit(nn.Module):
#    def __init__(self, channels_1, channels_2, channels_3, channels_4, out_channels=3):
#        super(ReconstructionUnit, self).__init__()
#        
#        self.up1 = nn.Sequential(
#            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
#            nn.BatchNorm2d(1024),
#            nn.ReLU(inplace=True),
#            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#        )
#        self.proj1 = nn.Conv2d(1024 * 2, 1024, kernel_size=1)
#        
#        self.up2 = nn.Sequential(
#            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(inplace=True),
#            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#        )
#        self.proj2 = nn.Conv2d(1024 * 2, 1024, kernel_size=1)
#        
#        self.up3 = nn.Sequential(
#            nn.Conv2d(512, 256, kernel_size=3, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#        )
#        self.proj3 = nn.Conv2d(1024 * 2, 1024, kernel_size=1)
#        
#        self.up4 = nn.Sequential(
#            nn.Conv2d(256, 128, kernel_size=3, padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
#            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#        )
#        
#        self.up5 = nn.Sequential(
#            nn.Conv2d(128, 64, kernel_size=3, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(inplace=True),
#            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#        )
#        
#        self.reconstruction_net = nn.Sequential(
#            nn.Conv2d(64, 64, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(64, 64, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
#            nn.Tanh()
#        )
#    
#    def forward(self, frs, comb_1, comb_2, comb_3, comb_4):
#        x = self.up1(comb_1)
#        x = torch.cat([x, comb_2], dim=1)
#        x = self.proj1(x)  
#        
#        x = self.up2(x)
#        x = torch.cat([x, comb_3], dim=1)
#        x = self.proj2(x)  
#        
#        x = self.up3(x)
#        x = torch.cat([x, comb_4], dim=1)
#        x = self.proj3(x)
#        
#        x = self.up4(x)
#        x = self.up5(x)
#        x = self.reconstruction_net(x)
#        return x

class ReconstructionUnit(nn.Module):
    def __init__(self, channels_1, channels_2, channels_3, channels_4, out_channels=3):
        super(ReconstructionUnit, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(channels_4, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.proj1 = nn.Conv2d(1024 + channels_3, 1024, kernel_size=1)
        
        self.up2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.proj2 = nn.Conv2d(512 + channels_2, 512, kernel_size=1)
        
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.proj3 = nn.Conv2d(256 + channels_1, 256, kernel_size=1)
        
        self.up4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.reconstruction_net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, pff1, pff2, pff3, pff4):
        x = self.up1(pff4)
        x = torch.cat([x, pff3], dim=1)
        x = self.proj1(x)
        x = self.up2(x)
        x = torch.cat([x, pff2], dim=1)
        x = self.proj2(x)
        x = self.up3(x)
        x = torch.cat([x, pff1], dim=1)
        x = self.proj3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.reconstruction_net(x)
        return x


class Dynamic_attention_model(nn.Module):
    def __init__(self, layer1_channels, layer2_channels, layer3_channels, layer4_channels):
        super().__init__()
        
        self.gamma_encoder = ResNetEncoder()
        self.underexposed_encoder = ResNetEncoder()
        self.overexposed_encoder = ResNetEncoder()
        self.original_encoder = ResNetEncoder()
        self.h_2_encoder = ResNetEncoder()
        self.histoEQ_encoder = ResNetEncoder()

        self.cpu_encoders = [self.original_encoder, self.h_2_encoder, self.histoEQ_encoder]

        
        self.gamma_dab1 = DAB_block(layer1_channels)
        self.gamma_dab2 = DAB_block(layer2_channels)
        self.gamma_dab3 = DAB_block(layer3_channels)
        self.gamma_dab4 = DAB_block(layer4_channels)
        self.gamma_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.gamma_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.gamma_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.gamma_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.underexposed_dab1 = DAB_block(layer1_channels)
        self.underexposed_dab2 = DAB_block(layer2_channels)
        self.underexposed_dab3 = DAB_block(layer3_channels)
        self.underexposed_dab4 = DAB_block(layer4_channels)
        self.underexposed_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.underexposed_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.underexposed_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.underexposed_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.overexposed_dab1 = DAB_block(layer1_channels)
        self.overexposed_dab2 = DAB_block(layer2_channels)
        self.overexposed_dab3 = DAB_block(layer3_channels)
        self.overexposed_dab4 = DAB_block(layer4_channels)
        self.overexposed_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.overexposed_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.overexposed_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.overexposed_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.original_dab1 = DAB_block(layer1_channels)
        self.original_dab2 = DAB_block(layer2_channels)
        self.original_dab3 = DAB_block(layer3_channels)
        self.original_dab4 = DAB_block(layer4_channels)
        self.original_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.original_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.original_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.original_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.h_2_dab1 = DAB_block(layer1_channels)
        self.h_2_dab2 = DAB_block(layer2_channels)
        self.h_2_dab3 = DAB_block(layer3_channels)
        self.h_2_dab4 = DAB_block(layer4_channels)
        self.h_2_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.h_2_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.h_2_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.h_2_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.histoEQ_dab1 = DAB_block(layer1_channels)
        self.histoEQ_dab2 = DAB_block(layer2_channels)
        self.histoEQ_dab3 = DAB_block(layer3_channels)
        self.histoEQ_dab4 = DAB_block(layer4_channels)
        self.histoEQ_proj1 = nn.Conv2d(layer1_channels * 2, layer1_channels, kernel_size=1)
        self.histoEQ_proj2 = nn.Conv2d(layer2_channels * 2, layer2_channels, kernel_size=1)
        self.histoEQ_proj3 = nn.Conv2d(layer3_channels * 2, layer3_channels, kernel_size=1)
        self.histoEQ_proj4 = nn.Conv2d(layer4_channels * 2, layer4_channels, kernel_size=1)
        
        self.pff_block_1 = PFF_block_pre(layer1_channels * 6, layer2_channels * 6)
        #self.pff_block_2 = PFF_block_3(layer1_channels * 6, layer2_channels * 6, layer3_channels * 6)
        #self.pff_block_3 = PFF_block_3(layer2_channels * 6, layer3_channels * 6, layer4_channels * 6)
        self.pff_block_2 = PFF_block_3(layer1_channels * 6, layer2_channels * 6, layer3_channels * 6, DAB_block)
        self.pff_block_3 = PFF_block_3(layer2_channels * 6, layer3_channels * 6, layer4_channels * 6, DAB_block)
        #self.pff_block_4 = PFF_block_next(layer3_channels * 6, layer4_channels * 6)
        self.pff_block_4 = PFF_block_next(layer4_channels * 6, layer3_channels * 6)

        
        self.reconstructed_image = ReconstructionUnit(layer1_channels * 6, layer2_channels * 6, layer3_channels * 6, layer4_channels * 6)
    
    def setup_cpu_offloading(self):
        """Move some encoders to CPU to save GPU memory"""
        self.original_encoder = self.original_encoder.cpu()
        self.h_2_encoder = self.h_2_encoder.cpu()
        self.histoEQ_encoder = self.histoEQ_encoder.cpu()
        print("✓ Moved 3 encoders to CPU to save GPU memory")


    def forward(self, gamma, underexposed, overexposed, original, h2, histoEQ):
        device = gamma.device  # Remember the GPU device
        
        # GPU encoders with checkpointing (saves memory)
        gamma_l1, gamma_l2, gamma_l3, gamma_l4 = checkpoint(self.gamma_encoder, gamma, use_reentrant=False)
        underexposed_l1, underexposed_l2, underexposed_l3, underexposed_l4 = checkpoint(self.underexposed_encoder, underexposed, use_reentrant=False)
        overexposed_l1, overexposed_l2, overexposed_l3, overexposed_l4 = checkpoint(self.overexposed_encoder, overexposed, use_reentrant=False)
        
        # CPU encoders - process on CPU to save GPU memory
        original_l1, original_l2, original_l3, original_l4 = self.original_encoder(original.cpu())
        original_l1 = original_l1.to(device)
        original_l2 = original_l2.to(device)
        original_l3 = original_l3.to(device)
        original_l4 = original_l4.to(device)
        
        h2_l1, h2_l2, h2_l3, h2_l4 = self.h_2_encoder(h2.cpu())
        h2_l1 = h2_l1.to(device)
        h2_l2 = h2_l2.to(device)
        h2_l3 = h2_l3.to(device)
        h2_l4 = h2_l4.to(device)
        
        histoEQ_l1, histoEQ_l2, histoEQ_l3, histoEQ_l4 = self.histoEQ_encoder(histoEQ.cpu())
        histoEQ_l1 = histoEQ_l1.to(device)
        histoEQ_l2 = histoEQ_l2.to(device)
        histoEQ_l3 = histoEQ_l3.to(device)
        histoEQ_l4 = histoEQ_l4.to(device)
        
        # Apply DAB blocks and projections for gamma
        gamma_l1 = self.gamma_proj1(torch.cat([gamma_l1, self.gamma_dab1(gamma_l1)], dim=1))
        gamma_l2 = self.gamma_proj2(torch.cat([gamma_l2, self.gamma_dab2(gamma_l2)], dim=1))
        gamma_l3 = self.gamma_proj3(torch.cat([gamma_l3, self.gamma_dab3(gamma_l3)], dim=1))
        gamma_l4 = self.gamma_proj4(torch.cat([gamma_l4, self.gamma_dab4(gamma_l4)], dim=1))
        
        # Apply DAB blocks and projections for underexposed
        underexposed_l1 = self.underexposed_proj1(torch.cat([underexposed_l1, self.underexposed_dab1(underexposed_l1)], dim=1))
        underexposed_l2 = self.underexposed_proj2(torch.cat([underexposed_l2, self.underexposed_dab2(underexposed_l2)], dim=1))
        underexposed_l3 = self.underexposed_proj3(torch.cat([underexposed_l3, self.underexposed_dab3(underexposed_l3)], dim=1))
        underexposed_l4 = self.underexposed_proj4(torch.cat([underexposed_l4, self.underexposed_dab4(underexposed_l4)], dim=1))
        
        # Apply DAB blocks and projections for overexposed
        overexposed_l1 = self.overexposed_proj1(torch.cat([overexposed_l1, self.overexposed_dab1(overexposed_l1)], dim=1))
        overexposed_l2 = self.overexposed_proj2(torch.cat([overexposed_l2, self.overexposed_dab2(overexposed_l2)], dim=1))
        overexposed_l3 = self.overexposed_proj3(torch.cat([overexposed_l3, self.overexposed_dab3(overexposed_l3)], dim=1))
        overexposed_l4 = self.overexposed_proj4(torch.cat([overexposed_l4, self.overexposed_dab4(overexposed_l4)], dim=1))
        
        # Apply DAB blocks and projections for original
        original_l1 = self.original_proj1(torch.cat([original_l1, self.original_dab1(original_l1)], dim=1))
        original_l2 = self.original_proj2(torch.cat([original_l2, self.original_dab2(original_l2)], dim=1))
        original_l3 = self.original_proj3(torch.cat([original_l3, self.original_dab3(original_l3)], dim=1))
        original_l4 = self.original_proj4(torch.cat([original_l4, self.original_dab4(original_l4)], dim=1))
        
        # Apply DAB blocks and projections for h2
        h2_l1 = self.h2_proj1(torch.cat([h2_l1, self.h2_dab1(h2_l1)], dim=1))
        h2_l2 = self.h2_proj2(torch.cat([h2_l2, self.h2_dab2(h2_l2)], dim=1))
        h2_l3 = self.h2_proj3(torch.cat([h2_l3, self.h2_dab3(h2_l3)], dim=1))
        h2_l4 = self.h2_proj4(torch.cat([h2_l4, self.h2_dab4(h2_l4)], dim=1))
        
        # Apply DAB blocks and projections for histoEQ
        histoEQ_l1 = self.histoEQ_proj1(torch.cat([histoEQ_l1, self.histoEQ_dab1(histoEQ_l1)], dim=1))
        histoEQ_l2 = self.histoEQ_proj2(torch.cat([histoEQ_l2, self.histoEQ_dab2(histoEQ_l2)], dim=1))
        histoEQ_l3 = self.histoEQ_proj3(torch.cat([histoEQ_l3, self.histoEQ_dab3(histoEQ_l3)], dim=1))
        histoEQ_l4 = self.histoEQ_proj4(torch.cat([histoEQ_l4, self.histoEQ_dab4(histoEQ_l4)], dim=1))
        
        # Concatenate all features
        layer1 = torch.cat([gamma_l1, underexposed_l1, overexposed_l1, original_l1, h2_l1, histoEQ_l1], dim=1)
        layer2 = torch.cat([gamma_l2, underexposed_l2, overexposed_l2, original_l2, h2_l2, histoEQ_l2], dim=1)
        layer3 = torch.cat([gamma_l3, underexposed_l3, overexposed_l3, original_l3, h2_l3, histoEQ_l3], dim=1)
        layer4 = torch.cat([gamma_l4, underexposed_l4, overexposed_l4, original_l4, h2_l4, histoEQ_l4], dim=1)
        
        # Apply PFF blocks
        pff1 = self.pff_block_1(layer1, layer2)
        pff2 = self.pff_block_2(layer1, layer2, layer3)
        pff3 = self.pff_block_3(layer2, layer3, layer4)
        pff4 = self.pff_block_4(layer4, layer3)
        
        # Reconstruct image
        output = self.reconstructed_image(pff1, pff2, pff3, pff4)
        
        return output


#    def forward(self, gamma, underexposed, overexposed, original, h_2, histoEQ):
#
#        gamma_l1, gamma_l2, gamma_l3, gamma_l4 = checkpoint(self.gamma_encoder, gamma, use_reentrant=False)
#        underexposed_l1, underexposed_l2, underexposed_l3, underexposed_l4 = checkpoint(self.underexposed_encoder, underexposed, use_reentrant=False)
#        overexposed_l1, overexposed_l2, overexposed_l3, overexposed_l4 = checkpoint(self.overexposed_encoder, overexposed, use_reentrant=False)
#        original_l1, original_l2, original_l3, original_l4 = checkpoint(self.original_encoder, original, use_reentrant=False)
#        h_2_l1, h_2_l2, h_2_l3, h_2_l4 = checkpoint(self.h_2_encoder, h_2, use_reentrant=False)
#        histoEQ_l1, histoEQ_l2, histoEQ_l3, histoEQ_l4 = checkpoint(self.histoEQ_encoder, histoEQ, use_reentrant=False)
#
##        gamma_l1, gamma_l2, gamma_l3, gamma_l4 = self.gamma_encoder(gamma)
##        underexposed_l1, underexposed_l2, underexposed_l3, underexposed_l4 = self.underexposed_encoder(underexposed)
##        overexposed_l1, overexposed_l2, overexposed_l3, overexposed_l4 = self.overexposed_encoder(overexposed)
##        original_l1, original_l2, original_l3, original_l4 = self.original_encoder(original)
##        h_2_l1, h_2_l2, h_2_l3, h_2_l4 = self.h_2_encoder(h_2)
##        histoEQ_l1, histoEQ_l2, histoEQ_l3, histoEQ_l4 = self.histoEQ_encoder(histoEQ)
#        #
#
#        device = gamma.device
#
#        with torch.no_grad():
#        
#            gamma_l1 = self.gamma_proj1(torch.cat([gamma_l1, self.gamma_dab1(gamma_l1)], dim=1))
#            gamma_l2 = self.gamma_proj2(torch.cat([gamma_l2, self.gamma_dab2(gamma_l2)], dim=1))
#            gamma_l3 = self.gamma_proj3(torch.cat([gamma_l3, self.gamma_dab3(gamma_l3)], dim=1))
#            gamma_l4 = self.gamma_proj4(torch.cat([gamma_l4, self.gamma_dab4(gamma_l4)], dim=1))
#            
#            underexposed_l1 = self.underexposed_proj1(torch.cat([underexposed_l1, self.underexposed_dab1(underexposed_l1)], dim=1))
#            underexposed_l2 = self.underexposed_proj2(torch.cat([underexposed_l2, self.underexposed_dab2(underexposed_l2)], dim=1))
#            underexposed_l3 = self.underexposed_proj3(torch.cat([underexposed_l3, self.underexposed_dab3(underexposed_l3)], dim=1))
#            underexposed_l4 = self.underexposed_proj4(torch.cat([underexposed_l4, self.underexposed_dab4(underexposed_l4)], dim=1))
#            
#            overexposed_l1 = self.overexposed_proj1(torch.cat([overexposed_l1, self.overexposed_dab1(overexposed_l1)], dim=1))
#            overexposed_l2 = self.overexposed_proj2(torch.cat([overexposed_l2, self.overexposed_dab2(overexposed_l2)], dim=1))
#            overexposed_l3 = self.overexposed_proj3(torch.cat([overexposed_l3, self.overexposed_dab3(overexposed_l3)], dim=1))
#            overexposed_l4 = self.overexposed_proj4(torch.cat([overexposed_l4, self.overexposed_dab4(overexposed_l4)], dim=1))
#            
#            original_l1 = self.original_proj1(torch.cat([original_l1, self.original_dab1(original_l1)], dim=1))
#            original_l2 = self.original_proj2(torch.cat([original_l2, self.original_dab2(original_l2)], dim=1))
#            original_l3 = self.original_proj3(torch.cat([original_l3, self.original_dab3(original_l3)], dim=1))
#            original_l4 = self.original_proj4(torch.cat([original_l4, self.original_dab4(original_l4)], dim=1))
#            
#            h_2_l1 = self.h_2_proj1(torch.cat([h_2_l1, self.h_2_dab1(h_2_l1)], dim=1))
#            h_2_l2 = self.h_2_proj2(torch.cat([h_2_l2, self.h_2_dab2(h_2_l2)], dim=1))
#            h_2_l3 = self.h_2_proj3(torch.cat([h_2_l3, self.h_2_dab3(h_2_l3)], dim=1))
#            h_2_l4 = self.h_2_proj4(torch.cat([h_2_l4, self.h_2_dab4(h_2_l4)], dim=1))
#            
#            histoEQ_l1 = self.histoEQ_proj1(torch.cat([histoEQ_l1, self.histoEQ_dab1(histoEQ_l1)], dim=1))
#            histoEQ_l2 = self.histoEQ_proj2(torch.cat([histoEQ_l2, self.histoEQ_dab2(histoEQ_l2)], dim=1))
#            histoEQ_l3 = self.histoEQ_proj3(torch.cat([histoEQ_l3, self.histoEQ_dab3(histoEQ_l3)], dim=1))
#            histoEQ_l4 = self.histoEQ_proj4(torch.cat([histoEQ_l4, self.histoEQ_dab4(histoEQ_l4)], dim=1))
#            
#            layer1 = torch.cat([gamma_l1, underexposed_l1, overexposed_l1, original_l1, h_2_l1, histoEQ_l1], dim=1)
#            layer2 = torch.cat([gamma_l2, underexposed_l2, overexposed_l2, original_l2, h_2_l2, histoEQ_l2], dim=1)
#            layer3 = torch.cat([gamma_l3, underexposed_l3, overexposed_l3, original_l3, h_2_l3, histoEQ_l3], dim=1)
#            layer4 = torch.cat([gamma_l4, underexposed_l4, overexposed_l4, original_l4, h_2_l4, histoEQ_l4], dim=1)
#            
#            pff1 = self.pff_block_1(layer1, layer2)
#            pff2 = self.pff_block_2(layer1, layer2, layer3)
#            pff3 = self.pff_block_3(layer2, layer3, layer4)
#            #pff4 = self.pff_block_4(layer3, layer4)
#            pff4 = self.pff_block_4(layer4, layer3)
#            
#            output = self.reconstructed_image(pff1, pff2, pff3, pff4)
#
#            
#            return output
#
