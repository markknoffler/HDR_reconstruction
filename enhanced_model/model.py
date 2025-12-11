import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
        f3_ = self.enc:3(x3)
        f4_ = self.enc4(x4)
        f5_ = self.enc5(x5)
        f6_ = self.enc6(x6)

        f1_deep = f1_[-1]
        f2_deep = f2_[-1]
        f3_deep = f3_[-1]
        f4_deep = f4_[-1]
        f5_deep = f5_[-1]
        f6_deep = f6_[-1]

        return f1_deep. f2_deep, f3_deep, f4_deep, f5_deep, f6_deep

class DAB_block(nn.Module):
    def __init__(self):
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
    def __init__(self, ch_pre, ch_cur, ch_nex, dab_block_cls):
        super().__init__()
        self.down_dw = DepthwiseConv2D(ch_pre, kernel_size=2, stride=2)
        self.down_1x1 = nn.Conv2d(ch_pre, ch_cur, kernel_size=1, stride =1, padding=0)

        self.up_1x1 = nn.Conv2d(ch_nex, ch_curr, kernel_size=1, size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        concat_channels =  ch_curr * 2 
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_cur, kernel_size=1, stride=1, padding=0)

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
        self.down_dw = DepthwiseConv2D(ch_prev, kernel_size=2, stride=2)
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



