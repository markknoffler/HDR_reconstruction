import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        
        resnet_gt = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        resnet_his = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.encoder_gt = nn.Sequential(
            resnet_gt.conv1,
            resnet_gt.bn1,
            resnet_gt.relu,
            resnet_gt.maxpool,
            resnet_gt.layer1,
            resnet_gt.layer2,
            resnet_gt.layer3,
            resnet_gt.layer4
        )
        
        self.encoder_his = nn.Sequential(
            resnet_his.conv1,
            resnet_his.bn1,
            resnet_his.relu,
            resnet_his.maxpool,
            resnet_his.layer1,
            resnet_his.layer2,
            resnet_his.layer3,
            resnet_his.layer4
        )
        
    def forward(self, ldr_gt, ldr_his):
        f_gt = self.encoder_gt(ldr_gt)
        f_his = self.encoder_his(ldr_his)
        fused_features = torch.cat([f_gt, f_his], dim=1)
        return fused_features

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels=4096):
        super(SpatialSelfAttention, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.scale = in_channels ** -0.5
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, C, H * W)
        K = K.view(B, C, H * W)
        V = V.view(B, C, H * W)
        
        Q = Q.permute(0, 2, 1)
        K = K.permute(0, 2, 1)
        V = V.permute(0, 2, 1)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2))
        
        attention_scores = attention_scores * self.scale
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        out = torch.bmm(attention_weights, V)
        
        out = out.permute(0, 2, 1)
        out = out.view(B, C, H, W)
        
        out = self.proj_out(out)
        
        return out

class ReconstructionUnit(nn.Module):
    def __init__(self, in_channels=4096, out_channels=3):
        super(ReconstructionUnit, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
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
    
    def forward(self, frs):
        x = self.up1(frs)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.reconstruction_net(x)
        return x


class HistoHDRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HistoHDRNet, self).__init__()

        self.encoder = Encoder(pretrained=pretrained)

        self.self_attention = SpatialSelfAttention(in_channels = 4096)

        self.reconstruction_unit = ReconstructionUnit(in_channels=4096, out_channels=3)
        #need to add upsampling before reconstruction_net

    def forward(self, ldr_gt, ldr_his):
        fused_features = self.encoder(ldr_gt, ldr_his)
        attention_weights = self.self_attention(fused_features)
        final_features = fused_features + attention_weights

        hdr_reconstructed = self.reconstruction_unit(final_features)

        return hdr_reconstructed

