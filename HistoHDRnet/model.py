import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HistoHDRNet_Encoder(nn.Module):
    """
    Encoder part of HistoHDR-Net with dual ResNet50 backbones
    
    Architecture:
    - Two parallel ResNet50 encoders
    - One processes original LDR (LDRGT)
    - One processes histogram-equalized LDR (LDRHis)
    - Features are concatenated along channel dimension
    """
    def __init__(self, pretrained=True):
        super(HistoHDRNet_Encoder, self).__init__()
        
        # Load pretrained ResNet50 for original LDR images (LDRGT)
        resnet_gt = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Load pretrained ResNet50 for histogram-equalized LDR images (LDRHis)
        resnet_his = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extract feature extraction layers (removing FC layers)
        # ResNet50 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        
        # Encoder for LDRGT (original LDR)
        self.encoder_gt = nn.Sequential(
            resnet_gt.conv1,      # 3 -> 64 channels, stride=2
            resnet_gt.bn1,
            resnet_gt.relu,
            resnet_gt.maxpool,    # stride=2
            resnet_gt.layer1,     # 64 -> 256 channels
            resnet_gt.layer2,     # 256 -> 512 channels, stride=2
            resnet_gt.layer3,     # 512 -> 1024 channels, stride=2
            resnet_gt.layer4      # 1024 -> 2048 channels, stride=2
        )
        
        # Encoder for LDRHis (histogram-equalized LDR)
        self.encoder_his = nn.Sequential(
            resnet_his.conv1,     # 3 -> 64 channels, stride=2
            resnet_his.bn1,
            resnet_his.relu,
            resnet_his.maxpool,   # stride=2
            resnet_his.layer1,    # 64 -> 256 channels
            resnet_his.layer2,    # 256 -> 512 channels, stride=2
            resnet_his.layer3,    # 512 -> 1024 channels, stride=2
            resnet_his.layer4     # 1024 -> 2048 channels, stride=2
        )
        
    def forward(self, ldr_gt, ldr_his):
        """
        Forward pass through dual encoders and concatenation
        
        Args:
            ldr_gt: Original LDR image, shape (B, 3, H, W)
            ldr_his: Histogram-equalized LDR image, shape (B, 3, H, W)
        
        Returns:
            fused_features: Concatenated features, shape (B, 4096, H/32, W/32)
            f_gt: Features from LDRGT encoder, shape (B, 2048, H/32, W/32)
            f_his: Features from LDRHis encoder, shape (B, 2048, H/32, W/32)
        
        Dimension Flow (for 512x512 input):
            Input:  (B, 3, 512, 512)
            ↓ conv1 + maxpool
            (B, 64, 128, 128)
            ↓ layer1
            (B, 256, 128, 128)
            ↓ layer2
            (B, 512, 64, 64)
            ↓ layer3
            (B, 1024, 32, 32)
            ↓ layer4
            (B, 2048, 16, 16)  ← Output features
        """
        # Extract features from original LDR
        # This implements: f_GT_LDR = Encoder_GT(LDRGT)
        f_gt = self.encoder_gt(ldr_gt)  # Shape: (B, 2048, H/32, W/32)
        
        # Extract features from histogram-equalized LDR
        # This implements: f_His_LDR = Encoder_His(LDRHis)
        f_his = self.encoder_his(ldr_his)  # Shape: (B, 2048, H/32, W/32)
        
        # Concatenate along channel dimension
        # This implements Equation 1 from paper: f_fuse = f_GT_LDR ⊕ f_His_LDR
        fused_features = torch.cat([f_gt, f_his], dim=1)  # Shape: (B, 4096, H/32, W/32)
        
        return fused_features, f_gt, f_his
