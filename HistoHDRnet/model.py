import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HistoHDRNet_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(HistoHDRNet_Encoder, self).__init__()
        
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
        return fused_features, f_gt, f_hi
