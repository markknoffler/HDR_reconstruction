import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np

class TransformationUnit(nn.Module):
    def __init__(self):
        super(TransformationUnit, self).__init__()
    
    def forward(self, ldr_input):
        ldr_ev0 = ldr_input
        
        ldr_ev_minus2 = torch.pow(ldr_input, 2.0)  
        
        ldr_ev_plus2 = torch.pow(ldr_input, 0.5)  
        
        ldr_ev_minus2 = torch.clamp(ldr_ev_minus2, 0.0, 1.0)
        ldr_ev_plus2 = torch.clamp(ldr_ev_plus2, 0.0, 1.0)
        
        return ldr_ev_minus2, ldr_ev0, ldr_ev_plus2

class FeatureUnit(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(FeatureUnit, self).__init__()
        self.base_channels = base_channels
        
        self.branch_minus2 = self._make_branch(in_channels)
        self.branch_0 = self._make_branch(in_channels)
        self.branch_plus2 = self._make_branch(in_channels)
    
    def _make_branch(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, ldr_ev_minus2, ldr_ev0, ldr_ev_plus2):
        fe_minus2 = self.branch_minus2(ldr_ev_minus2)
        fe_0 = self.branch_0(ldr_ev0)
        fe_plus2 = self.branch_plus2(ldr_ev_plus2)
        
        fe_all = fe_minus2 + fe_0 + fe_plus2
        
        return fe_minus2, fe_0, fe_plus2, fe_all

class DilatedDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, dilation_rate=3):
        super(DilatedDenseBlock, self).__init__()
        self.growth_rate = growth_rate
        
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=dilation_rate, 
                              dilation=dilation_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        
        self.compression_in = nn.Conv2d(in_channels, in_channels, 1)
        self.compression_out = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_compressed = self.compression_in(x)
        
        out1 = self.relu(self.conv1(x_compressed))
        concat1 = torch.cat([x_compressed, out1], 1)
        
        out2 = self.relu(self.conv2(concat1))
        concat2 = torch.cat([concat1, out2], 1)
        
        out3 = self.relu(self.conv3(concat2))
        concat3 = torch.cat([concat2, out3], 1)
        
        out4 = self.relu(self.conv4(concat3))
        concat4 = torch.cat([concat3, out4], 1)
        
        output = self.compression_out(concat4)
        
        return output

class FeedbackUnit(nn.Module):
    def __init__(self, in_channels=64, num_blocks=3, num_iterations=4):
        super(FeedbackUnit, self).__init__()
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        
        self.initial_compression = nn.Conv2d(in_channels, in_channels, 1)
        
        self.dilated_blocks = nn.ModuleList([
            DilatedDenseBlock(in_channels, growth_rate=32, dilation_rate=3)
            for _ in range(num_blocks)
        ])
        
        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.hidden_state = None
    
    def reset_hidden_state(self):
        self.hidden_state = None
    
    def forward(self, fe_all):
        if self.hidden_state is None:
            self.hidden_state = fe_all
        else:
            self.hidden_state = fe_all + self.hidden_state
            #self.hidden_state = self.hidden_state
        
        x = self.initial_compression(self.hidden_state)
        
        for block in self.dilated_blocks:
            x = block(x)
        
        fb_output = self.relu(self.final_conv(x))
        
        self.hidden_state = fb_output
        
        return fb_output

class ReconstructionUnit(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(ReconstructionUnit, self).__init__()
        
        self.reconstruction_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Tanh()  # Output in range [-1, 1] for HDR
        )
    
    def forward(self, frs):
        return self.reconstruction_net(frs)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        features = vgg.features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(2):  # relu1_1, relu1_2
            self.slice1.add_module(str(x), features[x])
        for x in range(2, 7):  # up to relu2_2
            self.slice2.add_module(str(x), features[x])
        for x in range(7, 12):  # up to relu3_4
            self.slice3.add_module(str(x), features[x])
        for x in range(12, 21):  # up to relu4_4
            self.slice4.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x_features = []
        y_features = []
        
        for slice_module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = slice_module(x)
            y = slice_module(y)
            x_features.append(x)
            y_features.append(y)
        
        perceptual_loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            perceptual_loss += F.l1_loss(x_feat, y_feat)
        
        return perceptual_loss

class ArtHDRNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_iterations=4):
        super(ArtHDRNet, self).__init__()
        self.num_iterations = num_iterations
        
        self.transformation_unit = TransformationUnit()
        self.feature_unit = FeatureUnit(in_channels, base_channels)
        self.feedback_unit = FeedbackUnit(base_channels, num_iterations=num_iterations)
        self.reconstruction_unit = ReconstructionUnit(base_channels, in_channels)
        
        self.fe0_1 = None  
        self.fe0_2 = None  
    
    def extract_mid_branch_features(self, ldr_ev0):
        x = self.feature_unit.branch_0[0](ldr_ev0)  
        x = self.feature_unit.branch_0[1](x)
        self.fe0_1 = x
        
        x = self.feature_unit.branch_0[2](x)  
        x = self.feature_unit.branch_0[3](x)  
        self.fe0_2 = x
    
    def forward(self, ldr_input):
        self.feedback_unit.reset_hidden_state()
        
        ldr_ev_minus2, ldr_ev0, ldr_ev_plus2 = self.transformation_unit(ldr_input)
        
        fe_minus2, fe_0, fe_plus2, fe_all = self.feature_unit(
            ldr_ev_minus2, ldr_ev0, ldr_ev_plus2
        )
        
        self.extract_mid_branch_features(ldr_ev0)
        
        hdr_outputs = []
        
        for t in range(self.num_iterations):
            fb_t = self.feedback_unit(fe_all)
            
            frs_t = self.fe0_1 + self.fe0_2 + fb_t
            
            hdr_t = self.reconstruction_unit(frs_t)
            hdr_outputs.append(hdr_t)
        
        return hdr_outputs

class ArtHDRLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.5, mu=5000):
        super(ArtHDRLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
    
#    def mu_law_compression(self, hdr):
#        return torch.log(1 + self.mu * hdr) / torch.log(1 + torch.tensor(self.mu).to(hdr.device))
#
    def mu_law_compression(self, hdr):
    # Denormalize from [-1,1] to [0,1] first
        hdr_positive = (hdr + 1.0) / 2.0  # Maps [-1,1] to [0,1]
        hdr_positive = torch.clamp(hdr_positive, min=0.0)  # Ensure non-negative
        return torch.log(1 + self.mu * hdr_positive) / torch.log(1 + torch.tensor(self.mu).to(hdr.device))

    
    def forward(self, hdr_outputs, hdr_gt):
        total_loss = 0
        
        hdr_gt_tm = self.mu_law_compression(hdr_gt)
        
        for t, hdr_pred in enumerate(hdr_outputs):
            hdr_pred_tm = self.mu_law_compression(hdr_pred)
            
            l1_loss = self.l1_loss(hdr_pred_tm, hdr_gt_tm)
            
            perceptual_loss = self.perceptual_loss(hdr_pred_tm, hdr_gt_tm)
            
            iteration_loss = self.lambda1 * l1_loss + self.lambda2 * perceptual_loss
            total_loss += iteration_loss
        
        return total_loss

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

