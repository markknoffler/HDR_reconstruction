import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models


class ToneMapper(nn.Module):
    def __init__(self, mu=5000.0):
        super(ToneMapper, self).__init__()
        self.mu = mu
        self.log_mu_plus_1 = np.log(1.0 + mu)  # Pre-compute constant
    
    def forward(self, hdr):
        # Clamp to safe range FIRST
        hdr = torch.clamp((hdr + 1.0) / 2.0, 0.0, 1.0)
        # Use log1p for numerical stability
        tone_mapped = torch.log1p(self.mu * hdr) / self.log_mu_plus_1
        return tone_mapped


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred, target):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred_vgg = self.vgg(pred)
        target_vgg = self.vgg(target)
        loss = 0
        for i in range(len(pred_vgg)):
            loss += self.weights[i] * self.criterion(pred_vgg[i], target_vgg[i].detach())
        return loss


class WeberPSNRLoss(nn.Module):
    def __init__(self, bit_depth=8, weber_fraction=0.02):
        super(WeberPSNRLoss, self).__init__()
        self.bit_depth = bit_depth
        self.max_val = 2 ** bit_depth - 1
        self.weber_fraction = weber_fraction
        self.eps = 1e-6
    
    def forward(self, pred, target):
        pred_scaled = torch.clamp(pred * self.max_val, 0, self.max_val)
        target_scaled = torch.clamp(target * self.max_val, 0, self.max_val)
        
        weber_weights = (self.weber_fraction * self.max_val) / (target_scaled + self.eps)
        weber_weights = torch.clamp(weber_weights, 0, 100)  # Prevent explosion
        
        squared_diff = (target_scaled - pred_scaled) ** 2
        weighted_squared_error = (weber_weights ** 2) * squared_diff
        
        mse_weighted = torch.mean(weighted_squared_error, dim=[1, 2, 3])
        # Clamp mse to at most max_val**2 so that psnr is always >= 0
        mse_weighted = torch.clamp(mse_weighted, max=self.max_val ** 2)
        mse_weighted = torch.clamp(mse_weighted, min=self.eps)  # avoid log(0)
        
        psnr_w = 10.0 * torch.log10((self.max_val ** 2) / mse_weighted)  # now always >= 0
        loss = torch.mean(1.0 / (psnr_w + self.eps))
        
        return loss


class MS_SSIM_Loss(nn.Module):
    def __init__(self, max_val=1.0, k1=0.01, k2=0.03, scales=5):
        super(MS_SSIM_Loss, self).__init__()
        self.max_val = max_val
        self.k1 = k1
        self.k2 = k2
        self.scales = scales
        self.C1 = (k1 * max_val) ** 2
        self.C2 = (k2 * max_val) ** 2
    
    def _ssim(self, pred, target):
        mu_pred = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
        mu_target = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = F.avg_pool2d(pred ** 2, kernel_size=11, stride=1, padding=5) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, kernel_size=11, stride=1, padding=5) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=11, stride=1, padding=5) - mu_pred_target
        luminance = (2 * mu_pred_target + self.C1) / (mu_pred_sq + mu_target_sq + self.C1)
        cs = (2 * sigma_pred_target + self.C2) / (sigma_pred_sq + sigma_target_sq + self.C2)
        return luminance, cs
    
    def forward(self, pred, target):
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(pred.device)
        ms_ssim = 1.0
        for i in range(self.scales):
            if i > 0:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
            luminance, cs = self._ssim(pred, target)
            
            # Clamp cs_mean to be non-negative to avoid NaN when raising to fractional power
            cs_mean = torch.mean(cs).clamp(min=0.0)
            
            if i < self.scales - 1:
                ms_ssim *= cs_mean ** weights[i]
            else:
                lum_mean = torch.mean(luminance)
                ms_ssim *= (lum_mean ** weights[i]) * (cs_mean ** weights[i])
        return 1.0 - ms_ssim


class ColorLoss(nn.Module):
    def __init__(self, bit_depth=8):
        super(ColorLoss, self).__init__()
        self.max_val = 2 ** bit_depth - 1
    
    def forward(self, pred, target):
        pred_scaled = pred * self.max_val
        target_scaled = target * self.max_val
        squared_diff = (pred_scaled - target_scaled) ** 2
        loss = torch.mean(squared_diff) / (self.max_val ** 2)
        return loss


class EnhancedModelLoss(nn.Module):
    def __init__(self, alpha=0.18, beta=0.5, gamma=0.82, delta=0.80, epsilon=0.82, mu=5000.0):
        super(EnhancedModelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tone_mapper = ToneMapper(mu=mu)
        self.l1_loss = L1Loss()
        self.vgg_loss = VGGPerceptualLoss()
        self.weber_loss = WeberPSNRLoss(bit_depth=8, weber_fraction=0.02)
        self.ms_ssim_loss = MS_SSIM_Loss(max_val=1.0, k1=0.01, k2=0.03, scales=5)
        self.color_loss = ColorLoss(bit_depth=8)
    
    def forward(self, hdr_pred, hdr_gt):
        pred_tm = self.tone_mapper(hdr_pred)
        gt_tm = self.tone_mapper(hdr_gt)
        loss_l1 = self.l1_loss(pred_tm, gt_tm)
        loss_vgg = self.vgg_loss(pred_tm, gt_tm)
        loss_weber = self.weber_loss(pred_tm, gt_tm)
        loss_ms_ssim = self.ms_ssim_loss(pred_tm, gt_tm)
        loss_color = self.color_loss(pred_tm, gt_tm)
        total_loss = (
            self.alpha * loss_l1 +
            self.beta * loss_vgg +
            self.gamma * loss_weber +
            self.delta * loss_ms_ssim +
            self.epsilon * loss_color
        )
        loss_dict = {
            'total': total_loss.item(),
            'l1': loss_l1.item(),
            'vgg': loss_vgg.item(),
            'weber': loss_weber.item(),
            'ms_ssim': loss_ms_ssim.item(),
            'color': loss_color.item()
        }
        return total_loss, loss_dict
