import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRadiometricConsistencyLoss(nn.Module):
    def __init__(
        self,
        alpha_crf=1.0,
        alpha_log=0.6,
        alpha_exp=0.4,
        alpha_rolloff=0.3,
        gamma=2.2,
        eps=1e-6,
    ):
        super().__init__()
        self.alpha_crf = alpha_crf
        self.alpha_log = alpha_log
        self.alpha_exp = alpha_exp
        self.alpha_rolloff = alpha_rolloff
        self.gamma = gamma
        self.eps = eps

    def _inverse_crf(self, x):
        x = x.clamp(0.0, 1.0)
        return torch.pow(x + self.eps, self.gamma)

    def _forward_crf(self, x):
        x = x.clamp(min=0.0)
        return torch.pow(x + self.eps, 1.0 / self.gamma).clamp(0.0, 1.0)

    def forward(self, pred_hdr, gt_hdr, ldr, gate):
        pred_hdr_lin = pred_hdr.clamp(min=self.eps, max=1.0)
        gt_hdr_lin = gt_hdr.clamp(min=self.eps, max=1.0)
        ldr_in = ldr.clamp(0.0, 1.0)
        non_clip = gate
        clip = 1.0 - gate

        pred_ldr = self._forward_crf(pred_hdr_lin)
        gt_ldr = self._forward_crf(gt_hdr_lin)
        crf_cycle = (non_clip * torch.abs(pred_ldr - ldr_in)).mean() + (non_clip * torch.abs(pred_ldr - gt_ldr)).mean()

        log_pred = torch.log(pred_hdr_lin + self.eps)
        log_gt = torch.log(gt_hdr_lin + self.eps)
        log_term = (non_clip * torch.abs(log_pred - log_gt)).mean()

        pred_ratio = torch.log((pred_hdr_lin[:, :, :, 1:] + self.eps) / (pred_hdr_lin[:, :, :, :-1] + self.eps))
        gt_ratio = torch.log((gt_hdr_lin[:, :, :, 1:] + self.eps) / (gt_hdr_lin[:, :, :, :-1] + self.eps))
        exp_term = torch.abs(pred_ratio - gt_ratio).mean()

        sat_mask = (ldr_in.max(dim=1, keepdim=True).values > 0.95).float()
        rolloff = (clip * sat_mask * torch.abs(torch.tanh(pred_hdr_lin) - torch.tanh(gt_hdr_lin))).mean()

        total = self.alpha_crf * crf_cycle + self.alpha_log * log_term + self.alpha_exp * exp_term + self.alpha_rolloff * rolloff
        return total, {
            "radiometric_total": total,
            "crf_cycle": crf_cycle,
            "log_term": log_term,
            "exp_term": exp_term,
            "rolloff": rolloff,
        }

