import torch
import torch.nn.functional as F


def sobel_grad_mag(x):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gray = x.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8), gx, gy


def structural_fidelity_loss(pred, target, lambda_edge=0.1):
    gp, _, _ = sobel_grad_mag(pred)
    gt, _, _ = sobel_grad_mag(target)
    edge = F.l1_loss((gp > 0.1).float(), (gt > 0.1).float())
    return F.l1_loss(gp, gt) + lambda_edge * edge


def seam_gradient_continuity_loss(pred, target, gate):
    _, gpx, gpy = sobel_grad_mag(pred)
    _, gtx, gty = sobel_grad_mag(target)
    boundary = gate * (1.0 - gate) * 4.0
    return (boundary * (torch.abs(gpx - gtx) + torch.abs(gpy - gty))).mean()

