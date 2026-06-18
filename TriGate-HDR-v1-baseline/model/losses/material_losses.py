import torch
import torch.nn.functional as F


def _gram(x):
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def material_consistency_loss(pred, ldr, gate):
    grad_p = torch.gradient(pred.mean(dim=1, keepdim=True), dim=(2, 3))
    grad_l = torch.gradient(ldr.mean(dim=1, keepdim=True), dim=(2, 3))
    gram_loss = F.mse_loss(_gram(grad_p[0]), _gram(grad_l[0])) + F.mse_loss(_gram(grad_p[1]), _gram(grad_l[1]))
    hist_loss = F.l1_loss(F.avg_pool2d(pred, 5, 1, 2), F.avg_pool2d(ldr, 5, 1, 2))
    return gate.mean() * (gram_loss + hist_loss)

