import torch
import torch.nn.functional as F


def luminance(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def stage1_diffusion_losses(
    pred_x0,
    target_x0,
    x_t,
    noise_lum,
    diffusion_module,
    t,
    w_x0=1.0,
    w_eps=1.0,
):
    """
    Standard Gaussian diffusion supervision (DDPM-style):
      - pixel L1 on predicted clean HDR (x0)
      - MSE on luminance noise epsilon implied by (x_t, pred_x0)
    """
    loss_x0 = F.l1_loss(pred_x0, target_x0)

    y0_pred = luminance(pred_x0.clamp(-1.0, 1.0))
    eps_pred = diffusion_module.estimate_noise_from_x0(x_t, y0_pred, t)
    loss_eps = F.mse_loss(eps_pred, noise_lum)

    total = w_x0 * loss_x0 + w_eps * loss_eps
    return total, {"loss_x0": loss_x0, "loss_eps": loss_eps, "loss_diff": total}
