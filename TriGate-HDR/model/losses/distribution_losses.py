import torch


def _histogram(x, bins=64, min_val=0.0, max_val=1.0):
    x = x.clamp(min_val, max_val).reshape(-1)
    h = torch.histc(x, bins=bins, min=min_val, max=max_val)
    return h / (h.sum() + 1e-8)


def wasserstein_hist_loss(pred, target, clipped_gate, bins=64):
    loss = 0.0
    for c in range(pred.shape[1]):
        hp = _histogram(pred[:, c : c + 1], bins=bins)
        ht = _histogram(target[:, c : c + 1], bins=bins)
        cdf_p = torch.cumsum(hp, dim=0)
        cdf_t = torch.cumsum(ht, dim=0)
        loss = loss + torch.mean(torch.abs(cdf_p - cdf_t))
    return (clipped_gate.mean() * loss)

