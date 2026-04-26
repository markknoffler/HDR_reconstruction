import torch


def _histogram(x, bins=64, min_val=0.0, max_val=1.0):
    x = x.clamp(min_val, max_val).reshape(-1)
    h = torch.histc(x, bins=bins, min=min_val, max=max_val)
    return h / (h.sum() + 1e-8)


def wasserstein_hist_loss(pred, target, clipped_gate, class_probs=None, class_masks=None, bins=64):
    loss = 0.0
    class_tensor = class_masks if class_masks is not None else class_probs
    if class_tensor is None:
        for c in range(pred.shape[1]):
            hp = _histogram(pred[:, c : c + 1], bins=bins)
            ht = _histogram(target[:, c : c + 1], bins=bins)
            cdf_p = torch.cumsum(hp, dim=0)
            cdf_t = torch.cumsum(ht, dim=0)
            loss = loss + torch.mean(torch.abs(cdf_p - cdf_t))
        return clipped_gate.mean() * loss

    num_classes = class_tensor.shape[1]
    for cls_idx in range(num_classes):
        cls_mask = class_tensor[:, cls_idx : cls_idx + 1] * clipped_gate
        cls_weight = cls_mask.mean()
        if cls_weight.item() <= 1e-8:
            continue
        hp = _histogram(pred * cls_mask, bins=bins)
        ht = _histogram(target * cls_mask, bins=bins)
        cdf_p = torch.cumsum(hp, dim=0)
        cdf_t = torch.cumsum(ht, dim=0)
        loss = loss + cls_weight * torch.mean(torch.abs(cdf_p - cdf_t))
    return loss

