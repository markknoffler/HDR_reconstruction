import torch


def kl_codebook_loss(mus, logvars):
    loss = 0.0
    for mu, logvar in zip(mus, logvars):
        loss = loss + (-0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp()))
    return loss

