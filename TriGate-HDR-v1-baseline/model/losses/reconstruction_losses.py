import torch


def gated_l2_loss(pred, target, gate):
    return ((gate * (pred - target) ** 2).mean())

