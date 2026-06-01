"""
Verify TriGate compute_psnr_ssim matches FHDR/test.py line-for-line on the same tensors.

Run from TriGate-HDR/:
  PYTHONPATH=$(pwd) python -m model.training_scripts.verify_fhdr_metrics
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import compare_ssim

from .common_training import compute_psnr_ssim, compute_psnr_ssim_fhdr, mse_loss, mu_tonemap


def _fhdr_test_py_inline(pred: torch.Tensor, gt: torch.Tensor):
    """Copy of FHDR/test.py metric block for one CHW pair."""
    mu_tonemap_gt = mu_tonemap(gt)
    mse = mse_loss(mu_tonemap(pred), mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())
    generated = (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = compare_ssim(generated, real, multichannel=True)
    return float(psnr), float(ssim)


def _assert_close(a: float, b: float, name: str, tol: float = 1e-6):
    if not np.isfinite(a) or not np.isfinite(b):
        raise AssertionError(f"{name}: non-finite {a} vs {b}")
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} != {b} (diff {abs(a - b)})")


def main():
    torch.manual_seed(0)
    h, w = 64, 64
    cases = [
        ("random", torch.randn(3, h, w).clamp(-1, 1), torch.randn(3, h, w).clamp(-1, 1)),
        ("identical", torch.rand(3, h, w) * 2 - 1, torch.rand(3, h, w) * 2 - 1),
        ("offset", torch.rand(3, h, w) * 2 - 1, torch.rand(3, h, w) * 2 - 1),
    ]
    cases[1] = (cases[1][0], cases[1][1].clone())
    pred_o, gt_o = cases[2][1], cases[2][2]
    cases[2] = ("offset", pred_o, (pred_o + 0.15 * torch.randn_like(pred_o)).clamp(-1, 1))

    for name, pred, gt in cases:
        ref = _fhdr_test_py_inline(pred, gt)
        a = compute_psnr_ssim_fhdr(pred, gt)
        b = compute_psnr_ssim(pred, gt)
        for label, got in (("fhdr_fn", a), ("wrapper", b)):
            _assert_close(ref[0], got[0], f"{name} PSNR {label}")
            _assert_close(ref[1], got[1], f"{name} SSIM {label}")
        print(f"OK {name}: PSNR={ref[0]:.6f} SSIM={ref[1]:.6f}")

    print("All FHDR metric checks passed.")


if __name__ == "__main__":
    main()
