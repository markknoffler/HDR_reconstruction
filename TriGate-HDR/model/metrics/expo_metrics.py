"""
ExpoCM Table 1 metrics (expoCM_final_followup.pdf §4.1).

Columns: PSNR-μ, SSIM-μ, PSNR-PU, SSIM-PU, PSNR-l, SSIM-l, MS-SSIM,
         HDR-VDP-2, HDR-VDP-3, LPIPS, ΔE2000.

Domains:
  -l  : linear HDR in [0, 1] (relative radiance after (x+1)/2)
  -μ  : μ-law tonemap (μ=5000), FHDR/test.py
  -PU : PU21 banding_glare on peak-normalized cd/m² RGB + SI-HDR CRF correction
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..training_scripts.common_training import compute_psnr_ssim_fhdr, fhdr_compare_ssim, mu_tonemap
from .hdrvdp_official import tensor_pair_to_linear_rgb_cd_m2
from .pu21_encoding import compute_pu_psnr_ssim_pair, pu21_encode_absolute

# Back-compat aliases (tests / imports)
pu21_encode_rgb_cd_m2 = pu21_encode_absolute

_LPIPS_MODEL = None
_LPIPS_WARNED = False


@dataclass
class ExpoMetricVector:
    psnr_mu: float = 0.0
    ssim_mu: float = 0.0
    psnr_pu: float = 0.0
    ssim_pu: float = 0.0
    psnr_l: float = 0.0
    ssim_l: float = 0.0
    ms_ssim: float = 0.0
    hdrvdp2: float = 0.0
    hdrvdp3: float = 0.0
    lpips: float = 0.0
    delta_e2000: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @staticmethod
    def csv_header(prefix: str = "") -> List[str]:
        return [f"{prefix}{f.name}" for f in fields(ExpoMetricVector)]

    def csv_values(self) -> List[str]:
        return [f"{getattr(self, f.name):.6f}" for f in fields(ExpoMetricVector)]


def _chw_to_hwc01(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().numpy()
    return np.clip((np.transpose(x, (1, 2, 0)) + 1.0) * 0.5, 0.0, 1.0)


def _mse_psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((peak * peak) / mse))


def pu21_encode_rgb_cd_m2(rgb_cd_m2: np.ndarray) -> np.ndarray:
    """PU21 banding_glare encode on HxWx3 cd/m² (official gfxdisp/pu21 params)."""
    return pu21_encode_absolute(rgb_cd_m2)


def _pu21_from_tensors(pred: torch.Tensor, gt: torch.Tensor, display_peak: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    pred_cd, gt_cd = tensor_pair_to_linear_rgb_cd_m2(pred, gt, display_peak=display_peak)
    return pred_cd, gt_cd


def _ms_ssim_hwc(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """5-scale MS-SSIM (Wang 2003) on HWC float RGB."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    levels = 5
    mssim = 1.0
    x1, x2 = img1.astype(np.float64), img2.astype(np.float64)
    for i in range(levels):
        ssim_val = fhdr_compare_ssim(
            np.clip(x1, 0.0, data_range),
            np.clip(x2, 0.0, data_range),
        )
        mssim *= max(ssim_val, 1e-8) ** weights[i]
        if i < levels - 1:
            x1 = x1[::2, ::2, :]
            x2 = x2[::2, ::2, :]
            if min(x1.shape[0], x1.shape[1]) < 7:
                break
    return float(mssim)


def _srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB linear [0,1] -> CIE Lab (D65)."""
    x = rgb01.astype(np.float64)
    mask = x <= 0.04045
    linear = np.empty_like(x)
    linear[mask] = x[mask] / 12.92
    linear[~mask] = np.power((x[~mask] + 0.055) / 1.055, 2.4)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float64,
    )
    xyz = linear @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Mean CIEDE2000 over pixels (Sharma et al. 2005)."""
    l1, a1, b1 = [lab1[..., i] for i in range(3)]
    l2, a2, b2 = [lab2[..., i] for i in range(3)]
    avg_lp = 0.5 * (l1 + l2)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    avg_c = 0.5 * (c1 + c2)
    g = 0.5 * (1.0 - np.sqrt(np.power(avg_c, 7) / (np.power(avg_c, 7) + np.power(25.0, 7))))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    avg_cp = 0.5 * (c1p + c2p)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((c1p * c2p) < 1e-8, 0.0, dhp)
    dhp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2.0))
    avg_hp = np.where(np.abs(h1p - h2p) > 180.0, 0.5 * (h1p + h2p + 360.0), 0.5 * (h1p + h2p))
    avg_hp = np.where((c1p * c2p) < 1e-8, h1p + h2p, avg_hp)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )
    sl = 1.0 + (0.015 * np.square(avg_lp - 50.0)) / np.sqrt(20.0 + np.square(avg_lp - 50.0))
    sc = 1.0 + 0.045 * avg_cp
    sh = 1.0 + 0.015 * avg_cp * t
    rt = (
        -2.0
        * np.sqrt(np.power(avg_cp, 7) / (np.power(avg_cp, 7) + np.power(25.0, 7)))
        * np.sin(np.radians(60.0 * np.exp(-np.square((avg_hp - 275.0) / 25.0))))
    )
    de = np.sqrt(
        np.square(dlp / sl)
        + np.square(dcp / sc)
        + np.square(dhp / sh)
        + rt * (dcp / sc) * (dhp / sh)
    )
    return float(np.mean(de))


def _get_lpips():
    global _LPIPS_MODEL, _LPIPS_WARNED
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # type: ignore

        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad = False
        return _LPIPS_MODEL
    except Exception as exc:
        if not _LPIPS_WARNED:
            print(f"WARNING: LPIPS unavailable ({exc}); reporting 0.0. pip install lpips")
            _LPIPS_WARNED = True
        return None


def _finite(x: float, default: float = 0.0) -> float:
    x = float(x)
    return x if np.isfinite(x) else default


def compute_expo_metrics_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    hdrvdp_calculator=None,
) -> ExpoMetricVector:
    """
    All Table 1 metrics for one CHW pair in [-1, 1].
    """
    pred = pred.detach().float()
    gt = gt.detach().float()

    psnr_mu, ssim_mu = compute_psnr_ssim_fhdr(pred, gt)

    lin_p = _chw_to_hwc01(pred)
    lin_g = _chw_to_hwc01(gt)
    psnr_l = _mse_psnr(lin_p, lin_g, peak=1.0)
    ssim_l = fhdr_compare_ssim(lin_p, lin_g)

    pu_pred_cd, pu_gt_cd = _pu21_from_tensors(pred, gt)
    psnr_pu, ssim_pu = compute_pu_psnr_ssim_pair(pu_pred_cd, pu_gt_cd, apply_crf=True)

    mu_p = mu_tonemap(pred).cpu().numpy()
    mu_g = mu_tonemap(gt).cpu().numpy()
    mu_p_hwc = np.transpose(mu_p, (1, 2, 0))
    mu_g_hwc = np.transpose(mu_g, (1, 2, 0))
    ms_ssim = _ms_ssim_hwc(
        np.clip((mu_p_hwc + 1) * 0.5, 0, 1),
        np.clip((mu_g_hwc + 1) * 0.5, 0, 1),
    )

    h2, h3 = 0.0, 0.0
    if hdrvdp_calculator is not None:
        h2 = float(hdrvdp_calculator.compute_hdrvdp2(pred, gt))
        h3 = float(hdrvdp_calculator.compute_hdrvdp3(pred, gt))

    lpips_val = 0.0
    model = _get_lpips()
    if model is not None:
        # LPIPS on μ-law displayable [0,1] -> [-1,1]
        mp = torch.from_numpy(np.clip((mu_p_hwc + 1) * 0.5, 0, 1)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        mg = torch.from_numpy(np.clip((mu_g_hwc + 1) * 0.5, 0, 1)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_val = float(model(mp, mg).mean().item())

    lab_p = _srgb_to_lab(lin_p)
    lab_g = _srgb_to_lab(lin_g)
    de00 = _delta_e2000(lab_p, lab_g)

    return ExpoMetricVector(
        psnr_mu=_finite(psnr_mu),
        ssim_mu=_finite(ssim_mu),
        psnr_pu=_finite(psnr_pu),
        ssim_pu=_finite(ssim_pu),
        psnr_l=_finite(psnr_l),
        ssim_l=_finite(ssim_l),
        ms_ssim=_finite(ms_ssim),
        hdrvdp2=_finite(h2),
        hdrvdp3=_finite(h3),
        lpips=_finite(lpips_val),
        delta_e2000=_finite(de00),
    )


def average_expo_metrics(vectors: List[ExpoMetricVector]) -> ExpoMetricVector:
    if not vectors:
        return ExpoMetricVector()
    keys = [f.name for f in fields(ExpoMetricVector)]
    out = {}
    for k in keys:
        vals = [getattr(v, k) for v in vectors if np.isfinite(getattr(v, k))]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return ExpoMetricVector(**out)


def format_expo_table_row(m: ExpoMetricVector) -> str:
    return (
        f"PSNR-μ={m.psnr_mu:.2f} SSIM-μ={m.ssim_mu:.4f} | "
        f"PSNR-PU={m.psnr_pu:.2f} SSIM-PU={m.ssim_pu:.4f} | "
        f"PSNR-l={m.psnr_l:.2f} SSIM-l={m.ssim_l:.4f} MS-SSIM={m.ms_ssim:.4f} | "
        f"VDP2={m.hdrvdp2:.2f} VDP3={m.hdrvdp3:.2f} | "
        f"LPIPS={m.lpips:.4f} ΔE00={m.delta_e2000:.2f}"
    )
