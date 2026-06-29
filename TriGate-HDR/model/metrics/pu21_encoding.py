"""
Official PU21 banding_glare encoding + SI-HDR CRF correction (gfxdisp/pu21).

Reference:
  - pu21_encoder.m / pu21_metric.m (banding_glare params, peak=256, SSIM on Y)
  - crf_correction.m (SI-HDR evaluation; enabled for Table 1 PU metrics)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# gfxdisp/pu21 matlab/pu21_encoder.m — banding_glare (2020-02-06 params)
PU21_BANDING_GLARE_PARAMS = np.array(
    [
        0.353487901,
        0.3734658629,
        8.277049286e-05,
        0.9062562627,
        0.09150303166,
        0.9099517204,
        596.3148142,
    ],
    dtype=np.float64,
)

PU21_L_MIN = 0.005
PU21_L_MAX = 10000.0
# pu21_metric.m uses psnr(..., 256) and ssim(..., 'DynamicRange', 256)
PU21_METRIC_PEAK = 256.0

_BT709_TO_XYZ = np.array(
    [
        [0.412424, 0.357579, 0.180464],
        [0.212656, 0.715158, 0.072186],
        [0.019333, 0.119193, 0.950444],
    ],
    dtype=np.float64,
)
_XYZ_TO_BT709 = np.linalg.inv(_BT709_TO_XYZ)


def pu21_encode_absolute(linear_rgb_cd_m2: np.ndarray) -> np.ndarray:
    """Encode HxWx3 linear RGB in cd/m² to PU21 banding_glare codes."""
    p = PU21_BANDING_GLARE_PARAMS
    y = np.clip(linear_rgb_cd_m2.astype(np.float64), PU21_L_MIN, PU21_L_MAX)
    inner = (p[0] + p[1] * np.power(y, p[3])) / (1.0 + p[2] * np.power(y, p[3]))
    encoded = np.maximum(p[6] * (np.power(inner, p[4]) - p[5]), 0.0)
    return encoded.astype(np.float32)


def pu21_encode_luminance(luminance_cd_m2: np.ndarray) -> np.ndarray:
    """Encode HxW luminance (cd/m²) to PU21 codes."""
    return pu21_encode_absolute(luminance_cd_m2[..., None]).squeeze(-1)


def linear_rgb_to_luminance(linear_rgb_cd_m2: np.ndarray) -> np.ndarray:
    """BT.709 luminance from linear cd/m² RGB (pu21_metric get_luminance)."""
    rgb = linear_rgb_cd_m2.astype(np.float64)
    return (
        0.212656 * rgb[..., 0]
        + 0.715158 * rgb[..., 1]
        + 0.072186 * rgb[..., 2]
    )


def _rgb_to_luv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float64), 1e-6, None)
    linear = np.empty_like(rgb)
    for c in range(3):
        linear[..., c] = rgb[..., c]
    xyz = linear @ _BT709_TO_XYZ.T
    xyz[..., 0] = np.clip(xyz[..., 0], 1e-4, 1e8)
    xyz[..., 1] = np.clip(xyz[..., 1], 1e-4, 1e8)
    xyz[..., 2] = np.clip(xyz[..., 2], 1e-4, 1e8)
    s = np.sum(xyz, axis=-1, keepdims=True)
    x = xyz[..., 0:1] / s
    y = xyz[..., 1:2] / s
    luv = np.zeros_like(rgb)
    luv[..., 0] = xyz[..., 1]
    luv[..., 1] = 4.0 * x[..., 0] / (-2.0 * x[..., 0] + 12.0 * y[..., 0] + 3.0) * (410.0 / 255.0)
    luv[..., 2] = 9.0 * y[..., 0] / (-2.0 * x[..., 0] + 12.0 * y[..., 0] + 3.0) * (410.0 / 255.0)
    return luv


def _luv_to_rgb(luv: np.ndarray) -> np.ndarray:
    luv = luv.astype(np.float64)
    l_ = np.clip(luv[..., 0], 1e-4, 1e8)
    u = luv[..., 1] * 255.0 / 410.0
    v = luv[..., 2] * 255.0 / 410.0
    x = 9.0 * u / (6.0 * u - 16.0 * v + 12.0)
    y = 4.0 * v / (6.0 * u - 16.0 * v + 12.0)
    big_y = l_
    big_x = np.clip((x / np.maximum(y, 1e-8)) * big_y, 1e-4, 1e8)
    big_z = np.clip(((1.0 - x - y) / np.maximum(y, 1e-8)) * big_y, 1e-4, 1e8)
    xyz = np.stack([big_x, big_y, big_z], axis=-1)
    rgb = xyz @ _XYZ_TO_BT709.T
    return np.clip(rgb, PU21_L_MIN, None)


def _pq_forward(x: np.ndarray, l_max: float = PU21_L_MAX) -> np.ndarray:
    m, n = 78.8438, 0.1593
    c1, c2, c3 = 0.8359, 18.8516, 18.6875
    lp = np.power(np.clip(x / l_max, 0.0, None), n)
    return np.power((c1 + c2 * lp) / (1.0 + c3 * lp), m)


def _pq_inverse(y: np.ndarray, l_max: float = PU21_L_MAX) -> np.ndarray:
    m, n = 78.8438, 0.1593
    c1, c2, c3 = 0.8359, 18.8516, 18.6875
    y = np.clip(y, 0.0, None)
    lp = (c1 - np.power(y, 1.0 / m)) / (c3 * np.power(y, 1.0 / m) - c2)
    return np.clip(l_max * np.power(np.clip(lp, 0.0, None), 1.0 / n), PU21_L_MIN, l_max)


def _poly_design(x: np.ndarray, deg: int) -> np.ndarray:
    """Polynomial design matrix (matlab lin_matrix, single-channel)."""
    flat = x.reshape(-1, 1).astype(np.float64)
    cols = [flat]
    for d in range(2, deg + 1):
        cols.insert(0, flat**d)
    cols.append(np.ones((flat.shape[0], 1), dtype=np.float64))
    return np.concatenate(cols, axis=1)


def _corr_opt_1d(ir: np.ndarray, igt: np.ndarray, deg: int = 3, lambda_reg: float = 0.01) -> np.ndarray:
    y = _pq_forward(igt)
    x = _pq_forward(ir)
    x_mat = _poly_design(x, deg)
    n, p = x_mat.shape
    reg = lambda_reg * n / max(p, 1)
    w0 = np.zeros((p, 1), dtype=np.float64)
    w0[-2, 0] = 1.0
    y_flat = y.reshape(-1, 1)
    lhs = x_mat.T @ x_mat + reg * np.eye(p)
    rhs = x_mat.T @ y_flat + reg * w0
    w = np.linalg.solve(lhs, rhs)
    out = (x_mat @ w).reshape(ir.shape)
    out = np.maximum(out, _pq_forward(np.array(PU21_L_MIN)))
    return _pq_inverse(out)


def crf_correction_si_hdr(pred_cd: np.ndarray, gt_cd: np.ndarray) -> np.ndarray:
    """
    Align test radiance to reference before PU / HDR metrics (pu21/crf_correction.m defaults).
    """
    pred = np.clip(pred_cd.astype(np.float64), PU21_L_MIN, PU21_L_MAX)
    gt = np.clip(gt_cd.astype(np.float64), PU21_L_MIN, PU21_L_MAX)
    pred_luv = _rgb_to_luv(pred)
    gt_luv = _rgb_to_luv(gt)
    out_luv = np.zeros_like(pred_luv)
    out_luv[..., 0] = _corr_opt_1d(pred_luv[..., 0], gt_luv[..., 0], deg=3, lambda_reg=0.0)
    uv_pred = pred_luv[..., 1:3]
    uv_gt = gt_luv[..., 1:3]
    out_luv[..., 1] = _corr_opt_1d(uv_pred[..., 0], uv_gt[..., 0], deg=3, lambda_reg=0.01)
    out_luv[..., 2] = _corr_opt_1d(uv_pred[..., 1], uv_gt[..., 1], deg=3, lambda_reg=0.01)
    return np.clip(_luv_to_rgb(out_luv), PU21_L_MIN, PU21_L_MAX)


def compute_pu_psnr_ssim_pair(
    pred_cd: np.ndarray,
    gt_cd: np.ndarray,
    *,
    apply_crf: bool = True,
) -> Tuple[float, float]:
    """
    ExpoCM Table 1 PU21 PSNR (RGB) and SSIM (encoded luminance), peak/dynamic range 256.
    """
    test = crf_correction_si_hdr(pred_cd, gt_cd) if apply_crf else np.clip(pred_cd, PU21_L_MIN, PU21_L_MAX)
    ref = np.clip(gt_cd, PU21_L_MIN, PU21_L_MAX)

    pu_pred_rgb = pu21_encode_absolute(test)
    pu_ref_rgb = pu21_encode_absolute(ref)
    mse = float(np.mean((pu_pred_rgb.astype(np.float64) - pu_ref_rgb.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        psnr_pu = 100.0
    else:
        psnr_pu = float(10.0 * np.log10((PU21_METRIC_PEAK * PU21_METRIC_PEAK) / mse))

    y_pred = pu21_encode_luminance(linear_rgb_to_luminance(test))
    y_ref = pu21_encode_luminance(linear_rgb_to_luminance(ref))
    try:
        from skimage.metrics import structural_similarity

        ssim_pu = float(structural_similarity(y_pred, y_ref, data_range=PU21_METRIC_PEAK))
    except Exception:
        from ..training_scripts.common_training import fhdr_compare_ssim

        ssim_pu = float(
            fhdr_compare_ssim(
                np.clip(y_pred / PU21_METRIC_PEAK, 0.0, 1.0)[..., None].repeat(3, axis=2),
                np.clip(y_ref / PU21_METRIC_PEAK, 0.0, 1.0)[..., None].repeat(3, axis=2),
            )
        )
    return psnr_pu, ssim_pu
