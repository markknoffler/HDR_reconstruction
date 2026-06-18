"""
Official HDR-VDP-2 and HDR-VDP-3 metrics via bundled SourceForge releases + Octave.

Benchmark protocol (ExpoCM CVPR 2026 / SingleHDR evaluation):
  - Linear RGB in cd/m^2 with per-image peak mapped to display_peak (default 1000).
  - HDR-VDP-2: hdrvdp() Q correlate on 0-100 scale, 30 PPD, 0.5 m viewing.
  - HDR-VDP-3: hdrvdp3('quality') Q_JOD on ~0-10 scale, display-native PPD (~11 for 512px).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch

_TRIGATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_THIRD_PARTY = os.path.join(_TRIGATE_ROOT, "third_party")
_V2_ROOT = os.path.join(_THIRD_PARTY, "hdrvdp-2.2.2")
_V3_ROOT = os.path.join(_THIRD_PARTY, "hdrvdp-3.0.7")
_OCTAVE_SCRIPT = os.path.join(_THIRD_PARTY, "octave", "compute_hdrvdp_pair.m")

# Match data_loader.py GLOBAL_HDR_SCALE and ExpoCM peak-luminance mapping.
GLOBAL_HDR_SCALE = 100.0
_DEFAULT_DISPLAY_PEAK = 1000.0
_DEFAULT_PIXELS_PER_DEGREE = 30.0
_DEFAULT_VIEWING_DISTANCE_M = 0.5
_DEFAULT_DISPLAY_INCHES = 24.0


def _octave_env(octave_executable: str) -> dict:
    """Environment so conda Octave finds pkg/image (needs OCTAVE_HOME)."""
    env = os.environ.copy()
    octave_bin = os.path.dirname(os.path.abspath(octave_executable))
    prefix = os.path.dirname(octave_bin)
    env["OCTAVE_HOME"] = prefix
    env["PATH"] = f"{octave_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _find_octave_executable() -> Optional[str]:
    env_bin = os.environ.get("HDRVDP_OCTAVE_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    found = shutil.which("octave")
    if found:
        return found
    conda_base = os.environ.get("CONDA_PREFIX", "")
    if conda_base:
        candidate = os.path.join(conda_base, "bin", "octave")
        if os.path.isfile(candidate):
            return candidate
    home = os.path.expanduser("~")
    for name in ("trigate-hdrvdp",):
        candidate = os.path.join(home, "anaconda3", "envs", name, "bin", "octave")
        if os.path.isfile(candidate):
            return candidate
    return None


def tensor_pair_to_linear_rgb_cd_m2(
    hdr_pred: torch.Tensor,
    hdr_gt: torch.Tensor,
    display_peak: float = _DEFAULT_DISPLAY_PEAK,
    *,
    align_pred_exposure: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pred/gt tensors (C,H,W) in [-1, 1] to linear RGB cd/m^2 (H,W,3).

    Benchmark convention:
      1. Align pred exposure to gt peak (removes global gain bias; shape errors remain).
      2. Map gt peak to ``display_peak`` cd/m^2 (default 1000); pred uses the same scale.
    """
    pred = ((hdr_pred.detach().float().cpu() + 1.0) * 0.5).clamp(min=0.0)
    gt = ((hdr_gt.detach().float().cpu() + 1.0) * 0.5).clamp(min=0.0)
    pred = pred.permute(1, 2, 0).numpy()
    gt = gt.permute(1, 2, 0).numpy()

    gt_peak = max(float(gt.max()), 1e-8)
    if align_pred_exposure:
        pred_peak = max(float(pred.max()), 1e-8)
        pred = pred * (gt_peak / pred_peak)

    scale = float(display_peak) / gt_peak
    pred_cd = np.clip(pred * scale, 1e-6, None)
    gt_cd = np.clip(gt * scale, 1e-6, None)
    if pred_cd.ndim == 2:
        pred_cd = np.stack([pred_cd, pred_cd, pred_cd], axis=-1)
        gt_cd = np.stack([gt_cd, gt_cd, gt_cd], axis=-1)
    return pred_cd.astype(np.float64), gt_cd.astype(np.float64)


def tensor_to_linear_rgb_cd_m2(
    hdr_tensor: torch.Tensor,
    display_peak: float = _DEFAULT_DISPLAY_PEAK,
    reference_tensor: Optional[torch.Tensor] = None,
    *,
    use_peak_normalization: bool = True,
) -> np.ndarray:
    """Single-image helper; prefer ``tensor_pair_to_linear_rgb_cd_m2`` for metric pairs."""
    if reference_tensor is None:
        reference_tensor = hdr_tensor
    test, _ = tensor_pair_to_linear_rgb_cd_m2(
        hdr_tensor,
        reference_tensor,
        display_peak,
        align_pred_exposure=use_peak_normalization,
    )
    return test


class OfficialHDRVDPBackend:
    """Runs official HDR-VDP-2.2.2 and HDR-VDP-3.0.7 through Octave."""

    def __init__(
        self,
        display_peak: float = _DEFAULT_DISPLAY_PEAK,
        pixels_per_degree: float = _DEFAULT_PIXELS_PER_DEGREE,
        display_inches: float = _DEFAULT_DISPLAY_INCHES,
        viewing_distance_m: float = _DEFAULT_VIEWING_DISTANCE_M,
        octave_executable: Optional[str] = None,
        *,
        peak_luminance: Optional[float] = None,
    ):
        # Back-compat alias used by older call sites.
        if peak_luminance is not None:
            display_peak = float(peak_luminance)
        self.display_peak = float(display_peak)
        self.pixels_per_degree = float(pixels_per_degree)
        self.display_inches = display_inches
        self.viewing_distance_m = viewing_distance_m
        self.octave_executable = octave_executable or _find_octave_executable()
        self.available = self._check_installation()

    def _check_installation(self) -> bool:
        if not self.octave_executable:
            return False
        required = (_V2_ROOT, _V3_ROOT, _OCTAVE_SCRIPT)
        return all(os.path.isdir(p) or os.path.isfile(p) for p in required)

    def compute_pair(
        self,
        hdr_pred: torch.Tensor,
        hdr_gt: torch.Tensor,
    ) -> Tuple[float, float]:
        if not self.available:
            raise RuntimeError(
                "Official HDR-VDP backend unavailable. Install Octave in env "
                "'trigate-hdrvdp' (conda-forge) or set HDRVDP_OCTAVE_BIN."
            )

        test, ref = tensor_pair_to_linear_rgb_cd_m2(
            hdr_pred, hdr_gt, self.display_peak, align_pred_exposure=True
        )

        with tempfile.TemporaryDirectory(prefix="hdrvdp_") as tmp:
            mat_path = os.path.join(tmp, "pair.mat")
            out_path = os.path.join(tmp, "result.json")
            try:
                import scipy.io

                scipy.io.savemat(mat_path, {"test": test, "ref": ref})
            except ImportError as exc:
                raise RuntimeError("scipy is required to run official HDR-VDP metrics") from exc

            cmd = [
                self.octave_executable,
                "--no-gui",
                "--quiet",
                "--eval",
                (
                    f"compute_hdrvdp_pair('{mat_path}', '{out_path}', "
                    f"'{_V2_ROOT}', '{_V3_ROOT}', "
                    f"{self.display_inches}, {self.viewing_distance_m}, "
                    f"{self.display_peak}, {self.pixels_per_degree});"
                ),
            ]
            proc = subprocess.run(
                cmd,
                cwd=os.path.dirname(_OCTAVE_SCRIPT),
                env=_octave_env(self.octave_executable),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Octave HDR-VDP failed (code {proc.returncode}):\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )

            if os.path.isfile(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                q2 = float(payload.get("hdrvdp2", float("nan")))
                q3 = float(payload.get("hdrvdp3", float("nan")))
                return q2, q3

            q2, q3 = float("nan"), float("nan")
            for line in (proc.stdout or "").splitlines():
                m2 = re.match(r"HDRVDP2=([-\d.eE+]+)", line.strip())
                m3 = re.match(r"HDRVDP3=([-\d.eE+]+)", line.strip())
                if m2:
                    q2 = float(m2.group(1))
                if m3:
                    q3 = float(m3.group(1))
            if np.isfinite(q2) or np.isfinite(q3):
                return q2, q3
            raise RuntimeError(
                f"Could not parse HDR-VDP output.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
