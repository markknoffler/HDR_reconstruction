"""
Official HDR-VDP-2 and HDR-VDP-3 metrics via bundled SourceForge releases + Octave.

Paper (HistoHDR-Net / ICIP): HDR-VDP-2 on linear HDR images; Q is reported on ~0-100.
HDR-VDP-3 quality uses task='quality' and Q_JOD (Just-Objectionable-Differences).
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

# Default peak luminance (cd/m^2) when tensors are max-normalized relative HDR in [0, 1].
_DEFAULT_PEAK_LUMINANCE = 1000.0
_DEFAULT_DISPLAY_INCHES = 24.0
_DEFAULT_VIEWING_DISTANCE_M = 0.5


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


def tensor_to_linear_rgb_cd_m2(
    hdr_tensor: torch.Tensor,
    peak_luminance: float = _DEFAULT_PEAK_LUMINANCE,
    reference_tensor: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Convert model HDR (C,H,W) in [-1, 1] to linear RGB in cd/m^2 (H,W,3).

    Dataloader max-normalizes HDR per image; we map back to [0, 1] then scale by
    peak_luminance so pred and ref share the same photometric units (paper: linear domain).
    """
    t = hdr_tensor.detach().float().cpu()
    rgb = (t + 1.0) * 0.5
    rgb = rgb.clamp(min=0.0)
    rgb = rgb.permute(1, 2, 0).numpy()

    scale = float(peak_luminance)
    if reference_tensor is not None:
        ref = reference_tensor.detach().float().cpu()
        ref_rgb = (ref + 1.0) * 0.5
        ref_max = float(ref_rgb.max().item())
        if ref_max > 1e-8:
            scale = ref_max * float(peak_luminance)

    rgb = np.clip(rgb * scale, 1e-6, None)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return rgb.astype(np.float64)


class OfficialHDRVDPBackend:
    """Runs official HDR-VDP-2.2.2 and HDR-VDP-3.0.7 through Octave."""

    def __init__(
        self,
        peak_luminance: float = _DEFAULT_PEAK_LUMINANCE,
        display_inches: float = _DEFAULT_DISPLAY_INCHES,
        viewing_distance_m: float = _DEFAULT_VIEWING_DISTANCE_M,
        octave_executable: Optional[str] = None,
    ):
        self.peak_luminance = peak_luminance
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

        test = tensor_to_linear_rgb_cd_m2(hdr_pred, self.peak_luminance, hdr_gt)
        ref = tensor_to_linear_rgb_cd_m2(hdr_gt, self.peak_luminance, hdr_gt)

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
                    f"{self.display_inches}, {self.viewing_distance_m}, {self.peak_luminance});"
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

            # Fallback: parse stdout HDRVDP2=/HDRVDP3= lines
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
