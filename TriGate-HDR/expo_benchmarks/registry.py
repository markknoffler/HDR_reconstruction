"""
ExpoCM (CVPR 2026) benchmark registry.

Paper: expoCM_final_followup.pdf
Benchmarks: HDR-REAL [24], HDR-EYE [36], AIM2025 [49]
Protocol: within-dataset train/val split (80/20, seed 42) — same as TriGate dataset_splits.
Metrics: FHDR/test.py PSNR-μ, SSIM-μ + official HDR-VDP-2/3 (see common_training.HDRVDPMetrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExpoTargets:
    """ExpoCM Table 1 'Ours' row — targets to beat on each benchmark."""

    psnr_mu: float
    ssim_mu: float
    hdrvdp2: float = 0.0
    hdrvdp3: float = 0.0
    psnr_pu: float = 0.0
    ssim_pu: float = 0.0
    delta_e2000: float = 999.0
    lpips: float = 999.0


@dataclass(frozen=True)
class ExpoDatasetSpec:
    name: str
    slug: str
    ldr_dir: Path
    hdr_dir: Path
    description: str
    expected_pairs: int
    max_dim: int
    val_ratio: float = 0.2
    split_seed: int = 42
    train_patch: int = 0  # 0 = use max_dim resize (inference at full res)
    expo_targets: ExpoTargets = field(default_factory=lambda: ExpoTargets(0.0, 0.0))
    download_url: str = ""
    download_zip: str = ""

    def resolve_paths(self) -> "ExpoDatasetSpec":
        root = _project_root()
        return ExpoDatasetSpec(
            name=self.name,
            slug=self.slug,
            ldr_dir=(root / self.ldr_dir).resolve(),
            hdr_dir=(root / self.hdr_dir).resolve(),
            description=self.description,
            expected_pairs=self.expected_pairs,
            max_dim=self.max_dim,
            val_ratio=self.val_ratio,
            split_seed=self.split_seed,
            train_patch=self.train_patch,
            expo_targets=self.expo_targets,
            download_url=self.download_url,
            download_zip=self.download_zip,
        )

    def is_ready(self) -> bool:
        s = self.resolve_paths()
        return s.ldr_dir.is_dir() and s.hdr_dir.is_dir() and any(s.ldr_dir.iterdir())


def _ds(
    slug: str,
    name: str,
    rel: str,
    pairs: int,
    max_dim: int,
    targets: ExpoTargets,
    url: str = "",
    zip_name: str = "",
    desc: str = "",
) -> ExpoDatasetSpec:
    base = Path("datasets") / rel
    return ExpoDatasetSpec(
        slug=slug,
        name=name,
        ldr_dir=base / "LDR_in",
        hdr_dir=base / "HDR_gt",
        description=desc or f"ExpoCM benchmark: {name}",
        expected_pairs=pairs,
        max_dim=max_dim,
        expo_targets=targets,
        download_url=url,
        download_zip=zip_name,
    )


# ExpoCM Table 1 — 'Ours' row per dataset (HDR-REAL values from paper Table 1 / Table 3)
EXPO_DATASETS: Dict[str, ExpoDatasetSpec] = {
    "hdr_real": _ds(
        "hdr_real",
        "HDR-REAL",
        "HDR-REAL",
        pairs=1838,
        max_dim=512,
        targets=ExpoTargets(
            psnr_mu=28.66,
            ssim_mu=0.8684,
            hdrvdp2=44.27,
            hdrvdp3=7.72,
            psnr_pu=30.07,
            ssim_pu=0.8935,
            delta_e2000=4.02,
            lpips=0.1919,
        ),
        url="https://www.cmlab.csie.ntu.edu.tw/~yulunliu/hdr/SingleHDR_results/HDR-Real.zip",
        zip_name="HDR-Real.zip",
        desc="SingleHDR HDR-REAL benchmark (1838 pairs, 512×512).",
    ),
    "hdr_real_full": _ds(
        "hdr_real_full",
        "HDR-REAL-FullTrain",
        "HDR-REAL-FullTrain",
        pairs=9786,
        max_dim=512,
        targets=ExpoTargets(psnr_mu=28.66, ssim_mu=0.8684, hdrvdp2=44.27, hdrvdp3=7.72),
        desc="Full SingleHDR training HDR-REAL (symlink from SingleHDR_training_data).",
    ),
    "hdr_eye": _ds(
        "hdr_eye",
        "HDR-EYE",
        "HDR-EYE",
        pairs=46,
        max_dim=512,
        targets=ExpoTargets(
            psnr_mu=20.75,
            ssim_mu=0.8017,
            hdrvdp2=37.84,
            hdrvdp3=7.08,
            psnr_pu=16.56,
            ssim_pu=0.7496,
            delta_e2000=14.05,
            lpips=0.2811,
        ),
        url="https://www.cmlab.csie.ntu.edu.tw/~yulunliu/hdr/SingleHDR_results/HDR-Eye.zip",
        zip_name="HDR-Eye.zip",
        desc="HDR-EYE fixation dataset (46 pairs, 512×512).",
    ),
    "aim2025": _ds(
        "aim2025",
        "AIM2025",
        "AIM2025",
        pairs=18898,
        max_dim=256,
        targets=ExpoTargets(
            psnr_mu=29.02,
            ssim_mu=0.8922,
            hdrvdp2=0.0,
            hdrvdp3=0.0,
            delta_e2000=3.90,
            lpips=0.1504,
        ),
        desc="AIM 2025 Inverse Tone Mapping challenge (~19k train pairs @256²). "
        "Download manually from Codabench after registration — see datasets/AIM2025/README.md",
    ),
    "hdr_synth": _ds(
        "hdr_synth",
        "HDR-Synth",
        "HDR-Synth",
        pairs=0,
        max_dim=512,
        targets=ExpoTargets(psnr_mu=0.0, ssim_mu=0.0),
        url="https://www.cmlab.csie.ntu.edu.tw/~yulunliu/hdr/SingleHDR_results/HDR-Synth.zip",
        zip_name="HDR-Synth.zip",
        desc="HDR-Synth benchmark (optional, large).",
    ),
}


def get_dataset_spec(slug: str) -> ExpoDatasetSpec:
    key = slug.lower().replace("-", "_")
    if key not in EXPO_DATASETS:
        known = ", ".join(sorted(EXPO_DATASETS))
        raise KeyError(f"Unknown dataset '{slug}'. Known: {known}")
    return EXPO_DATASETS[key]


def list_datasets() -> list[str]:
    return sorted(EXPO_DATASETS.keys())
