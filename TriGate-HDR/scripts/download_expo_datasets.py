#!/usr/bin/env python3
"""
Download and prepare ExpoCM paper benchmarks into datasets/ at project root.

Usage:
  cd TriGate-HDR && python scripts/download_expo_datasets.py --all
  python scripts/download_expo_datasets.py --dataset hdr_real hdr_eye
  python scripts/download_expo_datasets.py --link-existing

Does NOT modify the deeplearning conda env.
"""

from __future__ import annotations

import argparse
import ftplib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
sys.path.insert(0, str(ROOT / "TriGate-HDR"))

from expo_benchmarks.registry import EXPO_DATASETS, get_dataset_spec  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _normalize_singlehdr_zip(extract_dir: Path, out_ldr: Path, out_hdr: Path) -> int:
    """Convert SingleHDR zip layout to flat LDR_in/ + HDR_gt/."""
    out_ldr.mkdir(parents=True, exist_ok=True)
    out_hdr.mkdir(parents=True, exist_ok=True)

    # HDR-Eye benchmark: HDR-Eye/00000/input.jpg + gt.hdr (46 scenes)
    eye_root = None
    for candidate in (extract_dir / "HDR-Eye", extract_dir):
        if candidate.is_dir() and any(candidate.glob("*/input.jpg")):
            eye_root = candidate
            break
    if eye_root is not None:
        count = 0
        for scene_dir in sorted(eye_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            ldr_src = scene_dir / "input.jpg"
            hdr_src = scene_dir / "gt.hdr"
            if not ldr_src.is_file() or not hdr_src.is_file():
                continue
            stem = scene_dir.name
            shutil.copy2(ldr_src, out_ldr / f"{stem}.jpg")
            shutil.copy2(hdr_src, out_hdr / f"{stem}.hdr")
            count += 1
        return count

    count = 0
    for ldr_path in extract_dir.rglob("*"):
        if not ldr_path.is_file():
            continue
        if ldr_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = ldr_path.stem
        hdr_candidates = [
            ldr_path.with_suffix(".hdr"),
            ldr_path.with_suffix(".exr"),
            ldr_path.parent / f"{stem}.hdr",
            ldr_path.parent / f"{stem}.exr",
        ]
        hdr_src = None
        for c in hdr_candidates:
            if c.is_file():
                hdr_src = c
                break
        if hdr_src is None:
            # Common SingleHDR layout: LDR_in/ and HDR_gt/ subfolders
            for sub in ("HDR_gt", "hdr_gt", "HDR", "gt"):
                alt = ldr_path.parent.parent / sub / f"{stem}.hdr"
                if alt.is_file():
                    hdr_src = alt
                    break
                alt = ldr_path.parent.parent / sub / f"{stem}.exr"
                if alt.is_file():
                    hdr_src = alt
                    break
        if hdr_src is None:
            continue
        dst_ldr = out_ldr / ldr_path.name
        dst_hdr = out_hdr / f"{stem}.hdr"
        if not dst_ldr.exists():
            shutil.copy2(ldr_path, dst_ldr)
        if not dst_hdr.exists():
            shutil.copy2(hdr_src, dst_hdr)
        count += 1
    return count


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _download_url(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        _run(["wget", "--no-check-certificate", "-c", "-O", str(dest), url])
    elif shutil.which("curl"):
        _run(["curl", "-k", "-L", "-o", str(dest), url])
    else:
        raise RuntimeError("Need wget or curl to download datasets.")


def _link_hdr_real_from_existing(spec) -> bool:
    """Symlink existing SingleHDR_training_data/HDR-Real when NTU zip is unavailable."""
    src = ROOT / "SingleHDR_training_data" / "HDR-Real"
    if not src.is_dir():
        return False
    spec.ldr_dir.parent.mkdir(parents=True, exist_ok=True)
    for sub, name in [(spec.ldr_dir, "LDR_in"), (spec.hdr_dir, "HDR_gt")]:
        if sub.exists() or sub.is_symlink():
            if sub.is_symlink():
                sub.unlink()
            elif sub.is_dir() and any(sub.iterdir()):
                continue
        target = src / name
        if sub.is_dir() and not sub.is_symlink():
            shutil.rmtree(sub)
        sub.symlink_to(target, target_is_directory=True)
        print(f"  linked {sub} -> {target}")
    return True


def _download_hdr_eye_epfl(out_root: Path) -> int:
    """
    Fetch HDR-Eye from EPFL MMSPG FTP and build LDR_in/HDR_gt pairs.
    FTP: tremplin.epfl.ch, user download@hdr-eye, folder HDREye (~1.5 GB).
    """
    staging = out_root / "_staging" / "hdr_eye_epfl"
    staging.mkdir(parents=True, exist_ok=True)
    print("  Trying EPFL FTP (tremplin.epfl.ch/HDREye)...")
    try:
        ftp = ftplib.FTP("tremplin.epfl.ch", timeout=120)
        ftp.login("download@hdr-eye", "ohsh9jah4T")
        ftp.cwd("HDREye")

        def _recurse(path: str):
            items = []
            ftp.retrlines("NLST", items.append)
            for name in items:
                if name in (".", ".."):
                    continue
                local = staging / path / name
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    ftp.cwd(name)
                    _recurse(str(Path(path) / name))
                    ftp.cwd("..")
                except ftplib.error_perm:
                    with open(local, "wb") as f:
                        ftp.retrbinary(f"RETR {name}", f.write)
                    print(f"    {path}/{name}")

        _recurse("")
        ftp.quit()
    except Exception as exc:
        print(f"  EPFL FTP failed: {exc}")
        return 0

    out_ldr = out_root / "HDR-EYE" / "LDR_in"
    out_hdr = out_root / "HDR-EYE" / "HDR_gt"
    out_ldr.mkdir(parents=True, exist_ok=True)
    out_hdr.mkdir(parents=True, exist_ok=True)
    count = 0
    for hdr_path in staging.rglob("*"):
        if not hdr_path.is_file():
            continue
        if hdr_path.suffix.lower() not in (".hdr", ".exr", ".tif", ".tiff"):
            continue
        stem = hdr_path.stem
        ldr_candidates = list(hdr_path.parent.glob(f"{stem}*.jpg")) + list(
            hdr_path.parent.glob(f"{stem}*.png")
        )
        if not ldr_candidates:
            for sub in ("LDR", "ldr", "sdr"):
                ldr_candidates = list((hdr_path.parent / sub).glob(f"{stem}*"))
                if ldr_candidates:
                    break
        if not ldr_candidates:
            continue
        ldr_src = ldr_candidates[0]
        dst_ldr = out_ldr / ldr_src.name
        dst_hdr = out_hdr / f"{stem}.hdr"
        if not dst_ldr.exists():
            shutil.copy2(ldr_src, dst_ldr)
        if not dst_hdr.exists():
            shutil.copy2(hdr_path, dst_hdr)
        count += 1
    return count


def _normalize_aim2025_layout(extract_dir: Path, out_ldr: Path, out_hdr: Path) -> int:
    """
    Normalize AIM2025 ITM challenge layouts to LDR_in/ + HDR_gt/.
    Handles Codabench bundles with im_XXXXXX_YYYYYY.jpg + matching .hdr/.exr.
    """
    out_ldr.mkdir(parents=True, exist_ok=True)
    out_hdr.mkdir(parents=True, exist_ok=True)
    count = 0

    # Layout A: paired subdirs train/ldr + train/hdr with same stem
    for ldr_root, hdr_root in (
        (extract_dir / "train" / "LDR", extract_dir / "train" / "HDR"),
        (extract_dir / "train" / "ldr", extract_dir / "train" / "hdr"),
        (extract_dir / "LDR_in", extract_dir / "HDR_gt"),
        (extract_dir / "LDR", extract_dir / "HDR"),
    ):
        if ldr_root.is_dir() and hdr_root.is_dir():
            for ldr_path in ldr_root.rglob("*"):
                if not ldr_path.is_file() or ldr_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                stem = ldr_path.stem
                hdr_src = None
                for ext in (".hdr", ".exr", ".EXR", ".HDR"):
                    c = hdr_root / f"{stem}{ext}"
                    if c.is_file():
                        hdr_src = c
                        break
                if hdr_src is None:
                    continue
                shutil.copy2(ldr_path, out_ldr / ldr_path.name)
                shutil.copy2(hdr_src, out_hdr / f"{stem}.hdr")
                count += 1
            if count:
                return count

    # Layout B: flat tree im_*.jpg with hdr sibling or in HDR_gt folder
    for ldr_path in extract_dir.rglob("*"):
        if not ldr_path.is_file():
            continue
        if ldr_path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        stem = ldr_path.stem
        hdr_src = None
        for c in (
            ldr_path.with_suffix(".hdr"),
            ldr_path.with_suffix(".exr"),
            ldr_path.parent / f"{stem}.hdr",
            ldr_path.parent / f"{stem}.exr",
        ):
            if c.is_file():
                hdr_src = c
                break
        if hdr_src is None:
            for sub in ("HDR_gt", "hdr_gt", "HDR", "hdr", "gt"):
                for ext in (".hdr", ".exr"):
                    c = ldr_path.parent.parent / sub / f"{stem}{ext}"
                    if c.is_file():
                        hdr_src = c
                        break
                if hdr_src:
                    break
        if hdr_src is None:
            continue
        dst_ldr = out_ldr / ldr_path.name
        dst_hdr = out_hdr / f"{stem}.hdr"
        if not dst_ldr.exists():
            shutil.copy2(ldr_path, dst_ldr)
        if not dst_hdr.exists():
            shutil.copy2(hdr_src, dst_hdr)
        count += 1
    return count


def _prepare_aim2025(spec, skip_download: bool, aim2025_url: str = "") -> None:
    readme = spec.ldr_dir.parent / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    if spec.is_ready():
        n = len(list(spec.ldr_dir.glob("*")))
        print(f"  AIM2025 ready: {n} LDR files in {spec.ldr_dir.parent}")
        return

    url = aim2025_url or os.environ.get("AIM2025_DATA_URL", "").strip()
    downloads = DATASETS / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # User-dropped zips from Codabench (after login)
    local_zips = sorted(
        list(downloads.glob("AIM2025*.zip"))
        + list(downloads.glob("aim2025*.zip"))
        + list(downloads.glob("*ITM*train*.zip"))
        + list(downloads.glob("*inverse*tone*.zip"))
    )

    if url and not skip_download:
        zip_path = downloads / "AIM2025_train.zip"
        try:
            _download_url(url, zip_path)
            local_zips = [zip_path]
        except Exception as exc:
            print(f"  AIM2025 URL download failed: {exc}")

    if local_zips:
        zip_path = local_zips[0]
        staging = DATASETS / "_staging" / "aim2025"
        if staging.exists():
            shutil.rmtree(staging)
        _extract_zip(zip_path, staging)
        n = _normalize_aim2025_layout(staging, spec.ldr_dir, spec.hdr_dir)
        print(f"  AIM2025 normalized {n} pairs from {zip_path.name}")
        if n == 0:
            print("  [WARN] Zip extracted but no LDR/HDR pairs found — check layout in README.")
        return

    if not readme.is_file():
        readme.write_text(
            "# AIM2025 Inverse Tone Mapping Dataset\n\n"
            "**Not a public direct download.** The ~19k-pair training set is gated on Codabench.\n\n"
            "## Steps\n\n"
            "1. Create a free account: https://www.codabench.org/accounts/signup/\n"
            "2. Open the challenge: https://www.codabench.org/competitions/8231/\n"
            "3. Click **Download** for the **training / development** data bundle.\n"
            "4. Save the zip here (any of these names work):\n\n"
            "```\n"
            "datasets/_downloads/AIM2025_train.zip\n"
            "datasets/_downloads/AIM2025.zip\n"
            "```\n\n"
            "5. Ingest and normalize:\n\n"
            "```bash\n"
            "cd TriGate-HDR\n"
            "python scripts/download_expo_datasets.py --dataset aim2025\n"
            "```\n\n"
            "Or pass a direct URL (from Codabench after login):\n\n"
            "```bash\n"
            "export AIM2025_DATA_URL='https://...signed-url-from-codabench...'\n"
            "python scripts/download_expo_datasets.py --dataset aim2025\n"
            "```\n\n"
            "Expected layout after ingest:\n\n"
            "```\n"
            "datasets/AIM2025/LDR_in/\n"
            "datasets/AIM2025/HDR_gt/\n"
            "```\n\n"
            "**Disk:** ~19k pairs @ 256² needs several GB free (your disk was nearly full).\n",
            encoding="utf-8",
        )
    print(f"  AIM2025 not downloaded — no public URL without Codabench login.")
    print(f"  See: {readme}")
    print("  After you download the zip from Codabench, place it in datasets/_downloads/ and re-run.")


def prepare_dataset(slug: str, skip_download: bool = False, prefer_existing: bool = False, aim2025_url: str = "") -> None:
    spec = get_dataset_spec(slug).resolve_paths()
    print(f"\n=== {spec.name} ({slug}) ===")

    if slug == "hdr_real_full":
        src = ROOT / "SingleHDR_training_data" / "HDR-Real"
        if not src.is_dir():
            print(f"[SKIP] Missing {src}")
            return
        spec.ldr_dir.parent.mkdir(parents=True, exist_ok=True)
        for sub, name in [(spec.ldr_dir, "LDR_in"), (spec.hdr_dir, "HDR_gt")]:
            sub.parent.mkdir(parents=True, exist_ok=True)
            if sub.exists() or sub.is_symlink():
                if sub.is_symlink():
                    sub.unlink()
                elif sub.is_dir():
                    print(f"  exists: {sub}")
                    continue
            target = src / name
            sub.symlink_to(target, target_is_directory=True)
            print(f"  linked {sub} -> {target}")
        return

    if slug == "aim2025":
        _prepare_aim2025(spec, skip_download=skip_download, aim2025_url=aim2025_url)
        return

    if spec.download_url and not skip_download:
        if prefer_existing and slug in ("hdr_real", "hdr_real_full"):
            src = ROOT / "SingleHDR_training_data" / "HDR-Real"
            if src.is_dir() and _link_hdr_real_from_existing(spec):
                print("  --prefer-existing: skipped NTU zip (using local HDR-Real).")
                return
        zip_path = DATASETS / "_downloads" / spec.download_zip
        try:
            _download_url(spec.download_url, zip_path)
            staging = DATASETS / "_staging" / slug
            if staging.exists():
                shutil.rmtree(staging)
            _extract_zip(zip_path, staging)
            n = _normalize_singlehdr_zip(staging, spec.ldr_dir, spec.hdr_dir)
            print(f"  Normalized {n} pairs -> {spec.ldr_dir.parent}")
        except (subprocess.CalledProcessError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"  NTU download failed ({exc}); trying fallbacks...")
            if slug in ("hdr_real", "hdr_real_full"):
                _link_hdr_real_from_existing(spec.resolve_paths())
            elif slug == "hdr_eye":
                n = _download_hdr_eye_epfl(DATASETS)
                if n == 0:
                    readme = spec.ldr_dir.parent / "README.md"
                    readme.parent.mkdir(parents=True, exist_ok=True)
                    readme.write_text(
                        "# HDR-EYE manual download\n\n"
                        "NTU mirror and EPFL FTP may be offline.\n\n"
                        "1. https://www.epfl.ch/labs/mmspg/downloads/hdr-eye/\n"
                        "2. Or request HDR-Eye.zip from SingleHDR authors.\n"
                        "3. Extract to LDR_in/ and HDR_gt/ then re-run --prepare-only.\n",
                        encoding="utf-8",
                    )
                    print(f"  See {readme}")
                else:
                    print(f"  EPFL: {n} pairs -> {spec.ldr_dir.parent}")
    elif spec.is_ready():
        print(f"  Already prepared at {spec.ldr_dir.parent}")
    else:
        print(f"  [WARN] No data and no download URL for {slug}")


def main():
    parser = argparse.ArgumentParser(description="Download ExpoCM benchmark datasets.")
    parser.add_argument("--dataset", nargs="*", default=[], help="Dataset slug(s), e.g. hdr_real hdr_eye")
    parser.add_argument("--all", action="store_true", help="Download all datasets with public URLs")
    parser.add_argument("--link-existing", action="store_true", help="Symlink full HDR-REAL training data")
    parser.add_argument("--prepare-only", action="store_true", help="Skip downloads, only normalize/link")
    parser.add_argument(
        "--prefer-existing",
        action="store_true",
        help="Skip NTU zip downloads when SingleHDR_training_data/HDR-Real exists; symlink instead.",
    )
    parser.add_argument("--list", action="store_true", help="List registered datasets")
    parser.add_argument(
        "--aim2025-url",
        type=str,
        default="",
        help="Direct/signed Codabench URL for AIM2025 training zip (or set AIM2025_DATA_URL).",
    )
    args = parser.parse_args()

    if args.list:
        for slug in sorted(EXPO_DATASETS):
            spec = get_dataset_spec(slug)
            print(f"  {slug:16} {spec.name:20} pairs≈{spec.expected_pairs}  target PSNR-μ={spec.expo_targets.psnr_mu}")
        return

    slugs = args.dataset
    if args.all:
        slugs = [s for s in EXPO_DATASETS if EXPO_DATASETS[s].download_url or s == "aim2025"]
    if args.link_existing:
        slugs = list(set(slugs + ["hdr_real_full"]))
    if not slugs:
        slugs = ["hdr_real", "hdr_eye", "hdr_real_full", "aim2025"]

    DATASETS.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        prepare_dataset(
            slug,
            skip_download=args.prepare_only,
            prefer_existing=args.prefer_existing,
            aim2025_url=args.aim2025_url,
        )

    print("\nDone. Dataset root:", DATASETS)


if __name__ == "__main__":
    main()
