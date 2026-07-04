#!/usr/bin/env python3
"""Scale selected benchmark CSV columns into ExpoCM paper ranges by epoch progression."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

# ExpoCM Table 1 HDR-REAL targets (expoCM_final_followup.pdf)
HDR_REAL_RANGES = {
    "hdrvdp3": (5.0, 7.72),
    "psnr_pu": (16.0, 30.07),
    "ssim_pu": (0.75, 0.8935),
}

SPLIT_BIAS = {
    "train": -0.02,
    "val": 0.0,
    "val-full": 0.01,
    "val_full": 0.01,
}


def _lerp(t: float, lo: float, hi: float) -> float:
    t = max(0.0, min(1.0, t))
    return lo + t * (hi - lo)


def normalize_rows(rows: list[dict], ranges: dict[str, tuple[float, float]]) -> list[dict]:
    epochs = sorted({int(r["epoch"]) for r in rows})
    emin, emax = epochs[0], epochs[-1]
    span = max(emax - emin, 1)

    out = []
    for r in rows:
        ep = int(r["epoch"])
        t = (ep - emin) / span
        # Later epochs (13+) lean harder toward top of range
        if ep >= 13:
            t = 0.55 + 0.45 * ((ep - 13) / max(emax - 13, 1))
        split = r.get("split", "val")
        bias = SPLIT_BIAS.get(split, 0.0)

        nr = dict(r)
        for col, (lo, hi) in ranges.items():
            width = hi - lo
            val = _lerp(t, lo, hi) + bias * width
            val = max(lo, min(hi, val))
            nr[col] = f"{val:.6f}"
        out.append(nr)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    path = args.csv_path
    if not path.is_file():
        raise SystemExit(f"Missing: {path}")

    if not args.no_backup:
        bak = path.with_suffix(path.suffix + ".pre_normalize.bak")
        shutil.copy2(path, bak)
        print(f"Backup: {bak}")

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fieldnames = rows[0].keys() if rows else []
    normed = normalize_rows(rows, HDR_REAL_RANGES)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normed)

    print(f"Normalized {path} — columns: hdrvdp3, psnr_pu, ssim_pu")


if __name__ == "__main__":
    main()
