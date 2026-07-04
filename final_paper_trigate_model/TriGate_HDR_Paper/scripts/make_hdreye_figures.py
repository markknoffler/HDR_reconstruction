#!/usr/bin/env python3
"""Generate HDR-EYE figures for the TriGate-HDR paper.

Three self-designed figures (not copied from any reference paper), all derived from
the HDR-EYE comparison table (ExpoCM-protocol baselines + our best checkpoint):
  1. hdreye_radar.pdf     -- normalised multi-metric radar profile (all 11 sub-metrics)
                             of TriGate-HDR against the strongest prior methods.
  2. hdreye_advantage.pdf -- per-metric relative margin of TriGate-HDR over the strongest
                             competing method (diverging horizontal bars, win = green).
  3. hdreye_dynamics.pdf  -- HDR-EYE training dynamics from the measured benchmark CSV.

Figures are vector PDFs written to ../figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "benchmark_metrics_hdreye.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

TRUST = "#1f5f8b"   # cold / trusted path blue
GEN = "#c0392b"     # generative path red
ACCENT = "#e08e0b"  # amber accent (ours)
GREEN = "#2e7d32"
LOSS = "#b03a2e"

SELECTED_EPOCH = 77  # reported checkpoint (epoch number not shown in tables)

# ---- HDR-EYE comparison table (ExpoCM-protocol baselines; ours = reported checkpoint) ----
# columns: PSNR-mu, SSIM-mu, PSNR-PU, SSIM-PU, PSNR-l, SSIM-l, MS-SSIM, HDR-VDP-2, HDR-VDP-3, LPIPS, dE2000
METRICS = ["PSNR-$\\mu$", "SSIM-$\\mu$", "PSNR-PU", "SSIM-PU", "PSNR-$\\ell$", "SSIM-$\\ell$",
           "MS-SSIM", "HDR-VDP-2", "HDR-VDP-3", "LPIPS$\\downarrow$", "$\\Delta E_{2000}\\downarrow$"]
LOWER_BETTER = [False, False, False, False, False, False, False, False, False, True, True]
TABLE = {
    "HDRCNN":    [15.55, 0.5986, 16.12, 0.5673, 22.84, 0.7030, 0.8049, 37.84, 7.08, 0.2811, 14.05],
    "SingleHDR": [15.04, 0.6535, 14.36, 0.5536, 19.04, 0.5612, 0.8813, 45.23, 7.66, 0.2436, 19.28],
    "ExpandNet": [16.09, 0.7023, 15.15, 0.6073, 17.05, 0.5605, 0.8878, 27.97, 7.32, 0.3105, 17.55],
    "HDRUNet":   [14.81, 0.6883, 13.99, 0.6289, 17.69, 0.6014, 0.8149, 26.56, 5.79, 0.3054, 15.35],
    "DDPM":      [17.45, 0.7496, 16.56, 0.6859, 23.38, 0.6191, 0.9040, 53.12, 7.99, 0.2005, 13.81],
    "DDIM":      [16.98, 0.7647, 16.03, 0.7062, 21.57, 0.6270, 0.9044, 53.47, 7.92, 0.2007, 13.19],
    "HDR-Trans.":[17.23, 0.7453, 16.47, 0.6889, 20.85, 0.6576, 0.8801, 44.62, 7.51, 0.2537, 12.78],
    "Reti-Diff": [15.36, 0.6944, 14.97, 0.6163, 18.77, 0.5626, 0.8974, 46.26, 7.74, 0.2475, 17.52],
    "ExpoCM":    [20.75, 0.8017, 19.32, 0.7638, 21.30, 0.7424, 0.9053, 44.09, 7.94, 0.2353, 9.68],
    "TriGate-HDR":[27.87, 0.7221, 22.93, 0.8093, 18.85, 0.7551, 0.9177, 52.00, 8.02, 0.1433, 10.57],
}
# subset shown as polygons on the radar (readability)
RADAR_METHODS = ["TriGate-HDR", "ExpoCM", "DDPM", "DDIM"]
COLORS = {"TriGate-HDR": ACCENT, "ExpoCM": TRUST, "DDPM": GEN, "DDIM": GREEN}


def load():
    df = pd.read_csv(DATA)
    df["split"] = df["split"].str.replace("-", "_", regex=False)
    df = df.drop_duplicates(subset=["epoch", "split"], keep="last")
    return df


def series(df, split, col):
    s = df[df["split"] == split].sort_values("epoch")
    return s["epoch"].to_numpy(), s[col].to_numpy()


def ema(y, beta=0.6):
    out = np.empty_like(y, dtype=float)
    acc = y[0]
    for i, v in enumerate(y):
        acc = beta * acc + (1 - beta) * v
        out[i] = acc
    return out


def fig_radar():
    names = RADAR_METHODS
    arr = np.array([TABLE[n] for n in names], dtype=float)
    norm = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi - lo < 1e-9 else (col - lo) / (hi - lo)
        if LOWER_BETTER[j]:
            norm[:, j] = 1.0 - norm[:, j]

    N = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, name in enumerate(names):
        v = norm[i].tolist()
        v += v[:1]
        lw = 2.4 if name == "TriGate-HDR" else 1.2
        z = 5 if name == "TriGate-HDR" else 3
        ax.plot(angles, v, color=COLORS[name], lw=lw, label=name, zorder=z)
        if name == "TriGate-HDR":
            ax.fill(angles, v, color=ACCENT, alpha=0.18, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS, fontsize=6.8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.35, linewidth=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              frameon=False, fontsize=7.3, handlelength=1.3, columnspacing=1.2)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "hdreye_radar.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_advantage():
    """Relative margin of TriGate-HDR over the strongest competing method, per metric."""
    ours = np.array(TABLE["TriGate-HDR"], dtype=float)
    others = np.array([TABLE[m] for m in TABLE if m != "TriGate-HDR"], dtype=float)
    labels = ["PSNR-$\\mu$", "SSIM-$\\mu$", "PSNR-PU", "SSIM-PU", "PSNR-$\\ell$", "SSIM-$\\ell$",
              "MS-SSIM", "HDR-VDP-2", "HDR-VDP-3", "LPIPS", "$\\Delta E_{2000}$"]
    margins = []
    for j in range(len(labels)):
        if LOWER_BETTER[j]:
            best_other = others[:, j].min()
            margins.append((best_other - ours[j]) / best_other * 100.0)
        else:
            best_other = others[:, j].max()
            margins.append((ours[j] - best_other) / best_other * 100.0)
    margins = np.array(margins)
    order = np.argsort(margins)          # worst -> best, so best sits on top
    margins, labels = margins[order], [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    y = np.arange(len(labels))
    colors = [GREEN if m >= 0 else LOSS for m in margins]
    ax.barh(y, margins, color=colors, alpha=0.85, height=0.68,
            edgecolor="k", linewidth=0.3)
    for yi, m in zip(y, margins):
        ax.text(m + (0.6 if m >= 0 else -0.6), yi, f"{m:+.1f}%",
                va="center", ha="left" if m >= 0 else "right", fontsize=6.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("relative margin over strongest baseline (%)", fontsize=8.5)
    ax.set_xlim(margins.min() - 10, margins.max() + 12)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "hdreye_advantage.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_dynamics(df):
    panels = [
        ("psnr_mu", r"PSNR-$\mu$ (dB) $\uparrow$"),
        ("hdrvdp2", r"HDR-VDP-2 $\uparrow$"),
        ("ms_ssim", r"MS-SSIM $\uparrow$"),
        ("lpips", r"LPIPS $\downarrow$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0))
    for ax, (col, ylab) in zip(axes.ravel(), panels):
        et, yt = series(df, "train", col)
        ev, yv = series(df, "val", col)
        ax.plot(et, yt, color=TRUST, lw=0.7, alpha=0.28)
        ax.plot(et, ema(yt), color=TRUST, lw=1.6, label="train")
        ax.plot(ev, yv, color=GEN, lw=0.7, alpha=0.28)
        ax.plot(ev, ema(yv), color=GEN, lw=1.6, label="val")
        si = int(np.argmin(np.abs(ev - SELECTED_EPOCH)))
        ax.axvline(ev[si], color=ACCENT, ls=":", lw=0.9, alpha=0.8)
        ax.scatter([ev[si]], [yv[si]], color=ACCENT, s=24, zorder=5,
                   edgecolor="k", linewidth=0.4)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.tick_params(labelsize=7.5)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "hdreye_dynamics.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    df = load()
    outs = [fig_radar(), fig_advantage(), fig_dynamics(df)]
    print("Wrote:")
    for o in outs:
        print("  ", os.path.relpath(o, HERE))
    ev, yv = series(df, "val", "psnr_mu")
    si = int(np.argmin(np.abs(ev - SELECTED_EPOCH)))
    print(f"selected epoch {int(ev[si])}: PSNR-mu(val)={yv[si]:.3f}")


if __name__ == "__main__":
    main()
