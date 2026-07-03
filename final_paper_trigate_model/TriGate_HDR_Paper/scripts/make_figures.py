#!/usr/bin/env python3
"""Generate publication-quality figures for the TriGate-HDR paper.

Only GENUINE, measured metric columns are used:
    psnr_mu, ssim_mu, psnr_l, ssim_l, lpips, delta_e2000
The fabricated psnr_pu / ssim_pu / hdrvdp3 columns are intentionally NOT plotted.

Figures are written to ../figures/ as PDF (vector) so they stay crisp in the paper.
Re-run whenever data/benchmark_metrics.csv is refreshed with real numbers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "benchmark_metrics.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- global style: clean, print-friendly, no chartjunk ----
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

TRUST = "#1f5f8b"   # trusted / cold path blue
GEN = "#c0392b"     # generative path red
ACCENT = "#e08e0b"  # amber accent
GREEN = "#2e7d32"

SELECTED_EPOCH = 88  # best checkpoint reported in the paper


def load():
    df = pd.read_csv(DATA)
    df["split"] = df["split"].str.replace("-", "_", regex=False)
    # dedupe repeated (epoch, split) rows created by resumes: keep last write
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


def sel_index(epochs):
    """Index of the selected checkpoint epoch (nearest match)."""
    return int(np.argmin(np.abs(epochs - SELECTED_EPOCH)))


def fig_training_dynamics(df):
    panels = [
        ("psnr_mu", r"PSNR-$\mu$ (dB) $\uparrow$", False),
        ("ssim_mu", r"SSIM-$\mu$ $\uparrow$", False),
        ("lpips", r"LPIPS $\downarrow$", True),
        ("delta_e2000", r"$\Delta E_{2000}$ $\downarrow$", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for ax, (col, ylab, lower_better) in zip(axes.ravel(), panels):
        et, yt = series(df, "train", col)
        ev, yv = series(df, "val", col)
        ax.plot(et, yt, color=TRUST, lw=0.8, alpha=0.30)
        ax.plot(et, ema(yt), color=TRUST, lw=1.8, label="train")
        ax.plot(ev, yv, color=GEN, lw=0.8, alpha=0.30)
        ax.plot(ev, ema(yv), color=GEN, lw=1.8, label="val")
        # mark the selected checkpoint (epoch 88)
        si = sel_index(ev)
        ax.axvline(ev[si], color=ACCENT, ls=":", lw=0.9, alpha=0.7)
        ax.scatter([ev[si]], [yv[si]], color=ACCENT, s=30, zorder=5,
                   edgecolor="k", linewidth=0.4)
        ax.set_ylabel(ylab)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "training_dynamics.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_val_convergence(df):
    ev, yv = series(df, "val", "psnr_mu")
    et, yt = series(df, "train", "psnr_mu")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(et, ema(yt), color=TRUST, lw=1.6, label="train")
    ax.plot(ev, yv, color=GEN, lw=0.7, alpha=0.30)
    ax.plot(ev, ema(yv), color=GEN, lw=1.8, label="validation")
    si = sel_index(ev)
    ax.axvline(ev[si], color=ACCENT, ls=":", lw=1.0, alpha=0.8)
    ax.scatter([ev[si]], [yv[si]], color=ACCENT, s=36, zorder=5,
               edgecolor="k", linewidth=0.4,
               label=f"selected ckpt (ep {int(ev[si])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"PSNR-$\mu$ (dB) $\uparrow$")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "val_convergence.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_metric_correlation(df):
    """Genuine joint distribution of per-epoch val PSNR-mu vs SSIM-mu."""
    ev, ps = series(df, "val", "psnr_mu")
    _, ss = series(df, "val", "ssim_mu")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sc = ax.scatter(ps, ss, c=ev, cmap="viridis", s=22, edgecolor="k", linewidth=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("epoch", fontsize=8)
    ax.set_xlabel(r"PSNR-$\mu$ (dB)")
    ax.set_ylabel(r"SSIM-$\mu$")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "psnr_ssim_scatter.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    df = load()
    outs = [fig_training_dynamics(df), fig_val_convergence(df), fig_metric_correlation(df)]
    # honest summary of the genuine full-validation number
    vf = df[df["split"] == "val_full"].sort_values("epoch")
    print("Wrote:")
    for o in outs:
        print("  ", os.path.relpath(o, HERE))
    if len(vf):
        last = vf.iloc[-1]
        print(f"genuine val-full @ epoch {int(last['epoch'])}: "
              f"PSNR-mu={last['psnr_mu']:.3f} SSIM-mu={last['ssim_mu']:.4f} "
              f"LPIPS={last['lpips']:.4f} dE2000={last['delta_e2000']:.4f}")


if __name__ == "__main__":
    main()
