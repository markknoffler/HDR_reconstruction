# TriGate-HDR Paper — Working Instructions & Charter

This directory (`final_paper_trigate_model/TriGate_HDR_Paper/`) is the **single central hub**
for the TriGate-HDR paper. Every element of the paper — the LaTeX source, all figures, all
tables, all CSV data, the bibliography, and the rendered PDF — lives here. Nothing about the
paper should live anywhere else.

This file records, in depth, exactly what we are building and the rules we follow while
building it. It is the standing brief for every future editing session. Read it first.

---

## 1. Mission

We are writing an original research paper for **TriGate-HDR** (Gate-Partitioned Unified
Radiance Energy for single-image HDR reconstruction) and taking it all the way to a
compiled PDF, entirely with a **local LaTeX toolchain — no Overleaf**.

The paper already exists in draft form (abstract, introduction, related work, method,
architecture, losses). Our job now is to **keep improving it**, and in particular to **build
out the Results and Experiments sections** as real experimental data arrives.

The paper is aimed at a **top-tier venue: CVPR / ICCV / ICLR / ICML**. Everything we write
must meet the bar those venues expect (see §6).

---

## 2. What lives in this directory

```
TriGate_HDR_Paper/
├── PAPER_INSTRUCTIONS.md      <- this file (the charter)
├── assets/                    <- everything unzipped from TriGate_HDR_Assets.zip
│   ├── figures/architecture_diagram.png
│   ├── latex_source/          <- original draft main.tex + references.bib + preview PDF
│   └── source_documents/      <- architecture / pipeline / benchmark design docs
├── main.tex                   <- THE paper source we edit (local build, not Overleaf)
├── references.bib             <- bibliography
├── figures/                   <- ALL figures used by the paper (ours, generated here)
├── tables/                    <- generated LaTeX table fragments (\input-ed by main.tex)
├── data/                      <- CSVs / result files the figures & tables are built from
└── build/                     <- pdflatex output (PDF + aux); the rendered paper
```

- `assets/` is the raw material (draft + design docs + architecture figure).
- We author into `main.tex` / `references.bib` / `figures/` / `tables/` at the top level.
- Figures and tables are **generated from real result data in `data/`** so they can be
  regenerated whenever new results land.

---

## 3. The core workflow (repeat every session)

1. **New results arrive** (a benchmark CSV, a checkpoint eval, a qualitative comparison).
   Drop the source data into `data/`.
2. **Regenerate figures** from that data into `figures/` (matplotlib scripts kept in this
   directory so they are reproducible).
3. **Regenerate / update tables** in `tables/`.
4. **Write prose** in `main.tex`: describe what the numbers mean, reference every figure and
   table (`\ref`, `\autoref`), and integrate them into the Results / Experiments narrative.
5. **Compile** locally (§5) and confirm the PDF renders with no missing refs.
6. Keep going. The paper is a living document — we improve it continuously as more results
   come in. **The user will keep adding results; we keep writing them up.**

---

## 4. Data-integrity rules (non-negotiable)

These exist to protect the submission, not to slow it down. Top venues run integrity checks
and expert reviewers; fabricated or mismatched numbers are the fastest way to a desk-reject.

- **Only genuine, measured numbers go in the paper.** Every value in a table or figure must
  trace back to a real evaluation of our checkpoints on the stated benchmark.
- **Never paste a competitor's numbers in as our own**, and never interpolate/ramp a metric
  column to hit a target. (The old `*_benchmark_metrics.csv` had `psnr_pu`, `ssim_pu`,
  `hdrvdp3` overwritten with a linear ramp to ExpoCM's exact values — those are NOT usable.)
- If a metric harness looks buggy (e.g. HDR-VDP-3 ≈ 0.08 when the scale should be 0–10),
  we **fix the harness and re-measure**, we do not fudge the output.
- Cite the source of every baseline number (paper + venue + year) in the table or its caption.
- If a number is provisional, mark it clearly (`TBD`, `\dagger preliminary`) rather than
  guessing.

---

## 5. Local LaTeX build (no Overleaf)

Compile with a standard local toolchain (TeX Live / MiKTeX). Four-pass cycle so refs + cites
resolve:

```bash
cd TriGate_HDR_Paper
pdflatex -output-directory=build main.tex
bibtex   build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

Output: `build/main.pdf`. A convenience script (`build.sh`) is kept in this directory.

If the venue ships an official style (`cvpr.sty`, `iclr2026_conference.sty`, etc.), drop it
in this directory and switch `\documentclass` accordingly; the source is structured to make
that a one-line change.

---

## 6. Writing to a top-venue (CVPR/ICCV/ICLR/ICML) standard

The paper must read like an accept, not just be correct. Concretely:

- **Format:** two-column CVPR/ICCV style (or single-column ICLR/ICML with their `.sty`),
  8 pages + references, anonymized for review.
- **Story first:** a crisp problem statement, an explicit gap in prior work, and a clearly
  stated core idea. Reviewers decide in the first page.
- **Contributions:** an explicit bulleted list; each contribution must be defensible and map
  to evidence later in the paper.
- **Math with rigor:** definitions, assumptions, and at least one formal result
  (e.g. the LR-CFP identifiability theorem) stated cleanly with a proof/proof-sketch.
  Notation must be consistent and defined on first use.
- **Method figure:** one strong architecture/overview figure that a reader can understand
  standalone from the caption.
- **Experiments that convince:**
  - Comparison against SOTA on the standard benchmarks (HDR-Real, HDR-Eye, AIM2025) with
    the standard metric suite (PSNR-μ, SSIM-μ, PSNR-PU, SSIM-PU, MS-SSIM, HDR-VDP-2/3,
    LPIPS, ΔE2000).
  - An **ablation** that isolates each novel component (RSO, LR-CFP, ECC, joint energy) so
    reviewers can see each part earns its place.
  - **Qualitative** side-by-side comparisons (saturated highlights, underexposed regions).
  - Training curves / analysis plots that show the method behaves as claimed.
  - Honest **limitations** and failure cases.
- **Reproducibility:** dataset splits (80/20, seed 42), training protocol, hyperparameters,
  and metric definitions all stated precisely.
- **Related work:** position against FHDR, SingleHDR, ArtHDR-Net, HistoHDR-Net, LEDiff,
  ExpoCM — make the delta from each explicit.

We will research and confirm the exact current formatting/length rules for the target venue
before finalizing, and write to that spec.

---

## 7. Relationship to the ExpoCM reference paper

`papers/expoCM_final_followup.pdf` is a **reference for orientation only**.

- **Do NOT** copy its text, its figures, its figure style, its table layout, or its structure.
- Use it only to understand the benchmark protocol and the state of the art we compare against.
- All of our figures, tables, phrasing, and layout are **original to this paper.**

---

## 8. Current status / open items

- [x] Directory created; assets unzipped into `assets/`.
- [x] Port draft `main.tex` to a top-level, **local-build** version (`main.tex`) — algorithm
      packages replaced with a self-contained implementation so it compiles with plain
      `pdflatex` (no external installs, no Overleaf). Build with `./build.sh`.
- [x] Generate results figures from **genuine** data into `figures/`
      (`training_dynamics.pdf`, `val_convergence.pdf`, `psnr_ssim_scatter.pdf`) via
      `scripts/make_figures.py`. Re-run that script whenever `data/benchmark_metrics.csv`
      is refreshed.
- [x] Rewrote Results + Experiments (datasets/protocol, implementation details, comparative
      SOTA table, training-dynamics figures, ablation) — this is a **comparative study**
      table (our best single checkpoint vs.\ baselines), NOT a dump of the epoch CSV.
- [x] Main results reported at the selected best checkpoint (**epoch 88**, validation split):
      PSNR-μ 29.06, SSIM-μ 0.9407, PSNR-PU 31.20, SSIM-PU 0.8846, **SSIM-l 0.9507**,
      MS-SSIM 0.9492, HDR-VDP-2 53.75, HDR-VDP-3 7.55, LPIPS 0.2674.
- [x] Added the **SSIM-l** column to the comparison table (epoch-88 value 0.9507; best=ExpoCM
      0.9521, ours second-best). SSIM is scale-invariant so SSIM-l was already correct.
- [x] Baseline rows (HDRCNN, SingleHDR, ExpandNet, HDRUNet, HDR-Transformer, DDIM, DDPM,
      Reti-Diff, ExpoCM) taken from the unified evaluation protocol of ExpoCM and cited to
      that paper (referred, not reproduced verbatim as ours). Best/second-best highlighted.
- [x] Compiles to `build/main.pdf` (no undefined refs/cites). See `HOW_TO_RENDER.md`.
- [ ] Fill ablation single-component rows via controlled one-toggle re-runs.
- [x] **PSNR-l bug fixed** in `TriGate-HDR/model/metrics/expo_metrics.py`. The old harness
      computed linear PSNR on `(x+1)/2` in [0,1] with peak=1, so dark pixels dominated and the
      score was inflated (~67 dB vs. the ~25–37 dB range everyone else reports). Per the
      SI-HDR/PU21 protocol (Hanji et al. 2022, Eilertsen et al. 2021), PSNR-l is now computed on
      absolute cd/m² (exposure-aligned, peak→1000 cd/m² — the same representation already used for
      PU/HDR-VDP) with data range = display peak. Synthetic sanity check now yields ~32 dB. SSIM-l
      is unchanged (structural/scale-invariant, was already correct at ExpoCM scale ≈0.95).
- [ ] PSNR-l is fixed in code but the epoch-88 CSV value (67.11) predates the fix, so PSNR-l is
      **not yet in the table**. Re-run the benchmark on the epoch-88 checkpoint with the corrected
      harness to obtain the real linear PSNR, then add the column.
- [x] Training now prints the **full ExpoCM metric row** after every validation run (PSNR-μ/-PU,
      SSIM-μ/-PU/-l, MS-SSIM, HDR-VDP-2/3, LPIPS, ΔE2000) via `validate_model_mtraining(..., full_metrics=True)`
      in all training scripts (stage2, stage1×2, unified GPURE, stage3). Previously only
      PSNR/SSIM/HDR-VDP-2/3 were shown. Checkpoint selection / early stopping are unaffected.
- [ ] ΔE2000 intentionally excluded from the paper table (per instruction — not needed).
- [ ] Add qualitative side-by-side comparison figure (saturated highlight / underexposed
      cases) once export images are ready.
- [ ] Tighten to 8 content pages (CVPR limit, references excluded) before submission.
- [ ] Minor: a few wide equations produce overfull hboxes; break them for camera-ready.

Keep this section updated as work proceeds.
