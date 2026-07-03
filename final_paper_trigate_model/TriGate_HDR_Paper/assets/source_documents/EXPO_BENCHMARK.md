# ExpoCM benchmark (expoCM_final_followup.pdf)

## Metrics (Table 1 — all columns)

Implemented in `model/metrics/expo_metrics.py`:

| Column | Domain / notes |
|--------|----------------|
| PSNR-μ, SSIM-μ | μ-law (μ=5000), FHDR/test.py |
| PSNR-PU, SSIM-PU | PU21 banding_glare (official gfxdisp params), peak=256, SI-HDR CRF correction |
| PSNR-l, SSIM-l | Linear relative radiance [0,1] |
| MS-SSIM | 5-scale Wang MS-SSIM on μ-law RGB |
| HDR-VDP-2 / HDR-VDP-3 | Official Octave, **30 PPD** for both, CRF-corrected cd/m², Q_JOD 0–10 |
| LPIPS ↓ | Alex backbone on μ-law display (`pip install lpips`) |
| ΔE2000 ↓ | CIEDE2000 mean in CIE Lab |

Logged each epoch to `expo_metrics.csv` with splits: **train**, **val**, **val_full** (when `--expo_metrics`).

## Datasets

| Slug | Status |
|------|--------|
| `hdr_real` / `hdr_real_full` | Ready (symlink / 9786 pairs) |
| `hdr_eye` | Ready (46 pairs) |
| `aim2025` | Manual Codabench download |

`datasets/` is in `.gitignore`.

## Environment

```bash
conda activate deeplearning
cd TriGate-HDR
export PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0
export HDRVDP_OCTAVE_BIN=$HOME/anaconda3/envs/trigate-hdrvdp/bin/octave
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install lpips   # optional, for LPIPS column
```

## Train (per dataset)

```bash
# HDR-REAL (full training set, ExpoCM metrics + 500k-iter-scaled epochs)
python -m model.training_scripts.train_expo_benchmark \
  --dataset hdr_real_full --arch_v2 --use_rso --use_lr_cfp --amp

# HDR-EYE (46 pairs)
python -m model.training_scripts.train_expo_benchmark \
  --dataset hdr_eye --arch_v2 --amp

# AIM2025 (after manual download to datasets/AIM2025/)
python -m model.training_scripts.train_expo_benchmark \
  --dataset aim2025 --arch_v2 --amp --batch_size 2
```

Checkpoints: `experiments/expo_<slug>/`  
- `training_metrics.csv` — PSNR-μ / SSIM / HDR-VDP (fast)  
- `expo_metrics.csv` — full Table 1 per split  

## Post-hoc evaluation (Table 1 report)

```bash
python -m model.training_scripts.evaluate_expo_benchmark \
  --dataset hdr_eye \
  --checkpoint experiments/expo_hdr_eye/best.pt \
  --split all --arch_v2
```

Writes `expo_eval_<dataset>.json` next to the checkpoint.

## ExpoCM targets (HDR-REAL “Ours”)

PSNR-μ **28.66** | SSIM-μ **0.8684** | PSNR-PU **30.07** | SSIM-PU **0.8935**  
HDR-VDP-2 **44.27** / HDR-VDP-3 **7.72** | LPIPS **0.1919** | ΔE2000 **4.02**

## Architecture note

TriGate uses **your** GPURE / cold-diffusion stack — not ExpoCM’s consistency model.  
Training is Stage-2 orchestration (`train_expo_benchmark` → `train_stage2_crf_recovery`).  
Optional joint GPURE trainer: `train_unified_gpure.py` (combined optimization, not required for Expo runs).

Send the critic when ready — we can map each concern to whether GPURE + exposure-aware losses address it.
