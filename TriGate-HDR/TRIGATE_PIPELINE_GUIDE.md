# TriGate-HDR: Full Pipeline Implementation & Design Guide

This document tracks architectural decisions, training status, and the **Stage-2 Cold HDR debugging log**.

---

## 1. Project Objective

Three-stage HDR reconstruction pipeline:

| Stage | Method | Role |
|-------|--------|------|
| **Stage 1** | InstructPix2Pix + TriGate encoders | Generative hallucination / structure |
| **Stage 2** | ColdEfficient-LORCD (latent cold diffusion) | Radiometric expansion from LDR anchor |
| **Stage 3** | Gated WGAN | Seam blending |

---

## 2. Metrics (FHDR-comparable)

All stages use **`compute_psnr_ssim_fhdr`** in `model/training_scripts/common_training.py`, matching **`FHDR/test.py`**:

- **PSNR-μ:** `mu_tonemap` (μ=5000) → MSE → `10 * log10(1 / mse)`
- **SSIM:** `(tensor + 1) / 2` → HWC float RGB → `compare_ssim(..., multichannel=True)` (legacy skimage) or `structural_similarity(..., channel_axis=2, data_range=1.0)`

No sanitization/clamp before metric computation (sanitization is for saved previews only).

---

## 3. Stage 2 — ColdEfficient-LORCD

### Architecture

- **MiniHDR-VAE** (train from scratch, /8 latent): encodes HDR and LDR-in-HDR-space
- **MonoLift (`mln`)**: `z_lift = z_ldr + net(z_ldr)` — LDR anchor in latent space
- **Expansion**: `z_exp = z_hdr - z_lift` — radiometric headroom to recover
- **Cold corruption**: `z_exp_t = (1 - α_t) · z_exp_0`, `z_t = z_lift + z_exp_t`
- **ColdEfficientLatentUNet**: predicts `z_exp_0` from `(z_t, z_ldr, t, trust_gate)`

### Inference (`restore_hdr`)

Reverse cold chain from `z_exp = 0` (fully corrupted) down to `z_exp_hat_0`, then `decode(z_lift + z_exp)`.

---

## 4. Stage 2 Debug Log — Why Results Were Bad

### User request (May 2026)

Stage 1 (InstructPix2) training is progressing; **Stage 2 cold HDR validation PSNR stuck ~10–11 dB / SSIM ~0.55–0.70** despite long runs. Goal: diagnose root cause, fix implementation, re-train on GPU 1 (`deeplearning` env, `max_dim=512`).

### Symptoms observed

| Signal | Value | Interpretation |
|--------|-------|----------------|
| VAE oracle decode (`z_lift + true z_exp`) | **~27 dB PSNR, ~1.0 SSIM** | VAE + decomposition are **not** the bottleneck |
| `restore_hdr` / UNet prediction | **~7 dB PSNR, ~0.60 SSIM** | Expansion UNet predicts wrong latent **direction** |
| UNet output vs timestep `t` | **~same PSNR for t=0…99** | Model is **timestep-invariant** (ignores cold step) |
| `exp_loss` at eval | **~0.39** (high) | Latent expansion not matching GT |
| Epochs 54+ in old CSV | **all zeros** | Training process crashed / OOM (not a metric bug) |

### Root causes (confirmed by diagnostics)

1. **`restore_hdr` final-step bug (fixed)**  
   Old last-step update `z_exp = z_exp - cold_at_t + z_exp_hat_0` is a **no-op when t=0** (`cold_at_t = z_exp_hat_0`), so the final denoised expansion was never applied. Replaced with explicit `z_exp = cold_forward(z_hat, t_prev)` on intermediate steps and `z_exp = z_exp_hat_0` on the last step.

2. **Trust loss + VAE drift encouraged trivial solutions**  
   High trust weight penalizes `|z_exp|` in well-exposed regions; a drifting VAE/MLN can absorb error so the UNet outputs a weak, timestep-independent expansion.

3. **Uniform random `t` sampling under-trains the inference start state**  
   At inference we start from `z_t ≈ z_lift` (high `t`). Uniform `t` sampling over-emphasizes easy low-`t` steps where `z_t ≈ z_hdr`.

4. **VAE kept trainable during cold phase**  
   Gradients split between VAE and UNet; UNet does not fully own expansion prediction.

### Fixes applied (May 2026)

| Fix | File | What |
|-----|------|------|
| Correct cold reverse loop + 1-step fast path | `cold_hdr_diffusion_decoder.py` | Intermediate: `z_exp = cold_forward(z_hat, t_prev)`; final: `z_exp = z_hat` |
| High-`t` biased timestep sampling | `cold_hdr_diffusion_decoder.py` | `sample_timesteps()` — `u^0.5` bias toward `t → T-1` |
| Timestep-weighted `hdr/exp/cold` losses | `cold_hdr_diffusion_decoder.py` | Weight `0.25 + 0.75·α_t` |
| Freeze VAE after warmup (default on) | `cold_hdr_diffusion_decoder.py`, `train_stage2_crf_recovery.py` | `set_vae_trainable(False)`; rebuild optimizer for UNet only |
| Loss weights tuned | `train_stage2_crf_recovery.py` | `exp_loss_weight=2.0`, `trust_loss_weight=0.01`, `radiometric_weight=0.1`, `vae_warmup_epochs=8`, `inference_timesteps=50` |
| Radiometric loss scale | `train_stage2_crf_recovery.py` | Default weight was 1.0 but `log_term` ~11 dominated; reduced to 0.1 |
| Final test export | `train_stage2_crf_recovery.py` | 5 random val LDR→HDR after training |

### Training command (full run, GPU 1)

```bash
conda activate deeplearning
cd ~/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1

python -u -m model.training_scripts.train_stage2_crf_recovery \
  --ldr_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in \
  --hdr_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt \
  --checkpoint_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold_lorcd_v2 \
  --epochs 60 \
  --batch_size 1 \
  --max_dim 512 \
  --num_workers 4 \
  --vae_warmup_epochs 8 \
  --full_val_every 10 \
  --train_eval_samples 50 \
  --inference_timesteps 50
```

Checkpoints: `experiments/stage2_cold_lorcd_v2/`  
Metrics CSV: FHDR PSNR-μ + SSIM every epoch (train-probe); full val every 10 epochs.

### Run v2 failure (NaN at epoch 9)

`stage2_cold_lorcd_v2` trained epochs 1–8 VAE warmup OK, then **NaN at epoch 9 batch ~1707** when cold UNet + AMP started. Metrics CSV shows `nan` from epoch 9 onward. Run killed.

**NaN fixes (v3):**
- Disable AMP during cold phase (AMP only for VAE warmup)
- Grad clip `max_norm=1.0`
- Skip non-finite loss batches
- Cold LR = `lr * 0.1` after VAE freeze (was full `2e-4`)
- Clamp radiometric inputs; `nan_to_num` on VAE decode

### v3 run results (May–Jun 2026)

| Milestone | Full-val PSNR | Full-val SSIM | Notes |
|-----------|---------------|---------------|-------|
| Epoch 10 | 6.73 | 0.37 | Cold phase just started |
| Epoch 20 | 10.50 | 0.67 | Improving |
| **Epoch 30 (best)** | **11.36** | **0.71** | Matches FHDR paper SSIM on HDR-Real; PSNR still ~6 dB below FHDR (17.11) |
| Epoch 40 | 9.52 | 0.53 | **Divergence begins** |
| Epoch 60 | 3.51 | 0.09 | Collapsed weights in `latest.pt` |

**Why final epochs looked terrible (user report ~4–6 dB):**

1. **Metrics are correct** — `compute_psnr_ssim_fhdr` matches `FHDR/test.py` (μ-tonemap PSNR + SSIM on `(x+1)/2` HWC RGB). Not a formula bug.
2. **Oracle VAE decode** still ~28 dB / 1.0 SSIM — bottleneck is the **cold UNet**, not the VAE.
3. **Training instability** after epoch 30: no LR decay, no early stopping; `latest.pt` kept overwriting with worse weights while `best.pt` stayed at epoch 30.
4. **`--batch_size 16 --continue_train`** from epoch 60 made it worse (490 steps/epoch vs 7829, different optimization).

**Paper targets (ICIP HistoHDR-Net, Table 1, HDR-Real):**

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| FHDR | 17.11 | 0.71 |
| Diffusion-based | 33.52 | 0.90 |

Stage 2 goal: approach **FHDR baseline (17+ dB, 0.71+ SSIM)** on the same HDR-Real split with FHDR-comparable metrics.

### v4 fixes (Jun 2026 — user request: improve PSNR/SSIM, GPU 0, `deeplearning` env)

| Change | Why |
|--------|-----|
| **μ-tonemap MSE loss** (`mu_psnr_loss_weight=2`) | Training optimized L1 in linear HDR; PSNR metric uses μ-law — align objective with FHDR/test.py |
| **Inference-anchor losses** at `t=T-1` | Matches `restore_hdr` start (`z_exp=0`, `z_t=z_lift`) |
| **Higher `hdr_loss_weight=3`, `exp_loss_weight=3`** | Stronger pixel + latent supervision |
| **Cosine LR** after VAE freeze | Prevent late-epoch divergence (v3 collapse) |
| **EMA (0.999)** for validation | Smoother eval checkpoints |
| **Early stopping** (3 full-val plateaus) | Stop before collapse; keep `best.pt` |
| **Radiometric decay to 0** over 25 cold epochs | Radiometric `log_term` was destabilizing late training |
| **`--warm_start_from` v3 `best_epoch_30.pt`** | Resume from best weights, fresh optimizer/schedule |
| **`batch_size 1–2` only** | Warn if >2; large batches hurt HDR at 512px |

**v4 training command (GPU 0):**

```bash
conda activate deeplearning
cd ~/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python -u -m model.training_scripts.train_stage2_crf_recovery \
  --ldr_dir .../SingleHDR_training_data/HDR-Real/LDR_in \
  --hdr_dir .../SingleHDR_training_data/HDR-Real/HDR_gt \
  --checkpoint_dir .../experiments/stage2_cold_lorcd_v4 \
  --warm_start_from .../experiments/stage2_cold_lorcd_v3/best_epoch_30.pt \
  --epochs 80 --batch_size 1 --max_dim 512 --num_workers 4 \
  --vae_warmup_epochs 0 --full_val_every 5 --train_eval_samples 50 \
  --inference_timesteps 50 --amp --early_stop_patience 3
```

(`vae_warmup_epochs=0` when warm-starting from a trained checkpoint — cold UNet + frozen VAE already learned.)

### v4 run milestones (auto-updated)

| Milestone | Full-val PSNR | Full-val SSIM | Notes |
|-----------|---------------|---------------|-------|
| Epoch 0 (trial) | 7.64 | 0.4754 | trial_val before epoch 1; warm-start from v3 best_epoch_30 |
| — | — | — | Training started GPU 0, PID ~367128, Jun 10 2026 |


### Autonomous loop log

- **2026-06-10 10:25**: Started **stage2_cold_lorcd_v6** (warm_start=/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt, args=--cold_lr 5e-5 --mu_psnr_loss_weight 0.5 --hdr_loss_weight 1.0 --exp_loss_weight 2.0 --anchor_exp_weight 0.5 --anchor_hdr_weight 0.5 --radiometric_weight 0.02 --early_stop_patience 5 --inference_timesteps 50)
- **2026-06-10 10:25**: Autonomous training loop started (target PSNR≥17, SSIM≥0.71, GPU 0, deeplearning env)
- **2026-06-10 06:00**: **v6 started** — conservative recipe after v4/v4b collapse (heavy mu_psnr/hdr losses destroyed warm-start weights). cold_lr=5e-5, mu_psnr=0.5, hdr=1.0, early_stop=5, warm_start=v3/best_epoch_30.pt
- **2026-06-10 05:53**: Started **stage2_cold_lorcd_v5** (warm_start=/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt, args=--base_ch 96 --inference_timesteps 100 --mu_psnr_loss_weight 5.0 --hdr_loss_weight 6.0 --cold_lr 5e-6)
- **2026-06-10 05:53**: **stage2_cold_lorcd_v4b** finished: full-val epoch=25 PSNR=2.6432 SSIM=0.0269
- **2026-06-10 03:53**: **stage2_cold_lorcd_v4b** epoch 10 full-val: PSNR=6.1405 SSIM=0.2679
- **2026-06-10 03:13**: **stage2_cold_lorcd_v4b** epoch 5 full-val: PSNR=8.9813 SSIM=0.4870
- **2026-06-10 02:28**: Started **stage2_cold_lorcd_v4b** (warm_start=/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt, args=--mu_psnr_loss_weight 4.0 --hdr_loss_weight 4.0 --exp_loss_weight 3.0 --ssim_rgb_l1_weight 3.0 --cold_lr 5e-6 --radiometric_weight 0.01 --early_stop_patience 5)
- **2026-06-10 02:28**: **stage2_cold_lorcd_v4** finished: full-val epoch=10 PSNR=2.8413 SSIM=0.0295
- **2026-06-10 02:28**: **stage2_cold_lorcd_v4** epoch 10 full-val: PSNR=2.8413 SSIM=0.0295
- **2026-06-10 01:48**: **stage2_cold_lorcd_v4** epoch 5 full-val: PSNR=5.3451 SSIM=0.2010
- **2026-06-10 01:46**: **v4** epoch 5 full-val collapsed PSNR=5.35 SSIM=0.20 (early-stop streak 2/3); v4b tuned (cold_lr=5e-6, ssim_rgb_l1=3.0)
- **2026-06-10 01:35**: Stage-2 trainer agent active — monitoring v4 (PID 367128), added `--ssim_rgb_l1_weight` for v4b, epoch 1 full-val PSNR=10.63 SSIM=0.6484
- **2026-06-10 01:23**: Started **stage2_cold_lorcd_v4** (warm_start=/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt, args=default)
- **2026-06-10 01:23**: Autonomous training loop started (target PSNR≥17, SSIM≥0.71, GPU 0, deeplearning env)
### Verification status

- [x] Root-cause diagnostics (oracle VAE vs UNet vs timestep sweep)
- [x] Metrics verified FHDR/test.py identical
- [x] v3 completed — best epoch 30, then collapsed
- [ ] **v4** — training with stability + μ-PSNR alignment (GPU 0) — **in progress**
- [ ] Target: full-val PSNR **≥17 dB** (FHDR HDR-Real), SSIM **≥0.71**

---

## 5. Stage 1 — InstructPix2Pix (reference)

Dataset paths (same for all stages):

```
SingleHDR_training_data/HDR-Real/LDR_in
SingleHDR_training_data/HDR-Real/HDR_gt
SingleHDR_training_data/segmented_masks   # Stage 1 / Stage 3 SAM
```

Stage 1 command reference:

```bash
python -u -m model.training_scripts.train_stage1_instruct_finetune \
  --ldr_dir .../LDR_in --hdr_dir .../HDR_gt --sam_mask_dir .../segmented_masks \
  --checkpoint_dir .../experiments/stage1_instruct \
  --epochs 60 --batch_size 1 --max_dim 512 --num_workers 4 \
  --torch_dtype float32 --val_inference_steps 20 --full_val_every 10 \
  --train_eval_samples 50
```

Stage 1 fixes (global HDR scale, CFG in `restore_hdr`, `res_scale=0` init) — see git history / `trainable_stage1_system.py`.

---

## 6. Data normalization

`data_loader.py` uses **global HDR scale** (`GLOBAL_HDR_SCALE = 100`) instead of per-image max normalization, preserving scene-to-scene radiance ratios.

---

*Last updated: Stage 2 cold HDR diagnosis & fix pass (May 2026).*
