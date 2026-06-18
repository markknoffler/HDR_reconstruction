# TriGate-HDR — Implementation History & Design Philosophy

Living document: **what** each stage does, **why** it exists, **what changed**, and **what results mean**.

---

## 0. Paper targets (ICIP HistoHDR-Net, Table 1 — HDR-Real)

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| FHDR (weak baseline) | 17.11 | 0.71 |
| SingleHDR | 26.33 | 0.85 |
| ArtHDR-Net | **33.45** | 0.88 |
| Diffusion-based [21] | **33.52** | 0.90 |
| **HistoHDR-Net (best in paper)** | **33.48** | **0.91** |

**User goal:** Beat **~33.5 dB PSNR / ~0.90+ SSIM** on HDR-Real (not merely match FHDR 17 dB).

**Metrics:** Same as FHDR `test.py` — μ-law PSNR (μ=5000) + SSIM on `(x+1)/2` RGB. Implemented in `common_training.compute_psnr_ssim_fhdr`.

---

## 1. Pipeline philosophy (three stages)

### Stage 1 — Full diffusion (InstructPix2Pix + TriGate encoders)

| Aspect | Detail |
|--------|--------|
| **Idea** | Generative **full HDR reconstruction** from LDR using diffusion — can hallucinate structure in clipped regions. |
| **Philosophy** | "Grounded imagination" — diffusion explores the solution space; TriGate encoders inject spatial/semantic structure. |
| **Role** | Produce a **coarse global HDR** with plausible content in over/under-exposed areas. |
| **Weakness** | Radiometric accuracy and fine detail — diffusion is not anchored to physical exposure expansion. |

### Stage 2 — Cold diffusion (ColdEfficient-LORCD) ← **current focus**

| Aspect | Detail |
|--------|--------|
| **Idea** | **Spatially preserved** reconstruction: LDR latent is a fixed **anchor** (`z_lift`); only an **expansion latent** `z_exp` is cold-corrupted and denoised. |
| **Philosophy** | Theoretically elegant — separates **geometry/spatial layout** (from LDR) from **radiometric headroom** (expansion). Cold diffusion is deterministic corruption `z_exp_t = (1-α_t)·z_exp_0`, not Gaussian noise. |
| **Role** | Recover **true HDR radiance** in clipped regions while preserving well-exposed pixels (trust gate). |
| **Inference** | `restore_hdr`: reverse cold chain from `z_exp=0` over 50 steps, VAE decode, optional pixel refiner. |
| **Why results were ~15 dB (not 33 dB)** | See Section 3. |

### Stage 3 — Seaming GAN (Gated WGAN)

| Aspect | Detail |
|--------|--------|
| **Idea** | Blend Stage-1 generative output with Stage-2 radiometric output along **clip boundaries** (gate/seam band). |
| **Philosophy** | Stage 1 gives plausible textures in dead zones; Stage 2 gives radiometric truth elsewhere; GAN hides the seam. |
| **Role** | Final composite quality — not relevant until Stage 2 reaches strong PSNR. |

---

## 2. Stage 2 architecture (ColdEfficient-LORCD)

```
LDR ──► VAE encode ──► z_ldr ──► MonoLift ──► z_lift (anchor)
HDR ──► VAE encode ──► z_hdr ──► z_exp = z_hdr - z_lift (expansion)

Training: corrupt z_exp_t = (1-α_t)·z_exp, z_t = z_lift + z_exp_t
          UNet(z_t, z_ldr, t, trust) → z_exp_pred → decode → HDR

Inference: z_exp = 0 → reverse cold chain (50 steps) → decode → [Pixel Refiner] → HDR
```

**Components:**

| Module | Purpose |
|--------|---------|
| `MiniHDRVAE` | /8 latent compression; encode HDR & LDR-in-HDR-space |
| `MonoLift (mln)` | `z_lift = z_ldr + net(z_ldr)` — LDR-anchored latent |
| `ColdEfficientLatentUNet` | Dual-stream (cold + anchor), RGCF trust fusion, predicts `z_exp_0` |
| `PixelHDRRefiner` *(v2)* | Pixel residual correction after VAE decode — recovers HF detail |

---

## 3. Why Stage 2 plateaued ~15–16 dB (not a metric bug)

### Confirmed facts (v8 run, epochs 37–57)

- Full val: **~15.7 dB PSNR / ~0.86 SSIM**
- Train loss still decreasing → **optimization plateau**, not broken code
- Oracle VAE (`z_lift + true z_exp` decode): **~28 dB** → VAE is **not** the bottleneck
- `restore_hdr` UNet path: **~11–16 dB** → **UNet + multi-step inference** is the bottleneck

### Root causes

1. **Train/inference mismatch** — Training: single random-t forward. Validation: 50-step `restore_hdr`. Model optimizes one-step denoising, not the full chain.
2. **Latent bottleneck** — 4-ch latent at /8 resolution smears fine detail (text, edges, saturation).
3. **No pixel-space refinement** — ArtHDR-Net / HistoHDR-Net operate largely in pixel/feature space; our pipeline ended at VAE decode.
4. **Capacity** — `base_ch=64`, `latent_ch=4` is small vs ResNet50 + decoder in paper methods.
5. **Loss mismatch** — Paper uses Weber PSNR, MS-SSIM, VGG, color ΔE losses; we used mostly L1 in latent/pixel space.

### What is NOT the problem

- PSNR/SSIM formula (verified against FHDR `test.py`)
- Dataset split (same HDR-Real paths)
- VAE warmup ~2 dB (expected; not comparable to final metrics)

---

## 4. Architectural improvements (Jun 2026 — Stage 2 v2)

**User request:** Improve Stage 2 architecture to push past 15 dB toward paper-level 33+ dB.

### Change 1: `PixelHDRRefiner` (new)

| | |
|--|--|
| **File** | `model/decoders/pixel_hdr_refiner.py` |
| **What** | 6-block residual CNN: `concat(LDR, coarse_HDR) → ΔHDR`; output = coarse + Δ |
| **Why** | VAE decode caps quality ~16 dB; pixel refiner recovers edges/textures like ArtHDR-Net's decoder head |
| **Where** | Applied in `forward()` and `restore_hdr()` after VAE decode |
| **Flag** | `--use_pixel_refiner` or `--arch_v2` |

### Change 2: Sobel gradient loss (`hf_loss`)

| | |
|--|--|
| **What** | L1 on Sobel gradients of pred vs GT |
| **Why** | Paper methods emphasize structural similarity (MS-SSIM); gradient loss directly improves edges/SSIM |
| **Flag** | `--hf_loss_weight 0.5` (in `--arch_v2`) |

### Change 3: Wider VAE + UNet (`--arch_v2` preset)

| Parameter | v1 (old) | v2 (`--arch_v2`) |
|-----------|----------|------------------|
| `latent_ch` | 4 | **8** |
| `vae_base_ch` | 32 | **48** |
| `base_ch` (UNet) | 64 | **96** |
| `pixel_refiner` | off | **on** |
| `mu_psnr_loss_weight` | 0 | **0.25** |
| `ssim_rgb_l1_weight` | 0 | **0.35** |
| `anchor_exp/hdr_weight` | 0 | **0.5** |
| `inference_loss_weight` | 0 | **0.1** (25 steps, every 50 batches) |

### Change 4: Metric-aligned training losses

| Loss | Why |
|------|-----|
| `mu_psnr_loss` | Directly optimizes the μ-tonemap MSE used in PSNR metric |
| `ssim_rgb_l1` | L1 in [0,1] RGB space — same space as SSIM computation |
| `anchor_*` at t=T-1 | Strengthens inference start state (`z_exp=0`) |
| `inference_loss` (mild) | Aligns training with multi-step `restore_hdr` without destabilizing (weight 0.1, not 1.0) |

---

## 5. Training history (experiments)

| Run | Best full-val PSNR | Best SSIM | Notes |
|-----|-------------------|-----------|-------|
| `stage2_cold_lorcd_v3` | 11.36 (ep 30) | 0.71 | Collapsed after ep 30 |
| `stage2_cold_lorcd_v8` | **15.66 (ep 55)** | **0.86** | Fresh train; plateau ~15–16 dB |
| `stage2_lorcd_v2_arch` | *pending* | *pending* | `--arch_v2` with pixel refiner |

---

## 6. Changelog (code changes)

### 2026-06 — Stage 2 v2 architecture

- Added `PixelHDRRefiner` + `sobel_gradient_loss`
- Integrated refiner into `ColdHDRDiffusion.forward` and `restore_hdr`
- Added `--arch_v2`, `--use_pixel_refiner`, `--vae_base_ch`, `--hf_loss_weight`
- Resume: restore optimizer state; sanity val on resume; warn if collapsed weights

### 2026-05 — Stage 2 debugging

- Fixed `restore_hdr` reverse loop final step
- High-t biased timestep sampling
- VAE freeze after warmup
- Per-epoch train + val metrics in CSV
- `best.pt` vs `latest.pt` resume logic

---

## 7. Training commands

### A. Full v2 architecture (recommended — target 25–33+ dB)

Fresh training with all improvements. **Cannot** load v8 weights (different `latent_ch`/width).

```bash
conda activate deeplearning
cd ~/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR
export PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m model.training_scripts.train_stage2_crf_recovery \
  --ldr_dir "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in" \
  --hdr_dir "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt" \
  --checkpoint_dir "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_lorcd_v2_arch" \
  --arch_v2 \
  --epochs 120 \
  --batch_size 1 \
  --max_dim 512 \
  --num_workers 2 \
  --vae_warmup_epochs 8 \
  --skip_trial_validation \
  --full_val_every 5 \
  --train_eval_samples 50 \
  --val_eval_samples 50 \
  --inference_timesteps 50 \
  --cold_lr 1e-5 \
  --early_stop_patience 8 \
  --ema_decay 0
```

### B. Refiner upgrade from v8 best (quick — keep v1 UNet/VAE dims)

Loads v8 `best.pt` weights; trains new pixel refiner + HF loss on top.

```bash
python -u -m model.training_scripts.train_stage2_crf_recovery \
  --ldr_dir ".../SingleHDR_training_data/HDR-Real/LDR_in" \
  --hdr_dir ".../SingleHDR_training_data/HDR-Real/HDR_gt" \
  --checkpoint_dir ".../experiments/stage2_lorcd_v8_refiner" \
  --warm_start_from ".../experiments/stage2_cold_lorcd_v8/best.pt" \
  --use_pixel_refiner \
  --hf_loss_weight 0.5 \
  --mu_psnr_loss_weight 0.2 \
  --ssim_rgb_l1_weight 0.3 \
  --vae_warmup_epochs 0 \
  --skip_trial_validation \
  --epochs 60 \
  --batch_size 1 \
  --max_dim 512 \
  --num_workers 2 \
  --full_val_every 5 \
  --inference_timesteps 50 \
  --cold_lr 5e-6 \
  --early_stop_patience 5
```

---

## 8. Realistic expectations

| Target | Feasibility |
|--------|-------------|
| Beat FHDR 17 dB | **Done** (v8 reached ~15.7 full val; subset ~16 dB) |
| Reach 20–25 dB | **Likely** with pixel refiner + v2 width + metric losses |
| Match paper 33.5 dB | **Hard** — paper methods use ResNet50 fusion, VGG/MS-SSIM/Weber losses, direct pixel pipelines. May need Stage 2 + perceptual loss + longer training + possibly larger `max_dim` (768). |
| Beat paper 33.5 dB | Requires sustained v2 training + possible further additions (VGG perceptual loss, multi-scale inputs). |

---

## 9. Files reference

| File | Role |
|------|------|
| `model/decoders/cold_hdr_diffusion_decoder.py` | Stage 2 main model |
| `model/decoders/pixel_hdr_refiner.py` | Pixel refinement head (v2) |
| `model/decoders/cold_efficient_blocks.py` | Latent UNet + RGCF |
| `model/decoders/mini_hdr_vae.py` | Mini HDR VAE + MonoLift |
| `model/training_scripts/train_stage2_crf_recovery.py` | Training script |
| `model/training_scripts/common_training.py` | FHDR metrics |
| `TRIGATE_PIPELINE_GUIDE.md` | Operational guide |
| `TRIGATE_IMPLEMENTATION_HISTORY.md` | This file |
