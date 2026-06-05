# Stage 2 — ColdEfficient-LORCD: Deep Dive Guide

This document explains **what** Stage 2 does, **why** it was designed this way, **how** training and evaluation work, **why metrics can look “stuck”**, and **what limitations remain**. It is written so someone new to the project can follow the full chain of thought without reading the entire codebase.

---

## 1. What problem Stage 2 solves

**Input:** a single low-dynamic-range image (LDR), typically in \([0,1]\) after loading.

**Output:** a high-dynamic-range image (HDR) in \([-1,1]\) (per-channel, after per-image peak normalization), comparable to FHDR / ARThdrNet evaluation.

**Role in the full TriGate pipeline:**

| Stage | Mechanism | Role |
|-------|-----------|------|
| Stage 1 | Pretrained latent **Gaussian** diffusion (InstructPix2Pix-style) | Hallucinate clipped / saturated regions using semantic conditioning |
| **Stage 2** | **Cold** (deterministic) latent diffusion | Recover **global radiance structure** from LDR as a cold boundary |
| Stage 3 | GAN seaming | Blend Stage 1 + Stage 2 at clip boundaries |

Stage 2 is **not** “add Gaussian noise to LDR pixels.” It is **invert a deterministic corruption** that gradually removes an **expansion latent** while keeping a **monotone lift** of the LDR anchor fixed.

---

## 2. Mental model: cold diffusion vs Stage 1

| | Stage 1 | Stage 2 (LORCD) |
|---|---------|-----------------|
| Space | Pretrained SD VAE latent | **Train-from-scratch** mini VAE latent |
| Corruption | Gaussian ε | **Linear cold blend on expansion only:** \(z^E_t = (1-\alpha_t) z^E_0\) |
| Conditioning | CLIP + LDR latent concat | **Dual-stream anchor** from \(z_{LDR}\) + trust gate \(\tau\) |
| Foundation | Stable Diffusion | **None** |

**Cold boundary semantics:** at the last timestep, expansion is zero, so the latent state equals **MonoLift(\(z_{LDR}\))** — a tone-lifted version of the observable LDR in latent space — not a raw pixel copy of LDR.

---

## 3. Architecture walkthrough

### 3.1 Data entering the model

From `model/training_scripts/data_loader.py`:

- **LDR:** RGB \([0,1]\)
- **HDR:** load `.hdr`, clip \(\ge 0\), **divide by per-image max**, then `2 * x - 1` → \([-1,1]\)
- **Gate \(\tau\):** `gate = (ldr.max(channel) < 0.98)` → **1 = well-exposed (trust LDR)**, **0 = clipped / saturated**

This gate definition is critical. It is **not** “confidence.” It marks where the LDR is **not** saturated and therefore where we **trust** the measurement.

### 3.2 MiniHDR-VAE (`model/decoders/mini_hdr_vae.py`)

- Encoder: 3× conv downsample → **/8** spatial, 4-channel latent
- Decoder: symmetric upsample → 3ch **Tanh** (HDR in \([-1,1]\))
- Loss: L1 recon + small KL

**Why from scratch?** Stage 2 must not depend on Stable Diffusion. The latent must represent **HDR-Real ITM** pairs, not generic LAION images.

**Limitation:** a small VAE trained jointly with diffusion is **much weaker** than SD’s VAE. Tanh caps extremes; highlight detail is hard to encode.

### 3.3 Latent radiance decomposition

Given encoded \(z_{HDR}\), \(z_{LDR}\):

```
z_lift = MonoLift(z_ldr)     # invertible tone lift (3-layer conv residual)
z_exp  = z_hdr - z_lift      # expansion / ill-posed residual
```

**Design intent:** separate what LDR already explains (lift) from what must be **invented** (expansion in clipped areas).

### 3.4 Expansion-only cold forward (training)

```
z_exp_t = (1 - alpha_t) * z_exp_0
z_t     = z_lift + z_exp_t
alpha_t = linspace(0, 1, T)[t]
```

The UNet sees `concat(z_t, z_ldr)` on the **cold stream** and uncorrupted `z_ldr` on the **anchor stream**, and predicts **`z_exp_0`** (not full HDR latent).

### 3.5 RGCF fusion (`model/decoders/cold_efficient_blocks.py`)

At each UNet level, cold and anchor features fuse with trust \(\tau\):

- \(\tau \to 1\) (well-exposed): **lock toward anchor** — discourage hallucination
- \(\tau \to 0\) (clipped): **allow cross-gating** from anchor context into cold stream

**Implementation note:** full spatial attention was replaced with **conv cross-gate** after 12GB GPU OOM at 64×64 latent. Behavior is similar in intent (trust-gated fusion) but less expressive.

### 3.6 Training losses

| Loss | What it enforces |
|------|------------------|
| `L_vae` | VAE can encode/decode HDR and LDR |
| `L_hdr` | Pixel L1: `hdr ≈ decode(z_lift + z_exp_pred)` at **one random** \(t\) |
| `L_exp` | Latent L1 on expansion |
| `L_cold` | Cold consistency on expansion |
| `L_trust` | \(\|\tau \odot z\_exp\_pred\|\) — **zero expansion where LDR is trusted** |
| `L_ms` | Multi-scale cold on down levels 1–3 |
| `L_mono` | MonoLift monotonicity |
| `L_rad` | Hybrid radiometric consistency (linear HDR, gated) |

**Important:** training optimizes **single-step** denoising at a **random** \(t\). Evaluation uses **`restore_hdr`**: multi-step Algorithm 2 reverse chain (default **25** subsampled steps).

### 3.7 Inference: `restore_hdr(ldr, gate=...)`

1. Encode LDR → \(z_{LDR}\), compute \(z_{lift}\)
2. Start \(z_{exp} = 0\)
3. For subsampled \(t = T-1 \ldots 0\): predict \(\hat z^E_0\), apply cold drift correction on expansion only
4. Decode \(z_{lift} + z_{exp}\), clamp \([-1,1]\)

**Gate must be passed at inference** so clipped regions (\(\tau=0\)) can recover expansion. See Section 5.

---

## 4. How PSNR and SSIM are computed (FHDR alignment)

Implementation: `model/training_scripts/common_training.py` → `compute_psnr_ssim_fhdr`, mirroring `FHDR/test.py` lines 101–119.

**PSNR-μ:**

```python
mse = MSE(mu_tonemap(pred), mu_tonemap(gt))
psnr = 10 * log10(1 / mse)
```

`mu_tonemap` matches `FHDR/util.py` (μ=5000).

**SSIM:**

```python
generated = (pred_chw + 1) / 2   # HWC in [0,1]
real      = (gt_chw + 1) / 2
ssim = compare_ssim(generated, real, multichannel=True)
```

No extra clamp/sanitize before metrics (same as FHDR test path).

**Verification script:** `model/training_scripts/verify_fhdr_metrics.py`

---

## 5. Why your metrics looked “stuck” (~11 dB PSNR, ~0.51 SSIM)

### 5.1 Critical bug (fixed): gate missing at validation

**Before fix:** `make_stage2_predictor` called `model.restore_hdr(ldr)` **without** `batch["gate"]`.

Effect:

- Training used real \(\tau\) from the dataset
- Validation forced \(\tau = 1\) **everywhere**
- RGCF + trust behavior at inference **did not match training**
- Clipped highlights could **not** receive expansion at eval time
- Output ≈ `decode(MonoLift(z_ldr))` — essentially a tone-mapped LDR in HDR space

**Fix (in `val_export.py`):** pass `gate` into `restore_hdr` for Stage 2 validation and Stage 3 composition.

**Action for you:** re-run validation on a checkpoint **after pulling this fix**. Epochs 59–73 numbers mostly reflect the **old** inference path.

### 5.2 Most epoch summaries are NOT full validation

Training loop behavior:

| When | Metric source |
|------|----------------|
| Every epoch | **Train-probe:** random **50** training images (`val_export_seed + epoch`) |
| Every 10 epochs (+ last) | **Full val:** entire validation split (~1957 images) |

So the printed `Validation PSNR/SSIM` on epochs 61, 62, 63… is usually **probe noise**, not the full set. That is why you see ~11.2–12.1 dB bouncing while training loss stays ~3.94.

**Identical full-val at epochs 60 and 70:** `11.5823 / 0.5156` both times — that **is** a real plateau on the full split under the old inference bug.

The trainer now prints a note when the summary uses train-probe rather than full val.

### 5.3 Trust loss suppresses expansion on most pixels

Gate is 1 wherever **no channel** is near saturation (\(< 0.98\)). That is **most of the image** in typical LDRs.

`L_trust = mean(|tau * z_exp_pred|)` pushes expansion toward **zero** in all trusted regions.

**Intent:** do not hallucinate where LDR is already correct.

**Side effect:** the model spends capacity on a **small clipped fraction**; average PSNR-μ over the full frame moves slowly.

### 5.4 Train vs eval objective mismatch

- **Train:** one random \(t\), single forward, `L_hdr` on `decode(z_lift + z_exp_pred)`
- **Eval:** 25-step `restore_hdr` chain

If single-step predictions are inconsistent across \(t\), multi-step drift does not improve — metrics plateau even while `L_hdr` on random steps fluctuates (your `hdr=0.01..0.47` in the progress bar).

### 5.5 Scratch VAE bottleneck

Joint VAE + diffusion from scratch on HDR-Real is hard. Signs of VAE-limited performance:

- Low PSNR (~11 dB) but **non-terrible** SSIM (~0.5) → structurally similar, wrong **tone/peak** mapping
- `hdr_loss` can be small in latent/pixel L1 while μ-tonemapped PSNR stays low

### 5.6 Partial checkpoint resume

If you resumed after the RGCF architecture change (`strict=False`, missing≈50 keys), RGCF / mid-fusion layers **re-initialized**. VAE weights from epoch 5 were kept, but the denoiser fusion path was fresh — convergence can stall in a suboptimal basin.

### 5.7 HDR-VDP always 0.0000

Stage 2 trainer uses:

```python
HDRVDPMetrics(use_real_hdrvdp=False)
```

So HDR-VDP is **disabled**; `_finite_metric` turns NaN into **0.0**. This is **not** a model score — it means “metric not computed.”

Use `--use_real_hdrvdp` (and Octave backend) on Stage 1-style scripts if you need real HDR-VDP.

### 5.8 Loss ~3.94 “stuck”

Total loss combines latent terms (`cold`, `exp`, `trust`, `ms`, `vae`, `rad`). These can **plateau in latent units** while **PSNR-μ** (nonlinear μ-tonemap) barely moves. A small change in highlights affects PSNR more than L1 in \([-1,1]\).

---

## 6. What “good” looks like vs your numbers

Rough context on HDR-Real (μ-PSNR, FHDR-style):

| PSNR-μ | Interpretation |
|--------|----------------|
| ~5 dB | VAE warmup only / broken inference |
| ~11–12 dB | LDR-like or weak expansion (your plateau) |
| ~25–35+ dB | Competitive FHDR / ARThdrNet territory (after long training) |

Your ~0.5 SSIM with ~11 dB PSNR suggests **layout is partly preserved** but **luminance mapping** (especially highlights) is wrong — consistent with “predict LDR lift, little expansion” and the gate-at-inference bug.

---

## 7. Recommended actions (in order)

1. **Pull the gate fix** and re-run **full validation** on `epoch_70.pt` (or continue training).
2. Compare `validation_results/epoch_70/pred_hdr_*.hdr` vs `gt_hdr_*.hdr` visually.
3. Run metrics verifier:  
   `python -m model.training_scripts.verify_fhdr_metrics`
4. For a quick A/B on one checkpoint:

   ```python
   # With vs without gate at inference — PSNR should rise on clipped-heavy images when gate is passed
   pred_no_gate = model.restore_hdr(ldr)
   pred_gate    = model.restore_hdr(ldr, gate=batch["gate"])
   ```

5. If still plateaued after fix:
   - Lower `trust_loss_weight` slightly (e.g. 0.5 → 0.2) so expansion is not over-suppressed
   - Add occasional `restore_hdr` loss on a micro-batch (future work)
   - Train VAE longer (`vae_warmup_epochs` 10+)
   - Increase `inference_timesteps` at val (25 → 50) to match training \(T=100\)

---

## 8. Training command (your paths)

```bash
cd /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR
export PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m model.training_scripts.train_stage2_crf_recovery \
  --ldr_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in \
  --hdr_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt \
  --checkpoint_dir /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold \
  --continue_train \
  --resume_from /home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/stage2_cold/epoch_70.pt \
  --epochs 100 --batch_size 1 --max_dim 512 \
  --timesteps 100 --inference_timesteps 25 \
  --base_ch 64 --latent_ch 4 --vae_warmup_epochs 5 --amp
```

After the gate fix, **re-validate existing checkpoints** without retraining 70 epochs if weights are unchanged.

---

## 9. File map

| File | Purpose |
|------|---------|
| `model/decoders/mini_hdr_vae.py` | VAE + MonoLift |
| `model/decoders/cold_efficient_blocks.py` | Dual-stream UNet + RGCF |
| `model/decoders/cold_hdr_diffusion_decoder.py` | `ColdHDRDiffusion` orchestrator |
| `model/training_scripts/train_stage2_crf_recovery.py` | Training loop |
| `model/training_scripts/val_export.py` | Validation + `make_stage2_predictor` |
| `model/training_scripts/common_training.py` | FHDR PSNR/SSIM |
| `FHDR/test.py` | Reference metric implementation |
| `model_architecture.md` | Shorter architecture spec (§5) |

---

## 10. Known limitations (honest)

1. **No pretrained diffusion** — slower convergence, lower ceiling than LEDiff/LatentHDR.
2. **Mini VAE capacity** — highlight detail and absolute scale are constrained.
3. **RGCF is conv-gated, not attention** — memory tradeoff reduces fusion expressiveness.
4. **Single-step train / multi-step eval** — classic diffusion mismatch.
5. **Per-image HDR max normalization** — matches ARThdrNet/TriGate loader; absolute radiance is not preserved (FHDR test uses its own loader but same μ-PSNR formula on \([-1,1]\) tensors).
6. **Trust gate heuristic** — fixed 0.98 threshold; no learned confidence.
7. **HDR-VDP off by default** in Stage 2 trainer.

---

## 11. Design rationale summary (chain of thought)

1. **Why cold diffusion?** LDR is a **deterministic** degradation of HDR, not Gaussian noise. Cold forward with LDR as endpoint matches the physics of clipping / tone mapping.
2. **Why latent?** Pixel UNet at 512² is expensive; expansion in a **compressed** space is cheaper and separates lift vs hallucination.
3. **Why expansion-only?** Corrupting full \(z_{HDR}\) would destroy the LDR anchor. Only \(z^E\) is corrupted; \(z^{lift}\) stays fixed.
4. **Why dual stream?** Anchor pyramid preserves **measurement structure** at all scales; cold stream predicts **residual**.
5. **Why trust gate?** Do not invent radiance where LDR is already trustworthy; focus capacity on clipped regions (Stage 1 handles semantic inpainting; Stage 2 handles global radiance).
6. **Why VAE warmup?** Random cold UNet on untrained latents is unstable; VAE must map LDR/HDR into a usable manifold first.

---

*Last updated after identifying and fixing the validation `gate` omission in `make_stage2_predictor`.*
