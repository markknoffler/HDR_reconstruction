# TriGate-HDR: Full Pipeline Implementation & Design Guide

This document tracks the "chain of thought," architectural decisions, and the history of tasks for the TriGate-HDR project.

---

## 1. Project Objective
To build a three-stage HDR reconstruction pipeline:
- **Stage 1:** Generative hallucination using pretrained Diffusion (InstructPix2Pix).
- **Stage 2:** Radiometric recovery using Cold Diffusion (LORCD).
- **Stage 3:** Seamless composition using Gated WGAN.

---

## 2. Implementation History & Chain of Thought

### Phase 1: Stage 1 Baseline (Inquiry & Failure)
- **Goal:** Fine-tune InstructPix2Pix on HDR-Real dataset.
- **Initial Setup:** Used `GroundedHDRUNet` with 3-channel input, clean LDR as conditioning, and no forward diffusion process.
- **Failure:** The model was a plain mapper, not a denoiser. Results were "green/highlight garbage."
- **Correction:** Rewrote Stage 1 to use standard Gaussian diffusion. Input: `[x_t, LDR_cond]`. Loss: $\epsilon$-MSE + $x_0$-L1 + Novelty (W1/SFL).

### Phase 2: Stage 1 Training Stagnation (Current Status)
- **Observation:** Training PSNR is stuck at **~7.5 dB**. Training loss is low (~0.04) but metrics do not improve.
- **Diagnosis:**
    1. **Inference without CFG:** The `restore_hdr` function lacked Classifier-Free Guidance. InstructPix2Pix zero-shots poorly without it.
    2. **Per-Image Normalization:** `hdr = hdr / hdr.max()` in `data_loader.py` destroys relative radiance between different scenes.
    3. **Encoder Noise:** `res_scale` in the latent injector was set to 0.1, injecting untrained noise into the SD prior immediately.

### Phase 3: The "TriGate Fixes" (Applied now)
- **Fix A (Normalization):** Moving to global HDR scaling (e.g., factor of 100) to preserve radiometric consistency across the dataset.
- **Fix B (Guidance):** Implementing 3-way Classifier-Free Guidance in `restore_hdr`. This allows the model to "sharpen" its HDR estimate by comparing the prompt-conditioned output with the null-conditioned output.
- **Fix C (Residual Scaling):** Initializing `res_scale` to 0.0. The model starts as a pure InstructPix2Pix and gradually learns to incorporate material/structural/semantic features.

---

## 3. Stage 1 — InstructPix2Pix Fine-tuning Spec

### Conditioning
- **Text:** `instruction` (Text Cross-Attn)
- **Image:** `LDR` (VAE Latent Concat + CLIP Vision IP-Adapter)
- **Novelty:** TriGate Encoders (Material, Structural, Semantic) fused into a latent residual.

### Training Loop
- **Warm-up:** LoRA only (Epochs 1-5).
- **Phase 2:** LoRA + Encoders + Novelty Losses (W1, SFL, KL).

---

## 4. Stage 2 — ColdEfficient-LORCD Spec

| | Detail |
|---|---|
| Space | Train-from-scratch mini VAE latent |
| Corruption | **Cold blend:** $z^E_t = (1-\alpha_t) z^E_0$ |
| Objective | Recover radiometric expansion $z^E$ from $z_{LDR}$ |

*(Refer to previous Stage 2 sections for full math on RGCF and Trust Gates.)*

---

## 5. Metrics & Validation
- **PSNR-μ:** mu-tonemap (μ=5000) -> MSE -> 10*log10(1/MSE).
- **SSIM:** (x+1)/2 -> [0,1] -> skimage SSIM.
- **Validation:** Every 10 epochs. Train-probe every epoch.

---

## 6. Current Task Chain
1. [x] Rename guide to `TRIGATE_PIPELINE_GUIDE.md`.
2. [ ] Apply Fix A: Global normalization in `data_loader.py`.
3. [ ] Apply Fix B: CFG + Uncond encoding in `trainable_stage1_system.py`.
4. [ ] Apply Fix C: Initialize `res_scale=0.0` in `latent_cond_injector.py`.
5. [ ] User to re-run Stage 1 and report new PSNR.
