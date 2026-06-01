# InstructPix2Pix pretrained + fine-tuned Stage 1

Stage-1 HDR diffusion built on **InstructPix2Pix** (`timbrooks/instruct-pix2pix`) with TriGate novelty:

| Component | Training |
|-----------|----------|
| VAE, CLIP text encoder | Frozen |
| UNet | Pretrained + **LoRA** (ε MSE on HDR latents) |
| Material / structural / semantic / mask encoders | Trainable via `LatentCondInjector` |
| Loss curriculum | Diffusion-only warm-up → ramp W1 + SFL + KL on decoded HDR |

## Layout

```
instructpix2pix_pretrained_finetuned_stage1/
  trainable_stage1_system.py   # TrainableTriGateInstructPix2PixStage1
  latent_cond_injector.py      # Tri-stream fusion → image latents
  tri_encoder_bundle.py
  losses.py
  constants.py
```

## Install

```bash
pip install -r requirements-stable-diffusion.txt
```

## Train

From `TriGate-HDR/` (default **60 epochs**, same logging style as `ARThdrNet/m_training.py`):

```bash
python -m model.training_scripts.train_stage1_instruct_finetune \
  --ldr_dir /path/to/ldr \
  --hdr_dir /path/to/hdr \
  --checkpoint_dir checkpoints_stage1_instruct \
  --batch_size 1 \
  --max_dim 512 \
  --epochs 60 \
  --save_ckpt_after 5 \
  --continue_train   # optional resume from latest.pt
```

**Each epoch:** full validation PSNR / SSIM / HDR-VDP-2 / HDR-VDP-3 → `training_metrics.csv`; `validation_results/epoch_N/`; `latest.pt`.

**Every 5 epochs:** `epoch_5.pt`, `epoch_10.pt`, … (plus `best.pt` when PSNR improves).

**After epoch 60:** `val_exports/final_test_exports/` — 5 random val LDR→HDR samples (tonemap PNG + `.hdr`).

Smoke test:

```bash
python -m model.training_scripts.train_stage1_instruct_finetune --smoke_test
```

## Validation

Uses `validate_model_mtraining` (PSNR-μ, SSIM, HDR-VDP-2/3) and exports `validation_results/epoch_*` like ARThdrNet `m_training.py`.

**PSNR / SSIM** follow `FHDR/test.py` (`mu_tonemap` + `10*log10(1/mse)`, `compare_ssim(..., multichannel=True)`). On newer scikit-image, `compare_ssim` lives in `skimage.metrics.structural_similarity` (same call via `fhdr_compare_ssim`). No sanitize, PSNR caps, or adaptive-window SSIM on the metric path.

Verify on the training machine (same `skimage` as FHDR):

```bash
PYTHONPATH=$(pwd) python -m model.training_scripts.verify_fhdr_metrics
```

Inference during validation runs the **same latent path as training** (`restore_hdr` with TriGate injection), not a single-step `t=0` UNet pass.

## Notes

- Use a **new** checkpoint directory; weights are not compatible with the legacy `Stage1TriEncoderDiffusionSystem`.
- Prefer `--torch_dtype float32` on 12GB GPUs to avoid VAE decode NaNs.
- Add `--use_real_hdrvdp` when Octave HDR-VDP is installed.
