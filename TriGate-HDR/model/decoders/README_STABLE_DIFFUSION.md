# Stable Diffusion Stage-1 Baseline (Frozen)

Frozen diffusion for testing **LDR → image** before fine-tuning with TriGate encoders and losses.

## Default: InstructPix2Pix (image + text native)

**Recommended.** Trained on `(input image, text instruction) → edited image`. The UNet learned jointly with **both** conditioning types — not text-only SD with zero-shot image adapters.

| Setting | Default |
|---------|---------|
| Model | `timbrooks/instruct-pix2pix` |
| Pipeline | `StableDiffusionInstructPix2PixPipeline` |
| Input image | LDR (what to preserve / reconstruct) |
| Text | HDR expansion **instruction** (how to edit) |

### Install

```bash
pip install -r requirements-stable-diffusion.txt
```

### Run

```bash
cd TriGate-HDR
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python -m model.training_scripts.infer_stable_diffusion_baseline \
  --ldr_dir "/path/to/LDR_in" \
  --hdr_dir "/path/to/HDR_gt" \
  --output_dir "./experiments/instruct_pix2pix_baseline" \
  --pipeline instruct_pix2pix \
  --num_samples 5 \
  --max_dim 512 \
  --max_side 512 \
  --image_guidance_scale 1.5 \
  --guidance_scale 7.5 \
  --steps 30
```

**Tuning (frozen):**

- `--image_guidance_scale` **↑** (e.g. 1.8–2.0): stick closer to LDR layout/appearance  
- **↓** (e.g. 1.0–1.2): allow more change toward instruction (HDR look)  
- `--prompt "..."`: override default HDR instruction string  

### Legacy: SD 2.1 text img2img

Text-primary model; less aligned with LDR→HDR conditioning:

```bash
python -m model.training_scripts.infer_stable_diffusion_baseline \
  --pipeline legacy_img2img \
  --model_id stabilityai/stable-diffusion-2-1-base \
  --strength 0.55 \
  ...
```

## Code layout

| File | Purpose |
|------|---------|
| `stable_diffusion_instruct_pix2pix_decoder.py` | **`FrozenInstructPix2PixStage1`** (default) |
| `stable_diffusion_stage1_decoder.py` | `FrozenStableDiffusionStage1` (legacy img2img) |
| `stable_diffusion_components.py` | VAE / UNet / CLIP loaders for future fine-tune |
| `infer_stable_diffusion_baseline.py` | CLI |

## Python API

```python
from model.decoders import FrozenInstructPix2PixStage1

model = FrozenInstructPix2PixStage1.from_pretrained(device="cuda")
pred = model.restore_hdr(ldr_batch)  # [-1, 1] TriGate tensor range
```

## Caveat

InstructPix2Pix was **not** trained on linear HDR ground truth. Results are **sRGB edit proxies** until you fine-tune on HDR-Real with ε-loss + W1/SFL + your three encoders.
