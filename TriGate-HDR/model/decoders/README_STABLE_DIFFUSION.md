# Stable Diffusion Stage-1 Baseline (Frozen)

Frozen **Stable Diffusion 2.1 img2img** for testing LDR → image without training TriGate weights.

## Install dependencies

```bash
pip install -r requirements-stable-diffusion.txt
```

Optional: login for gated models (not required for SD 2.1 base):

```bash
huggingface-cli login
```

## Download weights (automatic on first run)

Weights are pulled from Hugging Face Hub and cached locally (default `~/.cache/huggingface/hub`).

| Setting | Default |
|---------|---------|
| Model ID | `stabilityai/stable-diffusion-2-1-base` |
| Pipeline | `StableDiffusionImg2ImgPipeline` |

Pre-download without running inference:

```bash
huggingface-cli download stabilityai/stable-diffusion-2-1-base \
  --include "vae/*" "unet/*" "text_encoder/*" "tokenizer/*" "scheduler/*" \
  --local-dir ./checkpoints/sd21-base
```

Then use `--model_id ./checkpoints/sd21-base` or set `local_files_only=True` in code.

### Smaller GPU (12 GB)

- `--max_side 512`
- `--steps 20`
- `--dtype float16` (default)
- Code enables attention slicing automatically

## Run baseline inference

```bash
cd TriGate-HDR
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python -m model.training_scripts.infer_stable_diffusion_baseline \
  --ldr_dir "/path/to/LDR_in" \
  --hdr_dir "/path/to/HDR_gt" \
  --output_dir "./experiments/sd21_baseline" \
  --num_samples 5 \
  --max_dim 512 \
  --strength 0.55 \
  --steps 30
```

Outputs per sample:

- `*_input_ldr.png` — input
- `*_pred_tonemap.png` — SD img2img result (viewable)
- `*_pred_hdr.hdr` — same tensor saved in TriGate [-1,1] convention (not true linear HDR)
- `*_gt_*` — ground truth for comparison

## Code layout

| File | Purpose |
|------|---------|
| `stable_diffusion_stage1_decoder.py` | `FrozenStableDiffusionStage1` — main API |
| `stable_diffusion_components.py` | VAE / UNet / CLIP / scheduler (explicit LDM stack) |
| `stable_diffusion_utils.py` | Freeze, PIL/tensor, resize helpers |
| `infer_stable_diffusion_baseline.py` | CLI test script |

## Important caveat

Pretrained SD outputs **tonemapped sRGB-like** images, not radiance-calibrated HDR. This baseline checks plumbing and priors; **fine-tuning** (LoRA / ControlNet + your encoders + HDR losses) is required for real LDR→HDR.

## Python API

```python
import torch
from model.decoders.stable_diffusion_stage1_decoder import FrozenStableDiffusionStage1

model = FrozenStableDiffusionStage1.from_pretrained(device="cuda")
ldr = torch.rand(1, 3, 512, 512)  # [0, 1]
pred = model.restore_hdr(ldr)       # [-1, 1] TriGate range
```

All parameters have `requires_grad=False` after load.
