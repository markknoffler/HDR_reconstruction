"""
Stable Diffusion component layout (Latent Diffusion Model).

Standard SD is not a single matrix — it is:
  - VAE: encode/decode between RGB and 4-channel latents
  - UNet: denoise latents conditioned on timestep + text (CLIP)
  - Scheduler: DDIM / Euler / etc. stepping rule
  - Text encoder: CLIP for cross-attention in UNet

Weights are loaded from Hugging Face (diffusers). We wrap submodules here so the
architecture is visible in-repo; full sampling uses diffusers pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from .stable_diffusion_utils import freeze_module, require_diffusers


@dataclass
class StableDiffusionComponentConfig:
    """Default public checkpoint (SD 2.1 base — widely used, no SDXL VRAM)."""

    model_id: str = "stabilityai/stable-diffusion-2-1-base"
    revision: Optional[str] = None
    torch_dtype: str = "float16"  # float16 | float32
    variant: Optional[str] = None  # e.g. "fp16" when using half weights


class StableDiffusionLatentStack(nn.Module):
    """
    Explicit SD building blocks (frozen after load).

    This mirrors the Intrinsic Image Diffusion / LDM stack:
      epsilon_theta(z_t, t, text) with z = E(image).
    """

    def __init__(self, config: Optional[StableDiffusionComponentConfig] = None):
        super().__init__()
        require_diffusers()
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        cfg = config or StableDiffusionComponentConfig()
        dtype = _resolve_dtype(cfg.torch_dtype)

        load_kw = {"torch_dtype": dtype, "revision": cfg.revision}
        if cfg.variant:
            load_kw["variant"] = cfg.variant

        self.vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder="vae", **load_kw)
        self.unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder="unet", **load_kw)
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.model_id, subfolder="text_encoder", **load_kw
        )
        self.scheduler = DDIMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder="tokenizer")
        self.model_id = cfg.model_id

        freeze_module(self.vae)
        freeze_module(self.unet)
        freeze_module(self.text_encoder)

    @property
    def latent_channels(self) -> int:
        return int(self.unet.config.in_channels)

    @property
    def default_sample_size(self) -> int:
        return int(getattr(self.unet.config, "sample_size", 64))


def _resolve_dtype(name: str):
    import torch

    name = (name or "float16").lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32
