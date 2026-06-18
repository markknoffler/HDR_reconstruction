"""
Frozen Stable Diffusion img2img backbone for Stage-1 baseline tests.

This does NOT replace TriGate's custom tri-encoder diffusion yet — it provides a
standard pretrained denoiser you can run without training to sanity-check:
  LDR in -> (frozen SD img2img) -> enhanced sRGB-like image out.

True linear HDR radiance is NOT guaranteed without fine-tuning; output is still
useful to verify weights, VRAM, and the LDR->image plumbing.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn

from .stable_diffusion_components import StableDiffusionComponentConfig, StableDiffusionLatentStack
from .stable_diffusion_utils import (
    check_sd_output_stats,
    freeze_module,
    pil_list_to_tensor_bchw,
    prepare_diffusion_pipeline_for_inference,
    require_diffusers,
    resize_ldr_for_sd,
    sd_output_to_trigate_hdr_range,
    tensor_bchw_to_pil_list,
)

# Default prompt nudges SD toward HDR-like appearance (still sRGB, not .hdr radiance).
DEFAULT_HDR_PROMPT = (
    "high dynamic range photograph, natural lighting, detailed highlights and shadows, "
    "photorealistic, sharp, high quality"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, oversaturated, cartoon, artifacts"


class FrozenStableDiffusionStage1(nn.Module):
    """
    Stage-1 baseline: Hugging Face Stable Diffusion 2.1 img2img, all weights frozen.

    Usage:
        model = FrozenStableDiffusionStage1.from_pretrained()
        pred_01 = model(ldr_bchw)  # ldr in [0, 1], returns [0, 1]
        pred_hdr_range = model(ldr_bchw, output_range="trigate")  # [-1, 1]
    """

    def __init__(
        self,
        pipeline,
        config: StableDiffusionComponentConfig,
        default_prompt: str = DEFAULT_HDR_PROMPT,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.default_prompt = default_prompt
        self.default_negative_prompt = default_negative_prompt
        self._freeze_pipeline()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: str = "float16",
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        use_safetensors: bool = True,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        default_prompt: str = DEFAULT_HDR_PROMPT,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> "FrozenStableDiffusionStage1":
        """
        Download (first run) and load SD weights from Hugging Face Hub.

        Weights are cached under ~/.cache/huggingface/hub by default.
        Set HF_HOME or cache_dir to change location.
        """
        require_diffusers()
        from diffusers import StableDiffusionImg2ImgPipeline

        cfg = StableDiffusionComponentConfig(
            model_id=model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            variant=variant,
        )
        dtype = _resolve_dtype(torch_dtype)

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        if device is not None:
            dev = torch.device(device)
            pipe = pipe.to(dev)
        elif torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")
            if dtype != torch.float32:
                pipe = pipe.to(torch.float32)

        if enable_attention_slicing:
            pipe.enable_attention_slicing()
        prepare_diffusion_pipeline_for_inference(pipe, force_vae_fp32=True)
        if enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()

        return cls(
            pipeline=pipe,
            config=cfg,
            default_prompt=default_prompt,
            default_negative_prompt=default_negative_prompt,
        )

    def load_explicit_components(self) -> StableDiffusionLatentStack:
        """Load VAE/UNet/CLIP/scheduler as separate frozen modules (for future fine-tune hooks)."""
        return StableDiffusionLatentStack(self.config)

    def _freeze_pipeline(self) -> None:
        for name in ("vae", "unet", "text_encoder", "text_encoder_2"):
            if hasattr(self.pipeline, name):
                mod = getattr(self.pipeline, name)
                if mod is not None:
                    freeze_module(mod)

    @property
    def device(self) -> torch.device:
        return self.pipeline.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pipeline.unet.dtype

    @torch.no_grad()
    def forward(
        self,
        ldr: torch.Tensor,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        strength: float = 0.55,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        max_side: int = 768,
        output_range: str = "zero_one",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Args:
            ldr: BCHW float in [0, 1] (TriGate LDR convention).
            strength: img2img noise strength (0 = copy LDR, 1 = ignore LDR structure).
            output_range: "zero_one" | "trigate" ([-1, 1] like HDR tensors in dataloader).

        Returns:
            BCHW tensor on same device as input.
        """
        if ldr.dim() != 4 or ldr.shape[1] != 3:
            raise ValueError(f"ldr must be BCHW with 3 channels, got {tuple(ldr.shape)}")

        orig_h, orig_w = ldr.shape[2], ldr.shape[3]
        ldr_resized = resize_ldr_for_sd(ldr, max_side=max_side)
        pil_in = tensor_bchw_to_pil_list(ldr_resized)

        prompt = prompt if prompt is not None else self.default_prompt
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self.default_negative_prompt
        )

        out_pils: List = []
        for im in pil_in:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=im,
                strength=float(strength),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
            out_pils.append(result.images[0])

        pred = pil_list_to_tensor_bchw(out_pils, device=ldr.device, dtype=torch.float32)
        check_sd_output_stats(pred, tag="sd_img2img")

        if pred.shape[2] != orig_h or pred.shape[3] != orig_w:
            import torch.nn.functional as F

            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        if output_range == "trigate":
            return sd_output_to_trigate_hdr_range(pred)
        if output_range == "zero_one":
            return pred
        raise ValueError(f"Unknown output_range: {output_range}")

    @torch.no_grad()
    def restore_hdr(self, ldr: torch.Tensor, **kwargs) -> torch.Tensor:
        """TriGate-compatible name: returns [-1, 1] tensor (sRGB proxy, not linear .hdr)."""
        kwargs.setdefault("output_range", "trigate")
        return self.forward(ldr, **kwargs)


def _resolve_dtype(name: str):
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32
