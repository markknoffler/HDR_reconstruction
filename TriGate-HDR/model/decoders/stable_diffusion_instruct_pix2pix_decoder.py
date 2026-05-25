"""
InstructPix2Pix — diffusion trained natively on (input image + text instruction) -> output image.

Unlike text-only SD + bolt-on IP-Adapter, the UNet was trained so the input image and edit
instruction jointly condition denoising. For LDR->HDR we use:
  - image: LDR (what to reconstruct / preserve layout)
  - prompt: short HDR expansion instruction (how to edit)

Still requires fine-tuning on HDR pairs for true radiance; this is the correct frozen prior.
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

# Default HF model: SD1.5-based InstructPix2Pix (image+text native edit diffusion).
DEFAULT_INSTRUCT_MODEL = "timbrooks/instruct-pix2pix"

DEFAULT_HDR_INSTRUCTION = (
    "Convert this to a high dynamic range photograph. Recover clipped highlights and shadow "
    "detail. Keep the same scene, geometry, and objects. Photorealistic, natural lighting."
)
DEFAULT_NEGATIVE_PROMPT = "blurry, cartoon, oversaturated, artifacts, different scene, warped"


class FrozenInstructPix2PixStage1(nn.Module):
    """
    Frozen InstructPix2Pix for Stage-1 baseline and future fine-tune hook.

    Native conditioning:
      - Input image encoded into UNet (concat + dedicated image guidance path in training)
      - Text instruction via CLIP cross-attention
    """

    def __init__(
        self,
        pipeline,
        config: StableDiffusionComponentConfig,
        default_instruction: str = DEFAULT_HDR_INSTRUCTION,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.default_instruction = default_instruction
        self.default_negative_prompt = default_negative_prompt
        self._freeze_pipeline()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_INSTRUCT_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: str = "float32",
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        use_safetensors: bool = True,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        default_instruction: str = DEFAULT_HDR_INSTRUCTION,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> "FrozenInstructPix2PixStage1":
        require_diffusers()
        from diffusers import StableDiffusionInstructPix2PixPipeline

        cfg = StableDiffusionComponentConfig(model_id=model_id, revision=revision, torch_dtype=torch_dtype, variant=variant)
        dtype = _resolve_dtype(torch_dtype)

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
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
            default_instruction=default_instruction,
            default_negative_prompt=default_negative_prompt,
        )

    def load_explicit_components(self) -> StableDiffusionLatentStack:
        return StableDiffusionLatentStack(self.config)

    def _freeze_pipeline(self) -> None:
        for name in ("vae", "unet", "text_encoder", "text_encoder_2", "image_encoder", "feature_extractor"):
            if hasattr(self.pipeline, name):
                mod = getattr(self.pipeline, name)
                if mod is not None:
                    freeze_module(mod)

    @property
    def device(self) -> torch.device:
        return self.pipeline.device

    @torch.no_grad()
    def forward(
        self,
        ldr: torch.Tensor,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        max_side: int = 768,
        output_range: str = "zero_one",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Args:
            ldr: BCHW float in [0, 1].
            prompt: edit instruction (default HDR expansion text).
            image_guidance_scale: higher -> follow input LDR structure more (typical 1.0-2.0).
            guidance_scale: text instruction strength.
        """
        if ldr.dim() != 4 or ldr.shape[1] != 3:
            raise ValueError(f"ldr must be BCHW with 3 channels, got {tuple(ldr.shape)}")

        instruction = prompt if prompt is not None else self.default_instruction
        neg = negative_prompt if negative_prompt is not None else self.default_negative_prompt

        orig_h, orig_w = ldr.shape[2], ldr.shape[3]
        ldr_resized = resize_ldr_for_sd(ldr, max_side=max_side)
        pil_list = tensor_bchw_to_pil_list(ldr_resized)

        out_pils: List = []
        for im in pil_list:
            call_kw = dict(
                prompt=instruction,
                image=im,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                image_guidance_scale=float(image_guidance_scale),
                generator=generator,
                output_type="pil",
            )
            # InstructPix2Pix supports negative_prompt; omit if empty to avoid edge cases.
            if neg:
                call_kw["negative_prompt"] = neg
            result = self.pipeline(**call_kw)
            out_pils.append(result.images[0])

        pred = pil_list_to_tensor_bchw(out_pils, device=ldr.device, dtype=torch.float32)
        check_sd_output_stats(pred, tag="instruct_pix2pix")
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
        kwargs.setdefault("output_range", "trigate")
        return self.forward(ldr, **kwargs)


def _resolve_dtype(name: str):
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32
