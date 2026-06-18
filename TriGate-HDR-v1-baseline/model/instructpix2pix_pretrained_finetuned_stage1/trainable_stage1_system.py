"""
Fine-tunable InstructPix2Pix + TriGate encoder conditioning for Stage-1 HDR diffusion.

Training anchor: native epsilon MSE on HDR latents with LDR image latents (InstructPix2Pix concat).
Novelty: W1 + SFL (+ KL) on decoded HDR, ramped after a diffusion-only warm-up.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.stable_diffusion_utils import (
    configure_vae_memory_efficient,
    freeze_module,
    prepare_diffusion_pipeline_for_inference,
    require_diffusers,
    resize_ldr_for_sd,
)
from .constants import DEFAULT_HDR_INSTRUCTION, DEFAULT_INSTRUCT_MODEL, DEFAULT_NEGATIVE_PROMPT
from .latent_cond_injector import LatentCondInjector
from .losses import curriculum_novelty_weight, diffusion_epsilon_loss, novelty_losses, predict_x0_from_noise


def _resolve_dtype(name: str):
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


class TrainableTriGateInstructPix2PixStage1(nn.Module):
    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        text_encoder: nn.Module,
        tokenizer,
        scheduler,
        cond_injector: Optional[LatentCondInjector] = None,
        instruction: str = DEFAULT_HDR_INSTRUCTION,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        train_lora: bool = True,
        train_cond_injector: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cond_injector = cond_injector or LatentCondInjector()
        self.instruction = instruction
        self.negative_prompt = negative_prompt
        self._scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

        if train_lora:
            self._attach_unet_lora(rank=lora_rank, alpha=lora_alpha)
        self._set_frozen_policy(train_lora=train_lora, train_cond_injector=train_cond_injector)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_INSTRUCT_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: str = "float32",
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        use_safetensors: bool = True,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        lora_rank: int = 8,
        train_lora: bool = True,
        train_cond_injector: bool = True,
    ) -> "TrainableTriGateInstructPix2PixStage1":
        require_diffusers()
        from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline

        dtype = _resolve_dtype(torch_dtype)
        load_kw = dict(
            revision=revision,
            variant=variant,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=dtype, **load_kw
        )
        prepare_diffusion_pipeline_for_inference(pipe, force_vae_fp32=True)

        if device is not None:
            dev = torch.device(device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        pipe = pipe.to(dev)

        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        model = cls(
            vae=pipe.vae,
            unet=pipe.unet,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            scheduler=scheduler,
            lora_rank=lora_rank,
            train_lora=train_lora,
            train_cond_injector=train_cond_injector,
        )
        # TriGate cond_injector is created after pipe.to(); move entire module tree to dev.
        model = model.to(dev)
        configure_vae_memory_efficient(model.vae)
        if hasattr(model.unet, "enable_gradient_checkpointing"):
            try:
                model.unet.enable_gradient_checkpointing()
            except Exception:
                pass
        model._inference_pipeline = pipe
        return model

    def _attach_unet_lora(self, rank: int, alpha: int) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "UNet LoRA requires peft: pip install peft (see requirements-stable-diffusion.txt)"
            ) from exc

        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.unet = get_peft_model(self.unet, cfg)

    def _set_frozen_policy(self, train_lora: bool, train_cond_injector: bool) -> None:
        freeze_module(self.vae)
        freeze_module(self.text_encoder)
        for p in self.unet.parameters():
            p.requires_grad = False
        if train_lora:
            for n, p in self.unet.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad = True
        for p in self.cond_injector.parameters():
            p.requires_grad = train_cond_injector

    def set_training_phase(self, phase: int) -> None:
        """
        1 = diffusion only (LoRA), 2 = + cond injector, 3 = full (same as 2; novelty via loss weight).
        """
        if phase <= 1:
            for p in self.cond_injector.parameters():
                p.requires_grad = False
            for n, p in self.unet.named_parameters():
                p.requires_grad = "lora" in n.lower()
        else:
            for p in self.cond_injector.parameters():
                p.requires_grad = True
            for n, p in self.unet.named_parameters():
                p.requires_grad = "lora" in n.lower()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode_prompt(self, batch_size: int, device: torch.device) -> torch.Tensor:
        prompts = [self.instruction] * batch_size
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        return self.text_encoder(input_ids)[0]

    def _encode_uncond_prompt(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Embed an empty string for Classifier-Free Guidance."""
        prompts = [""] * batch_size
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        return self.text_encoder(input_ids)[0]

    def _vae_encode(self, images_m11: torch.Tensor, sample: bool = True) -> torch.Tensor:
        images_m11 = images_m11.to(dtype=torch.float32)
        dist = self.vae.encode(images_m11).latent_dist
        latents = dist.sample() if sample else dist.mode()
        return latents * self._scaling_factor

    def _vae_decode_latents(self, latents: torch.Tensor, gradient_checkpoint: bool = False) -> torch.Tensor:
        latents = latents.to(dtype=torch.float32) / self._scaling_factor

        def _decode(z: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(z).sample

        if gradient_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(_decode, latents, use_reentrant=False)
        return _decode(latents)

    def _ldr_to_vae(self, ldr_01: torch.Tensor) -> torch.Tensor:
        return (2.0 * ldr_01.clamp(0.0, 1.0) - 1.0).clamp(-1.0, 1.0)

    def _hdr_to_vae(self, hdr: torch.Tensor) -> torch.Tensor:
        return hdr.clamp(-1.0, 1.0)

    def compute_training_loss(
        self,
        ldr: torch.Tensor,
        hdr: torch.Tensor,
        segmap=None,
        sam_class_masks=None,
        epoch: int = 1,
        diffusion_only_epochs: int = 5,
        novelty_ramp_epochs: int = 10,
        max_novelty_weight: float = 0.25,
        decode_for_novelty: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = ldr.device
        b = ldr.shape[0]

        prompt_embeds = self._encode_prompt(b, device)
        hdr_vae = self._hdr_to_vae(hdr)
        ldr_vae = self._ldr_to_vae(ldr)

        latents = self._vae_encode(hdr_vae, sample=True)
        image_latents = self._vae_encode(ldr_vae, sample=False)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (b,),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        image_latents, gate, class_probs, aux = self.cond_injector(
            ldr, image_latents, timesteps, segmap=segmap, return_aux=True
        )
        concat = torch.cat([noisy_latents, image_latents], dim=1)

        pred_noise = self.unet(
            concat,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        loss_diff = diffusion_epsilon_loss(pred_noise, noise)
        novelty_w = curriculum_novelty_weight(
            epoch, diffusion_only_epochs, novelty_ramp_epochs, max_novelty_weight
        )

        loss = loss_diff
        parts = {"diffusion": loss_diff.detach(), "novelty_weight": torch.tensor(novelty_w)}

        if novelty_w > 0 and decode_for_novelty:
            pred_x0 = predict_x0_from_noise(self.scheduler, noisy_latents, pred_noise, timesteps)
            pred_rgb = self._vae_decode_latents(pred_x0, gradient_checkpoint=True)
            pred_hdr = pred_rgb.clamp(-1.0, 1.0)
            loss_nov, nov_parts = novelty_losses(
                pred_hdr,
                hdr,
                gate,
                class_probs,
                sam_class_masks,
                aux,
            )
            loss = loss + novelty_w * loss_nov
            parts.update({f"nov_{k}": v.detach() if torch.is_tensor(v) else v for k, v in nov_parts.items()})

        parts["total"] = loss.detach()
        return loss, parts

    @torch.no_grad()
    def restore_hdr(
        self,
        ldr: torch.Tensor,
        segmap=None,
        num_inference_steps: int = 25,
        max_side: int = 768,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Reverse diffusion with fine-tuned UNet + TriGate latent conditioning.
        Implements 3-way Classifier-Free Guidance for InstructPix2Pix.
        """
        orig_h, orig_w = ldr.shape[2], ldr.shape[3]
        ldr_work = resize_ldr_for_sd(ldr, max_side=max_side)
        device = ldr.device
        b, _, h, w = ldr_work.shape

        ldr_vae = self._ldr_to_vae(ldr_work)
        image_latents = self._vae_encode(ldr_vae, sample=False)
        
        # 1. Encode prompts for 3-way guidance
        prompt_embeds = self._encode_prompt(b, device)
        uncond_prompt_embeds = self._encode_uncond_prompt(b, device)
        # Combined embeds for efficiency: [cond, uncond_text, uncond_all]
        # In this simplified 3-way, we often just do 2 passes if image_guidance is low, 
        # but here we follow the formal logic.
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            b,
            image_latents.shape[1],
            image_latents.shape[2],
            image_latents.shape[3],
            device=device,
            generator=generator,
            dtype=image_latents.dtype,
        )

        for t in self.scheduler.timesteps:
            t_batch = t.expand(b).long()
            img_cond = self.cond_injector(ldr_work, image_latents, t_batch, segmap=segmap)
            
            if do_classifier_free_guidance:
                # 3-way concat: [latents, latents, latents] with [cond_img, cond_img, zero_img]
                # but we use image_latents as the anchor.
                latent_model_input = torch.cat([latents] * 3)
                image_cond_input = torch.cat([img_cond, img_cond, torch.zeros_like(img_cond)])
                concat = torch.cat([latent_model_input, image_cond_input], dim=1)
                
                # Text: [cond, uncond, uncond]
                text_input = torch.cat([prompt_embeds, uncond_prompt_embeds, uncond_prompt_embeds])
                
                noise_pred = self.unet(
                    concat,
                    torch.cat([t_batch] * 3),
                    encoder_hidden_states=text_input,
                    return_dict=False,
                )[0]
                
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                
                noise_pred = noise_pred_uncond + \
                             guidance_scale * (noise_pred_text - noise_pred_image) + \
                             image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                concat = torch.cat([latents, img_cond], dim=1)
                noise_pred = self.unet(
                    concat,
                    t_batch,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        pred_rgb = self._vae_decode_latents(latents)
        pred_hdr = pred_rgb.clamp(-1.0, 1.0)
        if pred_hdr.shape[2] != orig_h or pred_hdr.shape[3] != orig_w:
            pred_hdr = F.interpolate(pred_hdr, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return pred_hdr
