"""
TriGateGPURESystem: unified wrapper over Path-G, Path-C, composer, and Path-S.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .gpure_energy import GPUREEnergyConfig, GPURELossParts, compute_gpure_energy
from .trigate_composer import TriGateComposer


@dataclass
class GPUREOutputs:
    x_final: torch.Tensor
    x_gen: torch.Tensor
    x_cold: torch.Tensor
    x_comp: torch.Tensor
    seam_mask: torch.Tensor
    clip_mask: torch.Tensor
    gate: torch.Tensor
    stage1_parts: Dict[str, torch.Tensor]
    stage2_parts: Dict[str, torch.Tensor]
    stage3_out: Optional[torch.Tensor] = None


class TriGateGPURESystem(nn.Module):
    """
    Gate-Partitioned Unified Radiance Energy system.

    Wraps Stage-1 (generative), Stage-2 (cold expansion), differentiable composer,
    and optional Stage-3 seam refiner under one forward + energy interface.
    """

    def __init__(
        self,
        stage1: nn.Module,
        stage2: nn.Module,
        stage3: Optional[nn.Module] = None,
        composer: Optional[TriGateComposer] = None,
        energy_cfg: Optional[GPUREEnergyConfig] = None,
        use_stage3: bool = False,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int | None = None,
        freeze_stage1: bool = False,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.composer = composer or TriGateComposer()
        self.energy_cfg = energy_cfg or GPUREEnergyConfig()
        self.use_stage3 = bool(use_stage3 and stage3 is not None)
        self.stage1_inference_steps = int(stage1_inference_steps)
        self.stage2_inference_steps = stage2_inference_steps
        self.freeze_stage1 = bool(freeze_stage1)

    def _resolve_gate(self, batch: dict, device: torch.device) -> torch.Tensor:
        gate = batch.get("gate")
        if gate is None:
            b, _, h, w = batch["ldr_image"].shape
            return torch.ones(b, 1, h, w, device=device)
        gate = gate.to(device)
        if gate.dim() == 3:
            gate = gate.unsqueeze(1)
        return gate.float()

    def _path_gen(self, batch: dict, ldr: torch.Tensor, mode: str) -> tuple[torch.Tensor, Dict[str, Any]]:
        segmap = batch.get("segmap", ldr)
        if not torch.is_tensor(segmap):
            segmap = ldr
        else:
            segmap = segmap.to(ldr.device)

        infer_only = mode == "eval" or self.freeze_stage1

        if mode == "train" and not infer_only and hasattr(self.stage1, "compute_training_loss"):
            hdr = batch["hdr_image"].to(ldr.device)
            loss, parts = self.stage1.compute_training_loss(ldr, hdr, segmap=segmap)
            with torch.no_grad():
                t0 = torch.zeros((ldr.shape[0],), device=ldr.device, dtype=torch.long)
                if hasattr(self.stage1, "forward"):
                    x_gen, _, _, _ = self.stage1(ldr, t0, segmap=segmap)
                else:
                    x_gen = self.stage1.restore_hdr(
                        ldr, segmap=segmap, num_inference_steps=1
                    )
            parts = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in parts.items()}
            parts["loss"] = loss
            return x_gen, parts

        ctx = torch.no_grad() if infer_only else torch.enable_grad()
        with ctx:
            if hasattr(self.stage1, "restore_hdr"):
                x_gen = self.stage1.restore_hdr(
                    ldr,
                    segmap=segmap,
                    num_inference_steps=self.stage1_inference_steps,
                )
            else:
                t0 = torch.zeros((ldr.shape[0],), device=ldr.device, dtype=torch.long)
                x_gen, _, _, _ = self.stage1(ldr, t0, segmap=segmap)
        return x_gen, {}

    def _path_cold(self, batch: dict, ldr: torch.Tensor, gate: torch.Tensor, mode: str) -> tuple[torch.Tensor, Dict[str, Any]]:
        hdr = batch.get("hdr_image")
        if mode == "train" and hdr is not None and hasattr(self.stage2, "forward"):
            hdr = hdr.to(ldr.device)
            x_cold, parts = self.stage2(hdr, ldr, gate=gate)
            return x_cold, parts

        if mode == "eval":
            if hasattr(self.stage2, "restore_hdr"):
                x_cold = self.stage2.restore_hdr(ldr, gate=gate)
            else:
                x_cold = self.stage2.restore_hdr_train(
                    ldr, gate=gate, n_steps=self.stage2_inference_steps
                )
            return x_cold, {}

        if hasattr(self.stage2, "restore_hdr_train"):
            x_cold = self.stage2.restore_hdr_train(
                ldr, gate=gate, n_steps=self.stage2_inference_steps
            )
        else:
            x_cold = self.stage2.restore_hdr(ldr, gate=gate)
        return x_cold, {}

    def forward(self, batch: dict, mode: str = "train") -> GPUREOutputs:
        device = batch["ldr_image"].device
        ldr = batch["ldr_image"].to(device)
        gate = self._resolve_gate(batch, device)

        x_gen, s1_parts = self._path_gen(batch, ldr, mode=mode)
        x_cold, s2_parts = self._path_cold(batch, ldr, gate, mode=mode)

        compose_out = self.composer(
            x_cold=x_cold,
            x_gen=x_gen,
            gate=gate,
            training_soft_blend=(mode == "train"),
        )

        x_final = compose_out.composed
        stage3_out = None
        if self.use_stage3 and self.stage3 is not None:
            stage3_out = self.stage3(compose_out.composed, x_gen, compose_out.seam_mask)
            x_final = stage3_out

        return GPUREOutputs(
            x_final=x_final,
            x_gen=x_gen,
            x_cold=x_cold,
            x_comp=compose_out.composed,
            seam_mask=compose_out.seam_mask,
            clip_mask=compose_out.clip_mask,
            gate=gate,
            stage1_parts=s1_parts,
            stage2_parts=s2_parts,
            stage3_out=stage3_out,
        )

    def compute_energy(self, outputs: GPUREOutputs, batch: dict) -> GPURELossParts:
        x_gt = batch["hdr_image"].to(outputs.x_final.device)
        cold_loss = outputs.stage2_parts.get("loss")
        gen_loss = outputs.stage1_parts.get("loss") or outputs.stage1_parts.get("total")
        return compute_gpure_energy(
            x_final=outputs.x_final,
            x_gt=x_gt,
            x_gen=outputs.x_gen,
            x_cold=outputs.x_cold,
            seam_mask=outputs.seam_mask,
            clip_mask=outputs.clip_mask,
            cold_loss=cold_loss,
            gen_loss=gen_loss,
            x_seam=outputs.stage3_out,
            x_comp=outputs.x_comp,
            cfg=self.energy_cfg,
        )
