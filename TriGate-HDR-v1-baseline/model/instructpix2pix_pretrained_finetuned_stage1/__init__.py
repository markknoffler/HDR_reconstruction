"""
Stage-1: InstructPix2Pix (pretrained image+text diffusion) + TriGate encoder streams + fine-tune.

Primary class: TrainableTriGateInstructPix2PixStage1
"""

from .constants import DEFAULT_HDR_INSTRUCTION, DEFAULT_INSTRUCT_MODEL, DEFAULT_NEGATIVE_PROMPT
from .trainable_stage1_system import TrainableTriGateInstructPix2PixStage1

__all__ = [
    "TrainableTriGateInstructPix2PixStage1",
    "DEFAULT_INSTRUCT_MODEL",
    "DEFAULT_HDR_INSTRUCTION",
    "DEFAULT_NEGATIVE_PROMPT",
]
