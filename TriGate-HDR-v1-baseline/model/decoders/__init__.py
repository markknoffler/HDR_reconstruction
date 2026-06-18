from .stable_diffusion_instruct_pix2pix_decoder import FrozenInstructPix2PixStage1
from .stable_diffusion_stage1_decoder import FrozenStableDiffusionStage1

# Default Stage-1 frozen diffusion baseline: image + text native edit model.
FrozenStableDiffusionStage1Default = FrozenInstructPix2PixStage1

__all__ = [
    "FrozenInstructPix2PixStage1",
    "FrozenStableDiffusionStage1",
    "FrozenStableDiffusionStage1Default",
]
