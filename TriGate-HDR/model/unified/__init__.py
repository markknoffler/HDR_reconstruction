"""
Gate-Partitioned Unified Radiance Energy (GPURE) framework.

Single variational objective coupling Path-G (generative), Path-C (cold expansion),
and Path-S (seam refinement) with optically grounded cold forward and radiometric synapses.
"""

from .gpure_energy import GPUREEnergyConfig, compute_gpure_energy
from .optical_cold_forward import OpticalColdForward
from .radiometric_synapse import RSOCell, RSOStem
from .trigate_composer import TriGateComposer, build_composited_input, build_seam_band
from .trigate_gpure_system import GPUREOutputs, TriGateGPURESystem

__all__ = [
    "GPUREEnergyConfig",
    "GPUREOutputs",
    "OpticalColdForward",
    "RSOCell",
    "RSOStem",
    "TriGateComposer",
    "TriGateGPURESystem",
    "build_composited_input",
    "build_seam_band",
    "compute_gpure_energy",
]
