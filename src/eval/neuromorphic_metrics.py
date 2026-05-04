"""
Neuromorphic evaluation metrics for spiking neural networks.

Reports energy, efficiency, and sparsity metrics that are critical
for the SNN deployability story. These complement accuracy metrics
to give a full picture of SNN quality.

Metrics:
    1. SynOps — synaptic operations per timestep (proportional to energy)
    2. Spike Sparsity — fraction of silent neurons per timestep
    3. Theoretical Energy — estimated energy in pJ using SynOps model
    4. Firing Rate — mean spikes per neuron per timestep
    5. Compute Ratio — SynOps(SNN) / FLOPs(ANN) for efficiency comparison

KOSMOS recommendation #5: Neuromorphic metrics alongside accuracy.

References:
    - Merolla et al. 2014 (TrueNorth): ~26 pJ per SynOp
    - Davies et al. 2018 (Loihi): ~0.9 pJ per SynOp (digital)
    - Yin et al. 2021: 0.9 pJ per spike event (Loihi 2)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Energy per synaptic operation on different hardware
ENERGY_PER_SYNOP = {
    "loihi2": 0.9e-12,     # 0.9 pJ (Intel Loihi 2)
    "truenorth": 26e-12,    # 26 pJ (IBM TrueNorth)
    "spinnaker2": 3.6e-12,  # 3.6 pJ (SpiNNaker 2)
    "gpu_fp32": 4.6e-12,    # 4.6 pJ (NVIDIA A100 FP32 MAC)
}


@dataclass
class NeuromorphicMetrics:
    """Container for neuromorphic evaluation results."""

    # Core metrics
    synops_per_timestep: float       # Mean synaptic operations per timestep
    spike_sparsity: float            # Fraction of neurons silent (0-1)
    mean_firing_rate: float          # Mean spikes per neuron per timestep
    total_spikes: int                # Total spike count across all time

    # Energy estimates
    energy_per_timestep_pj: float    # Estimated energy (pJ) per timestep
    energy_per_inference_pj: float   # Total energy for one full inference

    # Comparison metrics
    ann_flops_per_timestep: float    # Equivalent ANN FLOPs for comparison
    compute_ratio: float             # SynOps / ANN_FLOPs (lower = more efficient)

    # Hardware
    hardware: str                    # Reference hardware for energy estimate

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON/W&B serialization."""
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Formatted summary string."""
        return (
            f"Neuromorphic Metrics ({self.hardware}):\n"
            f"  Spike sparsity:     {self.spike_sparsity:.1%}\n"
            f"  Mean firing rate:   {self.mean_firing_rate:.4f} spikes/neuron/step\n"
            f"  SynOps/timestep:    {self.synops_per_timestep:,.0f}\n"
            f"  Energy/timestep:    {self.energy_per_timestep_pj:.2f} pJ\n"
            f"  Energy/inference:   {self.energy_per_inference_pj:.2f} pJ\n"
            f"  ANN FLOPs/step:     {self.ann_flops_per_timestep:,.0f}\n"
            f"  Compute ratio:      {self.compute_ratio:.4f}× (SNN/ANN)\n"
            f"  Total spikes:       {self.total_spikes:,}"
        )


def compute_synops(
    spikes: torch.Tensor,
    weight_shapes: list,
) -> float:
    """
    Compute synaptic operations (SynOps) for a spiking network.

    SynOps = sum over layers of (n_spikes_in × fan_out).
    Only the neurons that actually spike cause computation, so
    sparser activity = fewer SynOps = less energy.

    Args:
        spikes: (batch, T, H) binary spike tensor from SNN forward pass.
        weight_shapes: List of (in_features, out_features) tuples for
            each layer's weight matrix.

    Returns:
        Mean SynOps per timestep per sample.
    """
    batch, T, H = spikes.shape

    # Total spikes per timestep (averaged over batch)
    spikes_per_step = spikes.sum(dim=-1).mean(dim=0)  # (T,)

    total_synops = 0.0
    for in_f, out_f in weight_shapes:
        # Each input spike produces out_f synaptic operations
        # Scale by the fraction of the hidden size this layer uses
        scale = in_f / H if H > 0 else 1.0
        layer_synops = float(spikes_per_step.mean()) * out_f * scale
        total_synops += layer_synops

    return total_synops


def spike_sparsity(spikes: torch.Tensor) -> float:
    """
    Compute spike sparsity: fraction of neurons that are silent.

    Sparsity = 1 - (mean firing probability)
    Higher sparsity = fewer active neurons = more efficient.

    Args:
        spikes: (batch, T, H) binary spike tensor.

    Returns:
        Sparsity in [0, 1]. 0.95 means 95% of neurons are silent
        at any given timestep.
    """
    firing_prob = spikes.float().mean()
    return float(1.0 - firing_prob)


def ann_flops_estimate(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    architecture: str = "lru",
) -> float:
    """
    Estimate FLOPs for one timestep of the equivalent ANN.

    Args:
        input_size: Input dimension (M).
        hidden_size: Hidden dimension (H).
        num_layers: Number of recurrent layers.
        output_size: Output dimension.
        architecture: "lru", "lstm", or "mamba".

    Returns:
        FLOPs per timestep.
    """
    if architecture == "lstm":
        # LSTM: 4 gates × (input→hidden + hidden→hidden) + output
        flops = num_layers * (8 * input_size * hidden_size + 8 * hidden_size**2)
        flops += hidden_size * output_size  # Output projection
    elif architecture == "mamba":
        # Mamba: input proj + SSM computation + output proj
        d_inner = hidden_size * 2  # Mamba expand factor
        d_state = 16  # Default state dimension
        flops = num_layers * (
            2 * input_size * d_inner  # in_proj
            + d_inner * d_state * 3   # SSM (A, B, C)
            + d_inner * input_size    # out_proj
        )
    else:  # LRU
        # LRU: diagonal recurrence + input/output projections
        flops = num_layers * (
            2 * input_size * hidden_size   # Input projection
            + 2 * hidden_size              # Diagonal recurrence (element-wise)
        )
        flops += hidden_size * output_size  # Output projection

    return float(flops)


def compute_neuromorphic_metrics(
    spikes: torch.Tensor,
    model: torch.nn.Module,
    ann_architecture: str = "mamba",
    ann_input_size: int = 1240,
    ann_hidden_size: int = 256,
    ann_num_layers: int = 2,
    ann_output_size: int = 1240,
    hardware: str = "loihi2",
) -> NeuromorphicMetrics:
    """
    Compute all neuromorphic metrics for an SNN inference.

    Args:
        spikes: (batch, T, H) binary spike tensor from SNN forward pass.
        model: SNN model (for extracting weight shapes).
        ann_architecture: Reference ANN architecture for comparison.
        ann_input_size: ANN input dimension.
        ann_hidden_size: ANN hidden dimension.
        ann_num_layers: ANN number of layers.
        ann_output_size: ANN output dimension.
        hardware: Target neuromorphic hardware for energy estimate.

    Returns:
        NeuromorphicMetrics dataclass with all computed metrics.
    """
    batch, T, H = spikes.shape

    # Extract weight shapes from model
    weight_shapes = []
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() == 2:
            weight_shapes.append(param.shape)

    # If no weight shapes found, estimate from hidden size
    if not weight_shapes:
        weight_shapes = [(H, H)]

    # Core metrics
    synops = compute_synops(spikes, weight_shapes)
    sparsity = spike_sparsity(spikes)
    mean_rate = float(spikes.float().mean())
    total = int(spikes.sum().item())

    # Energy estimates
    e_per_synop = ENERGY_PER_SYNOP.get(hardware, 0.9e-12)
    energy_per_step = synops * e_per_synop * 1e12  # Convert to pJ
    energy_per_inference = energy_per_step * T

    # ANN comparison
    ann_flops = ann_flops_estimate(
        ann_input_size, ann_hidden_size, ann_num_layers,
        ann_output_size, ann_architecture,
    )
    ratio = synops / max(ann_flops, 1.0)

    metrics = NeuromorphicMetrics(
        synops_per_timestep=synops,
        spike_sparsity=sparsity,
        mean_firing_rate=mean_rate,
        total_spikes=total,
        energy_per_timestep_pj=energy_per_step,
        energy_per_inference_pj=energy_per_inference,
        ann_flops_per_timestep=ann_flops,
        compute_ratio=ratio,
        hardware=hardware,
    )

    logger.info("\n%s", metrics.summary())
    return metrics
