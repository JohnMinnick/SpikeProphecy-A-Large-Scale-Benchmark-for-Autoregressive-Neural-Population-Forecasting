"""
Student Spiking Neural Network (SNN) model.

Implements a recurrent SNN using snnTorch components, designed to learn
temporal dynamics similar to the Teacher LSTM but with sparse spiking
activity in the hidden layers.

Architecture:
    Input (T, M) → Linear Projection → LayerNorm → Dropout
                 → N × Recurrent Spiking Layer (RLeaky or RSynaptic)
                 → Exponential-weighted Readout (from last-layer spikes)
                 → Linear Readout → Softplus (Rate Output)

Config-gated features (all backward-compatible, defaults match v1):
    - learn_beta: Per-neuron learnable membrane decay (default False)
    - num_layers: Stacked spiking layers via ModuleList (default 1)
    - neuron_type: "rleaky" (default) or "rsynaptic" (two-compartment)
    - alpha: Synaptic current decay for RSynaptic (default 0.85)
    - use_layer_norm: LayerNorm on input currents (default False)
    - dropout: Dropout rate on continuous projections (default 0.0)
    - learn_threshold: Per-neuron learnable spike threshold (default False)
    - readout_mode: "mean" (default) or "exponential" weighted readout
    - auxiliary_heads: Optional list of auxiliary heads ["stimulus", "response"]
      These decode behavioral variables from the shared spiking backbone.
    - sgc_enabled: Smoothed Gradient Compensation bypass (default False).
      Adds a parallel smooth Tanh path alongside each spiking layer.
      Blend factor _sgc_lambda is controlled externally by the trainer.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.models.ti_lif import RecurrentTILIF
from src.models.selective_rsynaptic import SelectiveRSynaptic

logger = logging.getLogger(__name__)


class StudentSNN(nn.Module):
    """
    Recurrent SNN student model.

    Args:
        input_size: Number of input channels.
        hidden_size: Number of hidden spiking neurons per layer.
        beta: Membrane potential decay rate (0 < beta < 1).
        threshold: Spike threshold.
        output_size: Number of output channels (defaults to input_size).
        gradient_slope: Slope for the FastSigmoid surrogate gradient.
        learn_beta: If True, beta becomes a learnable per-neuron parameter.
        num_layers: Number of stacked spiking layers (default 1).
        neuron_type: Neuron model — "rleaky" or "rsynaptic" (default "rleaky").
        alpha: Synaptic current decay for RSynaptic neurons (default 0.85).
        use_layer_norm: Apply LayerNorm to input currents (default False).
        dropout: Dropout rate on continuous projections (default 0.0).
        learn_threshold: Per-neuron learnable spike threshold (default False).
        readout_mode: "mean" or "exponential" weighted readout (default "mean").
        auxiliary_heads: Optional list of auxiliary head names. Supported:
            - "stimulus": Linear head predicting left/right contrast (2 outputs).
            - "response": Linear head predicting response choice logits (3 classes).
        sgc_enabled: If True, create parallel smooth bypass paths (Tanh) for
            each spiking layer.  The blend factor _sgc_lambda is set externally
            by the trainer and anneals from 0.5 → 0.0 over training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        output_size: Optional[int] = None,
        gradient_slope: float = 25.0,
        learn_beta: bool = False,
        num_layers: int = 1,
        neuron_type: str = "rleaky",
        alpha: float = 0.85,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        learn_threshold: bool = False,
        readout_mode: str = "mean",
        auxiliary_heads: Optional[list] = None,
        sgc_enabled: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.beta = beta
        self.learn_beta = learn_beta
        self.num_spiking_layers = num_layers
        self.neuron_type = neuron_type.lower()
        self.alpha = alpha
        self.readout_mode = readout_mode

        # Surrogate gradient: FastSigmoid
        spike_grad = surrogate.fast_sigmoid(slope=gradient_slope)

        # Input projection: M → hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LayerNorm on input currents (stabilizes membrane dynamics)
        self.input_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # Dropout on continuous projections (not on binary spikes)
        self.dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # ------------------------------------------------------------------
        # Spiking layers — ModuleList for configurable depth
        # ------------------------------------------------------------------
        self.inter_proj = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()

        for i in range(num_layers):
            # Inter-layer projection (Identity for layer 0; input_proj handles that)
            if i > 0:
                self.inter_proj.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.inter_proj.append(nn.Identity())

            # Per-layer LayerNorm on currents before spiking
            if use_layer_norm and i > 0:
                self.layer_norms.append(nn.LayerNorm(hidden_size))
            else:
                self.layer_norms.append(nn.Identity())

            # Build the spiking neuron layer based on config
            if self.neuron_type == "rsynaptic":
                self.spiking_layers.append(snn.RSynaptic(
                    alpha=alpha,           # Synaptic current decay
                    beta=beta,             # Membrane potential decay
                    threshold=threshold,
                    spike_grad=spike_grad,
                    linear_features=hidden_size,
                    init_hidden=False,
                    learn_alpha=learn_beta,  # Tie alpha learnability to learn_beta
                    learn_beta=learn_beta,
                    learn_threshold=learn_threshold,
                    reset_mechanism="subtract",
                ))
            elif self.neuron_type == "ti_lif":
                # Ternary-Integer LIF: ternary spikes {-1, 0, +1}
                self.spiking_layers.append(RecurrentTILIF(
                    hidden_size=hidden_size,
                    beta=beta,
                    threshold=threshold,
                    learn_beta=learn_beta,
                    learn_threshold=learn_threshold,
                    slope=gradient_slope,
                ))
            elif self.neuron_type == "selective_rsynaptic":
                # Spiking Mamba: input-dependent β (selective gating)
                self.spiking_layers.append(SelectiveRSynaptic(
                    hidden_size=hidden_size,
                    alpha=alpha,
                    beta_init=beta,
                    threshold=threshold,
                    learn_alpha=learn_beta,
                    learn_threshold=learn_threshold,
                    slope=gradient_slope,
                ))
            else:
                # Default: RLeaky (original behavior)
                self.spiking_layers.append(snn.RLeaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad,
                    linear_features=hidden_size,
                    init_hidden=False,
                    learn_beta=learn_beta,
                    learn_threshold=learn_threshold,
                    reset_mechanism="subtract",
                ))

        # ------------------------------------------------------------------
        # Smoothed Gradient Compensation (SGC) bypass
        # ------------------------------------------------------------------
        # Parallel smooth path (Tanh) alongside each spiking layer.
        # Blended with spike output during training to prevent gradient
        # dead zones in ternary thresholding.  _sgc_lambda is set by
        # the trainer and annealed from 0.5 → 0.0 over the warmdown.
        self.sgc_enabled = sgc_enabled
        self._sgc_lambda = 0.0  # Default: pure spiking (trainer sets this)
        if sgc_enabled:
            self.sgc_smooth = nn.ModuleList()
            for _ in range(num_layers):
                # Tanh matches TI-LIF's bipolar [-1, +1] range
                self.sgc_smooth.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                ))

        # Readout: hidden_size (spikes) → output_size (rate)
        # For TI-LIF, we split ternary spikes into excitatory/inhibitory
        # channels (2× hidden_size) to preserve signed information.
        readout_dim = hidden_size * 2 if self.neuron_type == "ti_lif" else hidden_size
        self.output_proj = nn.Linear(readout_dim, self.output_size)

        # Output activation: Softplus for non-negative rates
        self.softplus = nn.Softplus()

        # ------------------------------------------------------------------
        # Auxiliary heads — decode behavioral variables from shared backbone
        # ------------------------------------------------------------------
        self.auxiliary_head_names = auxiliary_heads or []
        self.aux_heads = nn.ModuleDict()

        if "stimulus" in self.auxiliary_head_names:
            # Stimulus head: classify contrast pair into 16 classes.
            # Each class = one (left, right) contrast combo from
            # {0, 0.25, 0.5, 1.0} × {0, 0.25, 0.5, 1.0}.
            self.aux_heads["stimulus"] = nn.Linear(readout_dim, 16)
            logger.info("  + stimulus head: %d → 16 (contrast-pair classes)", readout_dim)

        if "response" in self.auxiliary_head_names:
            # Response head: predict animal's choice (classification, 3 classes)
            # Classes: -1 (left turn), 0 (no-go), +1 (right turn)
            self.aux_heads["response"] = nn.Linear(readout_dim, 3)
            logger.info("  + response head: %d → 3 (choice logits)", readout_dim)

        logger.info(
            "StudentSNN: input=%d, hidden=%d, beta=%.2f, output=%d, "
            "layers=%d, neuron=%s, learn_beta=%s, layer_norm=%s, "
            "dropout=%.2f, learn_threshold=%s, readout=%s, aux_heads=%s, "
            "sgc=%s",
            input_size, hidden_size, beta, self.output_size,
            num_layers, self.neuron_type, learn_beta,
            use_layer_norm, dropout, learn_threshold, readout_mode,
            self.auxiliary_head_names, sgc_enabled,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Processes input through stacked recurrent spiking layers one timestep
        at a time, then reads out the predicted rate from the weighted hidden
        spike activity (last layer) over the full window.

        Args:
            x: Input tensor (batch, T, M).

        Returns:
            rates: Predicted rates (batch, output_size) from hidden activity.
            spikes: Last-layer hidden spikes (batch, T, hidden_size) for regularization.
        """
        batch_size, T, _ = x.shape

        # Project input features to hidden dimension: (batch, T, M) → (batch, T, hidden)
        projected = self.input_proj(x)

        # Apply LayerNorm and Dropout to projected input currents
        projected = self.input_norm(projected)
        projected = self.dropout(projected)

        # ------------------------------------------------------------------
        # Initialize recurrent state for all layers
        # ------------------------------------------------------------------
        # Each layer needs spike and membrane state; RSynaptic also needs syn state
        spk_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]
        mem_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]
        # Synaptic current state — only used by RSynaptic, but allocated for all
        # layers to keep indexing consistent
        syn_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]

        all_last_layer_spikes = []
        all_last_layer_mems = []

        # ------------------------------------------------------------------
        # Step through time, propagating through all layers per timestep
        # ------------------------------------------------------------------
        for t in range(T):
            cur = projected[:, t, :]

            for layer_idx in range(self.num_spiking_layers):
                # Apply inter-layer projection (Identity for layer 0)
                cur = self.inter_proj[layer_idx](cur)

                # Apply per-layer LayerNorm and Dropout on currents
                cur = self.layer_norms[layer_idx](cur)
                cur = self.dropout(cur)

                # Save pre-spike input for SGC smooth bypass path
                pre_spike_input = cur

                # Step the spiking neuron
                if self.neuron_type == "rsynaptic":
                    # RSynaptic: 3 state variables (spk, syn, mem)
                    spk_states[layer_idx], syn_states[layer_idx], mem_states[layer_idx] = \
                        self.spiking_layers[layer_idx](
                            cur,
                            spk_states[layer_idx],
                            syn_states[layer_idx],
                            mem_states[layer_idx],
                        )
                elif self.neuron_type == "ti_lif":
                    # TI-LIF: 2 state variables (spk, mem), ternary output
                    spk_states[layer_idx], mem_states[layer_idx] = \
                        self.spiking_layers[layer_idx](
                            cur,
                            spk_states[layer_idx],
                            mem_states[layer_idx],
                        )
                elif self.neuron_type == "selective_rsynaptic":
                    # SelectiveRSynaptic: 3 state variables (spk, syn, mem)
                    # Same interface as RSynaptic
                    spk_states[layer_idx], syn_states[layer_idx], mem_states[layer_idx] = \
                        self.spiking_layers[layer_idx](
                            cur,
                            spk_states[layer_idx],
                            syn_states[layer_idx],
                            mem_states[layer_idx],
                        )
                else:
                    # RLeaky: 2 state variables (spk, mem)
                    spk_states[layer_idx], mem_states[layer_idx] = \
                        self.spiking_layers[layer_idx](
                            cur,
                            spk_states[layer_idx],
                            mem_states[layer_idx],
                        )

                # ---------------------------------------------------------
                # SGC: blend spiking output with smooth bypass for gradient
                # flow.  _sgc_lambda controls blend: 0.0 = pure spike,
                # 1.0 = pure smooth.  Only active during training.
                # ---------------------------------------------------------
                if (self.sgc_enabled and self.training
                        and self._sgc_lambda > 0):
                    smooth_out = self.sgc_smooth[layer_idx](pre_spike_input)
                    cur = (
                        (1 - self._sgc_lambda) * spk_states[layer_idx]
                        + self._sgc_lambda * smooth_out
                    )
                else:
                    # Pure spiking path (default / inference)
                    cur = spk_states[layer_idx]

            # Collect outputs from the last layer for readout.
            # When SGC is active, `cur` contains the blended spike+smooth
            # output which must flow into readout for gradient flow.
            # Raw spikes are still collected for regularization.
            all_last_layer_spikes.append(cur)
            # Also collect membrane potentials (for final_mem readout)
            all_last_layer_mems.append(mem_states[-1])

        # Stack spikes across time: (batch, T, hidden)
        all_hidden_spikes = torch.stack(all_last_layer_spikes, dim=1)

        # ------------------------------------------------------------------
        # Readout: aggregate hidden spikes into a rate summary
        # ------------------------------------------------------------------
        # For TI-LIF, split ternary spikes into separate excitatory (+1)
        # and inhibitory (-1) channels before readout.
        if self.neuron_type == "ti_lif":
            pos_spikes = torch.clamp(all_hidden_spikes, min=0)   # {0, +1}
            neg_spikes = torch.clamp(-all_hidden_spikes, min=0)  # {0, +1}
            # Concatenate along hidden dim: (batch, T, 2*hidden)
            readout_spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
        else:
            readout_spikes = all_hidden_spikes

        if self.readout_mode == "final_mem":
            # Use the final timestep's continuous membrane potential.
            # This bypasses the discrete spike bottleneck for readout,
            # letting the selective gating's causal accumulation shine.
            activity = all_last_layer_mems[-1]
        elif self.readout_mode == "exponential":
            weights = torch.exp(
                torch.linspace(-1.0, 0.0, T, device=x.device)
            )
            weights = weights / weights.sum()
            activity = (readout_spikes * weights[None, :, None]).sum(dim=1)
        else:
            activity = readout_spikes.mean(dim=1)

        # Decode to output rate via linear projection + softplus (non-negative)
        raw_out = self.output_proj(activity)
        rates = self.softplus(raw_out)

        # Stack membrane potentials across time for hidden-state alignment:
        # (batch, T, hidden_size) — pre-threshold continuous membrane values
        all_membrane_potentials = torch.stack(all_last_layer_mems, dim=1)

        # If no auxiliary heads, return backward-compatible tuple
        if not self.auxiliary_head_names:
            return rates, all_hidden_spikes

        # Multi-head output: run auxiliary heads on same activity vector
        result = {
            "rates": rates,
            "spikes": all_hidden_spikes,
            # Expose membrane potentials for hidden-state alignment loss.
            # Shape: (batch, T, hidden_size)
            "membrane_potentials": all_membrane_potentials,
        }

        if "stimulus" in self.aux_heads:
            # Contrast prediction (unbounded — can be 0..1 range)
            result["stimulus"] = self.aux_heads["stimulus"](activity)

        if "response" in self.aux_heads:
            # Response class logits (raw, CrossEntropy expects unnormalized)
            result["response"] = self.aux_heads["response"](activity)

        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_size: int) -> "StudentSNN":
        """Construct from config dict.

        Supports all architecture config keys with backward-compatible defaults.
        """
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 128),
            beta=model_cfg.get("beta", 0.9),
            threshold=model_cfg.get("threshold", 1.0),
            learn_beta=model_cfg.get("learn_beta", False),
            num_layers=model_cfg.get("num_layers", 1),
            neuron_type=model_cfg.get("neuron_type", "rleaky"),
            alpha=model_cfg.get("alpha", 0.85),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            dropout=model_cfg.get("dropout", 0.0),
            learn_threshold=model_cfg.get("learn_threshold", False),
            readout_mode=model_cfg.get("readout_mode", "mean"),
            auxiliary_heads=model_cfg.get("auxiliary_heads", None),
            sgc_enabled=model_cfg.get("sgc_enabled", False),
        )
