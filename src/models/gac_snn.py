"""
GAC-SNN: Gated-Aligned-Coupled Spiking Neural Network.

Extends the StudentSNN with two additional biophysical mechanisms aligned
to Mamba's internal signals:
  1. Short-Term Synaptic Plasticity (STP) ← aligned to Mamba B_t
  2. Dendritic Branch Gating ← aligned to Mamba C_t

The existing SelectiveRSynaptic (input-dependent β) already captures
the Mamba Δ ↔ τ(t) relationship. This module adds the remaining two
alignment pathways to complete the mechanism-aligned distillation.

Architecture:
    Input (batch, T, M)
    → Input Projection → LayerNorm → Dropout
    → STP (per-synapse facilitation/depression)       ← aligns to Mamba B
    → N × SelectiveRSynaptic layers (input-dependent β) ← aligns to Mamba Δ
    → Dendritic Branch Gating                          ← aligns to Mamba C
    → Linear Readout → Softplus (Rate Output)

KOSMOS Tier 2F: Mechanism-aligned Mamba→SNN distillation.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.models.selective_rsynaptic import SelectiveRSynaptic

logger = logging.getLogger(__name__)


class ShortTermPlasticity(nn.Module):
    """
    Short-term synaptic plasticity (STP) module.

    Implements Tsodyks-Markram STP dynamics: each synapse has a
    utilization parameter u (facilitation) and an available resource
    x (depression).  The effective synaptic weight is w * u * x.

    When aligned to Mamba's B_t projection, the STP parameters learn
    to modulate input gain in a similar way to how B_t controls how
    much new input enters the SSM state.

    Dynamics (per timestep):
        u_t = U + (1-U) * u_{t-1} * decay_f + U * (1-u_{t-1}) * s_{t-1}
        x_t = 1 - (1-x_{t-1}) * decay_d - u_t * x_{t-1} * s_{t-1}
        effective_input = input * u_t * x_t

    Simplified version for differentiable training:
        gain_t = σ(W_stp · input_t + b_stp) * (1 - fatigue_t)
        fatigue_t = ema(|input_t|, τ_fatigue)

    Args:
        hidden_size: Number of synapses.
        tau_facilitate: Facilitation time constant (learnable).
        tau_depress: Depression time constant (learnable).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Facilitation gate: controls how much synaptic efficacy increases
        # with recent activity (aligned to Mamba B input-projection)
        self.facilitate_gate = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.facilitate_gate.weight, std=0.01)
        nn.init.constant_(self.facilitate_gate.bias, 0.0)  # Start neutral

        # Depression gate: controls resource depletion with sustained input
        self.depress_gate = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.depress_gate.weight, std=0.01)
        nn.init.constant_(self.depress_gate.bias, 0.0)  # Start neutral

        # Learnable time constants for facilitation and depression
        # Initialized so sigmoid ≈ 0.9 (slow dynamics)
        self.log_tau_f = nn.Parameter(torch.full((hidden_size,), 2.2))
        self.log_tau_d = nn.Parameter(torch.full((hidden_size,), 2.2))

        # Store last STP state for alignment loss access
        self._last_gain = None

    def forward(
        self,
        x: torch.Tensor,
        u_prev: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single timestep STP update.

        Args:
            x: Input current (batch, hidden_size).
            u_prev: Previous facilitation state (batch, hidden_size).
            x_prev: Previous depression/resource state (batch, hidden_size).

        Returns:
            Tuple of (modulated_input, u_new, x_new).
        """
        # Facilitation: increase efficacy based on input
        # (analogous to Mamba B projecting input into state)
        tau_f = torch.sigmoid(self.log_tau_f)  # (0, 1) decay rate
        facilitate = torch.sigmoid(self.facilitate_gate(x))
        u_new = tau_f * u_prev + (1 - tau_f) * facilitate

        # Depression: deplete resources with sustained activity
        tau_d = torch.sigmoid(self.log_tau_d)
        depress = torch.sigmoid(self.depress_gate(x))
        x_new = tau_d * x_prev + (1 - tau_d) * (1 - depress * u_new)

        # Effective gain: facilitation × available resources
        gain = u_new * x_new.clamp(min=0.01)

        # Store for alignment loss
        self._last_gain = gain.detach()

        # Modulate input by STP gain
        return x * gain, u_new, x_new


class DendriticGate(nn.Module):
    """
    Dendritic branch gating module.

    Implements multiplicative gating on the readout, aligned to Mamba's
    C_t output projection.  Just as C_t controls how the SSM hidden state
    is projected to the output, the dendritic gate controls which
    information from the spiking activity reaches the output layer.

    Each "dendritic branch" learns an input-dependent gate that modulates
    the readout signal, enabling the SNN to selectively attend to
    different aspects of its hidden dynamics.

    Args:
        hidden_size: Size of the hidden representation.
        num_branches: Number of dendritic branches (default 4).
    """

    def __init__(self, hidden_size: int, num_branches: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_branches = num_branches

        # Each branch has its own gating projection
        # Branch gates: (hidden_size) → (hidden_size / num_branches) per branch
        assert hidden_size % num_branches == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_branches ({num_branches})"
        )
        branch_size = hidden_size // num_branches
        self.branch_size = branch_size

        self.branch_gates = nn.ModuleList([
            nn.Linear(hidden_size, branch_size)
            for _ in range(num_branches)
        ])

        # Initialize gates near 1.0 (pass-through) so the model
        # starts with behavior similar to a standard readout
        for gate in self.branch_gates:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.constant_(gate.bias, 2.0)  # sigmoid(2.0) ≈ 0.88

        # Store last gate values for alignment loss
        self._last_gates = None

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Apply dendritic branch gating to activity.

        Args:
            activity: Hidden activity (batch, hidden_size).

        Returns:
            Gated activity (batch, hidden_size).
        """
        # Split activity into branch-sized chunks
        branches = activity.split(self.branch_size, dim=-1)

        gated_branches = []
        gate_values = []

        for i, (branch, gate_layer) in enumerate(
            zip(branches, self.branch_gates)
        ):
            # Gate is computed from FULL activity (cross-branch interaction)
            gate = torch.sigmoid(gate_layer(activity))
            gate_values.append(gate)

            # Multiplicative gating on this branch
            gated_branches.append(branch * gate)

        # Store for alignment loss
        self._last_gates = torch.cat(gate_values, dim=-1).detach()

        return torch.cat(gated_branches, dim=-1)


class GacStudentSNN(nn.Module):
    """
    Gated-Aligned-Coupled SNN student with mechanism alignment.

    Extends the standard StudentSNN architecture with:
    1. STP (Short-Term Plasticity) before synaptic integration
    2. SelectiveRSynaptic neurons (input-dependent β = τ(t))
    3. DendriticGate on readout

    These three mechanisms map onto Mamba's three dynamic signals:
    - Δ_t → τ(t) via SelectiveRSynaptic.compute_beta()
    - B_t → STP gain via ShortTermPlasticity
    - C_t → Dendritic gate via DendriticGate

    Args:
        input_size: Number of input channels.
        hidden_size: Number of hidden spiking neurons per layer.
        output_size: Number of output channels (forecasting mode).
        beta_init: Initial membrane decay (default 0.9).
        alpha: Synaptic current decay (default 0.85).
        threshold: Spike threshold (default 1.0).
        num_layers: Number of stacked spiking layers (default 1).
        num_dendritic_branches: Number of dendritic branches (default 4).
        dropout: Dropout rate (default 0.2).
        gradient_slope: Surrogate gradient slope (default 25.0).
        learn_alpha: If True, alpha is learnable per-neuron.
        learn_threshold: If True, threshold is learnable per-neuron.
        readout_mode: "mean", "exponential", or "final_mem".
        enable_stp: Enable Short-Term Plasticity (default True).
        enable_dendrite: Enable Dendritic Gating (default True).
        task: "forecasting" or "classification".
        n_classes: Number of classes (required if task='classification').
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        output_size: Optional[int] = None,
        beta_init: float = 0.9,
        alpha: float = 0.85,
        threshold: float = 1.0,
        num_layers: int = 1,
        num_dendritic_branches: int = 4,
        dropout: float = 0.2,
        gradient_slope: float = 25.0,
        learn_alpha: bool = True,
        learn_threshold: bool = False,
        readout_mode: str = "mean",
        enable_stp: bool = True,
        enable_dendrite: bool = True,
        task: str = "forecasting",
        n_classes: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.num_spiking_layers = num_layers
        self.readout_mode = readout_mode
        self.enable_stp = enable_stp
        self.enable_dendrite = enable_dendrite
        self.task = task
        self.n_classes = n_classes

        # Warmup bypass mode: when True, skip STP and dendrite even if
        # enabled. This lets the base SNN train before biophysical modules
        # are active. Controlled by GacDistillTrainer during staged warmup.
        self._warmup_mode = False

        # Input projection: M → hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # STP module (aligned to Mamba B_t)
        if enable_stp:
            self.stp = ShortTermPlasticity(hidden_size)
            logger.info("GAC-SNN: STP enabled (aligned to Mamba B)")
        else:
            self.stp = None

        # Spiking layers using SelectiveRSynaptic (aligned to Mamba Δ_t)
        self.inter_proj = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()

        for i in range(num_layers):
            if i > 0:
                self.inter_proj.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.inter_proj.append(nn.Identity())

            self.layer_norms.append(
                nn.LayerNorm(hidden_size) if i > 0 else nn.Identity()
            )

            self.spiking_layers.append(SelectiveRSynaptic(
                hidden_size=hidden_size,
                alpha=alpha,
                beta_init=beta_init,
                threshold=threshold,
                learn_alpha=learn_alpha,
                learn_threshold=learn_threshold,
                slope=gradient_slope,
            ))

        # Dendritic gating on readout (aligned to Mamba C_t)
        if enable_dendrite:
            self.dendrite = DendriticGate(
                hidden_size, num_branches=num_dendritic_branches,
            )
            logger.info(
                "GAC-SNN: Dendritic gating enabled (%d branches, "
                "aligned to Mamba C)", num_dendritic_branches,
            )
        else:
            self.dendrite = None

        # Output head depends on task mode
        if task == "classification":
            assert n_classes > 0, "n_classes required for classification"
            self.output_proj = nn.Linear(hidden_size, n_classes)
            self.softplus = None  # Not used in classification
            logger.info(
                "GAC-SNN: Classification mode (%d classes)", n_classes,
            )
        else:
            # Forecasting mode: rate output via softplus
            self.output_proj = nn.Linear(hidden_size, self.output_size)
            self.softplus = nn.Softplus()

        # Log architecture
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "GacStudentSNN: input=%d, hidden=%d, output=%d, layers=%d, "
            "params=%d, stp=%s, dendrite=%s, readout=%s",
            input_size, hidden_size, self.output_size, num_layers,
            n_params, enable_stp, enable_dendrite, readout_mode,
        )

    def set_warmup_mode(self, enabled: bool) -> None:
        """
        Enable or disable warmup bypass mode.

        When enabled, STP and dendritic gating are bypassed even if
        the modules exist. This lets the base SNN architecture train
        without the extra biophysical complexity during warmup.

        Called by GacDistillTrainer at the start of each epoch.

        Args:
            enabled: True to bypass STP+dendrite, False to use them.
        """
        self._warmup_mode = enabled

    def forward(
        self, x: torch.Tensor, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with mechanism-aligned intermediate signals.

        Args:
            x: Input tensor (batch, T, M).

        Returns:
            rates: Predicted rates (batch, output_size).
            spikes: Last-layer hidden spikes (batch, T, hidden_size).
        """
        batch_size, T, _ = x.shape

        # Project input features to hidden dimension
        projected = self.input_proj(x)
        projected = self.input_norm(projected)
        projected = self.dropout(projected)

        # Initialize recurrent states for all layers
        spk_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]
        syn_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]
        mem_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_spiking_layers)
        ]

        # STP state (facilitation u, depression x)
        if self.stp is not None:
            u_state = torch.ones(
                batch_size, self.hidden_size, device=x.device,
            ) * 0.5  # Start at moderate facilitation
            x_state = torch.ones(
                batch_size, self.hidden_size, device=x.device,
            )  # Start with full resources

        all_last_layer_spikes = []
        all_last_layer_mems = []

        # Collect alignment signals for loss computation
        all_betas = []  # Mamba Δ ↔ τ(t) alignment
        all_stp_gains = []  # Mamba B ↔ STP alignment
        all_dendrite_gates = []  # Mamba C ↔ dendrite alignment

        # Step through time
        for t in range(T):
            cur = projected[:, t, :]

            # STP modulation (before spiking layers)
            # Skip during warmup bypass mode so the base SNN can train
            # without STP's multiplicative gating bottleneck.
            if self.stp is not None and not self._warmup_mode:
                cur, u_state, x_state = self.stp(cur, u_state, x_state)
                all_stp_gains.append(self.stp._last_gain)

            # Propagate through spiking layers
            for layer_idx in range(self.num_spiking_layers):
                cur = self.inter_proj[layer_idx](cur)
                cur = self.layer_norms[layer_idx](cur)
                cur = self.dropout(cur)

                # SelectiveRSynaptic: captures beta (τ alignment signal)
                spk_states[layer_idx], syn_states[layer_idx], \
                    mem_states[layer_idx] = self.spiking_layers[layer_idx](
                    cur, spk_states[layer_idx],
                    syn_states[layer_idx], mem_states[layer_idx],
                )

                # Collect beta from last layer for alignment
                if layer_idx == self.num_spiking_layers - 1:
                    all_betas.append(
                        self.spiking_layers[layer_idx]._last_beta
                    )

                cur = spk_states[layer_idx]

            all_last_layer_spikes.append(spk_states[-1])
            all_last_layer_mems.append(mem_states[-1])

        # Stack spikes across time: (batch, T, hidden)
        all_hidden_spikes = torch.stack(all_last_layer_spikes, dim=1)

        # Readout: aggregate hidden spikes into activity summary
        if self.readout_mode == "final_mem":
            activity = all_last_layer_mems[-1]
        elif self.readout_mode == "exponential":
            weights = torch.exp(
                torch.linspace(-1.0, 0.0, T, device=x.device)
            )
            weights = weights / weights.sum()
            activity = (
                all_hidden_spikes * weights[None, :, None]
            ).sum(dim=1)
        else:
            activity = all_hidden_spikes.mean(dim=1)

        # Dendritic gating on readout (from full activity)
        # Skip during warmup bypass mode for same reason as STP.
        if self.dendrite is not None and not self._warmup_mode:
            activity = self.dendrite(activity)

        # Output: rate prediction or class logits
        raw_out = self.output_proj(activity)
        if self.task == "classification":
            # Return logits directly (CE loss handles softmax)
            output = raw_out
        else:
            # Forecasting mode: softplus for non-negative rates
            output = self.softplus(raw_out)

        # Store alignment signals for mechanism loss access
        self._alignment_signals = {
            "betas": torch.stack(all_betas, dim=1) if all_betas else None,
            "stp_gains": (
                torch.stack(all_stp_gains, dim=1) if all_stp_gains else None
            ),
            "dendrite_gates": (
                self.dendrite._last_gates if self.dendrite else None
            ),
        }

        return output, all_hidden_spikes

    def get_alignment_signals(self) -> dict:
        """
        Get alignment signals from the last forward pass.

        Returns:
            Dict with keys:
            - 'betas': (batch, T, hidden) — Selective membrane decay
            - 'stp_gains': (batch, T, hidden) — STP facilitation × depression
            - 'dendrite_gates': (batch, hidden) — Dendritic gate values
        """
        return getattr(self, '_alignment_signals', {})

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], input_size: int,
    ) -> "GacStudentSNN":
        """Construct from config dict."""
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            output_size=model_cfg.get("output_size", None),
            beta_init=model_cfg.get("beta", 0.9),
            alpha=model_cfg.get("alpha", 0.85),
            threshold=model_cfg.get("threshold", 1.0),
            num_layers=model_cfg.get("num_layers", 1),
            num_dendritic_branches=model_cfg.get(
                "num_dendritic_branches", 4,
            ),
            dropout=model_cfg.get("dropout", 0.2),
            gradient_slope=model_cfg.get("gradient_slope", 25.0),
            learn_alpha=model_cfg.get("learn_alpha", True),
            learn_threshold=model_cfg.get("learn_threshold", False),
            readout_mode=model_cfg.get("readout_mode", "mean"),
            enable_stp=model_cfg.get("enable_stp", True),
            enable_dendrite=model_cfg.get("enable_dendrite", True),
            task=model_cfg.get("task", "forecasting"),
            n_classes=model_cfg.get("n_classes", 0),
        )
