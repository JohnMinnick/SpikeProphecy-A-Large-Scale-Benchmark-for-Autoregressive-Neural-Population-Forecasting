"""
SelectiveRSynaptic — Spiking Mamba neuron with input-dependent membrane decay.

Bridges the gap between Mamba's selective state space model and spiking
neural networks. The key insight: Mamba's power comes from input-dependent
state transitions (selective gating). Standard LIF/RSynaptic neurons use
a *static* decay constant β, which limits their expressiveness.

SelectiveRSynaptic makes β dynamic:
    β_t = σ(W_gate · x_t + b_gate)

This gives the neuron **selective memory** — it can choose to remember
(high β) or forget (low β) based on the current input, exactly like
Mamba's selective scan mechanism.

Mathematical connection:
    Mamba SSM:     h_t = Ā(x_t) · h_{t-1} + B̄(x_t) · x_t
    Standard LIF:  v_t = β · v_{t-1} + x_t            (β is static)
    This neuron:   v_t = β(x_t) · v_{t-1} + x_t       (β is input-dependent)

The selective gating is the architectural bridge between SSMs and SNNs.
When trained standalone on ground truth, this neuron can natively capture
the temporal processing that Mamba achieves through its selective scan,
without needing distillation.

Usage:
    neuron = SelectiveRSynaptic(hidden_size=256, learn_alpha=True)
"""

import torch
import torch.nn as nn
from snntorch import surrogate


class SelectiveSpikeFunction(torch.autograd.Function):
    """
    Standard binary spike with fast sigmoid surrogate gradient.

    Identical to snnTorch's FastSigmoid but implemented directly to
    avoid coupling to snnTorch internals while keeping the neuron
    self-contained.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, slope: float):
        """
        Compute binary spikes where membrane exceeds threshold.

        Args:
            membrane: Membrane potential tensor.
            threshold: Spike threshold.
            slope: Surrogate gradient slope.

        Returns:
            Binary spike tensor (0 or 1).
        """
        spikes = (membrane > threshold).float()
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        """Fast sigmoid surrogate gradient."""
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        slope = ctx.slope

        # Fast sigmoid: slope / (1 + slope * |v - threshold|)^2
        shifted = membrane - threshold
        surrogate_grad = slope / (1.0 + slope * shifted.abs()) ** 2

        return grad_output * surrogate_grad, None, None


class SelectiveRSynaptic(nn.Module):
    """
    Two-compartment recurrent spiking neuron with input-dependent decay.

    Like snnTorch's RSynaptic, but with a critical addition: the membrane
    decay β is computed dynamically from the current input via a learned
    linear gate, mirroring Mamba's selective state transition mechanism.

    Dynamics:
        # Selective gating (the Mamba mechanism)
        β_t = σ(W_beta · x_t + b_beta)          # Input-dependent decay

        # Two-compartment integration (standard RSynaptic)
        syn_t = α · syn_{t-1} + x_t + W_rec · s_{t-1}  # Synaptic current
        v_t   = β_t · v_{t-1} + syn_t                   # Membrane potential

        # Spike and reset
        s_t = Θ(v_t - threshold)
        v_t = v_t - s_t · threshold              # Subtract reset

    Args:
        hidden_size: Number of hidden neurons.
        alpha: Synaptic current decay (static, 0 < α < 1).
        beta_init: Initial bias for the beta gate (default 0.9).
        threshold: Spike threshold.
        learn_alpha: If True, alpha is learnable per-neuron.
        learn_threshold: If True, threshold is learnable per-neuron.
        slope: Surrogate gradient slope for FastSigmoid.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha: float = 0.85,
        beta_init: float = 0.9,
        threshold: float = 1.0,
        learn_alpha: bool = True,
        learn_threshold: bool = False,
        slope: float = 25.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.slope = slope

        # Recurrent weight (same as snnTorch's linear_features)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)

        # ===================================================================
        # THE KEY INNOVATION: Input-dependent β gate
        # β_t = sigmoid(W_beta · x_t + b_beta)
        #
        # The bias is initialized so that sigmoid(b) ≈ beta_init,
        # meaning the neuron starts with similar dynamics to a standard
        # RSynaptic neuron but can learn to selectively gate.
        # ===================================================================
        self.beta_gate = nn.Linear(hidden_size, hidden_size)
        # Initialize bias so sigmoid(b) ≈ beta_init
        beta_bias_init = torch.log(
            torch.tensor(beta_init / (1.0 - beta_init))
        )
        # Use small Normal init (not zeros!) to break symmetry saddle.
        # Zero-init creates sigmoid gradient squeeze: σ'(2.2) = 0.09,
        # choking gate gradients to 9% and causing ~15 epoch dead zone.
        # Normal(0, 0.01) gives gradients an asymmetric hook immediately.
        nn.init.normal_(self.beta_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_gate.bias, beta_bias_init.item())

        # Beta variance tracking — set during forward pass for external
        # monitoring of gate selectivity (is it actually input-dependent?).
        self._last_beta = None

        # Synaptic current decay α — optionally learnable
        if learn_alpha:
            alpha_logit = torch.log(torch.tensor(alpha / (1.0 - alpha)))
            self.alpha_logit = nn.Parameter(
                alpha_logit.expand(hidden_size).clone()
            )
        else:
            self.register_buffer(
                "alpha_val", torch.tensor(alpha).expand(hidden_size).clone()
            )
        self.learn_alpha = learn_alpha

        # Threshold — optionally learnable
        if learn_threshold:
            self.threshold_param = nn.Parameter(
                torch.full((hidden_size,), threshold)
            )
        else:
            self.register_buffer(
                "threshold_val", torch.full((hidden_size,), threshold)
            )
        self.learn_threshold = learn_threshold

    @property
    def alpha(self) -> torch.Tensor:
        """Effective synaptic current decay (constrained to (0, 1))."""
        if self.learn_alpha:
            return torch.sigmoid(self.alpha_logit)
        return self.alpha_val

    @property
    def threshold(self) -> torch.Tensor:
        """Effective spike threshold."""
        if self.learn_threshold:
            return nn.functional.softplus(self.threshold_param)
        return self.threshold_val

    def compute_beta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute input-dependent membrane decay β_t.

        This is the selective gating mechanism — the neuron decides
        how much of its membrane history to retain based on the
        current input.

        Args:
            x: Input current (batch, hidden_size).

        Returns:
            Per-neuron, per-timestep decay (batch, hidden_size) in (0, 1).
        """
        return torch.sigmoid(self.beta_gate(x))

    def init_state(self, batch_size: int, device: torch.device):
        """
        Initialize all states to zeros.

        Args:
            batch_size: Batch size for the hidden state.
            device: Device to create tensors on.

        Returns:
            Tuple of (spikes, synaptic_current, membrane) initialized to zeros.
        """
        spk = torch.zeros(batch_size, self.hidden_size, device=device)
        syn = torch.zeros(batch_size, self.hidden_size, device=device)
        mem = torch.zeros(batch_size, self.hidden_size, device=device)
        return spk, syn, mem

    def forward(
        self,
        x: torch.Tensor,
        spk_prev: torch.Tensor,
        syn_prev: torch.Tensor,
        mem_prev: torch.Tensor,
    ):
        """
        Single timestep forward pass.

        Args:
            x: Input current (batch, hidden_size).
            spk_prev: Previous spikes (batch, hidden_size).
            syn_prev: Previous synaptic current (batch, hidden_size).
            mem_prev: Previous membrane potential (batch, hidden_size).

        Returns:
            Tuple of (spikes, synaptic_current, membrane) for this timestep.
        """
        # Recurrent input from previous spikes
        rec_input = self.recurrent(spk_prev)

        # Synaptic current integration (same as standard RSynaptic)
        alpha = self.alpha
        syn = alpha * syn_prev + x + rec_input

        # THE KEY: Input-dependent membrane decay
        beta_t = self.compute_beta(x)

        # Store for external monitoring (gate selectivity diagnostic)
        self._last_beta = beta_t.detach()

        # Membrane integration with selective decay
        mem = beta_t * mem_prev + syn

        # Spike generation with surrogate gradient
        threshold_val = self.threshold.mean()
        spk = SelectiveSpikeFunction.apply(mem, threshold_val, self.slope)

        # Reset by subtraction
        mem = mem - spk * threshold_val

        return spk, syn, mem
