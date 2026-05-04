"""
Ternary-Integer LIF (TI-LIF) neuron for recurrent SNNs.

Outputs ternary spikes {-1, 0, +1} instead of binary {0, 1}.
- Positive threshold θ+: if membrane v > θ+, spike = +1 (excitatory)
- Negative threshold θ-: if membrane v < θ-, spike = -1 (inhibitory)
- Otherwise: spike = 0 (no spike)

This allows the SNN to represent both excitatory and inhibitory signals
directly, carrying ~58% more information per spike event (log₂(3) vs log₂(2)).

Surrogate gradient: piece-wise linear (from Ternary Spike, Guo et al. 2024).

Usage:
    neuron = RecurrentTILIF(
        beta=0.9, threshold=1.0, hidden_size=256, learn_beta=True
    )
"""

import torch
import torch.nn as nn


class TernarySpikeFunction(torch.autograd.Function):
    """
    Ternary spike with piece-wise linear surrogate gradient.

    Forward: outputs {-1, 0, +1} based on membrane potential vs thresholds.
    Backward: piece-wise linear surrogate within ±threshold region.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, slope: float):
        """
        Compute ternary spikes from membrane potential.

        Args:
            membrane: Membrane potential tensor.
            threshold: Positive spike threshold (negative is -threshold).
            slope: Surrogate gradient slope.

        Returns:
            Ternary spike tensor with values in {-1, 0, +1}.
        """
        # Positive spikes where v > +threshold
        pos_spikes = (membrane > threshold).float()
        # Negative spikes where v < -threshold
        neg_spikes = (membrane < -threshold).float()
        # Ternary output: +1, -1, or 0
        spikes = pos_spikes - neg_spikes

        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        """
        Piece-wise linear surrogate gradient for ternary spikes.

        Non-zero gradient in the region [-threshold, +threshold] for both
        positive and negative threshold crossings.
        """
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        slope = ctx.slope

        # Surrogate: wide triangular window around both thresholds.
        # Use 2/slope width (wider than standard 1/slope) to prevent
        # gradient death in the zone between thresholds.
        width = 2.0 / slope
        pos_mask = ((membrane - threshold).abs() < width).float()
        neg_mask = ((membrane + threshold).abs() < width).float()

        # Base gradient: small non-zero gradient everywhere to prevent
        # complete dead zones between thresholds
        base_grad = 0.1 * slope

        # Combined surrogate gradient
        surrogate_grad = slope * (pos_mask + neg_mask) + base_grad

        return grad_output * surrogate_grad, None, None


class RecurrentTILIF(nn.Module):
    """
    Recurrent Ternary-Integer LIF neuron layer.

    Implements a leaky integrate-and-fire neuron with recurrent connections
    and ternary spike output {-1, 0, +1}.

    Dynamics:
        I_t = W_rec @ s_{t-1}   (recurrent input from previous spikes)
        v_t = β * v_{t-1} + x_t + I_t  (leaky integration)
        s_t = TernarySpike(v_t, θ)    (ternary threshold)
        v_t = v_t - s_t * θ          (reset by subtraction)

    Args:
        hidden_size: Number of hidden neurons.
        beta: Membrane decay constant (0 < beta < 1).
        threshold: Positive spike threshold (negative is -threshold).
        learn_beta: If True, beta is a learnable per-neuron parameter.
        learn_threshold: If True, threshold is learnable per-neuron.
        slope: Surrogate gradient slope.
    """

    def __init__(
        self,
        hidden_size: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        slope: float = 25.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.slope = slope

        # Recurrent weight matrix (like snnTorch's linear_features)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)

        # Membrane decay — optionally learnable per-neuron
        if learn_beta:
            # Initialize near the given beta, constrained via sigmoid
            beta_init = torch.log(torch.tensor(beta / (1 - beta)))
            self.beta_logit = nn.Parameter(
                beta_init.expand(hidden_size).clone()
            )
        else:
            self.register_buffer(
                "beta_val", torch.tensor(beta).expand(hidden_size).clone()
            )
        self.learn_beta = learn_beta

        # Threshold — optionally learnable per-neuron
        if learn_threshold:
            self.threshold_param = nn.Parameter(
                torch.full((hidden_size,), threshold)
            )
        else:
            self.register_buffer(
                "threshold_val",
                torch.full((hidden_size,), threshold),
            )
        self.learn_threshold = learn_threshold

    @property
    def beta(self) -> torch.Tensor:
        """Effective membrane decay (constrained to (0, 1) via sigmoid)."""
        if self.learn_beta:
            return torch.sigmoid(self.beta_logit)
        return self.beta_val

    @property
    def threshold(self) -> torch.Tensor:
        """Effective threshold (positive, constrained via softplus)."""
        if self.learn_threshold:
            return nn.functional.softplus(self.threshold_param)
        return self.threshold_val

    def init_state(self, batch_size: int, device: torch.device):
        """
        Initialize membrane potential and spike state to zeros.

        Args:
            batch_size: Batch size for the hidden state.
            device: Device to create tensors on.

        Returns:
            Tuple of (spikes, membrane) initialized to zeros.
        """
        spk = torch.zeros(batch_size, self.hidden_size, device=device)
        mem = torch.zeros(batch_size, self.hidden_size, device=device)
        return spk, mem

    def forward(
        self,
        x: torch.Tensor,
        spk_prev: torch.Tensor,
        mem_prev: torch.Tensor,
    ):
        """
        Single timestep forward pass.

        Args:
            x: Input current (batch, hidden_size).
            spk_prev: Previous spikes (batch, hidden_size).
            mem_prev: Previous membrane potential (batch, hidden_size).

        Returns:
            Tuple of (spikes, membrane) for this timestep.
            Spikes are ternary: {-1, 0, +1}.
        """
        # Recurrent input from previous ternary spikes
        rec_input = self.recurrent(spk_prev)

        # Leaky integration: v_t = β * v_{t-1} + x_t + I_rec
        beta = self.beta
        mem = beta * mem_prev + x + rec_input

        # Ternary spike
        threshold = self.threshold.mean()  # scalar for the spike function
        spk = TernarySpikeFunction.apply(mem, threshold, self.slope)

        # Reset by subtraction (works for ternary: +1 and -1 both reset)
        mem = mem - spk * threshold

        return spk, mem
