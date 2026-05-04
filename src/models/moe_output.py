"""
DEPRECATED — Mixture-of-Experts output layer for neuron-conditioned routing.

⚠️ This module was NEVER DEPLOYED. It was a KOSMOS recommendation that was
not integrated into the production pipeline. Kept for reference only.

KOSMOS recommendation #8: Route sub-Poisson vs super-Poisson neurons
to specialized output heads. Sub-Poisson neurons have fundamentally
different statistics (regular spiking, FF < 1) than super-Poisson
neurons (bursty, FF > 1.5), and a single output head cannot optimally
serve both.

Architecture:
    shared_encoder → [hidden] → router (soft gating)
                                 ├─ expert_sub_poisson (Linear, M)
                                 ├─ expert_near_poisson (Linear, M)
                                 └─ expert_super_poisson (Linear, M)
                                 → weighted sum → output

The router produces soft weights per neuron, but is initialized
with a strong prior toward the correct expert based on Fano factors.
During training, it can learn to override this prior if beneficial.

Usage:
    from src.models.moe_output import MoEOutputLayer
    moe = MoEOutputLayer(hidden_size=256, output_size=1240,
                         fano_factors=fano_array)
    output = moe(hidden_state)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MoEOutputLayer(nn.Module):
    """
    Mixture-of-Experts output layer with Fano-conditioned routing.

    Three expert heads specialize in different neuron populations:
        1. Sub-Poisson expert (FF < 1): Regular spiking neurons
        2. Near-Poisson expert (1 ≤ FF ≤ 1.5): Standard neurons
        3. Super-Poisson expert (FF > 1.5): Bursty neurons

    A learnable router (gating network) produces soft weights
    per neuron per timestep, blending expert predictions.

    Args:
        hidden_size: Dimension of encoder hidden state (H).
        output_size: Number of output neurons (M_max).
        n_experts: Number of expert heads (default 3).
        fano_factors: Optional np.ndarray of per-neuron Fano factors
            for initializing the router with a strong prior.
        router_type: "soft" (learned per-timestep routing) or
            "static" (fixed Fano-based routing).
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_experts: int = 3,
        fano_factors: Optional[np.ndarray] = None,
        router_type: str = "soft",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_experts = n_experts
        self.router_type = router_type

        # Expert heads: each is a linear projection H → M
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, output_size)
            for _ in range(n_experts)
        ])

        if router_type == "soft":
            # Learnable router: H → n_experts per neuron
            # Produces (batch, M, n_experts) gating weights
            self.router = nn.Linear(hidden_size, output_size * n_experts)
        else:
            # Static routing: fixed Fano-based assignment
            self.router = None

        # Initialize static routing prior from Fano factors
        if fano_factors is not None:
            self._init_fano_prior(fano_factors)
        else:
            # Default: equal routing to all experts
            prior = torch.ones(output_size, n_experts) / n_experts
            self.register_buffer("fano_prior", prior)

        # Expert labels for logging
        self.expert_names = [
            "sub_poisson", "near_poisson", "super_poisson",
        ][:n_experts]

        logger.info(
            "MoEOutputLayer: H=%d, M=%d, %d experts, router=%s",
            hidden_size, output_size, n_experts, router_type,
        )

    def _init_fano_prior(self, fano_factors: np.ndarray):
        """
        Initialize routing prior from Fano factors.

        Creates a soft assignment matrix where each neuron is strongly
        assigned to its corresponding expert based on FF value.
        """
        M = self.output_size
        ff = fano_factors[:M] if len(fano_factors) >= M else np.pad(
            fano_factors, (0, M - len(fano_factors)),
            constant_values=1.0,
        )

        # Build prior: strong weight toward correct expert
        prior = np.zeros((M, self.n_experts), dtype=np.float32)

        for i in range(M):
            if ff[i] < 1.0:
                # Sub-Poisson → expert 0
                prior[i, 0] = 0.8
                prior[i, 1] = 0.15
                prior[i, 2] = 0.05
            elif ff[i] <= 1.5:
                # Near-Poisson → expert 1
                prior[i, 0] = 0.1
                prior[i, 1] = 0.8
                prior[i, 2] = 0.1
            else:
                # Super-Poisson → expert 2
                prior[i, 0] = 0.05
                prior[i, 1] = 0.15
                prior[i, 2] = 0.8

        self.register_buffer("fano_prior", torch.from_numpy(prior))

        n_sub = (ff < 1.0).sum()
        n_super = (ff > 1.5).sum()
        logger.info(
            "MoE Fano prior: %d sub, %d near, %d super-Poisson neurons",
            n_sub, M - n_sub - n_super, n_super,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: blend expert outputs via routing weights.

        Args:
            h: Hidden state from encoder, shape (batch, H).

        Returns:
            Predicted rates, shape (batch, M).
        """
        batch = h.shape[0]

        # Compute each expert's prediction
        expert_outputs = torch.stack(
            [expert(h) for expert in self.experts], dim=-1,
        )  # (batch, M, n_experts)

        # Compute routing weights
        if self.router_type == "soft" and self.router is not None:
            # Learnable routing: softmax over experts per neuron
            raw_gates = self.router(h)  # (batch, M * n_experts)
            gates = raw_gates.view(batch, self.output_size, self.n_experts)
            gates = F.softmax(gates, dim=-1)  # (batch, M, n_experts)

            # Blend with Fano prior (residual routing)
            # This ensures the model starts from a good prior and
            # can learn to deviate if beneficial.
            prior = self.fano_prior.unsqueeze(0)  # (1, M, n_experts)
            gates = 0.7 * gates + 0.3 * prior
        else:
            # Static routing: use Fano prior directly
            gates = self.fano_prior.unsqueeze(0).expand(
                batch, -1, -1,
            )  # (batch, M, n_experts)

        # Weighted sum of expert outputs
        # (batch, M, n_experts) * (batch, M, n_experts) → sum → (batch, M)
        output = (expert_outputs * gates).sum(dim=-1)

        # Apply softplus to ensure positive rates
        output = F.softplus(output)

        return output

    def get_routing_stats(self, h: torch.Tensor) -> dict:
        """
        Get routing statistics for analysis/logging.

        Args:
            h: Hidden state, shape (batch, H).

        Returns:
            Dict with per-expert mean weight, entropy, etc.
        """
        batch = h.shape[0]

        if self.router_type == "soft" and self.router is not None:
            raw_gates = self.router(h)
            gates = raw_gates.view(batch, self.output_size, self.n_experts)
            gates = F.softmax(gates, dim=-1)
            gates = 0.7 * gates + 0.3 * self.fano_prior.unsqueeze(0)
        else:
            gates = self.fano_prior.unsqueeze(0).expand(batch, -1, -1)

        # Per-expert average weight across all neurons and batch
        mean_weights = gates.mean(dim=(0, 1))  # (n_experts,)

        # Routing entropy (higher = more uniform routing)
        entropy = -(gates * (gates + 1e-8).log()).sum(dim=-1).mean()

        stats = {
            "routing_entropy": float(entropy),
        }
        for i, name in enumerate(self.expert_names):
            stats[f"routing_weight_{name}"] = float(mean_weights[i])

        return stats
