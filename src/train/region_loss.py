"""
Region-specific hybrid loss for heterogeneous neural populations.

KOSMOS finding: Negative binomial loss significantly improves
hippocampal neuron predictions (+6.93 ΔLL, p < 0.05) but provides
no benefit for the general population. This module implements a
hybrid loss that applies different likelihood functions to different
brain regions based on their statistical properties.

Default configuration:
    - Hippocampus (CA1, CA3, DG, SUB): Negative Binomial NLL
    - All other regions: Poisson NLL

The region assignments come from the trainer's region_map attribute,
which maps channel indices to brain region names.

Usage:
    from src.train.region_loss import RegionHybridLoss
    loss_fn = RegionHybridLoss(
        region_map={0: "VISp", 1: "CA1", ...},
        m_max=1240,
    )
    loss = loss_fn(y_hat, y, aux=dispersion_params)
"""

import logging
from typing import Dict, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Brain regions where NegBin loss improves predictions (KOSMOS finding)
HIPPOCAMPAL_REGIONS = {"CA1", "CA3", "DG", "SUB", "HPF", "ProS"}


class RegionHybridLoss(nn.Module):
    """
    Hybrid loss: NegBin for hippocampus, Poisson for rest.

    For hippocampal channels, uses negative binomial NLL which
    better models overdispersed spike counts. For all other channels,
    uses standard Poisson NLL.

    The loss is computed per-channel, then averaged across channels
    and batch, so hippocampal and non-hippocampal neurons contribute
    proportionally to their population size.

    Args:
        region_map: Dict mapping channel index to region name.
        m_max: Maximum number of channels (padded dimension).
        negbin_regions: Set of region names to use NegBin loss for.
            Defaults to HIPPOCAMPAL_REGIONS.
        negbin_weight: Relative weight for NegBin channels.
            Default 1.0 = equal weight. Set >1 to emphasize hippocampus.
    """

    def __init__(
        self,
        region_map: Dict[int, str],
        m_max: int,
        negbin_regions: Optional[Set[str]] = None,
        negbin_weight: float = 1.0,
    ):
        super().__init__()
        self.m_max = m_max
        self.negbin_weight = negbin_weight
        self.negbin_regions = negbin_regions or HIPPOCAMPAL_REGIONS

        # Build per-channel mask: True for NegBin, False for Poisson
        negbin_mask = torch.zeros(m_max, dtype=torch.bool)
        n_negbin = 0
        for ch_idx, region in region_map.items():
            if ch_idx < m_max and region in self.negbin_regions:
                negbin_mask[ch_idx] = True
                n_negbin += 1

        # Register as buffer (moves with model to GPU, not a parameter)
        self.register_buffer("negbin_mask", negbin_mask)
        self.register_buffer(
            "poisson_mask", ~negbin_mask,
        )

        self.n_negbin = n_negbin
        self.n_poisson = m_max - n_negbin

        logger.info(
            "RegionHybridLoss: %d NegBin channels (%s), "
            "%d Poisson channels, weight=%.2f",
            n_negbin,
            ", ".join(sorted(self.negbin_regions & set(region_map.values()))),
            self.n_poisson,
            negbin_weight,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hybrid loss.

        Args:
            y_hat: Predicted rates, shape (batch, M).
            y: Ground truth counts, shape (batch, M).
            aux: Optional dispersion parameters for NegBin, shape (batch, M).
                If None, uses fixed dispersion r=1.
            mask: Optional per-channel mask, shape (batch, M).
                1.0 for active channels, 0.0 for padding.

        Returns:
            Scalar loss (mean over batch and active channels).
        """
        batch, M = y_hat.shape

        # Ensure y_hat is positive (rates must be > 0)
        y_hat_safe = y_hat.clamp(min=1e-8)

        # Poisson NLL: -log P(y | λ) = λ - y·log(λ) + log(y!)
        # We omit log(y!) since it's constant w.r.t. parameters
        poisson_nll = y_hat_safe - y * torch.log(y_hat_safe)

        # NegBin NLL for hippocampal channels
        if aux is not None and self.n_negbin > 0:
            # aux = dispersion parameter r (must be > 0)
            r = aux.clamp(min=1e-4)
            # NegBin NLL: log Γ(r+y) - log Γ(r) - log(y!) +
            #             r·log(r/(r+λ)) + y·log(λ/(r+λ))
            negbin_nll = (
                torch.lgamma(r + y) - torch.lgamma(r)
                - torch.lgamma(y + 1)
                + r * torch.log(r / (r + y_hat_safe))
                + y * torch.log(y_hat_safe / (r + y_hat_safe))
            )
            # NegBin is a log-likelihood, negate for NLL
            negbin_nll = -negbin_nll
        else:
            # No dispersion params → use Poisson everywhere
            negbin_nll = poisson_nll

        # Build per-channel loss: NegBin for hippocampus, Poisson for rest
        # Expand masks to (1, M) for broadcasting with (batch, M)
        negbin_m = self.negbin_mask[:M].unsqueeze(0).float()
        poisson_m = self.poisson_mask[:M].unsqueeze(0).float()

        per_channel_loss = (
            negbin_nll * negbin_m * self.negbin_weight
            + poisson_nll * poisson_m
        )

        # Apply padding mask if provided
        if mask is not None:
            per_channel_loss = per_channel_loss * mask
            n_active = mask.sum().clamp(min=1.0)
        else:
            n_active = float(batch * M)

        return per_channel_loss.sum() / n_active


class FanoAdaptiveLoss(nn.Module):
    """
    Fano-adaptive loss: adjusts loss function per neuron based on Fano factor.

    Sub-Poisson (FF < 1): CMP NLL (handles underdispersion)
    Near-Poisson (1 ≤ FF ≤ 1.5): Poisson NLL (correct model)
    Super-Poisson (FF > 1.5): NegBin NLL (handles overdispersion)

    This is the "ideal" per-neuron loss that KOSMOS analysis suggests,
    though it requires the model to output distribution parameters
    for all three likelihoods. A simpler version (this one) uses
    Poisson everywhere but applies per-neuron loss weighting based
    on Fano factor to focus on neurons where the model can learn most.

    Args:
        fano_factors: np.ndarray of per-channel Fano factors.
        m_max: Maximum number of channels.
        sub_weight: Loss weight for sub-Poisson neurons.
        super_weight: Loss weight for super-Poisson neurons.
    """

    def __init__(
        self,
        fano_factors,
        m_max: int,
        sub_weight: float = 0.5,
        near_weight: float = 1.0,
        super_weight: float = 1.5,
    ):
        super().__init__()
        import numpy as np

        # Build per-channel weights based on Fano factor
        weights = np.ones(m_max, dtype=np.float32) * near_weight
        ff = fano_factors[:m_max] if len(fano_factors) >= m_max else np.pad(
            fano_factors, (0, m_max - len(fano_factors)),
            constant_values=1.0,
        )

        # Sub-Poisson: down-weight (model can't improve much)
        weights[ff < 1.0] = sub_weight
        # Super-Poisson: up-weight (most learnable signal)
        weights[ff > 1.5] = super_weight

        self.register_buffer(
            "channel_weights",
            torch.from_numpy(weights),
        )

        n_sub = (ff < 1.0).sum()
        n_super = (ff > 1.5).sum()
        logger.info(
            "FanoAdaptiveLoss: %d sub (w=%.1f), %d near (w=%.1f), "
            "%d super (w=%.1f)",
            n_sub, sub_weight,
            m_max - n_sub - n_super, near_weight,
            n_super, super_weight,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Fano-weighted Poisson NLL.

        Args:
            y_hat: Predicted rates (batch, M).
            y: Ground truth counts (batch, M).
            mask: Optional mask (batch, M).

        Returns:
            Scalar weighted loss.
        """
        y_hat_safe = y_hat.clamp(min=1e-8)
        nll = y_hat_safe - y * torch.log(y_hat_safe)

        # Apply per-channel Fano-based weights
        M = y_hat.shape[1]
        weights = self.channel_weights[:M].unsqueeze(0)  # (1, M)
        weighted_nll = nll * weights

        if mask is not None:
            weighted_nll = weighted_nll * mask
            n_active = mask.sum().clamp(min=1.0)
        else:
            n_active = float(y_hat.shape[0] * M)

        return weighted_nll.sum() / n_active
