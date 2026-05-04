"""
Conway-Maxwell-Poisson (CMP) loss function for spike count prediction.

The CMP distribution generalizes Poisson by adding a dispersion parameter
ν that handles both over-dispersion (ν < 1, super-Poisson) and
under-dispersion (ν > 1, sub-Poisson).  When ν = 1, CMP reduces to
Poisson.

This directly addresses KOSMOS Tier 2E: 28% of neurons are sub-Poisson
(Fano factor < 1) where the standard Poisson loss misspecifies the
variance, leading to suboptimal gradients.

Usage:
    from src.train.cmp_loss import cmp_nll, cmp_nll_per_element

    # Scalar loss (mean over batch × neurons)
    loss = cmp_nll(rates, counts, nu)

    # Per-element loss for masked training (batch, M)
    per_element = cmp_nll_per_element(rates, counts, nu)

References:
    - Shmueli et al. (2005) "A useful distribution for fitting discrete data"
    - Sellers & Shmueli (2010) "A flexible regression model for count data"
"""

import torch
import torch.nn as nn
from typing import Optional


# Maximum count value for truncating the normalizing constant sum.
# Spike counts at 50ms bins rarely exceed 20, so 50 is very safe.
_Z_TRUNCATION = 50


def _log_z_cmp(
    log_lambda: torch.Tensor,
    nu: torch.Tensor,
    k_max: int = _Z_TRUNCATION,
) -> torch.Tensor:
    """
    Compute log of the CMP normalizing constant Z(λ, ν) via logsumexp.

    Z(λ, ν) = Σ_{j=0}^{k_max} λ^j / (j!)^ν
            = Σ_{j=0}^{k_max} exp(j·log(λ) - ν·log(j!))

    Uses logsumexp for numerical stability.

    Args:
        log_lambda: Log of the rate parameter, shape (...).
        nu: Dispersion parameter, shape (...) or broadcastable.
        k_max: Truncation point for the infinite series.

    Returns:
        log Z(λ, ν), same shape as log_lambda.
    """
    # Precompute log(j!) for j = 0, ..., k_max
    # log(0!) = 0, log(1!) = 0, log(2!) = 0.693, ...
    j_vals = torch.arange(k_max + 1, device=log_lambda.device, dtype=log_lambda.dtype)
    log_factorials = torch.lgamma(j_vals + 1)  # lgamma(j+1) = log(j!)

    # Expand dimensions for broadcasting:
    # log_lambda: (...) -> (..., 1)
    # nu: (...) -> (..., 1)
    # j_vals, log_factorials: (k_max+1,)
    log_lambda_exp = log_lambda.unsqueeze(-1)  # (..., 1)
    nu_exp = nu.unsqueeze(-1)                  # (..., 1)

    # log terms: j * log(λ) - ν * log(j!)
    log_terms = j_vals * log_lambda_exp - nu_exp * log_factorials  # (..., k_max+1)

    # logsumexp over the series dimension
    return torch.logsumexp(log_terms, dim=-1)  # (...)


def cmp_nll_per_element(
    rates: torch.Tensor,
    counts: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-element CMP negative log-likelihood.

    -log P(y | λ, ν) = -y·log(λ) + ν·log(y!) + log Z(λ, ν)

    Args:
        rates: Predicted rates λ, shape (batch, M). Must be > 0.
        counts: Observed spike counts y, shape (batch, M).
        nu: Dispersion parameter, shape (M,) or (batch, M).
            ν > 1 = sub-Poisson (under-dispersed),
            ν < 1 = super-Poisson (over-dispersed),
            ν = 1 = Poisson.
        eps: Small constant for numerical stability.

    Returns:
        Per-element NLL, shape (batch, M).
    """
    # Ensure rates are positive
    lambda_safe = rates.clamp(min=eps)
    log_lambda = torch.log(lambda_safe)

    # log(y!) = lgamma(y + 1)
    log_y_factorial = torch.lgamma(counts + 1)

    # CMP NLL = -y * log(λ) + ν * log(y!) + log Z(λ, ν)
    log_z = _log_z_cmp(log_lambda, nu)

    nll = -counts * log_lambda + nu * log_y_factorial + log_z

    return nll


def cmp_nll(
    rates: torch.Tensor,
    counts: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute mean CMP negative log-likelihood (scalar loss).

    Args:
        rates: Predicted rates λ, shape (batch, M).
        counts: Observed spike counts y, shape (batch, M).
        nu: Dispersion parameter, shape (M,) or (batch, M).
        eps: Small constant for numerical stability.

    Returns:
        Scalar mean NLL.
    """
    return cmp_nll_per_element(rates, counts, nu, eps).mean()


class LearnableDispersion(nn.Module):
    """
    Learnable per-neuron dispersion parameter for CMP loss.

    Stores a raw (unconstrained) parameter that is mapped to ν > 0
    via softplus.  Initialized at ν = 1.0 (Poisson) so the CMP
    starts equivalent to standard Poisson NLL and learns to deviate.

    The dispersion is a property of the neuron, not the model's
    prediction quality, so it's stored separately from the model.

    Args:
        num_neurons: Number of output neurons (M_max).
    """

    def __init__(self, num_neurons: int):
        super().__init__()
        # Initialize raw parameter so that softplus(raw) ≈ 1.0
        # softplus(x) = log(1 + exp(x)); softplus(0.5413) ≈ 1.0
        init_val = 0.5413
        self.raw_nu = nn.Parameter(
            torch.full((num_neurons,), init_val)
        )

    def forward(self) -> torch.Tensor:
        """
        Return the dispersion parameter ν, shape (M_max,).

        Uses softplus to ensure ν > 0, with a small floor to
        prevent numerical issues near ν = 0.
        """
        return torch.nn.functional.softplus(self.raw_nu).clamp(min=0.01)

    @torch.no_grad()
    def get_stats(self) -> dict:
        """
        Return dispersion statistics for logging.

        Returns:
            Dict with mean, median, min, max, frac_sub_poisson
            (fraction of neurons with ν > 1).
        """
        nu = self.forward()
        return {
            "nu_mean": float(nu.mean()),
            "nu_median": float(nu.median()),
            "nu_min": float(nu.min()),
            "nu_max": float(nu.max()),
            "nu_frac_sub_poisson": float((nu > 1.0).float().mean()),
        }
