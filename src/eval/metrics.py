"""
Evaluation metrics and naive baselines for spike-count forecasting.

Provides the core metrics defined in soul.md §7 and two naive baselines:
    - Persistence: y_hat(t+1) = y(t)         (last observation carried forward)
    - Mean-rate:   y_hat(t+1) = mean(y[:t])   (historical average)

All metric functions accept (predicted, target) tensors of shape (*, M) and
return a scalar or per-channel result.

Usage:
    from src.eval.metrics import poisson_nll, pearson_r, r_squared, mae, mse
    from src.eval.metrics import persistence_baseline, mean_rate_baseline
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Core Metrics
# =============================================================================

def poisson_nll(
    predicted: torch.Tensor,
    target: torch.Tensor,
    log_input: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Poisson negative log-likelihood loss.

    This is the primary loss function for spike-count prediction, as spike
    counts are naturally Poisson-distributed.

    Args:
        predicted: Predicted values. If log_input=True, these are interpreted
            as log(rate) and will be exponentiated. Shape (*, M).
        target: Ground-truth spike counts, shape (*, M).
        log_input: If True, predicted is log(rate). If False, predicted is rate.
        eps: Small constant for numerical stability when log_input=False.

    Returns:
        Scalar loss (mean over all elements).
    """
    return F.poisson_nll_loss(
        predicted, target, log_input=log_input, eps=eps, reduction="mean",
    )


def pearson_r(
    predicted: torch.Tensor,
    target: torch.Tensor,
    per_channel: bool = False,
) -> torch.Tensor:
    """
    Pearson correlation coefficient between predicted and target.

    Computed per-channel, then averaged (unless per_channel=True).

    Args:
        predicted: Predicted values, shape (N, M) or (N,).
        target: Ground-truth values, shape (N, M) or (N,).
        per_channel: If True, return per-channel correlations of shape (M,).

    Returns:
        Scalar (mean) or (M,) tensor of correlations.
    """
    # Ensure 2D: (N, M)
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(1)
        target = target.unsqueeze(1)

    # Center the data
    pred_c = predicted - predicted.mean(dim=0, keepdim=True)
    targ_c = target - target.mean(dim=0, keepdim=True)

    # Numerator: sum of products (unnormalized covariance)
    numerator = (pred_c * targ_c).sum(dim=0)

    # Denominator: product of norms (unnormalized standard deviations)
    pred_norm = pred_c.pow(2).sum(dim=0).sqrt()
    targ_norm = targ_c.pow(2).sum(dim=0).sqrt()
    denom = pred_norm * targ_norm

    # Avoid division by zero (constant signal → r undefined → return 0)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    r = numerator / denom

    if per_channel:
        return r
    return r.mean()


def mae(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error.

    Args:
        predicted: Predicted values, shape (*, M).
        target: Ground-truth values, shape (*, M).

    Returns:
        Scalar MAE.
    """
    return F.l1_loss(predicted, target, reduction="mean")


def mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error.

    Args:
        predicted: Predicted values, shape (*, M).
        target: Ground-truth values, shape (*, M).

    Returns:
        Scalar MSE.
    """
    return F.mse_loss(predicted, target, reduction="mean")


def r_squared(
    predicted: torch.Tensor,
    target: torch.Tensor,
    per_channel: bool = False,
) -> torch.Tensor:
    """
    Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot

    Unlike Pearson R (which is scale-invariant), R² penalizes both
    correlation AND scale/bias errors. A model with perfect Pearson R
    but predictions 10x too large will have a negative R².

    Args:
        predicted: Predicted values, shape (N, M) or (N,).
        target: Ground-truth values, shape (N, M) or (N,).
        per_channel: If True, return per-channel R² of shape (M,).

    Returns:
        Scalar (mean across channels) or (M,) tensor of R² values.
    """
    # Ensure 2D: (N, M)
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(1)
        target = target.unsqueeze(1)

    # Total sum of squares (variance of target)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)

    # Residual sum of squares
    ss_res = ((target - predicted) ** 2).sum(dim=0)

    # R² = 1 - SS_res / SS_tot (handle constant target → R²=0)
    r2 = torch.where(
        ss_tot > 0,
        1.0 - ss_res / ss_tot,
        torch.zeros_like(ss_tot),
    )

    if per_channel:
        return r2
    return r2.mean()




# =============================================================================
# Population-Level Metrics
#
# These metrics capture system-wide dynamics rather than per-neuron prediction
# quality. They are more aligned with the neuromorphic twin's goal of
# replicating population-level neural dynamics.
#
# Rationale: per-neuron Pearson r penalizes constant/sparse neurons and misses
# the fact that models successfully capture population structure
# (spatial_pattern_r ~ 0.43 vs per-neuron r ~ 0.09). These metrics provide
# a more faithful assessment of whether a model captures the system's
# temporal dynamics and spatial activation patterns.
# =============================================================================

def population_rate_r(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Pearson r of total population firing rate over time.

    Sums all neurons per time bin to get a 1-D population rate signal,
    then correlates predicted vs. target.  Answers: "do the overall
    activity waves (peaks and valleys) match?"

    Args:
        predicted: Predicted rates, shape (N, M).
        target: Ground-truth spike counts, shape (N, M).

    Returns:
        Scalar Pearson r of the (N,) population rate signals.
    """
    # Sum across neurons -> (N,)
    pred_rate = predicted.sum(dim=-1)
    targ_rate = target.sum(dim=-1)
    return pearson_r(pred_rate, targ_rate)


def spatial_pattern_r(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-timebin correlation of the neuron activity vector.

    For each time bin, correlates the M-dimensional activity vector
    between predicted and target.  Answers: "at each moment, do the
    right neurons fire?"

    Bins where either vector is constant (zero variance) are excluded.

    Args:
        predicted: Predicted rates, shape (N, M).
        target: Ground-truth spike counts, shape (N, M).

    Returns:
        Scalar mean spatial pattern correlation.
    """
    # Center each time bin independently: (N, M)
    pred_c = predicted - predicted.mean(dim=-1, keepdim=True)
    targ_c = target - target.mean(dim=-1, keepdim=True)

    # Per-bin correlation numerator and denominator
    num = (pred_c * targ_c).sum(dim=-1)              # (N,)
    pred_norm = pred_c.pow(2).sum(dim=-1).sqrt()      # (N,)
    targ_norm = targ_c.pow(2).sum(dim=-1).sqrt()      # (N,)
    denom = pred_norm * targ_norm

    # Mask out bins where either signal is constant
    valid = denom > 0
    if valid.sum() == 0:
        return torch.tensor(0.0)

    r_per_bin = num[valid] / denom[valid]
    return r_per_bin.mean()


def population_cosine_sim(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean cosine similarity of population vectors across time.

    Measures how well the predicted population vector aligns with
    ground truth at each time bin, ignoring overall magnitude.

    Args:
        predicted: Predicted rates, shape (N, M).
        target: Ground-truth spike counts, shape (N, M).

    Returns:
        Scalar mean cosine similarity.
    """
    pred_norm = predicted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    targ_norm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos = (predicted / pred_norm * target / targ_norm).sum(dim=-1)
    return cos.mean()


def negative_binomial_nll(
    predicted_rate: torch.Tensor,
    predicted_dispersion: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative Binomial negative log-likelihood loss.

    Models spike counts with overdispersion (variance > mean), which is
    common in real neural data. Parameterized by rate λ and dispersion r:
        Var(X) = λ + λ²/r

    As r → ∞, the Negative Binomial converges to Poisson(λ), so this
    generalises the Poisson loss.

    NLL = -log NB(k | r, p) where p = λ / (λ + r):
        = -log Γ(k + r) + log Γ(r) + log k! - r·log(r/(λ+r)) - k·log(λ/(λ+r))

    Args:
        predicted_rate: Predicted firing rate λ > 0, shape (*, M).
        predicted_dispersion: Dispersion parameter r > 0, shape (*, M).
            Larger r → closer to Poisson. Typically Softplus output.
        target: Ground-truth spike counts k ≥ 0, shape (*, M).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss (mean over all elements).
    """
    # Clamp for numerical stability
    rate = predicted_rate.clamp(min=eps)
    r = predicted_dispersion.clamp(min=eps)
    k = target

    # Probability parameter: p = rate / (rate + r)
    p = rate / (rate + r)

    # NLL components:
    # -log Γ(k + r) + log Γ(r) + log k!
    # = -lgamma(k + r) + lgamma(r) + lgamma(k + 1)
    nll = (
        torch.lgamma(r)
        + torch.lgamma(k + 1)
        - torch.lgamma(k + r)
        - r * torch.log(1 - p + eps)     # -r · log(r / (λ + r))
        - k * torch.log(p + eps)          # -k · log(λ / (λ + r))
    )

    return nll.mean()


def zero_inflated_poisson_nll(
    predicted_rate: torch.Tensor,
    predicted_gate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Zero-Inflated Poisson (ZIP) negative log-likelihood loss.

    Models spike counts with excess zeros beyond what Poisson predicts,
    common in sparse real neural recordings. Parameterized by rate λ and
    zero-inflation probability π:
        P(X = 0) = π + (1 - π) · e^(-λ)
        P(X = k) = (1 - π) · Poisson(k | λ)    for k > 0

    As π → 0, ZIP converges to Poisson(λ), so this generalises the
    Poisson loss.

    Args:
        predicted_rate: Predicted firing rate λ > 0, shape (*, M).
        predicted_gate: Zero-inflation probability π ∈ [0, 1], shape (*, M).
            Typically Sigmoid output. π = P(extra zero).
        target: Ground-truth spike counts k ≥ 0, shape (*, M).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss (mean over all elements).
    """
    rate = predicted_rate.clamp(min=eps)
    gate = predicted_gate.clamp(min=eps, max=1.0 - eps)
    k = target

    # Poisson log-probability for all observations
    # log P_poisson(k | λ) = k·log(λ) - λ - log(k!)
    poisson_log_prob = k * torch.log(rate) - rate - torch.lgamma(k + 1)

    # Case 1: k == 0
    # log P(0) = log(π + (1 - π) · e^(-λ))
    #          = log(π + (1 - π) · exp(-λ))
    zero_prob = gate + (1 - gate) * torch.exp(-rate)
    log_prob_zero = torch.log(zero_prob + eps)

    # Case 2: k > 0
    # log P(k) = log(1 - π) + log P_poisson(k | λ)
    log_prob_nonzero = torch.log(1 - gate + eps) + poisson_log_prob

    # Select based on whether target is zero
    is_zero = (k < 0.5).float()  # Use < 0.5 instead of == 0 for float targets
    log_prob = is_zero * log_prob_zero + (1 - is_zero) * log_prob_nonzero

    # Return negative log-likelihood (mean over all elements)
    return -log_prob.mean()


# =============================================================================
# Naive Baselines
# =============================================================================

def persistence_baseline(
    spike_counts: np.ndarray,
    history_bins: int = 50,
) -> Dict[str, float]:
    """
    Persistence (last-value) baseline: y_hat(t+1) = y(t).

    Predicts the next time step as identical to the current one.
    This is the simplest baseline and establishes a floor for model performance.

    Args:
        spike_counts: Spike-count matrix, shape (M, T_total).
        history_bins: Number of history bins (to skip the initial window).

    Returns:
        Dict with metric names → values:
            poisson_nll, pearson_r, mae, mse
    """
    m, t_total = spike_counts.shape

    # Prediction starts from bin history_bins (first valid prediction point)
    start = max(history_bins, 1)
    predicted = spike_counts[:, start - 1 : t_total - 1]  # y(t)
    target = spike_counts[:, start : t_total]              # y(t+1)

    # Convert to tensors, shape (T, M) for metric functions
    pred_t = torch.tensor(predicted.T, dtype=torch.float32)
    targ_t = torch.tensor(target.T, dtype=torch.float32)

    # Compute all metrics
    # For Poisson NLL, use log_input=False since baseline outputs raw counts
    results = {
        "poisson_nll": float(poisson_nll(pred_t, targ_t, log_input=False)),
        "pearson_r": float(pearson_r(pred_t, targ_t)),
        "mae": float(mae(pred_t, targ_t)),
        "mse": float(mse(pred_t, targ_t)),
    }

    logger.info(
        "Persistence baseline: NLL=%.4f, r=%.4f, MAE=%.4f, MSE=%.4f",
        results["poisson_nll"], results["pearson_r"],
        results["mae"], results["mse"],
    )

    return results


def mean_rate_baseline(
    spike_counts: np.ndarray,
    history_bins: int = 50,
) -> Dict[str, float]:
    """
    Mean-rate baseline: y_hat(t+1) = cumulative mean of y(1:t).

    Predicts the next time step as the running average of all past observations.
    This baseline captures the overall firing rate but ignores temporal dynamics.

    Args:
        spike_counts: Spike-count matrix, shape (M, T_total).
        history_bins: Number of history bins (to skip the initial window).

    Returns:
        Dict with metric names → values:
            poisson_nll, pearson_r, mae, mse
    """
    m, t_total = spike_counts.shape
    start = max(history_bins, 1)

    # Compute cumulative mean up to each time bin
    cumsum = np.cumsum(spike_counts, axis=1).astype(np.float64)
    counts = np.arange(1, t_total + 1, dtype=np.float64)
    cumulative_mean = cumsum / counts[np.newaxis, :]

    # Predicted: cumulative mean at time t predicts time t+1
    predicted = cumulative_mean[:, start - 1 : t_total - 1]
    target = spike_counts[:, start : t_total]

    # Convert to tensors
    pred_t = torch.tensor(predicted.T, dtype=torch.float32)
    targ_t = torch.tensor(target.T, dtype=torch.float32)

    results = {
        "poisson_nll": float(poisson_nll(pred_t, targ_t, log_input=False)),
        "pearson_r": float(pearson_r(pred_t, targ_t)),
        "mae": float(mae(pred_t, targ_t)),
        "mse": float(mse(pred_t, targ_t)),
    }

    logger.info(
        "Mean-rate baseline: NLL=%.4f, r=%.4f, MAE=%.4f, MSE=%.4f",
        results["poisson_nll"], results["pearson_r"],
        results["mae"], results["mse"],
    )

    return results


def compute_all_baselines(
    spike_counts: np.ndarray,
    history_bins: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compute all naive baselines for comparison.

    Args:
        spike_counts: Spike-count matrix, shape (M, T_total).
        history_bins: Number of history bins.

    Returns:
        Dict mapping baseline name → metric dict.
    """
    return {
        "persistence": persistence_baseline(spike_counts, history_bins),
        "mean_rate": mean_rate_baseline(spike_counts, history_bins),
    }


def compute_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Compute standard metrics for a set of predictions.

    Args:
        target: Ground truth (N, M).
        predicted: Predicted rates (N, M).
        dt: Bin width in ms (unused in current metrics but good for API stability).

    Returns:
        Dict with poisson_nll, pearson_r, mae, mse.
    """
    # Convert to tensors for metric functions
    t_tens = torch.tensor(target, dtype=torch.float32)
    p_tens = torch.tensor(predicted, dtype=torch.float32)

    return {
        "poisson_nll": float(poisson_nll(p_tens, t_tens, log_input=False)),
        "pearson_r": float(pearson_r(p_tens, t_tens)),
        "r_squared": float(r_squared(p_tens, t_tens)),
        "mae": float(mae(p_tens, t_tens)),
        "mse": float(mse(p_tens, t_tens)),
        # Population-level metrics (system dynamics)
        "population_rate_r": float(population_rate_r(p_tens, t_tens)),
        "spatial_pattern_r": float(spatial_pattern_r(p_tens, t_tens)),
        "population_cosine_sim": float(population_cosine_sim(p_tens, t_tens)),
    }

