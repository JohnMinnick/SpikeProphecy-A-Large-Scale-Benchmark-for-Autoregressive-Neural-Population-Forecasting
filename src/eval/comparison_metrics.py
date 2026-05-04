"""
Comprehensive model comparison metrics for spike count prediction.

Goes beyond Pearson r to provide a multi-faceted view of prediction quality.
All metrics operate on (T, M) arrays: predictions vs ground truth.

Metrics:
    1. Pearson r (per-neuron, then averaged) — linear correlation
    2. R² (coefficient of determination) — explained variance
    3. MAE — mean absolute error
    4. RMSE — root mean squared error
    5. Poisson log-likelihood — the actual training objective
    6. Bits per spike (BPS) — information-theoretic, standard in neuroscience
    7. Population vector correlation — whole-state similarity per time bin
    8. SSIM — structural similarity on the (T, M) "image"
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------
@dataclass
class ComparisonMetrics:
    """Container for all comparison metrics between a model and ground truth."""

    # Per-neuron averaged metrics
    pearson_r: float         # Mean per-neuron Pearson correlation
    r_squared: float         # Mean per-neuron R² (coefficient of determination)
    mae: float               # Mean absolute error (across all bins × neurons)
    rmse: float              # Root mean squared error

    # Poisson-specific metrics
    poisson_nll: float       # Mean Poisson negative log-likelihood
    bits_per_spike: float    # Information gain over homogeneous Poisson (higher = better)

    # Population-level metrics
    pop_vector_corr: float   # Mean per-bin population vector correlation
    ssim: float              # Structural similarity index on the (T, M) heatmap

    # Counts
    n_neurons: int           # Number of neurons evaluated
    n_bins: int              # Number of time bins evaluated

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in self.__dict__.items()}

    def summary(self, name: str = "Model") -> str:
        """Return a formatted summary string."""
        return (
            f"{name}:\n"
            f"  Pearson r:       {self.pearson_r:.4f}\n"
            f"  R²:              {self.r_squared:.4f}\n"
            f"  MAE:             {self.mae:.4f}\n"
            f"  RMSE:            {self.rmse:.4f}\n"
            f"  Poisson NLL:     {self.poisson_nll:.4f}\n"
            f"  Bits/spike:      {self.bits_per_spike:.4f}\n"
            f"  Pop vector r:    {self.pop_vector_corr:.4f}\n"
            f"  SSIM:            {self.ssim:.4f}\n"
            f"  ({self.n_neurons} neurons × {self.n_bins} bins)"
        )


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def pearson_r_per_neuron(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Per-neuron Pearson r between (T, M) arrays.

    Returns (M,) array of correlations. Neurons with zero variance
    get r=0.
    """
    M = gt.shape[1]
    corrs = np.zeros(M)
    for n in range(M):
        g, p = gt[:, n], pred[:, n]
        if g.std() > 0 and p.std() > 0:
            r = np.corrcoef(g, p)[0, 1]
            corrs[n] = r if np.isfinite(r) else 0.0
    return corrs


def r_squared_per_neuron(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Per-neuron R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot. Can be negative if the model is worse
    than predicting the mean.
    """
    M = gt.shape[1]
    r2 = np.zeros(M)
    for n in range(M):
        g = gt[:, n]
        ss_tot = np.sum((g - g.mean()) ** 2)
        if ss_tot > 0:
            ss_res = np.sum((g - pred[:, n]) ** 2)
            r2[n] = 1.0 - ss_res / ss_tot
    return r2


def poisson_nll(
    gt: np.ndarray, pred_rates: np.ndarray, eps: float = 1e-8,
) -> float:
    """
    Mean Poisson negative log-likelihood.

    NLL = mean(pred - gt * log(pred + eps) + log(gt!))

    Lower is better. The log(gt!) term is constant w.r.t. the model
    so we omit it for comparison purposes.

    Args:
        gt: (T, M) ground truth spike counts (non-negative integers).
        pred_rates: (T, M) predicted Poisson rates (non-negative floats).
        eps: Small constant to avoid log(0).

    Returns:
        Scalar mean NLL (omitting the constant log(gt!) term).
    """
    rates = np.clip(pred_rates, eps, None)
    # Poisson NLL: lambda - k * log(lambda)  (ignoring log(k!) constant)
    nll = rates - gt * np.log(rates)
    return float(np.mean(nll))


def bits_per_spike(
    gt: np.ndarray, pred_rates: np.ndarray, eps: float = 1e-8,
) -> float:
    """
    Bits per spike (BPS) — information-theoretic metric.

    Measures how many bits of information the model provides per spike
    beyond a homogeneous Poisson baseline (neuron's mean rate).

    BPS = (1/N) * sum_i [ (1/n_i) * sum_t [ k_it * log2(lambda_it / mu_i) ] ]

    where:
        k_it = observed spike count for neuron i at time t
        lambda_it = predicted rate for neuron i at time t
        mu_i = mean rate of neuron i (homogeneous Poisson baseline)
        n_i = total spike count for neuron i

    Higher BPS = model captures more temporal structure beyond mean rate.
    BPS > 0 means the model is better than predicting the mean rate.

    Reference: Pillow et al. (2008), "Spatio-temporal correlations and
    visual signalling in a complete neuronal population"

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted Poisson rates.
        eps: Small constant to avoid log(0) or division by zero.

    Returns:
        Mean bits per spike across all neurons.
    """
    T, M = gt.shape
    bps_per_neuron = []

    for n in range(M):
        k = gt[:, n]  # Observed spikes
        total_spikes = k.sum()

        if total_spikes < 1:
            # Skip silent neurons
            continue

        mu = k.mean()  # Homogeneous Poisson baseline rate
        lam = np.clip(pred_rates[:, n], eps, None)  # Model rates

        # BPS = (1/n_spikes) * sum(k * log2(lambda / mu))
        log_ratio = np.log2(lam / max(mu, eps))
        bps = np.sum(k * log_ratio) / total_spikes
        bps_per_neuron.append(bps)

    return float(np.mean(bps_per_neuron)) if bps_per_neuron else 0.0


def population_vector_correlation(
    gt: np.ndarray, pred: np.ndarray,
) -> np.ndarray:
    """
    Per-bin population vector correlation.

    At each time bin, compute the Pearson r between the (M,) ground
    truth vector and the (M,) predicted vector. This measures whether
    the model captures the correct spatial (cross-neuron) pattern at
    each time step.

    Args:
        gt: (T, M) ground truth.
        pred: (T, M) predictions.

    Returns:
        (T,) array of per-bin correlations.
    """
    T = gt.shape[0]
    corrs = np.zeros(T)
    for t in range(T):
        g, p = gt[t], pred[t]
        if g.std() > 0 and p.std() > 0:
            r = np.corrcoef(g, p)[0, 1]
            corrs[t] = r if np.isfinite(r) else 0.0
    return corrs


def ssim_2d(
    gt: np.ndarray, pred: np.ndarray,
    C1: float = 0.01, C2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM) on (T, M) arrays.

    Treats the heatmap as a 2D "image" and computes global SSIM.
    Adapted from Wang et al. (2004) for neural data.

    SSIM = (2*mu_x*mu_y + C1²)(2*sigma_xy + C2²) /
           (mu_x² + mu_y² + C1²)(sigma_x² + sigma_y² + C2²)

    Args:
        gt: (T, M) ground truth.
        pred: (T, M) predictions.
        C1, C2: Stabilization constants (fraction of dynamic range).

    Returns:
        Global SSIM value in [0, 1]. Higher = more similar.
    """
    # Normalize to [0, 1] range for SSIM stability
    data_range = max(gt.max() - gt.min(), 1.0)
    C1_sq = (C1 * data_range) ** 2
    C2_sq = (C2 * data_range) ** 2

    mu_x = gt.mean()
    mu_y = pred.mean()
    sigma_x_sq = gt.var()
    sigma_y_sq = pred.var()
    sigma_xy = np.mean((gt - mu_x) * (pred - mu_y))

    numerator = (2 * mu_x * mu_y + C1_sq) * (2 * sigma_xy + C2_sq)
    denominator = (mu_x**2 + mu_y**2 + C1_sq) * (sigma_x_sq + sigma_y_sq + C2_sq)

    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_models(
    gt: np.ndarray,
    pred: np.ndarray,
    name: str = "Model",
) -> ComparisonMetrics:
    """
    Compute all comparison metrics between ground truth and predictions.

    Args:
        gt: (T, M) ground truth spike counts.
        pred: (T, M) predicted rates (continuous, non-negative).
        name: Model name for logging.

    Returns:
        ComparisonMetrics dataclass with all computed metrics.
    """
    T, M = gt.shape
    assert pred.shape == (T, M), f"Shape mismatch: gt={gt.shape}, pred={pred.shape}"

    # 1. Per-neuron Pearson r
    r_per_neuron = pearson_r_per_neuron(gt, pred)
    mean_r = float(np.mean(r_per_neuron))

    # 2. Per-neuron R²
    r2_per_neuron = r_squared_per_neuron(gt, pred)
    mean_r2 = float(np.mean(r2_per_neuron))

    # 3. MAE
    mae_val = float(np.mean(np.abs(gt - pred)))

    # 4. RMSE
    rmse_val = float(np.sqrt(np.mean((gt - pred) ** 2)))

    # 5. Poisson NLL
    nll = poisson_nll(gt, pred)

    # 6. Bits per spike
    bps = bits_per_spike(gt, pred)

    # 7. Population vector correlation
    pop_corrs = population_vector_correlation(gt, pred)
    mean_pop_corr = float(np.mean(pop_corrs))

    # 8. SSIM
    ssim_val = ssim_2d(gt, pred)

    metrics = ComparisonMetrics(
        pearson_r=mean_r,
        r_squared=mean_r2,
        mae=mae_val,
        rmse=rmse_val,
        poisson_nll=nll,
        bits_per_spike=bps,
        pop_vector_corr=mean_pop_corr,
        ssim=ssim_val,
        n_neurons=M,
        n_bins=T,
    )

    logger.info("\n%s", metrics.summary(name))
    return metrics
