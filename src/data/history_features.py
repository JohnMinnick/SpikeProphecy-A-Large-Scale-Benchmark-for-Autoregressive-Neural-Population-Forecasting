"""
Spike-history input features for temporal context augmentation.

Computes per-neuron derived features from binned spike-count matrices to
provide the teacher model with explicit temporal context. All features
are strictly causal (computed from past bins only) to prevent data leakage.

Features (ADR-0009 Batch B):
    - ISI: Normalized time since last spike per unit
    - EMA rate: Exponential moving average of recent spike counts
    - Refractory indicator: Binary flag for sub-refractory-period bins

Usage:
    from src.data.history_features import compute_history_features

    features, n_per_channel = compute_history_features(spike_counts, config)
    # features: (N * M, T_total) — N enabled features × M channels
    # n_per_channel: int — number of features per channel (0–3)
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_isi_features(
    counts: np.ndarray,
    bin_width_ms: float = 10.0,
    max_isi_ms: float = 500.0,
) -> np.ndarray:
    """
    Compute normalized inter-spike interval (time since last spike).

    For each unit and each time bin, computes the elapsed time (in ms) since
    the most recent bin with count >= 1. The result is clipped to max_isi_ms
    and normalized to [0, 1].

    Strictly causal: only looks at bins *before* the current bin.
    Bins with no prior spike get the maximum value (1.0 normalized).

    Args:
        counts: Spike-count matrix, shape (M, T_total).
        bin_width_ms: Width of each time bin in milliseconds.
        max_isi_ms: Maximum ISI value before clipping (in ms).

    Returns:
        ISI features, shape (M, T_total), values in [0, 1].
    """
    m, t_total = counts.shape
    max_isi_bins = max_isi_ms / bin_width_ms

    # Output array — initialize to max (no prior spike)
    isi = np.ones((m, t_total), dtype=np.float32)

    # Track bins since last spike for each unit (start at max)
    bins_since_spike = np.full(m, max_isi_bins, dtype=np.float32)

    for t in range(t_total):
        # Current ISI = bins since last spike (causal: uses state from t-1)
        isi[:, t] = np.minimum(bins_since_spike, max_isi_bins) / max_isi_bins

        # Update: reset counter for units that spiked at time t
        spiked = counts[:, t] >= 1
        bins_since_spike[spiked] = 0.0
        bins_since_spike[~spiked] += 1.0

    return isi


def compute_ema_rate(
    counts: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Compute exponential moving average of spike counts.

    Provides a smoothed recent-rate estimate for each unit. Uses the
    *previous* bin's count to maintain strict causality:
        ema[t] = alpha * counts[t-1] + (1 - alpha) * ema[t-1]

    Args:
        counts: Spike-count matrix, shape (M, T_total).
        alpha: Smoothing factor in (0, 1). Smaller = smoother.

    Returns:
        EMA rate features, shape (M, T_total), non-negative.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    m, t_total = counts.shape
    ema = np.zeros((m, t_total), dtype=np.float32)

    # ema[0] = 0 (no past data)
    for t in range(1, t_total):
        # Causal: use count from previous bin (t-1)
        ema[:, t] = alpha * counts[:, t - 1] + (1.0 - alpha) * ema[:, t - 1]

    return ema


def compute_refractory(
    counts: np.ndarray,
    refractory_bins: int = 1,
) -> np.ndarray:
    """
    Compute refractory period indicator.

    Binary flag that is 1 when a spike occurred within the last
    `refractory_bins` bins, indicating the unit is likely in its
    refractory period and less likely to fire.

    Strictly causal: only looks at past bins.

    Args:
        counts: Spike-count matrix, shape (M, T_total).
        refractory_bins: Number of bins to look back for refractory check.

    Returns:
        Refractory indicator, shape (M, T_total), values in {0.0, 1.0}.

    Raises:
        ValueError: If refractory_bins < 1.
    """
    if refractory_bins < 1:
        raise ValueError(
            f"refractory_bins must be >= 1, got {refractory_bins}"
        )

    m, t_total = counts.shape
    refractory = np.zeros((m, t_total), dtype=np.float32)

    for t in range(1, t_total):
        # Look back up to refractory_bins bins
        lookback_start = max(0, t - refractory_bins)
        # 1.0 if any spike occurred in the lookback window
        refractory[:, t] = (
            np.any(counts[:, lookback_start:t] >= 1, axis=1)
        ).astype(np.float32)

    return refractory


def compute_history_features(
    counts: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """
    Compute all enabled spike-history features.

    Dispatches to individual feature functions based on the
    ``history_features`` config section. Returns the feature matrix
    to be vertically concatenated with the original spike counts.

    Args:
        counts: Spike-count matrix, shape (M, T_total).
        config: Data config dict. Reads ``history_features`` sub-dict.

    Returns:
        Tuple of:
            - features: (N * M, T_total) float array, where N is the
              number of enabled features. Empty (0, T_total) if none enabled.
            - n_per_channel: int, number of features per channel (0–3).
    """
    hf_cfg = config.get("history_features", {})
    m, t_total = counts.shape

    # Master toggle — if disabled, return empty
    if not hf_cfg.get("enabled", False):
        logger.debug("History features disabled (master toggle OFF)")
        return np.empty((0, t_total), dtype=np.float32), 0

    bin_width_ms = config.get("bin_width_ms", 10.0)
    feature_blocks = []

    # ISI features
    isi_cfg = hf_cfg.get("isi", {})
    if isi_cfg.get("enabled", False):
        max_isi_ms = isi_cfg.get("max_isi_ms", 500.0)
        isi = compute_isi_features(counts, bin_width_ms, max_isi_ms)
        feature_blocks.append(isi)
        logger.info(
            "ISI features enabled: max_isi_ms=%.1f, bin_width_ms=%.1f",
            max_isi_ms, bin_width_ms,
        )

    # EMA rate features
    ema_cfg = hf_cfg.get("ema_rate", {})
    if ema_cfg.get("enabled", False):
        alpha = ema_cfg.get("alpha", 0.1)
        ema = compute_ema_rate(counts, alpha)
        feature_blocks.append(ema)
        logger.info("EMA rate features enabled: alpha=%.3f", alpha)

    # Refractory indicator
    ref_cfg = hf_cfg.get("refractory", {})
    if ref_cfg.get("enabled", False):
        refractory_bins = ref_cfg.get("refractory_bins", 1)
        ref = compute_refractory(counts, refractory_bins)
        feature_blocks.append(ref)
        logger.info(
            "Refractory features enabled: refractory_bins=%d",
            refractory_bins,
        )

    n_per_channel = len(feature_blocks)

    if n_per_channel == 0:
        logger.info("History features enabled but no sub-features turned on")
        return np.empty((0, t_total), dtype=np.float32), 0

    # Stack all feature blocks: each is (M, T_total) → (N*M, T_total)
    features = np.concatenate(feature_blocks, axis=0)
    logger.info(
        "History features computed: %d features × %d channels = %d rows",
        n_per_channel, m, features.shape[0],
    )

    return features, n_per_channel
