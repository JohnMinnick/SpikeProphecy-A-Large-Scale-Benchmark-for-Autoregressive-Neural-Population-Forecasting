"""
Per-channel loss weights derived from Poisson ceiling analysis.

Builds a weight tensor (M_max,) that scales each neuron's contribution
to the training loss based on its predictability ceiling.  Unpredictable
neurons (sub-Poisson, ceiling ≈ 0) get down-weighted so the model focuses
gradient signal on neurons with real rate modulation.

Three strategies:
    - ``ceiling``: raw analytical ceiling (continuous 0→1).
    - ``binary``:  1.0 if ceiling > threshold, else floor_weight.
    - ``softmax``: softmax-normalized ceilings (sum → M_max).

Usage:
    from src.eval.ceiling_weights import build_ceiling_weights
    weights = build_ceiling_weights("outputs/eval_analysis/per_neuron_stats.json",
                                    m_max=1240, strategy="binary")
"""

import json
import logging
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger(__name__)


def build_ceiling_weights(
    stats_path: Union[str, Path],
    m_max: int,
    strategy: str = "binary",
    floor_weight: float = 0.1,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Build per-channel loss weights from per-neuron ceiling stats.

    Loads ``per_neuron_stats.json`` (from ``analyze_evaluation.py``),
    averages ceiling values across sessions for each channel index,
    and applies the chosen weighting strategy.

    Args:
        stats_path: Path to the per_neuron_stats.json file.
        m_max: Maximum number of channels (padded dimension).
        strategy: Weighting strategy — ``"ceiling"``, ``"binary"``,
                  or ``"softmax"``.
        floor_weight: Minimum weight for any channel (prevents gradient
                      death).  Must be in (0, 1].
        threshold: For ``"binary"`` strategy only — ceiling values above
                   this get weight 1.0, below get ``floor_weight``.

    Returns:
        Tensor of shape (m_max,) with per-channel weights in
        [floor_weight, 1.0] (or softmax-scaled).

    Raises:
        FileNotFoundError: If stats_path does not exist.
        ValueError: If strategy is unknown or floor_weight is invalid.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Per-neuron stats file not found: {stats_path}. "
            f"Run scripts/analyze_evaluation.py first to generate it."
        )

    if floor_weight <= 0 or floor_weight > 1.0:
        raise ValueError(
            f"floor_weight must be in (0, 1], got {floor_weight}"
        )

    valid_strategies = ("ceiling", "binary", "softmax")
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy must be one of {valid_strategies}, got '{strategy}'"
        )

    # Load per-neuron stats
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Aggregate ceiling values per channel index across sessions.
    # Each session's neuron i occupies channel index i in the padded
    # M_max space (sessions are zero-padded beyond their N_i neurons).
    # We average across sessions so the weight tensor is session-agnostic.
    ceiling_sums = torch.zeros(m_max, dtype=torch.float32)
    ceiling_counts = torch.zeros(m_max, dtype=torch.float32)

    for entry in stats:
        neuron_idx = entry["neuron"]
        if neuron_idx < m_max:
            ceiling_val = entry.get("ceiling_analytical", 0.0)
            # Clamp negative ceilings to zero (should not happen, but safe)
            ceiling_sums[neuron_idx] += max(ceiling_val, 0.0)
            ceiling_counts[neuron_idx] += 1.0

    # Average ceiling per channel (avoid division by zero for unused indices)
    avg_ceiling = ceiling_sums / ceiling_counts.clamp(min=1.0)

    # Apply weighting strategy
    if strategy == "ceiling":
        # Continuous: use raw ceiling value, clamped to [floor_weight, 1.0]
        weights = avg_ceiling.clamp(min=floor_weight, max=1.0)

    elif strategy == "binary":
        # Sharp cutoff: above threshold → 1.0, below → floor_weight
        weights = torch.where(
            avg_ceiling > threshold,
            torch.ones_like(avg_ceiling),
            torch.full_like(avg_ceiling, floor_weight),
        )

    elif strategy == "softmax":
        # Softmax-normalized: convert ceiling to probability distribution,
        # then scale so weights sum to m_max (preserves loss magnitude)
        # Use temperature=1.0 (raw softmax on ceiling values)
        weights = torch.softmax(avg_ceiling, dim=0) * m_max
        # Clamp to ensure floor_weight minimum
        weights = weights.clamp(min=floor_weight)

    # Log summary statistics
    n_high = (avg_ceiling > threshold).sum().item()
    n_total = (ceiling_counts > 0).sum().item()
    logger.info(
        "Ceiling weights: strategy=%s, %d/%d neurons above threshold %.2f, "
        "weight range=[%.3f, %.3f], mean=%.3f",
        strategy, n_high, n_total, threshold,
        weights.min().item(), weights.max().item(), weights.mean().item(),
    )

    return weights
