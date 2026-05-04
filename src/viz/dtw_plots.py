"""
DTW (Dynamic Time Warping) visualization utilities for population dynamics.

Provides reusable plotting functions for evaluating temporal alignment
between ground truth and predicted population spike rates:

    1. Warping path overlay — Side-by-side naive vs DTW-aligned comparison
    2. Distance matrix — Pairwise cost heatmap with optimal DTW path
    3. Session comparison — Grouped bar chart of naive MAE vs DTW error

All functions return matplotlib Figure objects for flexible downstream
saving/display. Uses src/viz/style.py conventions (COLORS, apply_style).

Usage:
    from src.viz.dtw_plots import (
        plot_dtw_warping_path,
        plot_dtw_distance_matrix,
        plot_dtw_session_comparison,
    )

    fig = plot_dtw_warping_path(pop_gt, pop_pred, session_id="003")
    save_figure(fig, "dtw_warping_003", output_dir="plots/")
"""

import logging
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import COLORS, apply_style

logger = logging.getLogger(__name__)

# Apply consistent style on import
apply_style()

# Semantic color mapping for GT vs prediction
COLOR_GT = COLORS["blue"]       # Ground truth: blue
COLOR_PRED = COLORS["red"]      # Prediction: orange-red (D55E00)
COLOR_LINK = "#888888"           # Warping links: gray


def plot_dtw_warping_path(
    pop_gt: np.ndarray,
    pop_pred: np.ndarray,
    session_id: Optional[str] = None,
    link_step: int = 2,
) -> plt.Figure:
    """
    Two-panel DTW warping path visualization.

    Top panel: Naive (time-matched) overlay of GT and predicted population
    spike rates. Bottom panel: Same overlay with gray warping links showing
    how DTW realigns the two sequences.

    Args:
        pop_gt: 1D array of ground truth population spike counts per bin.
        pop_pred: 1D array of predicted population rates per bin.
        session_id: Optional session label for the title.
        link_step: Plot every Nth warping link to reduce clutter (default 2).

    Returns:
        matplotlib Figure with 2 subplots.
    """
    try:
        from fastdtw import fastdtw
    except ImportError:
        logger.error("plot_dtw_warping_path requires fastdtw. pip install fastdtw")
        raise

    # Compute DTW alignment
    dist, path_dtw = fastdtw(pop_gt.flatten(), pop_pred.flatten())
    avg_error = dist / max(len(path_dtw), 1)

    # Build figure
    session_label = f" (Session {session_id})" if session_id else ""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)

    # Top panel: Naive alignment
    axes[0].plot(pop_gt, label="Ground Truth", color=COLOR_GT, linewidth=1.5)
    axes[0].plot(pop_pred, label="Predicted", color=COLOR_PRED, linewidth=1.5)
    axes[0].set_title(f"Naive Alignment (Time-Matched){session_label}")
    axes[0].set_ylabel("Spikes / Bin")
    axes[0].legend(frameon=False)

    # Bottom panel: Warping links overlaid
    axes[1].plot(pop_gt, label="Ground Truth", color=COLOR_GT, linewidth=1.5)
    axes[1].plot(pop_pred, label="Predicted", color=COLOR_PRED, linewidth=1.5)
    axes[1].set_title(
        f"DTW Warping Path (avg error = {avg_error:.3f} spikes/bin)"
    )
    axes[1].set_xlabel("Time (Bins, 50ms each)")
    axes[1].set_ylabel("Spikes / Bin")

    # Draw warping links (every link_step-th to avoid visual clutter)
    for i, j in path_dtw[::link_step]:
        axes[1].plot(
            [i, j], [pop_gt[i], pop_pred[j]],
            color=COLOR_LINK, alpha=0.3, linewidth=0.5,
        )

    plt.tight_layout()
    logger.info(
        "DTW warping path plot: dist=%.2f, avg_error=%.4f%s",
        dist, avg_error, session_label,
    )
    return fig


def plot_dtw_distance_matrix(
    pop_gt: np.ndarray,
    pop_pred: np.ndarray,
    session_id: Optional[str] = None,
) -> plt.Figure:
    """
    DTW distance matrix heatmap with optimal path overlay.

    Shows the pairwise Euclidean distance matrix between GT and predicted
    population rates, with the optimal DTW path (white) and naive diagonal
    (dashed gray) overlaid. Marginal time-series traces are shown on the
    top (GT) and left (predicted) edges.

    Args:
        pop_gt: 1D array of ground truth population spike counts per bin.
        pop_pred: 1D array of predicted population rates per bin.
        session_id: Optional session label for the title.

    Returns:
        matplotlib Figure with distance matrix + marginal traces.
    """
    from scipy.spatial.distance import cdist

    try:
        from fastdtw import fastdtw
    except ImportError:
        logger.error("plot_dtw_distance_matrix requires fastdtw.")
        raise

    pop_gt = pop_gt.flatten()
    pop_pred = pop_pred.flatten()
    T = len(pop_gt)

    # Compute DTW path and pairwise distance matrix
    dist, path_dtw = fastdtw(pop_gt, pop_pred)
    D = cdist(pop_pred[:, None], pop_gt[:, None], metric="euclidean")

    # Build figure with gridspec for marginal traces
    fig = plt.figure(figsize=(9, 8), dpi=300)
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 5, 0.2],
        height_ratios=[1, 5],
        wspace=0.1, hspace=0.1,
    )

    ax_top = plt.subplot(gs[0, 1])    # GT trace (top)
    ax_left = plt.subplot(gs[1, 0])   # Predicted trace (left)
    ax_main = plt.subplot(gs[1, 1])   # Distance matrix heatmap
    ax_cbar = plt.subplot(gs[1, 2])   # Colorbar

    # Top marginal: Ground Truth time-series
    ax_top.plot(np.arange(T), pop_gt, color=COLOR_GT, linewidth=2)
    ax_top.axis("off")
    ax_top.margins(x=0)

    # Left marginal: Predicted time-series (rotated)
    ax_left.plot(pop_pred, np.arange(T), color=COLOR_PRED, linewidth=2)
    ax_left.invert_xaxis()  # Points rightward toward heatmap
    ax_left.axis("off")
    ax_left.margins(y=0)

    # Main heatmap
    im = ax_main.imshow(D, aspect="auto", origin="lower", cmap="viridis")

    # Extract DTW path coordinates
    path_gt_idx = [p[0] for p in path_dtw]
    path_pred_idx = [p[1] for p in path_dtw]

    # Naive diagonal reference
    ax_main.plot(
        [0, T - 1], [0, T - 1],
        color="lightgray", linestyle="--", linewidth=2,
        alpha=0.9, label="Naive Diagonal",
    )

    # Optimal DTW path
    ax_main.plot(
        path_gt_idx, path_pred_idx,
        color="white", linewidth=2.5, label="DTW Path",
    )

    # Axis labels and limits
    ax_main.set_xlabel("Time (Ground Truth)", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("Time (Twin Prediction)", fontsize=11, fontweight="bold")
    ax_main.set_xlim(-0.5, T - 0.5)
    ax_main.set_ylim(-0.5, T - 0.5)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(
        "Euclidean Distance", rotation=270, labelpad=15, fontweight="bold",
    )

    # Title with RMS warp metric
    rms_warp = dist / max(1, len(path_dtw))
    session_label = f" (Session {session_id})" if session_id else ""
    plt.suptitle(
        f"DTW Distance Matrix{session_label}\n"
        f"RMS warp = {rms_warp:.3f}",
        y=0.94, fontsize=14, fontweight="bold",
    )

    logger.info(
        "DTW distance matrix plot: T=%d, dist=%.2f, rms_warp=%.4f%s",
        T, dist, rms_warp, session_label,
    )
    return fig


def plot_dtw_session_comparison(
    session_data: List[Dict[str, object]],
) -> plt.Figure:
    """
    Grouped bar chart comparing naive MAE vs DTW average error across sessions.

    Shows that DTW-aligned error is consistently lower than naive bin-to-bin
    MAE, demonstrating the model captures the right rhythmic patterns but
    with minor temporal jitter.

    Args:
        session_data: List of dicts, each with keys:
            - "session_id": str (e.g., "003")
            - "naive_mae": float
            - "dtw_avg_error": float

    Returns:
        matplotlib Figure with grouped bar chart.
    """
    # Extract data from list of dicts
    session_ids = [d["session_id"] for d in session_data]
    naive_maes = [d["naive_mae"] for d in session_data]
    dtw_errors = [d["dtw_avg_error"] for d in session_data]

    n_sessions = len(session_ids)
    x = np.arange(n_sessions)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    # Grouped bars
    ax.bar(
        x - width / 2, naive_maes, width,
        label="Naive MAE", color=COLOR_GT,
    )
    ax.bar(
        x + width / 2, dtw_errors, width,
        label="DTW Avg Error", color=COLOR_PRED,
    )

    # Labels and formatting
    ax.set_ylabel("Absolute Error (Spikes/Bin)")
    ax.set_title(
        f"Population Alignment: Naive vs DTW ({n_sessions} Sessions)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(session_ids, rotation=90)
    ax.legend(frameon=False)

    # Mean improvement annotation
    mean_naive = np.mean(naive_maes)
    mean_dtw = np.mean(dtw_errors)
    improvement_pct = (1 - mean_dtw / mean_naive) * 100 if mean_naive > 0 else 0
    ax.annotate(
        f"Mean DTW improvement: {improvement_pct:.1f}%",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    logger.info(
        "DTW session comparison: %d sessions, mean improvement %.1f%%",
        n_sessions, improvement_pct,
    )
    return fig
