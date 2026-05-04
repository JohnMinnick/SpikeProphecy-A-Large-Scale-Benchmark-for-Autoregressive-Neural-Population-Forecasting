"""
Tier 2 visualizations: training diagnostics.

Provides plotting functions for monitoring model training:
    - Loss curves (train vs val)
    - Learning rate schedule
    - All evaluation metrics over epochs
    - Prediction vs actual time-series overlays
    - Split comparison bar charts (train vs val vs test sanity check)

All functions follow the decoupled convention: they accept data arrays/dicts
as input and never import from model or training code.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import COLOR_CYCLE, apply_style, reset_style, save_figure

logger = logging.getLogger(__name__)


def plot_loss_curves(
    history: Dict[str, List[float]],
    title: str = "Training Loss",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot train and validation loss curves over epochs.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists.
        title: Plot title.
        ax: Optional existing axes to plot on.

    Returns:
        Matplotlib figure.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot train and val loss
    ax.plot(epochs, history["train_loss"], label="Train",
            color=COLOR_CYCLE[0], linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val",
            color=COLOR_CYCLE[1], linewidth=2)

    # Mark best epoch
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    ax.axvline(best_epoch, color=COLOR_CYCLE[2], linestyle="--",
               alpha=0.7, label=f"Best (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val], color=COLOR_CYCLE[2],
               s=50, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Poisson NLL)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(1, len(history["train_loss"]))

    reset_style()
    return fig


def plot_lr_schedule(
    history: Dict[str, List[float]],
    title: str = "Learning Rate Schedule",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot learning rate over epochs.

    Args:
        history: Dict with 'learning_rate' list.
        title: Plot title.
        ax: Optional existing axes to plot on.

    Returns:
        Matplotlib figure.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    epochs = range(1, len(history["learning_rate"]) + 1)
    ax.plot(epochs, history["learning_rate"],
            color=COLOR_CYCLE[3], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.set_xlim(1, len(history["learning_rate"]))

    reset_style()
    return fig


def plot_metric_curves(
    history: Dict[str, List[float]],
    title: str = "Validation Metrics",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot all validation metrics over epochs in a 2×2 grid.

    Metrics: Poisson NLL, Pearson r, MAE, MSE.

    Args:
        history: Dict with val_poisson_nll, val_pearson_r, val_mae, val_mse.
        title: Overall figure title.
        ax: Ignored (always creates 2×2 subplots).

    Returns:
        Matplotlib figure.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Metric configs: (key, label, color_idx, lower_is_better)
    metrics = [
        ("val_poisson_nll", "Poisson NLL", 0, True),
        ("val_pearson_r", "Pearson r", 1, False),
        ("val_mae", "MAE", 4, True),
        ("val_mse", "MSE", 5, True),
    ]

    for ax_i, (key, label, c_idx, lower_better) in zip(axes.flat, metrics):
        if key not in history:
            ax_i.set_title(f"{label} (no data)")
            continue

        values = history[key]
        epochs = range(1, len(values) + 1)
        ax_i.plot(epochs, values, color=COLOR_CYCLE[c_idx], linewidth=2)

        # Mark best epoch
        best_fn = np.argmin if lower_better else np.argmax
        best_epoch = int(best_fn(values)) + 1
        best_val = values[best_epoch - 1]
        ax_i.axvline(best_epoch, color=COLOR_CYCLE[2], linestyle="--",
                     alpha=0.5)
        ax_i.scatter([best_epoch], [best_val], color=COLOR_CYCLE[2],
                     s=40, zorder=5)

        ax_i.set_xlabel("Epoch")
        ax_i.set_ylabel(label)
        ax_i.set_title(f"{label} (best: {best_val:.4f} @ ep {best_epoch})")
        ax_i.set_xlim(1, len(values))

    fig.tight_layout()
    reset_style()
    return fig


def plot_prediction_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    unit_indices: Optional[List[int]] = None,
    n_units: int = 4,
    n_bins: int = 200,
    bin_width_ms: float = 10.0,
    title: str = "Prediction vs Actual",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot predicted vs actual spike counts for selected units.

    Shows time-series overlay of predictions and ground truth for visual
    inspection of temporal structure capture.

    Args:
        predictions: Predicted rates, shape (N_samples, M).
        targets: Ground-truth counts, shape (N_samples, M).
        unit_indices: Which units to plot. If None, picks first n_units.
        n_units: Number of units to plot (if unit_indices is None).
        n_bins: Number of time bins to show.
        bin_width_ms: Bin width in ms for x-axis labeling.
        title: Overall figure title.
        ax: Ignored (creates multi-row figure).

    Returns:
        Matplotlib figure.
    """
    apply_style()

    n_samples, m = predictions.shape
    n_bins = min(n_bins, n_samples)

    # Select units to plot
    if unit_indices is None:
        unit_indices = list(range(min(n_units, m)))

    n_plots = len(unit_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots),
                             sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Handle single unit case
    if n_plots == 1:
        axes = [axes]

    # Time axis in seconds
    time_s = np.arange(n_bins) * bin_width_ms / 1000.0

    for ax_i, unit_idx in zip(axes, unit_indices):
        # Get data for this unit, first n_bins time steps
        actual = targets[:n_bins, unit_idx]
        predicted = predictions[:n_bins, unit_idx]

        # Plot actual as step (integer counts)
        ax_i.step(time_s, actual, where="mid", color=COLOR_CYCLE[0],
                  alpha=0.6, linewidth=1.0, label="Actual")

        # Plot predicted as smooth line
        ax_i.plot(time_s, predicted, color=COLOR_CYCLE[1],
                  linewidth=1.5, label="Predicted")

        ax_i.set_ylabel(f"Unit {unit_idx}")
        ax_i.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    reset_style()
    return fig


def plot_split_comparison(
    split_metrics: Dict[str, Dict[str, float]],
    title: str = "Train / Val / Test Comparison",
) -> plt.Figure:
    """
    Grouped bar chart comparing metrics across data splits.

    Produces a 2x2 grid of bar charts (NLL, Pearson r, MAE, MSE) with
    one bar per split, making it easy to spot whether the model is
    learning (train < val) and how large the generalization gap is.

    Args:
        split_metrics: Dict mapping split name (e.g. "train", "val", "test")
                       to a metric dict with keys like "{split}_poisson_nll",
                       "{split}_pearson_r", "{split}_mae", "{split}_mse".
        title: Overall figure title.

    Returns:
        Matplotlib figure.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Metric suffixes and labels
    metric_defs = [
        ("poisson_nll", "Poisson NLL"),
        ("pearson_r", "Pearson r"),
        ("mae", "MAE"),
        ("mse", "MSE"),
    ]

    split_names = list(split_metrics.keys())
    n_splits = len(split_names)
    x = np.arange(n_splits)
    bar_width = 0.5

    for ax_i, (suffix, label) in zip(axes.flat, metric_defs):
        values = []
        for sname in split_names:
            key = f"{sname}_{suffix}"
            values.append(split_metrics[sname].get(key, 0.0))

        # Color each bar with a different color from the palette
        colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(n_splits)]
        bars = ax_i.bar(x, values, bar_width, color=colors)

        # Add value annotations on bars
        for bar, val in zip(bars, values):
            ax_i.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax_i.set_xticks(x)
        ax_i.set_xticklabels([s.capitalize() for s in split_names])
        ax_i.set_ylabel(label)
        ax_i.set_title(label)

    fig.tight_layout()
    reset_style()
    return fig
