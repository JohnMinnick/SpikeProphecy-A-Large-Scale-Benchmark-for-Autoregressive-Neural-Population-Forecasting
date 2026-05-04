"""
Visualization tools for distillation targets and student performance.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.viz.style import apply_style, reset_style, save_figure

logger = logging.getLogger(__name__)


def plot_distillation_target(
    inputs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    teacher_rates: Union[np.ndarray, torch.Tensor],
    sample_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot input raster, ground truth counts, and teacher soft targets for a single sample.

    Args:
        inputs: Input spikes (N, T, M).
        targets: Ground truth spike counts (N, M).
        teacher_rates: Teacher predicted rates (N, M).
        sample_idx: Index of sample to plot.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    # Convert to numpy and select sample
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(teacher_rates, torch.Tensor):
        teacher_rates = teacher_rates.cpu().numpy()

    x = inputs[sample_idx].T  # (M, T)
    y = targets[sample_idx]   # (M,)
    y_hat = teacher_rates[sample_idx] # (M,)

    M, T = x.shape

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    # 1. Raster
    axes[0].imshow(
        x, aspect='auto', cmap='binary', interpolation='nearest', origin='lower'
    )
    axes[0].set_title(f"Input Spike Raster (Sample {sample_idx})")
    axes[0].set_ylabel("Channel")
    axes[0].set_xlabel("Time Bin")

    # 2. Ground Truth
    axes[1].bar(range(M), y, color='black', alpha=0.7, label='Ground Truth Counts')
    axes[1].set_title("Ground Truth Spike Counts (Target)")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc='upper right')

    # 3. Teacher Rates
    axes[2].bar(range(M), y_hat, color='red', alpha=0.7, label='Teacher Soft Targets')
    axes[2].set_title("Teacher Predicted Rates (Distillation Target)")
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Rate")
    axes[2].legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        save_figure(fig, name=path.stem, output_dir=path.parent)

    reset_style()
    return fig
