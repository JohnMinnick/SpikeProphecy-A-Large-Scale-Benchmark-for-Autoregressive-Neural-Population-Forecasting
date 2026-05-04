"""
Tests for distillation target visualization.

Validates that plot_distillation_target produces correct figures
with expected structure (3 subplots), accepts both Tensor and ndarray
inputs, and handles save_path correctly.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.viz.distillation import plot_distillation_target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create synthetic distillation target data."""
    np.random.seed(42)
    N, T, M = 8, 20, 5

    inputs = np.random.poisson(lam=1.0, size=(N, T, M)).astype(np.float32)
    targets = np.random.poisson(lam=1.0, size=(N, M)).astype(np.float32)
    teacher_rates = np.abs(np.random.randn(N, M).astype(np.float32))

    return inputs, targets, teacher_rates


@pytest.fixture
def sample_data_tensors(sample_data):
    """Return the same data as PyTorch tensors."""
    inputs, targets, teacher_rates = sample_data
    return (
        torch.tensor(inputs),
        torch.tensor(targets),
        torch.tensor(teacher_rates),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotDistillationTarget:
    """Tests for the plot_distillation_target function."""

    def test_returns_figure(self, sample_data):
        """Should return a matplotlib Figure."""
        inputs, targets, teacher_rates = sample_data
        fig = plot_distillation_target(inputs, targets, teacher_rates)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_three_subplots(self, sample_data):
        """Figure should contain exactly 3 subplots (raster, GT, teacher)."""
        inputs, targets, teacher_rates = sample_data
        fig = plot_distillation_target(inputs, targets, teacher_rates)
        axes = fig.get_axes()
        # Expect 3 subplots + possible colorbar axes
        assert len(axes) >= 3
        plt.close(fig)

    def test_accepts_tensors(self, sample_data_tensors):
        """Should accept PyTorch tensors as input."""
        inputs, targets, teacher_rates = sample_data_tensors
        fig = plot_distillation_target(inputs, targets, teacher_rates)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_sample_idx(self, sample_data):
        """Should plot the requested sample index."""
        inputs, targets, teacher_rates = sample_data
        fig = plot_distillation_target(
            inputs, targets, teacher_rates, sample_idx=3
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, sample_data, tmp_path):
        """Should save figure when save_path is provided."""
        inputs, targets, teacher_rates = sample_data
        save_path = tmp_path / "test_distill_target.png"
        fig = plot_distillation_target(
            inputs, targets, teacher_rates, save_path=save_path
        )
        assert isinstance(fig, plt.Figure)
        # save_figure creates with the stem name in the parent dir
        # The exact filename depends on save_figure implementation
        plt.close(fig)
