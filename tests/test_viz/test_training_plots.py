"""
Tests for Tier 2 training visualizations.

Validates that all training plot functions produce valid figures.
"""

import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np

from src.viz.training_plots import (
    plot_loss_curves,
    plot_lr_schedule,
    plot_metric_curves,
    plot_prediction_vs_actual,
    plot_split_comparison,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_history():
    """Create a mock training history dict."""
    np.random.seed(42)
    n_epochs = 20
    return {
        "train_loss": list(np.linspace(2.0, 0.5, n_epochs) + np.random.randn(n_epochs) * 0.05),
        "val_loss": list(np.linspace(2.2, 0.6, n_epochs) + np.random.randn(n_epochs) * 0.1),
        "val_poisson_nll": list(np.linspace(2.2, 0.6, n_epochs) + np.random.randn(n_epochs) * 0.1),
        "val_pearson_r": list(np.linspace(0.0, 0.8, n_epochs) + np.random.randn(n_epochs) * 0.05),
        "val_mae": list(np.linspace(1.0, 0.3, n_epochs) + np.random.randn(n_epochs) * 0.05),
        "val_mse": list(np.linspace(1.5, 0.2, n_epochs) + np.random.randn(n_epochs) * 0.05),
        "learning_rate": list(np.linspace(1e-3, 1e-5, n_epochs)),
    }


@pytest.fixture
def mock_predictions():
    """Create mock predictions and targets."""
    np.random.seed(42)
    n_samples, m = 100, 5
    targets = np.random.poisson(lam=1.0, size=(n_samples, m)).astype(np.float32)
    predictions = targets + np.random.randn(n_samples, m).astype(np.float32) * 0.3
    predictions = np.abs(predictions)  # Non-negative
    return predictions, targets


@pytest.fixture
def mock_split_metrics():
    """Create mock split metrics for comparison plotting."""
    return {
        "train": {
            "train_poisson_nll": 0.1386, "train_pearson_r": 0.2250,
            "train_mae": 0.0705, "train_mse": 0.0382,
        },
        "val": {
            "val_poisson_nll": 0.1477, "val_pearson_r": 0.1408,
            "val_mae": 0.0733, "val_mse": 0.0404,
        },
        "test": {
            "test_poisson_nll": 0.1572, "test_pearson_r": 0.1309,
            "test_mae": 0.0752, "test_mse": 0.0430,
        },
    }


# =============================================================================
# Tests
# =============================================================================

class TestLossCurves:
    """Tests for loss curve plotting."""

    def test_returns_figure(self, mock_history):
        """plot_loss_curves should return a figure."""
        fig = plot_loss_curves(mock_history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, mock_history):
        """Should accept a custom title."""
        fig = plot_loss_curves(mock_history, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLRSchedule:
    """Tests for LR schedule plotting."""

    def test_returns_figure(self, mock_history):
        """plot_lr_schedule should return a figure."""
        fig = plot_lr_schedule(mock_history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMetricCurves:
    """Tests for metric curve plotting."""

    def test_returns_figure(self, mock_history):
        """plot_metric_curves should return a 2x2 figure."""
        fig = plot_metric_curves(mock_history)
        assert isinstance(fig, plt.Figure)
        # Should have 4 subplots
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)


class TestPredictionVsActual:
    """Tests for prediction vs actual plot."""

    def test_returns_figure(self, mock_predictions):
        """plot_prediction_vs_actual should return a figure."""
        predictions, targets = mock_predictions
        fig = plot_prediction_vs_actual(predictions, targets)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_specific_units(self, mock_predictions):
        """Should plot specified units."""
        predictions, targets = mock_predictions
        fig = plot_prediction_vs_actual(
            predictions, targets, unit_indices=[0, 2]
        )
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_single_unit(self, mock_predictions):
        """Should handle single unit."""
        predictions, targets = mock_predictions
        fig = plot_prediction_vs_actual(
            predictions, targets, unit_indices=[0]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_limited_bins(self, mock_predictions):
        """Should respect n_bins limit."""
        predictions, targets = mock_predictions
        fig = plot_prediction_vs_actual(
            predictions, targets, n_bins=50
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSplitComparison:
    """Tests for split comparison bar chart."""

    def test_returns_figure(self, mock_split_metrics):
        """plot_split_comparison should return a figure."""
        fig = plot_split_comparison(mock_split_metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_four_subplots(self, mock_split_metrics):
        """Should create a 2x2 grid (NLL, Pearson r, MAE, MSE)."""
        fig = plot_split_comparison(mock_split_metrics)
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_two_splits_only(self):
        """Should work with only two splits (e.g. train + val)."""
        metrics = {
            "train": {"train_poisson_nll": 0.14, "train_pearson_r": 0.22,
                       "train_mae": 0.07, "train_mse": 0.04},
            "val": {"val_poisson_nll": 0.15, "val_pearson_r": 0.14,
                     "val_mae": 0.07, "val_mse": 0.04},
        }
        fig = plot_split_comparison(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, mock_split_metrics):
        """Should accept a custom title."""
        fig = plot_split_comparison(mock_split_metrics, title="Sanity Check")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
