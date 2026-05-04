"""
Tests for masked evaluation metric correctness.

Validates that zero-padded channels (from multi-session padding) do NOT
inflate Pearson r or skew loss/MAE/MSE metrics.  This is a Phase 1
regression guard — if the masking is broken, these tests will catch it.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.teacher import TeacherLSTM
from src.train.trainer import Trainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for deterministic testing."""
    return torch.device("cpu")


@pytest.fixture
def masked_data():
    """
    Create synthetic multi-session-style data with known masking.

    Returns (x, y, mask) where:
    - 10 channels total, first 5 are "real" (mask=1), last 5 are "padded" (mask=0)
    - Padded channels have y=0, which would correlate perfectly with a model
      predicting near-zero if masking is broken
    """
    torch.manual_seed(42)
    n_samples = 200
    T = 10
    M_total = 10
    M_real = 5

    # Real channels: random spike counts with signal
    x = torch.randn(n_samples, T, M_total).abs()
    y = torch.randn(n_samples, M_total).abs() * 0.5

    # Zero out padded channels in both x and y (simulates padding)
    x[:, :, M_real:] = 0.0
    y[:, M_real:] = 0.0

    # Build mask: 1 for real channels, 0 for padded
    mask = torch.zeros(n_samples, M_total)
    mask[:, :M_real] = 1.0

    return x, y, mask, M_total, M_real


@pytest.fixture
def config():
    """Minimal training config for masked evaluation."""
    return {
        "training": {
            "epochs": 1,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "patience": 100,
        },
        "loss": {"type": "poisson_nll"},
    }


# =============================================================================
# Tests
# =============================================================================

class TestMaskedPearsonR:
    """Tests that Pearson r is not inflated by zero-padded channels."""

    def test_padded_channels_dont_inflate_r(self, masked_data, config, device):
        """
        Pearson r with masking should be similar whether we include padded
        channels or not — because they should be excluded by the mask.

        If masking is broken, a model predicting small positive values for
        padded channels (where y=0) would see those channels contribute
        near-zero variance, inflating the overall r.
        """
        x, y, mask, M_total, M_real = masked_data

        # Build model and trainer with masked data
        model = TeacherLSTM(
            input_size=M_total, hidden_size=32, num_layers=1,
            output_size=M_total, use_attention=False,
        ).to(device)

        # Create masked DataLoader
        dataset = TensorDataset(x, y, mask)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader,
            config=config, device=device,
        )

        # Evaluate with full masked data (M_total channels, mask applied)
        metrics_masked = trainer.evaluate(loader, prefix="test")

        # Now evaluate with ONLY the real channels (no padding at all)
        x_real = x[:, :, :M_real]
        y_real = y[:, :M_real]
        model_real = TeacherLSTM(
            input_size=M_real, hidden_size=32, num_layers=1,
            output_size=M_real, use_attention=False,
        ).to(device)
        dataset_real = TensorDataset(x_real, y_real)
        loader_real = DataLoader(dataset_real, batch_size=32, shuffle=False)
        trainer_real = Trainer(
            model=model_real, train_loader=loader_real, val_loader=loader_real,
            config=config, device=device,
        )
        metrics_unmasked = trainer_real.evaluate(loader_real, prefix="test")

        # Both should yield finite, non-trivial Pearson r
        r_masked = metrics_masked["test_pearson_r"]
        r_unmasked = metrics_unmasked["test_pearson_r"]
        assert np.isfinite(r_masked), f"Masked r is not finite: {r_masked}"
        assert np.isfinite(r_unmasked), f"Unmasked r is not finite: {r_unmasked}"

        # Key assertion: masked r should NOT be suspiciously close to 1.0
        # (which would indicate padded zero-zero correlations inflating it)
        assert r_masked < 0.95, (
            f"Masked Pearson r={r_masked:.4f} is suspiciously high — "
            f"padded channels may be inflating the metric"
        )

    def test_all_masked_channels_get_zero_weight(self, masked_data, config, device):
        """
        Channels that are masked in ALL samples should contribute zero weight
        to the final Pearson r average.
        """
        x, y, mask, M_total, M_real = masked_data

        model = TeacherLSTM(
            input_size=M_total, hidden_size=32, num_layers=1,
            output_size=M_total, use_attention=False,
        ).to(device)

        dataset = TensorDataset(x, y, mask)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader,
            config=config, device=device,
        )

        metrics = trainer.evaluate(loader, prefix="test")
        r = metrics["test_pearson_r"]

        # r should be finite (not NaN from 0/0 in masked channels)
        assert np.isfinite(r), f"Pearson r is NaN or inf: {r}"
        # r should be in a reasonable range (not garbage from numerical issues)
        assert -1.0 <= r <= 1.0, f"Pearson r out of range: {r}"


class TestMaskedLoss:
    """Tests that loss only considers unmasked (real) channels."""

    def test_masked_loss_ignores_padded_channels(self, masked_data, config, device):
        """
        Loss should be the same whether padded channels predict 0 or 100,
        because the mask should exclude them entirely.
        """
        x, y, mask, M_total, M_real = masked_data

        model = TeacherLSTM(
            input_size=M_total, hidden_size=32, num_layers=1,
            output_size=M_total, use_attention=False,
        ).to(device)
        model.eval()

        dataset = TensorDataset(x, y, mask)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader,
            config=config, device=device,
        )

        # Get baseline metrics
        metrics_1 = trainer.evaluate(loader, prefix="test")

        # Now corrupt padded channels in y (set to huge values)
        y_corrupt = y.clone()
        y_corrupt[:, M_real:] = 999.0

        dataset_2 = TensorDataset(x, y_corrupt, mask)
        loader_2 = DataLoader(dataset_2, batch_size=32, shuffle=False)

        metrics_2 = trainer.evaluate(loader_2, prefix="test")

        # Loss should be identical because the corrupted channels are masked out
        assert abs(metrics_1["test_loss"] - metrics_2["test_loss"]) < 1e-5, (
            f"Loss changed when corrupting padded channels: "
            f"{metrics_1['test_loss']:.6f} vs {metrics_2['test_loss']:.6f}"
        )

    def test_masked_mae_ignores_padded_channels(self, masked_data, config, device):
        """MAE should not change when padded channel targets are corrupted."""
        x, y, mask, M_total, M_real = masked_data

        model = TeacherLSTM(
            input_size=M_total, hidden_size=32, num_layers=1,
            output_size=M_total, use_attention=False,
        ).to(device)
        model.eval()

        # Normal
        dataset = TensorDataset(x, y, mask)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader,
            config=config, device=device,
        )
        m1 = trainer.evaluate(loader, prefix="t")

        # Corrupt padded channels
        y_bad = y.clone()
        y_bad[:, M_real:] = 1000.0
        dataset_bad = TensorDataset(x, y_bad, mask)
        loader_bad = DataLoader(dataset_bad, batch_size=32, shuffle=False)
        m2 = trainer.evaluate(loader_bad, prefix="t")

        assert abs(m1["t_mae"] - m2["t_mae"]) < 1e-5, (
            f"MAE changed: {m1['t_mae']:.6f} vs {m2['t_mae']:.6f}"
        )
