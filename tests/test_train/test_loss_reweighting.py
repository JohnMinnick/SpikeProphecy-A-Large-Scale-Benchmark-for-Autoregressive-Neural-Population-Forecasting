"""
Tests for ceiling-based loss reweighting in the Trainer.

Validates that the Trainer correctly loads ceiling weights from config,
applies them during training loss computation, and handles edge cases
(disabled by default, uniform weights equal baseline, session-head slicing).
"""

import json
import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.teacher import TeacherLSTM
from src.train.trainer import Trainer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_stats_file(tmp_path):
    """Create a per_neuron_stats.json with known ceiling values."""
    m = 10  # 10 neurons
    stats = []
    for i in range(m):
        # Alternate: even neurons predictable (ceiling=0.5),
        # odd neurons unpredictable (ceiling=0.0)
        ceiling = 0.5 if i % 2 == 0 else 0.0
        stats.append({
            "session": 0,
            "neuron": i,
            "region": "test",
            "ceiling_analytical": ceiling,
        })
    stats_path = tmp_path / "per_neuron_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    return stats_path


@pytest.fixture
def synthetic_data():
    """Create small synthetic train/val data for Trainer."""
    m = 10
    n_train = 64
    n_val = 32
    t = 5  # history window

    # Training data: (x, y, mask)
    x_train = torch.randn(n_train, t, m).abs()
    y_train = torch.poisson(torch.ones(n_train, m) * 2.0)
    mask_train = torch.ones(n_train, m)

    x_val = torch.randn(n_val, t, m).abs()
    y_val = torch.poisson(torch.ones(n_val, m) * 2.0)
    mask_val = torch.ones(n_val, m)

    train_ds = TensorDataset(x_train, y_train, mask_train)
    val_ds = TensorDataset(x_val, y_val, mask_val)

    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    return train_loader, val_loader, m


@pytest.fixture
def weighted_config(mock_stats_file):
    """Config with ceiling weights enabled."""
    return {
        "model": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "output_size": 10,
            "output_distribution": "poisson",
        },
        "training": {
            "epochs": 3,
            "val_every_n": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "none",
            "warmup_steps": 0,
            "patience": 10,
            "grad_clip_norm": 1.0,
        },
        "loss": {
            "type": "poisson_nll",
            "log_input": False,
            "ceiling_weights": {
                "enabled": True,
                "stats_path": str(mock_stats_file),
                "strategy": "binary",
                "floor_weight": 0.1,
                "threshold": 0.1,
            },
        },
    }


@pytest.fixture
def baseline_config():
    """Config without ceiling weights (default)."""
    return {
        "model": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "output_size": 10,
            "output_distribution": "poisson",
        },
        "training": {
            "epochs": 3,
            "val_every_n": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "none",
            "warmup_steps": 0,
            "patience": 10,
            "grad_clip_norm": 1.0,
        },
        "loss": {
            "type": "poisson_nll",
            "log_input": False,
        },
    }


# =============================================================================
# Core tests
# =============================================================================


class TestLossReweighting:
    """Tests for ceiling-weighted loss in the Trainer."""

    def test_disabled_by_default(self, synthetic_data, baseline_config):
        """Trainer without ceiling_weights config should have None weights."""
        train_loader, val_loader, m = synthetic_data
        model = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        device = torch.device("cpu")
        trainer = Trainer(model, train_loader, val_loader, baseline_config, device)
        assert trainer.ceiling_weights is None

    def test_training_runs_with_ceiling_weights(
        self, synthetic_data, weighted_config,
    ):
        """Full training loop should complete with ceiling weights enabled."""
        train_loader, val_loader, m = synthetic_data
        model = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        device = torch.device("cpu")
        trainer = Trainer(model, train_loader, val_loader, weighted_config, device)

        # Verify weights are loaded
        assert trainer.ceiling_weights is not None
        assert trainer.ceiling_weights.shape == (m,)

        # Training should complete without errors
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        assert all(np.isfinite(loss) for loss in history["train_loss"])

    def test_weighted_loss_differs_from_baseline(
        self, synthetic_data, weighted_config, baseline_config,
    ):
        """Weighted loss should produce different values than unweighted."""
        train_loader, val_loader, m = synthetic_data
        device = torch.device("cpu")

        # Build weighted trainer
        torch.manual_seed(42)
        model_w = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        trainer_w = Trainer(model_w, train_loader, val_loader, weighted_config, device)

        # Build baseline trainer with same model init
        torch.manual_seed(42)
        model_b = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        trainer_b = Trainer(model_b, train_loader, val_loader, baseline_config, device)

        # Compute loss on same batch
        batch = next(iter(train_loader))
        x, y, mask = batch
        with torch.no_grad():
            y_hat_w = model_w(x)
            y_hat_b = model_b(x)
            loss_w = trainer_w._compute_loss(y_hat_w, y, mask=mask)
            loss_b = trainer_b._compute_loss(y_hat_b, y, mask=mask)

        # Losses should differ because weights are non-uniform
        # (even neurons weight=1.0, odd neurons weight=0.1)
        assert loss_w.item() != pytest.approx(loss_b.item(), abs=0.01)

    def test_uniform_weights_match_baseline(
        self, synthetic_data, baseline_config,
    ):
        """When all ceiling weights are 1.0, loss should match unweighted."""
        train_loader, val_loader, m = synthetic_data
        device = torch.device("cpu")

        # Build two trainers with identical model init
        torch.manual_seed(42)
        model = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        trainer = Trainer(model, train_loader, val_loader, baseline_config, device)
        # Manually set uniform ceiling weights
        trainer.ceiling_weights = torch.ones(m, device=device)

        # Also build unweighted trainer
        torch.manual_seed(42)
        model2 = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        trainer2 = Trainer(model2, train_loader, val_loader, baseline_config, device)

        # Compute loss on same batch
        batch = next(iter(train_loader))
        x, y, mask = batch
        with torch.no_grad():
            y_hat = model(x)
            y_hat2 = model2(x)
            loss_w = trainer._compute_loss(y_hat, y, mask=mask)
            loss_b = trainer2._compute_loss(y_hat2, y, mask=mask)

        # With uniform weights = 1.0, losses should be identical
        assert loss_w.item() == pytest.approx(loss_b.item(), abs=1e-5)

    def test_weights_sliced_for_session_heads(
        self, synthetic_data, baseline_config,
    ):
        """When output dim < M_max, ceiling weights should be sliced."""
        train_loader, val_loader, m = synthetic_data
        device = torch.device("cpu")

        model = TeacherLSTM(
            input_size=m, hidden_size=32, num_layers=1,
            output_distribution="poisson",
        )
        trainer = Trainer(model, train_loader, val_loader, baseline_config, device)
        # Set ceiling weights for full M_max=10
        trainer.ceiling_weights = torch.ones(m, device=device)
        trainer.ceiling_weights[0] = 0.5  # Mark first channel as low-weight

        # Simulate session-specific head: y_hat is (batch, 5) but y is (batch, 10)
        y_hat = torch.ones(4, 5, device=device) * 2.0  # Smaller output
        y = torch.ones(4, 10, device=device)
        mask = torch.ones(4, 10, device=device)

        # Should slice y, mask, AND ceiling_weights to match y_hat
        loss = trainer._compute_loss(y_hat, y, mask=mask)
        assert torch.isfinite(loss)
