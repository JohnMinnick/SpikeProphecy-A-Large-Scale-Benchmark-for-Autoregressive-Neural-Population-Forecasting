"""
Tests for the single-batch overfit sanity check.

Validates that Trainer.overfit_one_batch() can memorize a tiny
synthetic batch, confirming the training infra works end-to-end.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.teacher import TeacherLSTM, create_teacher_model
from src.train.trainer import Trainer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def overfit_data():
    """
    Create synthetic train/val data for overfit testing.

    Returns a tuple of (train_loader, val_loader) with small tensors that
    the model should easily memorize.
    """
    # 5 channels, history window of 10 bins, 16 samples
    m = 5
    t = 10
    n_samples = 16

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic spike-count-like data (non-negative integers)
    x = torch.from_numpy(
        np.random.poisson(lam=3.0, size=(n_samples, t, m)).astype(np.float32)
    )
    y = torch.from_numpy(
        np.random.poisson(lam=3.0, size=(n_samples, m)).astype(np.float32)
    )

    train_ds = TensorDataset(x, y)
    # Use all data for training AND validation (overfit = same batch OK)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    val_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    return train_loader, val_loader, m


@pytest.fixture
def overfit_config():
    """
    Minimal config tuned for rapid memorization.

    High learning rate, no scheduler, no warmup, no dropout.
    """
    return {
        "model": {
            "type": "lstm",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "use_layer_norm": False,
            "use_attention": False,
            "use_population_coupling": False,
            "output_distribution": "poisson",
        },
        "training": {
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "scheduler": "none",
            "warmup_steps": 0,
            "patience": 9999,
            "grad_clip_norm": 1.0,
        },
        "loss": {
            "type": "poisson_nll",
            "log_input": False,
        },
        "compute": {},
    }


@pytest.fixture
def overfit_trainer(overfit_data, overfit_config):
    """Create a Trainer wired up for overfit testing."""
    train_loader, val_loader, m = overfit_data

    # Set model dimensions
    overfit_config["model"]["input_size"] = m
    overfit_config["model"]["output_size"] = m

    model = TeacherLSTM(
        input_size=m,
        hidden_size=overfit_config["model"]["hidden_size"],
        num_layers=overfit_config["model"]["num_layers"],
        dropout=overfit_config["model"]["dropout"],
        use_layer_norm=overfit_config["model"]["use_layer_norm"],
        use_attention=overfit_config["model"]["use_attention"],
    )

    device = torch.device("cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=overfit_config,
        device=device,
    )
    return trainer


# =============================================================================
# Tests
# =============================================================================


class TestOverfitOneBatch:
    """Tests for the overfit_one_batch sanity check method."""

    def test_overfit_one_batch_loss_drops(self, overfit_trainer):
        """
        Loss should drop dramatically when training on one batch.

        If the model can memorize 16 samples over 100 iterations,
        the training infra is working. We check for at least a 50%
        drop (less strict than the script's 10x requirement, because
        the test fixture uses a smaller model and fewer iterations).
        """
        result = overfit_trainer.overfit_one_batch(n_iters=100, log_every=50)

        # Loss should decrease
        assert result["final_loss"] < result["initial_loss"], (
            f"Loss did not decrease: {result['initial_loss']:.4f} -> "
            f"{result['final_loss']:.4f}"
        )

        # At least 50% reduction (ratio < 0.5)
        assert result["loss_ratio"] < 0.5, (
            f"Loss ratio {result['loss_ratio']:.4f} >= 0.5: model is not "
            f"learning. initial={result['initial_loss']:.4f}, "
            f"final={result['final_loss']:.4f}"
        )

    def test_overfit_one_batch_returns_history(self, overfit_trainer):
        """
        overfit_one_batch should return a dict with the expected keys
        and correct list length.
        """
        n_iters = 50
        result = overfit_trainer.overfit_one_batch(n_iters=n_iters, log_every=25)

        # Check all expected keys are present
        assert "losses" in result
        assert "initial_loss" in result
        assert "final_loss" in result
        assert "loss_ratio" in result
        assert "converged" in result

        # Losses list should have exactly n_iters entries
        assert len(result["losses"]) == n_iters, (
            f"Expected {n_iters} loss entries, got {len(result['losses'])}"
        )

        # All losses should be finite
        for i, loss_val in enumerate(result["losses"]):
            assert np.isfinite(loss_val), (
                f"Loss at step {i+1} is not finite: {loss_val}"
            )

    def test_overfit_one_batch_converged_flag(self, overfit_trainer):
        """
        With enough iterations the converged flag should be True.

        Running 500 iterations on a tiny synthetic model/batch should
        achieve the 5x reduction threshold (ratio < 0.2).
        """
        result = overfit_trainer.overfit_one_batch(n_iters=500, log_every=250)

        # With 500 iters and high LR, a 5-channel synthetic model should converge
        assert result["converged"], (
            f"Model did not converge after 500 iters: "
            f"loss_ratio={result['loss_ratio']:.4f} "
            f"(need < 0.2)"
        )


# =============================================================================
# LRU Overfit Tests — same structure as LSTM, using create_teacher_model()
# =============================================================================


@pytest.fixture
def overfit_config_lru():
    """
    Minimal config for LRU rapid memorization.

    Same as LSTM overfit config but with architecture='lru'.
    """
    return {
        "model": {
            "architecture": "lru",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "use_layer_norm": False,
            "use_attention": False,
            "use_population_coupling": False,
            "output_distribution": "poisson",
            "gated": False,
        },
        "training": {
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "scheduler": "none",
            "warmup_steps": 0,
            "patience": 9999,
            "grad_clip_norm": 1.0,
        },
        "loss": {
            "type": "poisson_nll",
            "log_input": False,
        },
        "compute": {},
    }


@pytest.fixture
def overfit_trainer_lru(overfit_data, overfit_config_lru):
    """Create a Trainer wired up with an LRU model for overfit testing."""
    train_loader, val_loader, m = overfit_data

    # Use factory to create LRU model (same as production code path)
    model = create_teacher_model(overfit_config_lru, input_size=m)

    device = torch.device("cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=overfit_config_lru,
        device=device,
    )
    return trainer


class TestOverfitOneBatchLRU:
    """LRU overfit tests — mirrors TestOverfitOneBatch for LSTM."""

    def test_overfit_one_batch_lru_loss_drops(self, overfit_trainer_lru):
        """
        LRU loss should drop dramatically when training on one batch.

        Same threshold as the LSTM test: at least 50% drop.
        """
        result = overfit_trainer_lru.overfit_one_batch(n_iters=100, log_every=50)

        # Loss should decrease
        assert result["final_loss"] < result["initial_loss"], (
            f"LRU loss did not decrease: {result['initial_loss']:.4f} -> "
            f"{result['final_loss']:.4f}"
        )

        # At least 50% reduction (ratio < 0.5)
        assert result["loss_ratio"] < 0.5, (
            f"LRU loss ratio {result['loss_ratio']:.4f} >= 0.5: model is not "
            f"learning. initial={result['initial_loss']:.4f}, "
            f"final={result['final_loss']:.4f}"
        )

    def test_overfit_one_batch_lru_converged_flag(self, overfit_trainer_lru):
        """
        LRU should converge with 500 iterations (5x reduction).
        """
        result = overfit_trainer_lru.overfit_one_batch(n_iters=500, log_every=250)

        assert result["converged"], (
            f"LRU did not converge after 500 iters: "
            f"loss_ratio={result['loss_ratio']:.4f} "
            f"(need < 0.2)"
        )

