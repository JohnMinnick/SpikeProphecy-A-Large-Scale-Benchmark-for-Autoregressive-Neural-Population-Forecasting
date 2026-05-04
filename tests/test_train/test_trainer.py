"""
Tests for the training engine.

Validates training step, loss decrease, checkpoint save/load,
early stopping, and history logging.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.teacher import TeacherLSTM
from src.train.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_data():
    """Create synthetic train and val data."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create some structured data: x is history, y is the "next step"
    m = 5  # channels
    n_train, n_val = 100, 30
    T = 10  # history length

    # Training data
    x_train = torch.randn(n_train, T, m)
    y_train = torch.abs(torch.randn(n_train, m))  # Non-negative targets

    # Validation data
    x_val = torch.randn(n_val, T, m)
    y_val = torch.abs(torch.randn(n_val, m))

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader, m


@pytest.fixture
def config():
    """Minimal training config."""
    return {
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 5,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "patience": 10,  # Don't early stop in short tests
            "grad_clip_norm": 1.0,
        },
        "loss": {
            "log_input": False,
        },
    }


@pytest.fixture
def trainer(synthetic_data, config):
    """Create a trainer instance."""
    train_loader, val_loader, m = synthetic_data
    model = TeacherLSTM(input_size=m, hidden_size=16, num_layers=1, dropout=0.0)
    device = torch.device("cpu")
    return Trainer(model, train_loader, val_loader, config, device)


# =============================================================================
# Training step tests
# =============================================================================

class TestTrainingStep:
    """Tests for individual training steps."""

    def test_train_one_epoch_returns_loss(self, trainer):
        """_train_one_epoch should return a finite float loss."""
        loss = trainer._train_one_epoch()
        assert isinstance(loss, float)
        assert np.isfinite(loss), "Training loss is not finite"

    def test_validate_returns_all_metrics(self, trainer):
        """_validate should return all expected metric keys."""
        metrics = trainer._validate()
        required_keys = {
            "val_loss", "val_poisson_nll", "val_pearson_r",
            "val_mae", "val_mse",
        }
        assert required_keys.issubset(set(metrics.keys()))

    def test_validate_metrics_finite(self, trainer):
        """All validation metrics should be finite."""
        metrics = trainer._validate()
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"


# =============================================================================
# Full training tests
# =============================================================================

class TestFullTraining:
    """Tests for the complete training loop."""

    def test_training_runs(self, trainer):
        """Training should complete without errors."""
        history = trainer.train()
        assert len(history["train_loss"]) > 0

    def test_loss_decreases(self, synthetic_data, config):
        """Training loss should generally decrease."""
        train_loader, val_loader, m = synthetic_data

        # Use more epochs and higher LR for clear convergence
        config["training"]["epochs"] = 20
        config["training"]["learning_rate"] = 0.005

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu")
        )
        history = trainer.train()

        # First loss should be higher than last loss
        first_loss = history["train_loss"][0]
        last_loss = history["train_loss"][-1]
        assert last_loss < first_loss, (
            f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
        )

    def test_history_has_correct_length(self, trainer):
        """History should have entries for each trained epoch."""
        history = trainer.train()
        n_epochs = len(history["train_loss"])
        for key, values in history.items():
            # Skip non-list entries (e.g., population_metrics dict)
            if not isinstance(values, list):
                continue
            assert len(values) == n_epochs, (
                f"{key} has {len(values)} entries but expected {n_epochs}"
            )

    def test_learning_rate_recorded(self, trainer):
        """Learning rate should be recorded in history."""
        history = trainer.train()
        assert len(history["learning_rate"]) > 0
        assert all(lr > 0 for lr in history["learning_rate"])


# =============================================================================
# Checkpoint tests
# =============================================================================

class TestCheckpoints:
    """Tests for model checkpointing."""

    def test_save_and_load_checkpoint(self, synthetic_data, config, tmp_path):
        """Checkpoint save/load should restore model state."""
        train_loader, val_loader, m = synthetic_data
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        trainer = Trainer(
            model, train_loader, val_loader, config,
            torch.device("cpu"), exp_dir=tmp_path,
        )

        # Train a few epochs
        trainer.train()

        # Load the best checkpoint
        best_path = tmp_path / "best_model.pt"
        assert best_path.exists(), "Best model checkpoint not saved"

        # Create a new trainer and load checkpoint
        model2 = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        trainer2 = Trainer(
            model2, train_loader, val_loader, config,
            torch.device("cpu"), exp_dir=tmp_path,
        )
        epoch = trainer2.load_checkpoint(best_path)
        assert epoch > 0

        # Loaded model should produce identical output to what was checkpointed
        model2.eval()
        x_test = torch.randn(2, 10, m)
        with torch.no_grad():
            out_loaded = model2(x_test)

        # Reload same checkpoint into another model to confirm determinism
        model3 = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model3.load_state_dict(checkpoint["model_state_dict"])
        model3.eval()
        with torch.no_grad():
            out_direct = model3(x_test)

        torch.testing.assert_close(out_loaded, out_direct)

    def test_final_model_saved(self, synthetic_data, config, tmp_path):
        """Final model checkpoint should be saved."""
        train_loader, val_loader, m = synthetic_data
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        trainer = Trainer(
            model, train_loader, val_loader, config,
            torch.device("cpu"), exp_dir=tmp_path,
        )
        trainer.train()
        assert (tmp_path / "final_model.pt").exists()


# =============================================================================
# Early stopping tests
# =============================================================================

class TestEarlyStopping:
    """Tests for early stopping logic."""

    def test_early_stopping_triggers(self, synthetic_data):
        """Training should stop early when val loss plateaus."""
        train_loader, val_loader, m = synthetic_data

        config = {
            "training": {
                "epochs": 100,
                "learning_rate": 1e-6,   # Very low LR → no real improvement
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "patience": 3,
                "grad_clip_norm": 1.0,
            },
            "loss": {"log_input": False},
        }

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu")
        )
        history = trainer.train()

        # Should stop well before 100 epochs
        actual_epochs = len(history["train_loss"])
        assert actual_epochs < 100, (
            f"Early stopping didn't trigger: ran all {actual_epochs} epochs"
        )


# =============================================================================
# Loss selection tests (ADR-0009)
# =============================================================================

class TestLossSelection:
    """Tests for configurable loss function selection in Trainer."""

    def test_negbin_loss_training_runs(self, synthetic_data):
        """Training with NegBin loss should complete without errors."""
        train_loader, val_loader, m = synthetic_data
        config = {
            "model": {"hidden_size": 16, "output_distribution": "negbin"},
            "training": {
                "epochs": 3, "learning_rate": 0.01,
                "warmup_steps": 0, "patience": 10,
                "grad_clip_norm": 1.0, "weight_decay": 0.0,
            },
            "loss": {"type": "negbin_nll", "log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1,
            dropout=0.0, output_distribution="negbin",
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        # Both val_loss and val_poisson_nll should be present
        assert "val_loss" in history
        assert "val_poisson_nll" in history

    def test_zip_loss_training_runs(self, synthetic_data):
        """Training with ZIP loss should complete without errors."""
        train_loader, val_loader, m = synthetic_data
        config = {
            "model": {"hidden_size": 16, "output_distribution": "zip"},
            "training": {
                "epochs": 3, "learning_rate": 0.01,
                "warmup_steps": 0, "patience": 10,
                "grad_clip_norm": 1.0, "weight_decay": 0.0,
            },
            "loss": {"type": "zip_nll", "log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1,
            dropout=0.0, output_distribution="zip",
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 3

    def test_negbin_loss_decreases(self, synthetic_data):
        """NegBin training loss should decrease over epochs."""
        train_loader, val_loader, m = synthetic_data
        config = {
            "model": {"hidden_size": 16, "output_distribution": "negbin"},
            "training": {
                # NegBin needs more epochs/higher LR due to extra dispersion params
                "epochs": 30, "learning_rate": 0.01,
                "warmup_steps": 0, "patience": 35,
                "grad_clip_norm": 1.0, "weight_decay": 0.0,
            },
            "loss": {"type": "negbin_nll", "log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1,
            dropout=0.0, output_distribution="negbin",
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )
        history = trainer.train()
        # First loss should be higher than last
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_invalid_loss_type_raises(self, synthetic_data):
        """Invalid loss.type should raise ValueError."""
        train_loader, val_loader, m = synthetic_data
        config = {
            "training": {
                "epochs": 1, "learning_rate": 0.01,
                "warmup_steps": 0, "patience": 10,
                "grad_clip_norm": 1.0, "weight_decay": 0.0,
            },
            "loss": {"type": "invalid_loss"},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        with pytest.raises(ValueError, match="loss.type"):
            Trainer(
                model, train_loader, val_loader, config, torch.device("cpu"),
            )

    def test_default_loss_is_poisson(self, synthetic_data):
        """Default loss type should be poisson_nll."""
        train_loader, val_loader, m = synthetic_data
        config = {
            "training": {
                "epochs": 1, "learning_rate": 0.01,
                "warmup_steps": 0, "patience": 10,
                "grad_clip_norm": 1.0, "weight_decay": 0.0,
            },
            "loss": {},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )
        assert trainer.loss_type == "poisson_nll"


# =============================================================================
# Evaluate method tests (train-set validation / sanity check)
# =============================================================================

class TestEvaluate:
    """Tests for the public evaluate() method."""

    def test_evaluate_returns_metrics_with_prefix(self, trainer, synthetic_data):
        """evaluate() should return metric keys using the given prefix."""
        train_loader, val_loader, m = synthetic_data
        metrics = trainer.evaluate(train_loader, prefix="train")
        required_keys = {
            "train_loss", "train_poisson_nll", "train_pearson_r",
            "train_mae", "train_mse",
        }
        assert required_keys.issubset(set(metrics.keys()))

    def test_evaluate_custom_prefix(self, trainer, synthetic_data):
        """evaluate() should support arbitrary prefix strings."""
        train_loader, _, _ = synthetic_data
        metrics = trainer.evaluate(train_loader, prefix="foo")
        assert all(k.startswith("foo_") for k in metrics.keys())

    def test_evaluate_on_train_data_is_finite(self, trainer, synthetic_data):
        """All metrics should be finite when evaluating on training data."""
        train_loader, _, _ = synthetic_data
        metrics = trainer.evaluate(train_loader, prefix="train")
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_evaluate_matches_validate(self, trainer):
        """evaluate(val_loader, 'val') should match _validate()."""
        # Both should produce identical results since _validate delegates
        validate_metrics = trainer._validate()
        evaluate_metrics = trainer.evaluate(trainer.val_loader, prefix="val")
        for key in validate_metrics:
            assert abs(validate_metrics[key] - evaluate_metrics[key]) < 1e-6, (
                f"{key}: _validate={validate_metrics[key]:.8f} vs "
                f"evaluate={evaluate_metrics[key]:.8f}"
            )

    def test_train_loss_lower_after_training(self, synthetic_data, config):
        """After training, train-set loss should be lower than before."""
        train_loader, val_loader, m = synthetic_data

        config["training"]["epochs"] = 20
        config["training"]["learning_rate"] = 0.005

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        t = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu")
        )

        # Measure train-set loss before training
        before = t.evaluate(train_loader, prefix="train")

        # Train
        t.train()

        # Measure train-set loss after training
        after = t.evaluate(train_loader, prefix="train")

        assert after["train_loss"] < before["train_loss"], (
            f"Train loss did not decrease: {before['train_loss']:.4f} → "
            f"{after['train_loss']:.4f}"
        )


# =============================================================================
# Scheduler type tests
# =============================================================================


class TestSchedulerTypes:
    """Tests for configurable LR scheduler selection."""

    def _make_trainer(self, synthetic_data, scheduler_type, extra_cfg=None):
        """Helper to create a trainer with a specific scheduler type."""
        train_loader, val_loader, m = synthetic_data
        cfg = {
            "model": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
            "training": {
                "epochs": 5,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "patience": 10,
                "grad_clip_norm": 1.0,
                "scheduler": scheduler_type,
            },
            "loss": {"log_input": False},
        }
        if extra_cfg:
            cfg["training"].update(extra_cfg)
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        return Trainer(
            model, train_loader, val_loader, cfg, torch.device("cpu")
        )

    def test_cosine_scheduler_created(self, synthetic_data):
        """Setting scheduler='cosine' should create CosineAnnealingLR."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        t = self._make_trainer(synthetic_data, "cosine")
        assert isinstance(t.scheduler, CosineAnnealingLR)

    def test_cosine_restarts_scheduler_created(self, synthetic_data):
        """Setting scheduler='cosine_restarts' creates CosineAnnealingWarmRestarts."""
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        t = self._make_trainer(
            synthetic_data, "cosine_restarts",
            extra_cfg={"scheduler_t0": 3, "scheduler_t_mult": 1},
        )
        assert isinstance(t.scheduler, CosineAnnealingWarmRestarts)

    def test_none_scheduler_is_none(self, synthetic_data):
        """Setting scheduler='none' should result in None scheduler."""
        t = self._make_trainer(synthetic_data, "none")
        assert t.scheduler is None

    def test_none_scheduler_training_runs(self, synthetic_data):
        """Training with scheduler='none' should complete without errors."""
        t = self._make_trainer(synthetic_data, "none")
        history = t.train()
        assert len(history["train_loss"]) == 5

    def test_default_t0_and_t_mult(self, synthetic_data):
        """Default scheduler_t0=10 and scheduler_t_mult=2 should be applied."""
        t = self._make_trainer(synthetic_data, "cosine_restarts")
        assert t.scheduler_t0 == 10
        assert t.scheduler_t_mult == 2

    def test_warm_restarts_lr_resets(self, synthetic_data):
        """LR should reset (increase) after T_0 epochs with warm restarts.

        With T_0=3 and T_mult=1, LR should reset every 3 epochs.
        After epoch 3 the LR should jump back up toward base_lr.
        """
        t = self._make_trainer(
            synthetic_data, "cosine_restarts",
            extra_cfg={
                "scheduler_t0": 3,
                "scheduler_t_mult": 1,
                "epochs": 6,
            },
        )
        history = t.train()
        lrs = history["learning_rate"]

        assert lrs[3] > lrs[2], (
            f"LR should increase after restart: epoch 3 LR={lrs[2]:.6f}, "
            f"epoch 4 LR={lrs[3]:.6f}"
        )


# =============================================================================
# Callback tests (checkpoint + metrics callbacks)
# =============================================================================


class TestCallbacks:
    """Tests for checkpoint_callback and metrics_callback."""

    def test_metrics_callback_fires(self, synthetic_data, config):
        """metrics_callback should fire after each validation pass."""
        train_loader, val_loader, m = synthetic_data

        # Track callback invocations
        callback_calls = []

        def mock_metrics_cb(epoch, history):
            """Record callback invocations for assertion."""
            callback_calls.append({
                "epoch": epoch,
                "history_keys": list(history.keys()),
                "n_train_loss": len(history.get("train_loss", [])),
            })

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        config["training"]["epochs"] = 3
        trainer = Trainer(
            model, train_loader, val_loader, config,
            torch.device("cpu"),
            metrics_callback=mock_metrics_cb,
        )
        trainer.train()

        # With val_every_n=1 (default), callback should fire once per epoch
        assert len(callback_calls) == 3, (
            f"Expected 3 callback calls, got {len(callback_calls)}"
        )
        # Each call should have epoch and history with expected keys
        for call in callback_calls:
            assert "epoch" in call
            assert "train_loss" in call["history_keys"]
            assert "val_loss" in call["history_keys"]
        # Epochs should be 1, 2, 3
        epochs = [c["epoch"] for c in callback_calls]
        assert epochs == [1, 2, 3]

    def test_metrics_callback_failure_non_fatal(self, synthetic_data, config):
        """metrics_callback exceptions should not crash training."""
        train_loader, val_loader, m = synthetic_data

        def failing_callback(epoch, history):
            """Simulate a callback failure (e.g., S3 upload error)."""
            raise ConnectionError("Simulated S3 failure")

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        config["training"]["epochs"] = 3
        trainer = Trainer(
            model, train_loader, val_loader, config,
            torch.device("cpu"),
            metrics_callback=failing_callback,
        )
        # Should complete without raising
        history = trainer.train()
        assert len(history["train_loss"]) == 3

    def test_checkpoint_callback_fires_on_improvement(
        self, synthetic_data, config, tmp_path,
    ):
        """checkpoint_callback should fire when val_loss improves."""
        train_loader, val_loader, m = synthetic_data

        ckpt_calls = []

        def mock_ckpt_cb(path, epoch):
            """Record checkpoint callback invocations."""
            ckpt_calls.append({"path": str(path), "epoch": epoch})

        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0
        )
        config["training"]["epochs"] = 5
        trainer = Trainer(
            model, train_loader, val_loader, config,
            torch.device("cpu"), exp_dir=tmp_path,
            checkpoint_callback=mock_ckpt_cb,
        )
        trainer.train()

        # Callback should fire at least once (epoch 1 always improves)
        assert len(ckpt_calls) >= 1
        assert ckpt_calls[0]["epoch"] >= 1
