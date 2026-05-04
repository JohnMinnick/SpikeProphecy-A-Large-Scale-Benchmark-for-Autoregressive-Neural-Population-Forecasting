"""
End-to-end smoke test: Trainer + session-specific heads.

Simulates what NRP will do: a fake session-cycling loader with
current_session_id, a model with session_dims, and the real Trainer
running a few training + eval steps. If this passes, the pipeline
won't crash on NRP (barring data-specific issues).
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.teacher import create_teacher_model
from src.train.trainer import Trainer


class FakeSessionLoader:
    """
    Minimal mock of SessionCyclingLoader for testing.

    Yields batches from 2 fake sessions with different neuron counts,
    setting current_session_id before each batch.
    """

    def __init__(self, sessions, batch_size=4, T=10):
        self.sessions = sessions  # dict: session_id -> n_neurons
        self.batch_size = batch_size
        self.T = T
        self.current_session_id = None
        # M_max: all inputs are padded to the largest session
        self.m_max = max(sessions.values())
        # Estimate total batches (2 batches per session)
        self._total_batches = len(sessions) * 2

    def __len__(self):
        return self._total_batches

    def __iter__(self):
        for sid, n_neurons in self.sessions.items():
            self.current_session_id = sid
            # Yield 2 batches per session
            for _ in range(2):
                # Input is always padded to M_max (matching real data loader)
                # x: (batch, T, M_max), y: (batch, N_i), mask: (batch, N_i)
                x = torch.randn(self.batch_size, self.T, self.m_max)
                y = torch.rand(self.batch_size, n_neurons).abs() * 3.0
                mask = torch.ones(self.batch_size, n_neurons)
                yield x, y, mask


class TestSessionHeadsTrainingLoop:
    """End-to-end: Trainer trains and evaluates with session-specific heads."""

    @pytest.fixture
    def session_dims(self):
        """Two sessions with different neuron counts."""
        return {"session_000": 80, "session_001": 120}

    @pytest.fixture
    def lru_model(self, session_dims):
        """LRU model with session-specific heads."""
        config = {
            "model": {
                "architecture": "lru",
                "hidden_size": 32,
                "num_layers": 1,
            },
        }
        return create_teacher_model(
            config, input_size=120, session_dims=session_dims,
        )

    @pytest.fixture
    def lstm_model(self, session_dims):
        """LSTM model with session-specific heads."""
        config = {
            "model": {
                "architecture": "lstm",
                "hidden_size": 32,
                "num_layers": 1,
            },
        }
        return create_teacher_model(
            config, input_size=120, session_dims=session_dims,
        )

    def _run_training_loop(self, model, session_dims):
        """Helper: run 2 epochs of training + evaluation."""
        train_loader = FakeSessionLoader(session_dims, batch_size=4)
        val_loader = FakeSessionLoader(session_dims, batch_size=4)

        config = {
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "patience": 10,
                "val_every_n": 1,
            },
            "loss": {"type": "poisson_nll"},
        }

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )

        history = trainer.train()
        return history

    def test_lru_trains_without_crash(self, lru_model, session_dims):
        """LRU + session heads should complete 2 epochs without crashing."""
        history = self._run_training_loop(lru_model, session_dims)

        # Verify we got 2 epochs of data
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        # Loss should be finite (not NaN/Inf)
        assert all(
            torch.isfinite(torch.tensor(l))
            for l in history["train_loss"]
        ), "Train loss has NaN/Inf"

    def test_lstm_trains_without_crash(self, lstm_model, session_dims):
        """LSTM + session heads should complete 2 epochs without crashing."""
        history = self._run_training_loop(lstm_model, session_dims)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(
            torch.isfinite(torch.tensor(l))
            for l in history["train_loss"]
        ), "Train loss has NaN/Inf"

    def test_eval_returns_metrics(self, lru_model, session_dims):
        """Evaluate should return all expected metric keys."""
        train_loader = FakeSessionLoader(session_dims, batch_size=4)
        val_loader = FakeSessionLoader(session_dims, batch_size=4)

        config = {
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "patience": 10,
                "val_every_n": 1,
            },
            "loss": {"type": "poisson_nll"},
        }

        trainer = Trainer(
            model=lru_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )

        metrics = trainer.evaluate(val_loader, prefix="val")
        # All expected keys present
        assert "val_loss" in metrics
        assert "val_pearson_r" in metrics
        assert "val_mae" in metrics
        assert "val_mse" in metrics
        # Values should be finite
        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"{k}={v} is not finite"

    def test_loss_decreases_with_more_steps(self, session_dims):
        """Loss should decrease over more iterations (basic learning check)."""
        config = {
            "model": {
                "architecture": "lru",
                "hidden_size": 32,
                "num_layers": 1,
            },
        }
        model = create_teacher_model(
            config, input_size=120, session_dims=session_dims,
        )

        train_loader = FakeSessionLoader(session_dims, batch_size=8)
        val_loader = FakeSessionLoader(session_dims, batch_size=8)

        trainer_config = {
            "training": {
                "epochs": 10,
                "learning_rate": 0.01,
                "patience": 20,
                "val_every_n": 5,
            },
            "loss": {"type": "poisson_nll"},
        }

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=torch.device("cpu"),
        )

        history = trainer.train()
        # Loss at epoch 10 should be lower than epoch 1
        assert history["train_loss"][-1] < history["train_loss"][0], (
            f"Loss didn't decrease: {history['train_loss'][0]:.4f} -> "
            f"{history['train_loss'][-1]:.4f}"
        )
