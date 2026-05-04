"""
Tests for src/train/fgl_trainer.py

Validates the FGL training loop with a synthetic teacher-student setup.
Uses simple linear models as teacher/student to verify:
  - Teacher gradients are frozen
  - Composite loss is computed correctly
  - Student learns to track teacher predictions
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train.fgl_trainer import FGLTrainer


# =============================================================================
# Mock models
# =============================================================================

class MockTeacher(nn.Module):
    """
    Simple linear model pretending to be a teacher.

    Always returns softplus(linear(mean(x, dim=1))) to produce
    non-negative rates.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Forward: (B, T, M) -> (B, M)."""
        # Average over time dimension
        x_mean = x.mean(dim=1)  # (B, M)
        return self.softplus(self.linear(x_mean))


class MockStudent(nn.Module):
    """Same architecture as MockTeacher but with random init."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Forward: (B, T, M) -> (B, M)."""
        x_mean = x.mean(dim=1)
        return self.softplus(self.linear(x_mean))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fgl_setup():
    """
    Create a complete FGL training setup with synthetic data.

    Returns teacher, student, loaders, config, and device.
    """
    torch.manual_seed(42)

    M = 8
    T = 10
    N = 200

    # Generate synthetic FGL triplets: (x_student, x_teacher, y_target)
    x_student = torch.randn(N, T, M).abs()  # Non-negative
    x_teacher = torch.randn(N, T, M).abs()
    y_target = torch.poisson(torch.ones(N, M) * 2.0)

    dataset = TensorDataset(x_student, x_teacher, y_target)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    teacher = MockTeacher(M, M)
    student = MockStudent(M, M)

    config = {
        "training": {
            "learning_rate": 0.01,
            "epochs": 3,
            "patience": 10,
            "grad_clip_norm": 1.0,
            "scheduler": "none",
            "val_every_n_epochs": 1,
        }
    }

    device = torch.device("cpu")

    return {
        "teacher": teacher,
        "student": student,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "config": config,
        "device": device,
    }


# =============================================================================
# Tests
# =============================================================================

class TestFGLTrainerInit:
    """Tests for FGLTrainer initialization."""

    def test_teacher_frozen(self, fgl_setup):
        """Teacher parameters should have requires_grad=False after init."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.5,
        )
        for param in trainer.teacher.parameters():
            assert not param.requires_grad, "Teacher params should be frozen"

    def test_student_trainable(self, fgl_setup):
        """Student parameters should remain trainable."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.5,
        )
        for param in trainer.model.parameters():
            assert param.requires_grad, "Student params should be trainable"


class TestFGLTrainerTraining:
    """Tests for FGL training loop behavior."""

    def test_loss_decreases(self, fgl_setup):
        """Training loss should decrease over a few epochs."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.5,
        )
        loss_epoch1 = trainer._train_one_epoch()
        loss_epoch2 = trainer._train_one_epoch()
        loss_epoch3 = trainer._train_one_epoch()

        # Loss should generally decrease (allow some noise)
        assert loss_epoch3 < loss_epoch1 * 1.1, (
            f"Loss should decrease: epoch1={loss_epoch1:.4f}, "
            f"epoch3={loss_epoch3:.4f}"
        )

    def test_teacher_params_unchanged(self, fgl_setup):
        """Teacher weights should not change during training."""
        teacher = fgl_setup["teacher"]
        # Snapshot teacher params before training
        params_before = {
            name: p.clone() for name, p in teacher.named_parameters()
        }

        trainer = FGLTrainer(
            teacher=teacher,
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.5,
        )
        trainer._train_one_epoch()

        # Check teacher params are unchanged
        for name, param in trainer.teacher.named_parameters():
            torch.testing.assert_close(
                param, params_before[name],
                msg=f"Teacher param '{name}' was modified during training!",
            )


class TestFGLTrainerValidation:
    """Tests for FGL validation metrics."""

    def test_validate_returns_all_metrics(self, fgl_setup):
        """Validation should return all FGL metrics."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.5,
        )

        metrics = trainer._validate()
        expected_keys = [
            "val_loss", "val_supervised_loss", "val_distill_loss",
            "val_pearson_r", "val_poisson_nll", "val_mae", "val_mse",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert torch.isfinite(
                torch.tensor(metrics[key])
            ), f"{key} is not finite: {metrics[key]}"

    def test_alpha_zero_is_pure_distillation(self, fgl_setup):
        """With alpha=0, total loss should equal distillation loss."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=0.0,
        )
        metrics = trainer._validate()
        # val_loss should be close to val_distill_loss when alpha=0
        assert abs(metrics["val_loss"] - metrics["val_distill_loss"]) < 0.01

    def test_alpha_one_is_pure_supervised(self, fgl_setup):
        """With alpha=1, total loss should equal supervised loss."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
            alpha=1.0,
        )
        metrics = trainer._validate()
        # val_loss should be close to val_supervised_loss when alpha=1
        assert abs(metrics["val_loss"] - metrics["val_supervised_loss"]) < 0.01


class TestPoissonKL:
    """Tests for the _poisson_kl helper method."""

    def test_kl_zero_for_identical(self, fgl_setup):
        """KL divergence should be ~0 when teacher = student."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
        )
        rates = torch.ones(10, 5) * 3.0
        kl = trainer._poisson_kl(rates, rates)
        assert kl.item() < 1e-6, f"KL should be ~0 for identical rates, got {kl}"

    def test_kl_positive(self, fgl_setup):
        """KL divergence should be positive for different distributions."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
        )
        t_rates = torch.ones(10, 5) * 5.0
        s_rates = torch.ones(10, 5) * 2.0
        kl = trainer._poisson_kl(t_rates, s_rates)
        assert kl.item() > 0, f"KL should be positive, got {kl}"

    def test_kl_handles_zeros(self, fgl_setup):
        """KL should handle near-zero rates gracefully (clamped by EPS)."""
        trainer = FGLTrainer(
            teacher=fgl_setup["teacher"],
            student=fgl_setup["student"],
            train_loader=fgl_setup["train_loader"],
            val_loader=fgl_setup["val_loader"],
            config=fgl_setup["config"],
            device=fgl_setup["device"],
        )
        t_rates = torch.zeros(10, 5)
        s_rates = torch.ones(10, 5) * 2.0
        kl = trainer._poisson_kl(t_rates, s_rates)
        assert torch.isfinite(kl), f"KL should be finite, got {kl}"
