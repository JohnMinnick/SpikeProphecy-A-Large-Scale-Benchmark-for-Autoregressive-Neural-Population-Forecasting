"""
Tests for DistillationLoss.
"""

import pytest
import torch
from src.distill.loss import DistillationLoss


@pytest.fixture
def loss_fn():
    return DistillationLoss(distill_weight=0.5, reg_weight=0.1)


def test_loss_computation(loss_fn):
    """Loss should return a dict with total and components."""
    B, M, T, H = 4, 3, 10, 5
    
    # Mock data
    student_rates = torch.rand(B, M, requires_grad=True) * 10
    teacher_rates = torch.rand(B, M) * 10
    ground_truth = torch.randint(0, 10, (B, M)).float()
    student_spikes = torch.rand(B, T, H, requires_grad=True) # Soft spikes for testing grad
    
    losses = loss_fn(student_rates, student_spikes, ground_truth, teacher_rates)
    
    assert "loss" in losses
    assert "poisson" in losses
    assert "distill" in losses
    assert "reg" in losses
    
    # Check total
    total = losses["loss"]
    poisson = losses["poisson"]
    distill = losses["distill"]
    reg = losses["reg"]
    
    expected = poisson + 0.5 * distill + 0.1 * reg
    assert torch.isclose(total, expected)


def test_gradients(loss_fn):
    """Gradients should flow back to student inputs."""
    B, M, T, H = 2, 2, 5, 2
    student_rates = torch.rand(B, M, requires_grad=True)
    teacher_rates = torch.rand(B, M)
    ground_truth = torch.rand(B, M)
    student_spikes = torch.rand(B, T, H, requires_grad=True)
    
    losses = loss_fn(student_rates, student_spikes, ground_truth, teacher_rates)
    total = losses["loss"]
    
    total.backward()
    
    assert student_rates.grad is not None
    assert student_spikes.grad is not None
    assert torch.any(student_rates.grad != 0)
    assert torch.any(student_spikes.grad != 0)


def test_zero_reg_weight():
    """If reg_weight is 0, regularization term should be 0."""
    loss_fn = DistillationLoss(reg_weight=0.0)
    B, M, T, H = 2, 2, 5, 2
    student_rates = torch.rand(B, M)
    student_spikes = torch.rand(B, T, H)
    ground_truth = torch.rand(B, M)
    teacher_rates = torch.rand(B, M)
    
    losses = loss_fn(student_rates, student_spikes, ground_truth, teacher_rates)
    assert losses["reg"] == 0.0


def test_poisson_kl_identical_rates_is_zero():
    """KL divergence should be zero when teacher and student rates match."""
    loss_fn = DistillationLoss(distill_weight=1.0, reg_weight=0.0)
    B, M, T, H = 4, 3, 10, 5

    # Use identical teacher and student rates
    rates = torch.rand(B, M) * 10 + 0.1  # Avoid near-zero for stability
    student_spikes = torch.rand(B, T, H)

    losses = loss_fn(rates, student_spikes, torch.rand(B, M), rates)

    # Distillation term should be ~0 (within float precision)
    assert torch.isclose(losses["distill"], torch.tensor(0.0), atol=1e-6)


def test_poisson_kl_known_answer():
    """Verify Poisson KL against hand-computed values.

    KL(Poisson(λ_t) || Poisson(λ_s)) = λ_t * log(λ_t / λ_s) - λ_t + λ_s

    For λ_t=2.0, λ_s=4.0:
        KL = 2.0 * log(2.0 / 4.0) - 2.0 + 4.0
           = 2.0 * log(0.5) + 2.0
           = 2.0 * (-0.6931...) + 2.0
           = -1.3863 + 2.0
           = 0.6137
    """
    import math

    loss_fn = DistillationLoss(distill_weight=1.0, reg_weight=0.0)

    # Single-element tensors for easy hand-verification
    teacher_rates = torch.tensor([[2.0]])
    student_rates = torch.tensor([[4.0]], requires_grad=True)
    student_spikes = torch.zeros(1, 1, 1)
    ground_truth = torch.tensor([[1.0]])

    losses = loss_fn(student_rates, student_spikes, ground_truth, teacher_rates)

    expected_kl = 2.0 * math.log(2.0 / 4.0) - 2.0 + 4.0  # ≈ 0.6137
    assert torch.isclose(
        losses["distill"], torch.tensor(expected_kl), atol=1e-4
    ), f"Expected KL ≈ {expected_kl:.4f}, got {losses['distill'].item():.4f}"

    # Verify gradient flows through KL term
    losses["loss"].backward()
    assert student_rates.grad is not None
    assert torch.any(student_rates.grad != 0)

