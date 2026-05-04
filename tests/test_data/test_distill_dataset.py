"""
Tests for the DistillCollator (online teacher inference for distillation).

Uses a simple mock teacher model to validate:
- Output triplet shape: (x, y, teacher_rates)
- Teacher rates are non-negative (softplus output)
- Correct batch dimensions
- Handles both 3-tuple and 4-tuple (with covariates) inputs
"""

import torch
import torch.nn as nn
import pytest

from src.data.distill_dataset import DistillCollator


# =============================================================================
# Mock teacher model
# =============================================================================


class MockTeacher(nn.Module):
    """Minimal teacher that returns softplus'd linear projection."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict rates from the last timestep."""
        # x: (batch, T, M), take last timestep
        last = x[:, -1, :]
        return self.softplus(self.proj(last))


# =============================================================================
# Tests
# =============================================================================


class TestDistillCollator:
    """Tests for DistillCollator with a mock teacher."""

    @pytest.fixture
    def setup(self):
        """Create a mock teacher and collator."""
        input_size = 20
        output_size = 20
        teacher = MockTeacher(input_size, output_size)
        device = torch.device("cpu")
        collator = DistillCollator(teacher, device, output_channels=output_size)
        return collator, input_size, output_size

    def test_output_is_triplet(self, setup):
        """Collator should return (x, y, teacher_rates) 3-tuple."""
        collator, M, _ = setup
        T = 10
        batch_size = 4

        # Simulate a batch of 3-tuple samples from MaskedSpikeCountDataset
        batch = [
            (torch.randn(T, M), torch.randn(M), torch.ones(M))
            for _ in range(batch_size)
        ]

        result = collator(batch)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"

    def test_output_shapes(self, setup):
        """Output tensors should have correct batch dimensions."""
        collator, M, output_size = setup
        T = 10
        batch_size = 8

        batch = [
            (torch.randn(T, M), torch.randn(M), torch.ones(M))
            for _ in range(batch_size)
        ]

        x, y, teacher_rates = collator(batch)

        assert x.shape == (batch_size, T, M), (
            f"x shape: expected ({batch_size}, {T}, {M}), got {x.shape}"
        )
        assert y.shape == (batch_size, output_size), (
            f"y shape: expected ({batch_size}, {output_size}), got {y.shape}"
        )
        assert teacher_rates.shape == (batch_size, output_size), (
            f"teacher_rates shape: expected ({batch_size}, {output_size}), "
            f"got {teacher_rates.shape}"
        )

    def test_teacher_rates_non_negative(self, setup):
        """Teacher rates should be non-negative (softplus output)."""
        collator, M, _ = setup
        T = 10
        batch_size = 4

        batch = [
            (torch.randn(T, M), torch.randn(M), torch.ones(M))
            for _ in range(batch_size)
        ]

        _, _, teacher_rates = collator(batch)
        assert (teacher_rates >= 0).all(), "Teacher rates should be non-negative"

    def test_teacher_is_frozen(self, setup):
        """Teacher parameters should have requires_grad=False."""
        collator, _, _ = setup

        for param in collator.teacher.parameters():
            assert not param.requires_grad, (
                "Teacher params should be frozen (requires_grad=False)"
            )

    def test_handles_covariate_tuple(self, setup):
        """Collator should handle 4-tuple inputs (with covariates)."""
        collator, M, output_size = setup
        T = 10
        batch_size = 4
        n_cov = 5

        # 4-tuple: (x, y, mask, covariates)
        batch = [
            (
                torch.randn(T, M),
                torch.randn(M),
                torch.ones(M),
                torch.randn(T, n_cov),     # Temporal covariates
            )
            for _ in range(batch_size)
        ]

        result = collator(batch)
        assert len(result) == 3, "Should still return 3-tuple"

        x, y, teacher_rates = result
        assert x.shape == (batch_size, T, M)
        assert teacher_rates.shape == (batch_size, output_size)

    def test_output_on_cpu(self, setup):
        """All output tensors should be on CPU."""
        collator, M, _ = setup
        T = 10
        batch_size = 4

        batch = [
            (torch.randn(T, M), torch.randn(M), torch.ones(M))
            for _ in range(batch_size)
        ]

        x, y, teacher_rates = collator(batch)
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"
        assert teacher_rates.device.type == "cpu"

    def test_single_sample_batch(self, setup):
        """Single-sample batch should work correctly."""
        collator, M, output_size = setup
        T = 10

        batch = [(torch.randn(T, M), torch.randn(M), torch.ones(M))]

        x, y, teacher_rates = collator(batch)
        assert x.shape == (1, T, M)
        assert teacher_rates.shape == (1, output_size)
