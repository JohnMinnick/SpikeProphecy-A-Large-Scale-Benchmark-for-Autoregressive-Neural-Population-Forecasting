"""
Tests for hidden-state alignment in MultiHeadDistillationLoss.

Covers:
    - hidden_align_weight=0.0 produces zero alignment loss (backward compat)
    - hidden_align_weight>0 adds MSE term when teacher_hidden is provided
    - Alignment loss is correct against hand-computed MSE
    - Gradient flows through alignment loss to student membrane potentials
    - Dimension mismatch handling (truncation to min T)
    - No crash when teacher_hidden is None
"""

import pytest
import torch

from src.distill.multi_head_loss import MultiHeadDistillationLoss


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_student_output():
    """Minimal student output dict for testing."""
    B, M, T, H = 4, 5, 10, 16
    return {
        "rates": torch.rand(B, M) * 5 + 0.1,
        "spikes": torch.rand(B, T, H),
        "membrane_potentials": torch.randn(B, T, H, requires_grad=True),
    }


@pytest.fixture
def base_tensors():
    """Ground truth and teacher rates."""
    B, M = 4, 5
    return {
        "ground_truth": torch.randint(0, 10, (B, M)).float(),
        "teacher_rates": torch.rand(B, M) * 5 + 0.1,
    }


@pytest.fixture
def teacher_hidden():
    """Teacher hidden states (batch, T, H)."""
    B, T, H = 4, 10, 16
    return torch.randn(B, T, H)


# =============================================================================
# Backward Compatibility
# =============================================================================

class TestHiddenAlignBackwardCompat:
    """Verify hidden alignment doesn't break existing behavior."""

    def test_zero_weight_no_alignment(
        self, base_student_output, base_tensors, teacher_hidden
    ):
        """hidden_align_weight=0.0 should produce zero alignment loss."""
        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.0,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result = loss_fn(
            base_student_output,
            base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None,
            teacher_hidden=teacher_hidden,
        )
        assert result["hidden_align_loss"].item() == 0.0

    def test_none_teacher_hidden_no_crash(
        self, base_student_output, base_tensors
    ):
        """teacher_hidden=None should not crash, alignment loss = 0."""
        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result = loss_fn(
            base_student_output,
            base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None,
            teacher_hidden=None,
        )
        assert result["hidden_align_loss"].item() == 0.0

    def test_no_membrane_in_output_no_crash(self, base_tensors, teacher_hidden):
        """If student output lacks membrane_potentials, alignment = 0."""
        B, M, T, H = 4, 5, 10, 16
        student_output = {
            "rates": torch.rand(B, M),
            "spikes": torch.rand(B, T, H),
            # No "membrane_potentials" key
        }
        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result = loss_fn(
            student_output,
            base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None,
            teacher_hidden=teacher_hidden,
        )
        assert result["hidden_align_loss"].item() == 0.0


# =============================================================================
# Alignment Loss Correctness
# =============================================================================

class TestHiddenAlignCorrectness:
    """Tests for alignment loss computation."""

    def test_alignment_loss_nonzero(
        self, base_student_output, base_tensors, teacher_hidden
    ):
        """With weight>0 and teacher_hidden, alignment loss should be >0."""
        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result = loss_fn(
            base_student_output,
            base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None,
            teacher_hidden=teacher_hidden,
        )
        assert result["hidden_align_loss"].item() > 0.0

    def test_alignment_loss_known_answer(self):
        """Verify alignment loss matches hand-computed MSE."""
        B, M, T, H = 2, 3, 4, 8

        # Known student membrane and teacher hidden
        student_mem = torch.ones(B, T, H) * 2.0
        teacher_hidden = torch.ones(B, T, H) * 3.0

        student_output = {
            "rates": torch.rand(B, M) + 0.1,
            "spikes": torch.rand(B, T, H),
            "membrane_potentials": student_mem,
        }

        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=1.0,  # Weight=1 for easy verification
            distill_weight=0.0,       # Disable KL to isolate alignment
            reg_weight=0.0,           # Disable reg
        )

        result = loss_fn(
            student_output,
            torch.rand(B, M),
            torch.rand(B, M) + 0.1,
            behavior=None,
            teacher_hidden=teacher_hidden,
        )

        # MSE between 2.0 and 3.0 = (3-2)^2 = 1.0
        expected_mse = 1.0
        assert torch.isclose(
            result["hidden_align_loss"],
            torch.tensor(expected_mse),
            atol=1e-6,
        ), f"Expected MSE={expected_mse}, got {result['hidden_align_loss'].item()}"

    def test_identical_hidden_zero_loss(self):
        """Alignment loss should be ~0 when student membrane = teacher hidden."""
        B, M, T, H = 2, 3, 4, 8
        shared = torch.randn(B, T, H)

        student_output = {
            "rates": torch.rand(B, M) + 0.1,
            "spikes": torch.rand(B, T, H),
            "membrane_potentials": shared.clone(),
        }

        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=1.0,
            distill_weight=0.0,
            reg_weight=0.0,
        )

        result = loss_fn(
            student_output, torch.rand(B, M),
            torch.rand(B, M) + 0.1,
            behavior=None,
            teacher_hidden=shared.clone(),
        )

        assert torch.isclose(
            result["hidden_align_loss"],
            torch.tensor(0.0),
            atol=1e-6,
        )

    def test_alignment_contributes_to_total_loss(
        self, base_student_output, base_tensors, teacher_hidden
    ):
        """Total loss should increase when alignment weight > 0."""
        # No alignment
        loss_fn_no_align = MultiHeadDistillationLoss(
            hidden_align_weight=0.0,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result_no = loss_fn_no_align(
            base_student_output, base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None, teacher_hidden=teacher_hidden,
        )

        # With alignment
        loss_fn_align = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.5,
            reg_weight=0.001,
        )
        result_with = loss_fn_align(
            base_student_output, base_tensors["ground_truth"],
            base_tensors["teacher_rates"],
            behavior=None, teacher_hidden=teacher_hidden,
        )

        # Total loss should be higher with alignment (assuming non-zero MSE)
        assert result_with["loss"].item() > result_no["loss"].item()


# =============================================================================
# Dimension Handling
# =============================================================================

class TestHiddenAlignDimensions:
    """Tests for handling sequence length mismatches."""

    def test_truncation_when_teacher_shorter(self):
        """Should truncate to min(T_student, T_teacher) when teacher is shorter."""
        B, M, H = 2, 3, 8
        T_student = 10
        T_teacher = 6  # Shorter than student

        student_output = {
            "rates": torch.rand(B, M) + 0.1,
            "spikes": torch.rand(B, T_student, H),
            "membrane_potentials": torch.randn(B, T_student, H),
        }

        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.0,
            reg_weight=0.0,
        )

        # Should not crash despite T mismatch
        result = loss_fn(
            student_output, torch.rand(B, M),
            torch.rand(B, M) + 0.1,
            behavior=None,
            teacher_hidden=torch.randn(B, T_teacher, H),
        )
        assert result["hidden_align_loss"].item() > 0.0

    def test_truncation_when_student_shorter(self):
        """Should handle student T < teacher T."""
        B, M, H = 2, 3, 8
        T_student = 5
        T_teacher = 10

        student_output = {
            "rates": torch.rand(B, M) + 0.1,
            "spikes": torch.rand(B, T_student, H),
            "membrane_potentials": torch.randn(B, T_student, H),
        }

        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=0.1,
            distill_weight=0.0,
            reg_weight=0.0,
        )

        result = loss_fn(
            student_output, torch.rand(B, M),
            torch.rand(B, M) + 0.1,
            behavior=None,
            teacher_hidden=torch.randn(B, T_teacher, H),
        )
        assert result["hidden_align_loss"].item() > 0.0


# =============================================================================
# Gradient Flow
# =============================================================================

class TestHiddenAlignGradients:
    """Tests for gradient flow through alignment loss."""

    def test_gradient_to_membrane_potentials(self):
        """Alignment loss gradient should flow to student membrane potentials."""
        B, M, T, H = 2, 3, 4, 8
        membrane = torch.randn(B, T, H, requires_grad=True)

        student_output = {
            "rates": torch.rand(B, M, requires_grad=True) + 0.1,
            "spikes": torch.rand(B, T, H),
            "membrane_potentials": membrane,
        }

        loss_fn = MultiHeadDistillationLoss(
            hidden_align_weight=1.0,
            distill_weight=0.0,
            reg_weight=0.0,
        )

        result = loss_fn(
            student_output, torch.rand(B, M),
            torch.rand(B, M) + 0.1,
            behavior=None,
            teacher_hidden=torch.randn(B, T, H),
        )

        result["loss"].backward()

        assert membrane.grad is not None, (
            "Gradient should flow to membrane_potentials"
        )
        assert torch.any(membrane.grad != 0), (
            "Membrane potential gradients should be non-zero"
        )
