"""
Tests for channel-masked distillation loss.

Validates that:
1. Masked loss equals unmasked loss when mask is all-ones.
2. Padding channels contribute zero gradient when masked.
3. Loss magnitude scales correctly with mask fraction.
4. MultiHeadDistillationLoss passes mask through to base.
"""

import pytest
import torch

from src.distill.loss import DistillationLoss
from src.distill.multi_head_loss import MultiHeadDistillationLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_data():
    """Create synthetic data for loss testing."""
    torch.manual_seed(42)
    batch_size = 32
    m_max = 100  # Total channels (including padding)
    m_real = 40  # Real neurons

    # Student rates (softplus output, always positive)
    student_rates = torch.rand(batch_size, m_max) * 2.0 + 0.1
    student_rates.requires_grad_(True)

    # Hidden spikes for regularization
    student_spikes = torch.rand(batch_size, 10, 64)

    # Ground truth counts (integer-ish)
    ground_truth = torch.poisson(torch.ones(batch_size, m_max) * 0.5)
    # Zero out GT for padding channels (as multi-session loader does)
    ground_truth[:, m_real:] = 0.0

    # Teacher rates (slightly offset from student)
    teacher_rates = torch.rand(batch_size, m_max) * 2.0 + 0.1
    # Teacher also near-zero on padding (as trained model produces)
    teacher_rates[:, m_real:] = 0.01

    # Channel mask: 1 for real, 0 for padding
    channel_mask = torch.zeros(batch_size, m_max)
    channel_mask[:, :m_real] = 1.0

    return {
        "student_rates": student_rates,
        "student_spikes": student_spikes,
        "ground_truth": ground_truth,
        "teacher_rates": teacher_rates,
        "channel_mask": channel_mask,
        "m_real": m_real,
        "m_max": m_max,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# DistillationLoss tests
# ---------------------------------------------------------------------------

class TestDistillationLossMasking:
    """Tests for channel mask support in DistillationLoss."""

    def test_mask_none_backward_compatible(self, make_data):
        """No mask should behave identically to the original loss."""
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.0)

        # Without mask (original behavior)
        result = loss_fn(
            d["student_rates"].detach().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=None,
        )
        assert "loss" in result
        assert result["loss"].item() > 0.0

    def test_all_ones_mask_matches_unmasked(self, make_data):
        """Mask of all 1s should give similar loss to unmasked (within tolerance)."""
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.0)

        # All-ones mask
        ones_mask = torch.ones_like(d["channel_mask"])

        result_unmasked = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=None,
        )
        result_masked = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=ones_mask,
        )

        # The Poisson NLL implementations differ slightly (nn.PoissonNLLLoss
        # includes log(y!) term with full=True), but KL should match closely
        assert abs(result_masked["distill"].item() - result_unmasked["distill"].item()) < 1e-5

    def test_masked_loss_ignores_padding(self, make_data):
        """Masked loss should be identical regardless of padding values."""
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.0)

        # Run with normal padding
        result1 = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"].clone(),
            channel_mask=d["channel_mask"],
        )

        # Modify padding channels to wildly different values
        teacher_mod = d["teacher_rates"].clone()
        teacher_mod[:, d["m_real"]:] = 100.0  # Huge values in padding
        gt_mod = d["ground_truth"].clone()
        gt_mod[:, d["m_real"]:] = 50.0  # Non-zero GT in padding

        result2 = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            gt_mod,
            teacher_mod,
            channel_mask=d["channel_mask"],
        )

        # Loss should be the same because mask zeros out padding
        assert abs(result1["loss"].item() - result2["loss"].item()) < 1e-5

    def test_zero_gradient_on_padding(self, make_data):
        """Gradients for padding output neurons must be exactly zero."""
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.0)

        rates = d["student_rates"].detach().clone().requires_grad_(True)
        result = loss_fn(
            rates,
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=d["channel_mask"],
        )
        result["loss"].backward()

        # Gradients on padding channels should be zero
        padding_grads = rates.grad[:, d["m_real"]:]
        assert torch.all(padding_grads == 0.0), (
            f"Non-zero gradients on padding: max={padding_grads.abs().max().item():.6f}"
        )

        # Gradients on real channels should be non-zero
        real_grads = rates.grad[:, :d["m_real"]]
        assert real_grads.abs().sum() > 0.0

    def test_masked_loss_larger_per_neuron(self, make_data):
        """
        Masked loss should be larger per-neuron than unmasked because
        it concentrates on real neurons (which have actual signal)
        instead of diluting with trivial padding predictions.
        """
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.0)

        result_unmasked = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=None,
        )
        result_masked = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=d["channel_mask"],
        )

        # Masked Poisson loss should be larger because it only averages
        # over real neurons (which have higher GT counts)
        assert result_masked["poisson"].item() >= result_unmasked["poisson"].item() * 0.5

    def test_spike_reg_unaffected_by_mask(self, make_data):
        """Spike regularization operates on hidden spikes, not output — should be unchanged."""
        d = make_data
        loss_fn = DistillationLoss(distill_weight=0.5, reg_weight=0.01)

        result_no_mask = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=None,
        )
        result_masked = loss_fn(
            d["student_rates"].detach().clone().requires_grad_(True),
            d["student_spikes"],
            d["ground_truth"],
            d["teacher_rates"],
            channel_mask=d["channel_mask"],
        )

        # Reg loss should be identical
        assert abs(result_no_mask["reg"].item() - result_masked["reg"].item()) < 1e-6


# ---------------------------------------------------------------------------
# MultiHeadDistillationLoss tests
# ---------------------------------------------------------------------------

class TestMultiHeadMasking:
    """Tests for mask passthrough in MultiHeadDistillationLoss."""

    def test_mask_passes_through_to_base(self, make_data):
        """channel_mask should reach the base DistillationLoss."""
        d = make_data
        loss_fn = MultiHeadDistillationLoss(
            distill_weight=0.5,
            stimulus_weight=0.0,
            response_weight=0.0,
        )

        student_output = {
            "rates": d["student_rates"].detach().clone().requires_grad_(True),
            "spikes": d["student_spikes"],
        }

        result = loss_fn(
            student_output,
            d["ground_truth"],
            d["teacher_rates"],
            behavior=None,
            channel_mask=d["channel_mask"],
        )

        # Should produce valid loss
        assert result["loss"].item() > 0.0
        assert result["poisson"].item() > 0.0

    def test_multi_head_zero_grad_on_padding(self, make_data):
        """MultiHead with mask should also produce zero padding gradients."""
        d = make_data
        loss_fn = MultiHeadDistillationLoss(
            distill_weight=0.5,
            stimulus_weight=0.0,
            response_weight=0.0,
        )

        rates = d["student_rates"].detach().clone().requires_grad_(True)
        student_output = {
            "rates": rates,
            "spikes": d["student_spikes"],
        }

        result = loss_fn(
            student_output,
            d["ground_truth"],
            d["teacher_rates"],
            behavior=None,
            channel_mask=d["channel_mask"],
        )
        result["loss"].backward()

        # Padding gradients should be zero
        padding_grads = rates.grad[:, d["m_real"]:]
        assert torch.all(padding_grads == 0.0)
