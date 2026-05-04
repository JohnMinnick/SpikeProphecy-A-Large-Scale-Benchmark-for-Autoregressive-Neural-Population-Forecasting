"""
Tests for session-specific head shape handling in Trainer._compute_loss.

Verifies that the Trainer handles variable-size model outputs from
session-specific heads correctly. This test catches the shape mismatch
bug where y_hat has shape (batch, N_i) but y/mask are padded to (batch, M_max).
"""

import pytest
import torch

from src.train.trainer import Trainer


class MockSessionModel(torch.nn.Module):
    """
    Mock model that outputs variable-width predictions per session.

    Simulates session-specific heads by outputting different widths
    depending on the session_id passed to forward().
    """

    def __init__(self, session_dims):
        super().__init__()
        self.session_dims = session_dims
        # Minimal linear layer to make optimizer happy
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x, covariates=None, session_id=None):
        """Return a tensor of width N_i for the given session."""
        batch = x.shape[0]
        if session_id and session_id in self.session_dims:
            out_size = self.session_dims[session_id]
        else:
            out_size = 10  # default
        # Return softplus to ensure positive rates
        return torch.nn.functional.softplus(torch.randn(batch, out_size))

    def get_aux_output(self):
        return None


@pytest.fixture
def device():
    """Use CPU for all tests."""
    return torch.device("cpu")


@pytest.fixture
def session_dims():
    """Session dimensions: each session has a different neuron count."""
    return {"session_0": 500, "session_1": 677, "session_2": 1240}


@pytest.fixture
def m_max(session_dims):
    """Maximum neuron count across all sessions."""
    return max(session_dims.values())


@pytest.fixture
def trainer(session_dims, device):
    """Create a Trainer with a mock session-specific model."""
    model = MockSessionModel(session_dims)
    # Minimal config for Trainer init
    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 1e-3,
            "batch_size": 16,
        },
        "loss": {"type": "poisson_nll"},
    }
    # Create trainer (train/val loaders unused in these tests)
    return Trainer(
        model=model,
        train_loader=[],
        val_loader=[],
        config=config,
        device=device,
    )


class TestSessionHeadShapeAlignment:
    """Test that _compute_loss handles variable output sizes."""

    def test_matching_shapes_pass_through(self, trainer):
        """When y_hat and y have the same width, no slicing occurs."""
        y_hat = torch.nn.functional.softplus(torch.randn(8, 100))
        y = torch.randint(0, 5, (8, 100)).float()
        mask = torch.ones(8, 100)

        # Should not raise
        loss = trainer._compute_loss(y_hat, y, mask=mask)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0

    def test_narrow_output_wide_target(self, trainer, m_max):
        """
        Session-specific head outputs (batch, N_i) while targets
        are padded to (batch, M_max). Loss should slice targets.
        """
        n_i = 677  # smaller than M_max=1240
        y_hat = torch.nn.functional.softplus(torch.randn(8, n_i))
        # Targets padded to M_max
        y = torch.zeros(8, m_max)
        y[:, :n_i] = torch.randint(0, 5, (8, n_i)).float()
        # Mask: 1 for real channels, 0 for padding
        mask = torch.zeros(8, m_max)
        mask[:, :n_i] = 1.0

        # Should not raise (this was the crash before the fix)
        loss = trainer._compute_loss(y_hat, y, mask=mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_narrow_output_no_mask(self, trainer, m_max):
        """Without mask, loss should still handle shape mismatch."""
        n_i = 500
        y_hat = torch.nn.functional.softplus(torch.randn(8, n_i))
        y = torch.zeros(8, m_max)
        y[:, :n_i] = torch.randint(0, 5, (8, n_i)).float()

        # No mask — should slice y to match y_hat
        loss = trainer._compute_loss(y_hat, y, mask=None)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_multiple_sessions_different_widths(self, trainer, session_dims, m_max):
        """Simulate cycling through sessions with different neuron counts."""
        losses = []
        for session_id, n_i in session_dims.items():
            y_hat = torch.nn.functional.softplus(torch.randn(4, n_i))
            y = torch.zeros(4, m_max)
            y[:, :n_i] = torch.randint(0, 3, (4, n_i)).float()
            mask = torch.zeros(4, m_max)
            mask[:, :n_i] = 1.0

            loss = trainer._compute_loss(y_hat, y, mask=mask)
            assert loss.shape == ()
            losses.append(loss.item())

        # All losses should be valid positive numbers
        assert all(l > 0 for l in losses)

    def test_mask_sliced_correctly(self, trainer, m_max):
        """
        Verify that mask channels beyond N_i (which are all zeros)
        don't contribute to the loss after slicing.
        """
        n_i = 400
        y_hat = torch.nn.functional.softplus(torch.randn(8, n_i))

        # Targets: real data in first N_i channels, zeros in padding
        y = torch.zeros(8, m_max)
        y[:, :n_i] = torch.randint(0, 5, (8, n_i)).float()

        # Mask: 1 for real, 0 for padding
        mask = torch.zeros(8, m_max)
        mask[:, :n_i] = 1.0

        loss = trainer._compute_loss(y_hat, y, mask=mask)

        # Compare with manually sliced version
        y_sliced = y[:, :n_i]
        mask_sliced = mask[:, :n_i]
        loss_manual = trainer._compute_loss(y_hat, y_sliced, mask=mask_sliced)

        # Should be identical
        assert abs(loss.item() - loss_manual.item()) < 1e-6
