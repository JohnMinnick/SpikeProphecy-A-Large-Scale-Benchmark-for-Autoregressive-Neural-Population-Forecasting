"""
Tests for per-session SNN distillation (neuromorphic twins).

Validates:
- Unpadded data loading shapes (N_i, not M_max)
- Teacher padding + session_id routing
- Student output matches session dims
- End-to-end mini training loop (loss decreases)
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.models.student import StudentSNN


# =============================================================================
# Mock session-heads teacher
# =============================================================================


class MockSessionTeacher(nn.Module):
    """
    Minimal session-heads teacher for testing.

    Accepts M_max-padded input, uses session_id to select output head,
    returns softplus'd predictions with shape (batch, N_i).
    """

    def __init__(self, m_max: int, session_dims: dict):
        """
        Args:
            m_max: Maximum channel count (input dimension).
            session_dims: Dict mapping session_id -> N_i.
        """
        super().__init__()
        self.m_max = m_max
        self.session_dims = session_dims

        # Shared backbone: Linear(M_max, hidden)
        self.backbone = nn.Linear(m_max, 64)

        # Per-session output heads
        self.heads = nn.ModuleDict({
            sid: nn.Linear(64, n_i)
            for sid, n_i in session_dims.items()
        })
        self.softplus = nn.Softplus()

    def forward(self, x, session_id=None, **kwargs):
        """Forward pass: shared backbone + session-specific head."""
        assert session_id is not None, "session_id required"
        # x: (batch, T, M_max) -> take last timestep
        h = self.backbone(x[:, -1, :])
        return self.softplus(self.heads[session_id](h))


# =============================================================================
# Tests
# =============================================================================


class TestSessionDistillLoader:
    """Tests for the SessionDistillLoader wrapper."""

    @pytest.fixture
    def setup(self):
        """Create mock teacher, student, and sample data."""
        m_max = 100
        n_neurons = 30  # Session-specific neuron count
        session_id = "session_003"
        session_dims = {
            "session_000": 20,
            "session_003": n_neurons,
            "session_010": 50,
        }

        teacher = MockSessionTeacher(m_max, session_dims)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        return {
            "m_max": m_max,
            "n_neurons": n_neurons,
            "session_id": session_id,
            "session_dims": session_dims,
            "teacher": teacher,
        }

    def test_student_input_is_unpadded(self, setup):
        """Student receives (batch, T, N_i), NOT (batch, T, M_max)."""
        n_i = setup["n_neurons"]
        T = 10
        batch_size = 8

        # Simulate unpadded session data
        x = torch.randn(batch_size, T, n_i)
        student = StudentSNN(input_size=n_i, output_size=n_i, hidden_size=32)

        rates, spikes = student(x)
        assert rates.shape == (batch_size, n_i), (
            f"Student output shape {rates.shape} should use N_i={n_i}, not M_max"
        )

    def test_teacher_receives_padded_input(self, setup):
        """Teacher receives M_max-padded input and returns N_i-sized output."""
        teacher = setup["teacher"]
        m_max = setup["m_max"]
        n_i = setup["n_neurons"]
        session_id = setup["session_id"]
        T = 10
        batch_size = 4

        # Create M_max-padded input (simulate what SessionDistillLoader does)
        x_padded = torch.zeros(batch_size, T, m_max)
        x_padded[:, :, :n_i] = torch.randn(batch_size, T, n_i)

        with torch.no_grad():
            teacher_out = teacher(x_padded, session_id=session_id)

        assert teacher_out.shape == (batch_size, n_i), (
            f"Teacher output shape {teacher_out.shape} should be "
            f"(batch, N_i={n_i}), not (batch, M_max={m_max})"
        )

    def test_teacher_output_non_negative(self, setup):
        """Teacher rates should be non-negative (softplus)."""
        teacher = setup["teacher"]
        m_max = setup["m_max"]
        session_id = setup["session_id"]

        x = torch.randn(4, 10, m_max)
        with torch.no_grad():
            rates = teacher(x, session_id=session_id)

        assert (rates >= 0).all(), "Teacher rates must be non-negative"

    def test_student_output_matches_session_dims(self, setup):
        """Student output size should be exactly N_i for the session."""
        n_i = setup["n_neurons"]
        student = StudentSNN(input_size=n_i, output_size=n_i, hidden_size=32)

        x = torch.randn(4, 10, n_i)
        rates, spikes = student(x)

        assert rates.shape[1] == n_i
        assert spikes.shape[2] == 32  # hidden_size

    def test_distill_shapes_match(self, setup):
        """Student rates, y, and teacher_rates should all be (batch, N_i)."""
        teacher = setup["teacher"]
        m_max = setup["m_max"]
        n_i = setup["n_neurons"]
        session_id = setup["session_id"]
        T = 10
        batch_size = 4

        # Student sized to N_i
        student = StudentSNN(input_size=n_i, output_size=n_i, hidden_size=32)

        # Unpadded input (what student sees)
        x_raw = torch.randn(batch_size, T, n_i)
        # Padded input (what teacher sees)
        x_padded = torch.zeros(batch_size, T, m_max)
        x_padded[:, :, :n_i] = x_raw
        # Ground truth (N_i)
        y = torch.randn(batch_size, n_i).abs()

        # Run both models
        student_rates, student_spikes = student(x_raw)
        with torch.no_grad():
            teacher_rates = teacher(x_padded, session_id=session_id)

        # All should be (batch, N_i)
        assert student_rates.shape == (batch_size, n_i)
        assert teacher_rates.shape == (batch_size, n_i)
        assert y.shape == (batch_size, n_i)


class TestMiniTrainingLoop:
    """End-to-end test: 1-epoch distillation on synthetic data."""

    def test_loss_decreases_in_mini_loop(self):
        """Train for a few steps and verify loss decreases."""
        # Small synthetic setup
        n_i = 15
        m_max = 50
        T = 5
        n_samples = 64
        batch_size = 16
        session_id = "session_003"

        # Create mock teacher
        session_dims = {"session_003": n_i}
        teacher = MockSessionTeacher(m_max, session_dims)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Create student
        student = StudentSNN(
            input_size=n_i, output_size=n_i,
            hidden_size=32, num_layers=1,
        )

        # Create synthetic data
        x_data = torch.randn(n_samples, T, n_i).abs()
        y_data = torch.randn(n_samples, n_i).abs()

        # Simple distillation loss
        from src.distill.loss import DistillationLoss
        criterion = DistillationLoss(
            distill_weight=0.5, reg_weight=0.001, reg_type="l1",
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=0.01)

        # Collect losses over mini-batches
        losses = []
        student.train()

        for epoch in range(3):
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples - batch_size + 1, batch_size):
                x_batch = x_data[start:start + batch_size]
                y_batch = y_data[start:start + batch_size]

                # Teacher inference with padding
                x_padded = torch.zeros(batch_size, T, m_max)
                x_padded[:, :, :n_i] = x_batch
                with torch.no_grad():
                    teacher_rates = teacher(x_padded, session_id=session_id)

                # Student forward
                student_rates, spikes = student(x_batch)

                # Compute loss
                loss_dict = criterion(student_rates, spikes, y_batch, teacher_rates)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        # Loss should decrease over 3 epochs
        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_different_sessions_produce_different_outputs(self):
        """Different session heads should produce different teacher outputs."""
        m_max = 50
        session_dims = {
            "session_000": 10,
            "session_003": 30,
        }

        teacher = MockSessionTeacher(m_max, session_dims)
        teacher.eval()

        x = torch.randn(4, 5, m_max)

        with torch.no_grad():
            out_s0 = teacher(x, session_id="session_000")
            out_s3 = teacher(x, session_id="session_003")

        # Different sessions have different output sizes
        assert out_s0.shape[1] == 10
        assert out_s3.shape[1] == 30

    def test_student_params_scale_with_neurons(self):
        """Smaller sessions should produce smaller students."""
        small_student = StudentSNN(input_size=50, output_size=50, hidden_size=32)
        large_student = StudentSNN(input_size=500, output_size=500, hidden_size=32)

        small_params = sum(p.numel() for p in small_student.parameters())
        large_params = sum(p.numel() for p in large_student.parameters())

        assert small_params < large_params, (
            f"Small student ({small_params}) should have fewer params "
            f"than large student ({large_params})"
        )
