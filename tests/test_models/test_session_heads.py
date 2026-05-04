"""
Tests for session-specific read-in/read-out heads (Phase 1 Fix 1).

Architecture: shared input projection (data padded to M_max) +
per-session output projection (Linear(hidden, N_i)).

Validates that:
- Models accept session_dims and create per-session OUTPUT projections
- Input projection is always shared (data is always padded to M_max)
- Forward pass produces correct output shapes per session
- Gradients flow through both shared input and session-specific output
- Backward compat: session_dims=None matches original shared-head behavior
- create_teacher_model factory passes session_dims correctly
"""

import pytest
import torch
from typing import Dict

from src.models.teacher import TeacherLSTM, create_teacher_model
from src.models.lru import TeacherLRU


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def session_dims() -> Dict[str, int]:
    """Example session dimensions: 3 sessions with varying neuron counts."""
    return {
        "session_000": 100,
        "session_001": 250,
        "session_002": 50,
    }


@pytest.fixture
def m_max():
    """M_max: padded input dimension (max across sessions)."""
    return 250


# =============================================================================
# TeacherLSTM session-specific head tests
# =============================================================================

class TestLSTMSessionHeads:
    """Tests for session-specific heads in TeacherLSTM."""

    def test_creates_per_session_output_projections(self, session_dims, m_max):
        """Model should have per-session OUTPUT projections, shared INPUT."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )

        # Session-specific OUTPUT heads should exist
        assert model.session_output_projs is not None
        assert len(model.session_output_projs) == 3
        # Session-specific INPUT projs should NOT exist (input is shared)
        assert model.session_input_projs is None
        # Shared input proj always exists
        assert model.input_proj is not None
        # Shared output proj should be None (per-session replaces it)
        assert model.output_proj is None

    def test_correct_output_shapes(self, session_dims, m_max):
        """Each session should produce output matching its neuron count."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )
        model.eval()

        batch_size = 4
        T = 10

        for sid, n_neurons in session_dims.items():
            # Input is always M_max (padded), output is N_i (per-session)
            x = torch.randn(batch_size, T, m_max)
            out = model(x, session_id=sid)
            assert out.shape == (batch_size, n_neurons), (
                f"Session {sid}: expected ({batch_size}, {n_neurons}), "
                f"got {out.shape}"
            )

    def test_gradients_flow(self, session_dims, m_max):
        """Gradients should flow through shared input and session output."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )

        sid = "session_001"
        n_neurons = session_dims[sid]
        x = torch.randn(2, 5, m_max)
        y = torch.rand(2, n_neurons).abs() * 2.0

        out = model(x, session_id=sid)
        loss = (out - y).pow(2).mean()
        loss.backward()

        # Shared input proj should have gradients
        assert model.input_proj.weight.grad is not None, "Shared input proj has no gradients"
        assert model.input_proj.weight.grad.abs().sum() > 0, "Input proj gradients are zero"

        # Session-specific output proj should have gradients
        out_proj = model.session_output_projs[sid]
        assert out_proj.weight.grad is not None, "Output proj has no gradients"

    def test_backward_compat_shared_mode(self, m_max):
        """session_dims=None should behave identically to original model."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=None,
        )

        # Shared heads should exist
        assert model.input_proj is not None
        assert model.output_proj is not None
        assert model.session_input_projs is None
        assert model.session_output_projs is None

        # Forward should work without session_id
        x = torch.randn(2, 5, m_max)
        out = model(x)
        assert out.shape == (2, m_max)

    def test_session_id_required_when_session_dims_set(self, session_dims, m_max):
        """Forward without session_id should raise when session_dims is set."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )

        x = torch.randn(2, 5, m_max)
        with pytest.raises(AssertionError, match="session_id is required"):
            model(x)

    def test_all_positive_outputs(self, session_dims, m_max):
        """All outputs should be > 0 (Softplus enforced)."""
        model = TeacherLSTM(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )
        model.eval()

        for sid, n_neurons in session_dims.items():
            x = torch.randn(4, 10, m_max)
            out = model(x, session_id=sid)
            assert (out > 0).all(), f"Session {sid}: found non-positive rates"


# =============================================================================
# TeacherLRU session-specific head tests
# =============================================================================

class TestLRUSessionHeads:
    """Tests for session-specific heads in TeacherLRU."""

    def test_creates_per_session_output_projections(self, session_dims, m_max):
        """Model should have per-session OUTPUT projections, shared INPUT."""
        model = TeacherLRU(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )

        assert model.session_output_projs is not None
        assert len(model.session_output_projs) == 3
        assert model.session_input_projs is None
        assert model.input_proj is not None  # Always shared
        assert model.output_proj is None

    def test_correct_output_shapes(self, session_dims, m_max):
        """Each session should produce output matching its neuron count."""
        model = TeacherLRU(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )
        model.eval()

        for sid, n_neurons in session_dims.items():
            # Input always M_max, output N_i
            x = torch.randn(4, 10, m_max)
            out = model(x, session_id=sid)
            assert out.shape == (4, n_neurons), (
                f"Session {sid}: expected (4, {n_neurons}), got {out.shape}"
            )

    def test_gradients_flow(self, session_dims, m_max):
        """Gradients should flow through shared input and session output."""
        model = TeacherLRU(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=session_dims,
        )

        sid = "session_002"
        n_neurons = session_dims[sid]
        x = torch.randn(2, 5, m_max)
        y = torch.rand(2, n_neurons).abs() * 2.0

        out = model(x, session_id=sid)
        loss = (out - y).pow(2).mean()
        loss.backward()

        # Shared input proj should have gradients
        assert model.input_proj.weight.grad is not None, "Shared input proj has no gradients"

        # Session-specific output proj should have gradients
        out_proj = model.session_output_projs[sid]
        assert out_proj.weight.grad is not None, "Output proj has no gradients"

    def test_backward_compat_shared_mode(self, m_max):
        """session_dims=None should behave identically to original model."""
        model = TeacherLRU(
            input_size=m_max,
            hidden_size=32,
            num_layers=1,
            session_dims=None,
        )

        assert model.input_proj is not None
        assert model.output_proj is not None
        assert model.session_input_projs is None

        x = torch.randn(2, 5, m_max)
        out = model(x)
        assert out.shape == (2, m_max)


# =============================================================================
# Factory function tests
# =============================================================================

class TestCreateTeacherModel:
    """Tests for create_teacher_model with session_dims."""

    def test_lstm_with_session_dims(self, session_dims, m_max):
        """Factory should create LSTM with session-specific output heads."""
        config = {
            "model": {"architecture": "lstm", "hidden_size": 32, "num_layers": 1},
        }

        model = create_teacher_model(config, input_size=m_max, session_dims=session_dims)
        assert isinstance(model, TeacherLSTM)
        assert model.session_output_projs is not None
        assert len(model.session_output_projs) == 3
        assert model.input_proj is not None  # Shared

    def test_lru_with_session_dims(self, session_dims, m_max):
        """Factory should create LRU with session-specific output heads."""
        config = {
            "model": {"architecture": "lru", "hidden_size": 32, "num_layers": 1},
        }

        model = create_teacher_model(config, input_size=m_max, session_dims=session_dims)
        assert isinstance(model, TeacherLRU)
        assert model.session_output_projs is not None
        assert len(model.session_output_projs) == 3
        assert model.input_proj is not None  # Shared

    def test_factory_without_session_dims(self):
        """Factory without session_dims should create shared-head model."""
        config = {
            "model": {"architecture": "lstm", "hidden_size": 32, "num_layers": 1},
        }

        model = create_teacher_model(config, input_size=100)
        assert isinstance(model, TeacherLSTM)
        assert model.input_proj is not None
        assert model.session_input_projs is None
