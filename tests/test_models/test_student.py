"""
Tests for the StudentSNN model.

Covers v1 backward compatibility and v2 architecture improvements:
    - Learnable β (Change 1)
    - Multi-layer spiking (Change 2)
    - RSynaptic neurons (Change 3)
"""

import pytest
import torch

from src.models.student import StudentSNN


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def input_data():
    """Create a batch of input data (Batch, T, M)."""
    batch_size = 4
    T = 10
    M = 5
    return torch.randn(batch_size, T, M).abs()  # Non-negative rates/counts


@pytest.fixture
def model():
    """Create a student model (v1 defaults)."""
    return StudentSNN(input_size=5, hidden_size=16, output_size=5)


# =============================================================================
# Original v1 tests (backward compatibility)
# =============================================================================

class TestStudentSNN:
    """Tests for StudentSNN v1 behavior."""

    def test_output_shapes(self, model, input_data):
        """Forward pass should return rates (B, M) and spikes (B, T, H)."""
        rates, spikes = model(input_data)

        B, T, M = input_data.shape
        H = model.hidden_size

        # Rates: (B, Output)
        assert rates.shape == (B, M)
        # Spikes: (B, T, H)
        assert spikes.shape == (B, T, H)

    def test_rates_are_non_negative(self, model, input_data):
        """Output rates should be non-negative (softplus)."""
        rates, _ = model(input_data)
        assert (rates >= 0).all()

    def test_spikes_are_binary(self, model, input_data):
        """Spikes should be 0 or 1 (mostly)."""
        # Note: with surrogate gradients, they are 0/1 in forward pass.
        _, spikes = model(input_data)
        unique_vals = torch.unique(spikes)
        # Check standard values
        for v in unique_vals:
            assert v.item() in [0.0, 1.0]

    def test_gradient_flow(self, model, input_data):
        """Gradients should flow through the model (surrogate works)."""
        rates, spikes = model(input_data)
        loss = rates.sum() + spikes.sum()
        loss.backward()

        # Check gradients exist on projections
        assert model.input_proj.weight.grad is not None
        assert model.output_proj.weight.grad is not None
        # Check recurrent weights have gradients (any layer)
        has_grad = any(
            param.grad is not None
            for layer in model.spiking_layers
            for param in layer.parameters()
        )
        assert has_grad, "Spiking layer parameters should have gradients"

    def test_from_config(self):
        """Construction from config dict should set model parameters correctly."""
        config = {
            "model": {
                "hidden_size": 32,
                "beta": 0.8,
                "threshold": 0.5,
            }
        }
        model = StudentSNN.from_config(config, input_size=10)
        assert model.hidden_size == 32
        assert model.beta == 0.8
        assert model.input_size == 10
        assert model.output_size == 10  # defaults to input_size

        # Verify forward pass produces correct shapes
        x = torch.randn(2, 5, 10).abs()
        rates, spikes = model(x)
        assert rates.shape == (2, 10)
        assert spikes.shape == (2, 5, 32)


# =============================================================================
# Change 1: Learnable β
# =============================================================================

class TestLearnableBeta:
    """Tests for learnable membrane decay constants."""

    def test_learnable_beta_is_parameter(self):
        """When learn_beta=True, beta should be an nn.Parameter."""
        model = StudentSNN(input_size=5, hidden_size=16, learn_beta=True)

        # Check that beta is a learnable Parameter in each spiking layer
        for i, layer in enumerate(model.spiking_layers):
            assert isinstance(layer.beta, torch.nn.Parameter), (
                f"Layer {i}: beta should be nn.Parameter when learn_beta=True"
            )

    def test_learnable_beta_default_off(self):
        """Default learn_beta=False should NOT make beta a learnable parameter."""
        model = StudentSNN(input_size=5, hidden_size=16, learn_beta=False)

        # Beta should NOT be a Parameter
        for layer in model.spiking_layers:
            assert not isinstance(layer.beta, torch.nn.Parameter), (
                "beta should not be nn.Parameter when learn_beta=False"
            )

    def test_learnable_beta_forward_pass(self, input_data):
        """Model with learnable beta should produce valid output."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, learn_beta=True
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert (rates >= 0).all()

    def test_learnable_beta_gradient_flow(self, input_data):
        """Gradients should flow to learnable beta parameters."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, learn_beta=True
        )
        rates, spikes = model(input_data)
        loss = rates.sum()
        loss.backward()

        for layer in model.spiking_layers:
            assert layer.beta.grad is not None, "Learnable beta should get gradients"


# =============================================================================
# Change 2: Second Spiking Layer (Multi-layer)
# =============================================================================

class TestMultiLayer:
    """Tests for stacked spiking layers."""

    def test_multi_layer_output_shape(self, input_data):
        """2-layer model should produce valid output shapes."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, num_layers=2
        )
        rates, spikes = model(input_data)
        B, T, M = input_data.shape
        H = model.hidden_size

        assert rates.shape == (B, M)
        assert spikes.shape == (B, T, H)

    def test_single_layer_backward_compat(self, input_data):
        """num_layers=1 should behave identically to v1."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, num_layers=1
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert spikes.shape == (4, 10, 16)

    def test_multi_layer_gradient_flow(self, input_data):
        """All layers should receive gradients."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, num_layers=2
        )
        rates, spikes = model(input_data)
        loss = rates.sum() + spikes.sum()
        loss.backward()

        # Check each spiking layer has non-zero gradients
        for i, layer in enumerate(model.spiking_layers):
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in layer.parameters()
            )
            assert has_grad, f"Spiking layer {i} should have non-zero gradients"

    def test_multi_layer_param_count(self):
        """2-layer model should have more parameters than 1-layer."""
        model_1 = StudentSNN(input_size=5, hidden_size=16, num_layers=1)
        model_2 = StudentSNN(input_size=5, hidden_size=16, num_layers=2)

        params_1 = sum(p.numel() for p in model_1.parameters())
        params_2 = sum(p.numel() for p in model_2.parameters())
        assert params_2 > params_1, "2-layer model should have more parameters"

    def test_three_layers(self, input_data):
        """3-layer model should work too."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5, num_layers=3
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert spikes.shape == (4, 10, 16)


# =============================================================================
# Change 3: RSynaptic Neuron Type
# =============================================================================

class TestRSynaptic:
    """Tests for configurable neuron type."""

    def test_rsynaptic_forward_pass(self, input_data):
        """RSynaptic model should produce valid output shapes."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="rsynaptic",
        )
        rates, spikes = model(input_data)
        B, T, M = input_data.shape
        assert rates.shape == (B, M)
        assert spikes.shape == (B, T, 16)

    def test_rleaky_backward_compat(self, input_data):
        """Default neuron_type='rleaky' should work as before."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="rleaky",
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert (rates >= 0).all()

    def test_rsynaptic_learnable_alpha(self):
        """RSynaptic with learn_beta=True should also learn alpha."""
        model = StudentSNN(
            input_size=5, hidden_size=16,
            neuron_type="rsynaptic", learn_beta=True,
        )
        for layer in model.spiking_layers:
            assert isinstance(layer.alpha, torch.nn.Parameter), (
                "alpha should be nn.Parameter when learn_beta=True and neuron_type='rsynaptic'"
            )

    def test_rsynaptic_spikes_binary(self, input_data):
        """RSynaptic spikes should also be 0 or 1."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="rsynaptic",
        )
        _, spikes = model(input_data)
        unique_vals = torch.unique(spikes)
        for v in unique_vals:
            assert v.item() in [0.0, 1.0]


# =============================================================================
# Combined Features
# =============================================================================

class TestCombinedFeatures:
    """Tests for all features enabled simultaneously."""

    def test_all_features_enabled(self, input_data):
        """Model with all 3 improvements should produce valid output."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            learn_beta=True, num_layers=2, neuron_type="rsynaptic",
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert spikes.shape == (4, 10, 16)
        assert (rates >= 0).all()

    def test_from_config_new_fields(self):
        """from_config should handle all new architecture keys."""
        config = {
            "model": {
                "hidden_size": 32,
                "beta": 0.85,
                "threshold": 0.8,
                "learn_beta": True,
                "num_layers": 2,
                "neuron_type": "rsynaptic",
                "alpha": 0.9,
            }
        }
        model = StudentSNN.from_config(config, input_size=10)
        assert model.hidden_size == 32
        assert model.learn_beta is True
        assert model.num_spiking_layers == 2
        assert model.neuron_type == "rsynaptic"
        assert model.alpha == 0.9

        # Verify forward pass
        x = torch.randn(2, 5, 10).abs()
        rates, spikes = model(x)
        assert rates.shape == (2, 10)
        assert spikes.shape == (2, 5, 32)
