"""
Tests for TI-LIF + SGC features in StudentSNN.

Covers:
    - SGC bypass module creation and parameter count
    - SGC blending during training (lambda > 0)
    - SGC disabled at inference (lambda = 0 / eval mode)
    - Membrane potentials exposure in multi-head output dict
    - Gradient flow through SGC smooth path
    - from_config with sgc_enabled
    - Backward compatibility: sgc_enabled=False matches original behavior
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
    return torch.randn(batch_size, T, M).abs()


@pytest.fixture
def tilif_sgc_model():
    """Create a TI-LIF student with SGC enabled and auxiliary heads."""
    return StudentSNN(
        input_size=5,
        hidden_size=16,
        output_size=5,
        neuron_type="ti_lif",
        num_layers=2,
        sgc_enabled=True,
        auxiliary_heads=["stimulus", "response"],
    )


@pytest.fixture
def tilif_no_sgc_model():
    """Create a TI-LIF student WITHOUT SGC (baseline)."""
    return StudentSNN(
        input_size=5,
        hidden_size=16,
        output_size=5,
        neuron_type="ti_lif",
        num_layers=2,
        sgc_enabled=False,
        auxiliary_heads=["stimulus", "response"],
    )


# =============================================================================
# SGC Module Creation
# =============================================================================

class TestSGCModules:
    """Tests for SGC bypass module creation."""

    def test_sgc_modules_created_when_enabled(self, tilif_sgc_model):
        """SGC smooth bypass modules should be created for each layer."""
        assert hasattr(tilif_sgc_model, 'sgc_smooth')
        assert len(tilif_sgc_model.sgc_smooth) == tilif_sgc_model.num_spiking_layers

    def test_sgc_modules_not_created_when_disabled(self, tilif_no_sgc_model):
        """SGC modules should NOT exist when sgc_enabled=False."""
        assert not hasattr(tilif_no_sgc_model, 'sgc_smooth')

    def test_sgc_adds_parameters(self):
        """SGC-enabled model should have more parameters than without."""
        model_no_sgc = StudentSNN(
            input_size=5, hidden_size=16, num_layers=2,
            neuron_type="ti_lif", sgc_enabled=False,
        )
        model_sgc = StudentSNN(
            input_size=5, hidden_size=16, num_layers=2,
            neuron_type="ti_lif", sgc_enabled=True,
        )
        params_no = sum(p.numel() for p in model_no_sgc.parameters())
        params_sgc = sum(p.numel() for p in model_sgc.parameters())
        assert params_sgc > params_no, (
            f"SGC model ({params_sgc}) should have more params than non-SGC ({params_no})"
        )

    def test_sgc_smooth_has_tanh(self, tilif_sgc_model):
        """Each SGC smooth module should end with Tanh activation."""
        for smooth in tilif_sgc_model.sgc_smooth:
            # Sequential: [Linear, Tanh]
            last_layer = list(smooth.children())[-1]
            assert isinstance(last_layer, torch.nn.Tanh)

    def test_sgc_lambda_default_zero(self, tilif_sgc_model):
        """Default _sgc_lambda should be 0.0 (pure spiking)."""
        assert tilif_sgc_model._sgc_lambda == 0.0


# =============================================================================
# SGC Forward Pass Behavior
# =============================================================================

class TestSGCForward:
    """Tests for SGC blending behavior during forward pass."""

    def test_sgc_inactive_at_eval(self, tilif_sgc_model, input_data):
        """SGC should not blend at eval time, even if lambda > 0."""
        tilif_sgc_model._sgc_lambda = 0.5
        tilif_sgc_model.eval()

        result = tilif_sgc_model(input_data)
        assert "rates" in result
        assert result["rates"].shape == (4, 5)
        # Should produce valid output (no NaN/Inf)
        assert torch.isfinite(result["rates"]).all()

    def test_sgc_active_during_training(self, tilif_sgc_model, input_data):
        """SGC should produce valid output when active during training."""
        tilif_sgc_model._sgc_lambda = 0.5
        tilif_sgc_model.train()

        result = tilif_sgc_model(input_data)
        assert "rates" in result
        assert result["rates"].shape == (4, 5)
        assert (result["rates"] >= 0).all()

    def test_sgc_lambda_zero_matches_no_sgc(
        self, tilif_sgc_model, tilif_no_sgc_model, input_data
    ):
        """With lambda=0, SGC model should behave like non-SGC model.

        We can't compare outputs directly (different random weights), but
        we verify that SGC lambda=0 doesn't change behavior structurally.
        """
        tilif_sgc_model._sgc_lambda = 0.0
        tilif_sgc_model.train()

        result_sgc = tilif_sgc_model(input_data)
        result_no_sgc = tilif_no_sgc_model(input_data)

        # Same output structure
        assert set(result_sgc.keys()) == set(result_no_sgc.keys())
        # Same shapes
        assert result_sgc["rates"].shape == result_no_sgc["rates"].shape
        assert result_sgc["spikes"].shape == result_no_sgc["spikes"].shape

    def test_sgc_gradient_flow(self, tilif_sgc_model, input_data):
        """Gradients should flow through both spiking and SGC smooth paths."""
        tilif_sgc_model._sgc_lambda = 0.5
        tilif_sgc_model.train()

        result = tilif_sgc_model(input_data)
        loss = result["rates"].sum()
        loss.backward()

        # SGC smooth layers should have gradients
        for i, smooth in enumerate(tilif_sgc_model.sgc_smooth):
            for name, param in smooth.named_parameters():
                assert param.grad is not None, (
                    f"SGC smooth layer {i}.{name} should have gradients"
                )
                assert torch.any(param.grad != 0), (
                    f"SGC smooth layer {i}.{name} gradients should be non-zero"
                )

        # Regular spiking layers should also have gradients
        for i, layer in enumerate(tilif_sgc_model.spiking_layers):
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in layer.parameters()
            )
            assert has_grad, f"Spiking layer {i} should also have gradients"


# =============================================================================
# Membrane Potentials Exposure
# =============================================================================

class TestMembranePotentials:
    """Tests for membrane potential exposure in multi-head output."""

    def test_membrane_potentials_in_dict_output(
        self, tilif_sgc_model, input_data
    ):
        """Multi-head output should include membrane_potentials."""
        result = tilif_sgc_model(input_data)
        assert "membrane_potentials" in result

    def test_membrane_potentials_shape(self, tilif_sgc_model, input_data):
        """membrane_potentials should be (batch, T, hidden_size)."""
        B, T, M = input_data.shape
        result = tilif_sgc_model(input_data)
        expected_shape = (B, T, tilif_sgc_model.hidden_size)
        assert result["membrane_potentials"].shape == expected_shape, (
            f"Expected {expected_shape}, got {result['membrane_potentials'].shape}"
        )

    def test_membrane_potentials_are_continuous(
        self, tilif_sgc_model, input_data
    ):
        """Membrane potentials should be continuous (not discrete 0/1/-1)."""
        result = tilif_sgc_model(input_data)
        mem = result["membrane_potentials"]
        unique_vals = torch.unique(mem)
        # Should have many unique values (continuous), not just {-1, 0, 1}
        assert len(unique_vals) > 5, (
            "Membrane potentials should be continuous, not discrete"
        )

    def test_no_membrane_in_tuple_output(self, input_data):
        """Without auxiliary heads, output should be tuple (no membrane key)."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="ti_lif", sgc_enabled=True,
            # No auxiliary_heads → returns tuple
        )
        result = model(input_data)
        # Should be a tuple (rates, spikes), not a dict
        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# from_config with SGC
# =============================================================================

class TestFromConfigSGC:
    """Tests for from_config handling of SGC parameters."""

    def test_from_config_sgc_enabled(self):
        """from_config should create SGC modules when configured."""
        config = {
            "model": {
                "hidden_size": 16,
                "neuron_type": "ti_lif",
                "num_layers": 2,
                "sgc_enabled": True,
            }
        }
        model = StudentSNN.from_config(config, input_size=5)
        assert model.sgc_enabled is True
        assert hasattr(model, 'sgc_smooth')
        assert len(model.sgc_smooth) == 2

    def test_from_config_sgc_disabled_default(self):
        """from_config should default to sgc_enabled=False."""
        config = {
            "model": {
                "hidden_size": 16,
                "neuron_type": "ti_lif",
                "num_layers": 2,
            }
        }
        model = StudentSNN.from_config(config, input_size=5)
        assert model.sgc_enabled is False


# =============================================================================
# Backward Compatibility
# =============================================================================

class TestSGCBackwardCompat:
    """Verify SGC doesn't break existing non-SGC behavior."""

    def test_rsynaptic_unaffected(self, input_data):
        """RSynaptic model with sgc_enabled=False should work as before."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="rsynaptic", sgc_enabled=False,
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert (rates >= 0).all()

    def test_rleaky_unaffected(self, input_data):
        """RLeaky model should work unchanged."""
        model = StudentSNN(
            input_size=5, hidden_size=16, output_size=5,
            neuron_type="rleaky",
        )
        rates, spikes = model(input_data)
        assert rates.shape == (4, 5)
        assert (rates >= 0).all()
