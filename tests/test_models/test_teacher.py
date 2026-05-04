"""
Tests for the teacher LSTM model.

Validates forward pass shapes, softplus positivity, gradient flow,
config-based construction, and architecture improvements:
    - LayerNorm (Change 4)
    - Attention readout (Change 5)
    - Population coupling (ADR-0009 Batch C)
"""

import pytest
import torch
import numpy as np

from src.models.teacher import TeacherLSTM, PopulationCouplingLayer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """Create a small teacher model for testing (v1 defaults)."""
    return TeacherLSTM(
        input_size=10,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
    )


@pytest.fixture
def batch():
    """Create a dummy batch: (batch=4, T=50, M=10)."""
    torch.manual_seed(42)
    return torch.randn(4, 50, 10)


# =============================================================================
# Forward pass tests (v1 backward compat)
# =============================================================================

class TestForwardPass:
    """Tests for forward pass correctness."""

    def test_output_shape(self, model, batch):
        """Output shape should be (batch, M)."""
        output = model(batch)
        assert output.shape == (4, 10)

    def test_output_positive(self, model, batch):
        """All output rates must be positive (softplus enforces this)."""
        output = model(batch)
        assert (output > 0).all(), "Softplus should ensure all outputs > 0"

    def test_output_dtype(self, model, batch):
        """Output dtype should match input dtype."""
        output = model(batch)
        assert output.dtype == batch.dtype

    def test_single_sample(self, model):
        """Model should handle batch_size=1."""
        x = torch.randn(1, 50, 10)
        output = model(x)
        assert output.shape == (1, 10)

    def test_different_history_lengths(self, model):
        """Model should handle different history window sizes."""
        for T in [1, 10, 50, 100]:
            x = torch.randn(2, T, 10)
            output = model(x)
            assert output.shape == (2, 10), f"Failed for T={T}"

    def test_large_input_values(self, model):
        """Model should not produce NaN/inf for large inputs."""
        x = torch.ones(2, 50, 10) * 100.0
        output = model(x)
        assert torch.isfinite(output).all(), "Outputs contain NaN or inf"

    def test_zero_input(self, model):
        """Model should handle zero inputs gracefully."""
        x = torch.zeros(2, 50, 10)
        output = model(x)
        assert torch.isfinite(output).all()
        assert (output > 0).all()  # Softplus(anything) > 0


# =============================================================================
# Gradient tests (v1 backward compat)
# =============================================================================

class TestGradients:
    """Tests for gradient flow through the model."""

    def test_gradient_flow(self, model, batch):
        """Gradients should flow to all parameters."""
        output = model(batch)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient for {name}"

    def test_gradient_not_zero(self, model, batch):
        """At least some gradients should be non-zero."""
        output = model(batch)
        loss = output.mean()
        loss.backward()

        has_nonzero = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"


# =============================================================================
# Construction tests (v1 backward compat)
# =============================================================================

class TestConstruction:
    """Tests for model construction and configuration."""

    def test_default_output_size(self):
        """output_size defaults to input_size."""
        model = TeacherLSTM(input_size=10)
        assert model.output_size == 10

    def test_custom_output_size(self):
        """Custom output_size should work."""
        model = TeacherLSTM(input_size=10, output_size=5)
        x = torch.randn(2, 50, 10)
        output = model(x)
        assert output.shape == (2, 5)

    def test_single_layer_no_dropout(self):
        """Single-layer LSTM should have dropout=0 (PyTorch requirement)."""
        model = TeacherLSTM(input_size=10, num_layers=1, dropout=0.5)
        # Should not raise — dropout is set to 0 internally for single layer
        x = torch.randn(2, 50, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_from_config(self):
        """from_config should create the model correctly."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 3,
                "dropout": 0.3,
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.hidden_size == 64
        assert model.num_layers == 3

    def test_from_config_defaults(self):
        """from_config should handle missing keys with defaults."""
        config = {}
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.hidden_size == 128
        assert model.num_layers == 2

    def test_parameter_count(self):
        """Verify approximate parameter count."""
        model = TeacherLSTM(input_size=10, hidden_size=32, num_layers=1)
        n_params = sum(p.numel() for p in model.parameters())
        # Input proj: 10*32+32 = 352
        # LSTM: 4*(32*32+32*32+32) = 8320
        # Output proj: 32*10+10 = 330
        # Total: ~9002
        assert 8000 < n_params < 12000, f"Unexpected param count: {n_params}"


# =============================================================================
# Determinism tests (v1 backward compat)
# =============================================================================

class TestDeterminism:
    """Tests for reproducible outputs."""

    def test_deterministic_output(self, model, batch):
        """Same input should give same output in eval mode."""
        model.eval()
        with torch.no_grad():
            out1 = model(batch).clone()
            out2 = model(batch).clone()
        assert torch.allclose(out1, out2), "Non-deterministic output in eval mode"


# =============================================================================
# Change 4: LayerNorm
# =============================================================================

class TestLayerNorm:
    """Tests for optional LayerNorm in teacher model."""

    def test_layer_norm_present(self):
        """LayerNorm modules should exist when use_layer_norm=True."""
        model = TeacherLSTM(input_size=10, hidden_size=32, use_layer_norm=True)
        assert isinstance(model.input_norm, torch.nn.LayerNorm)
        assert isinstance(model.output_norm, torch.nn.LayerNorm)

    def test_layer_norm_default_off(self):
        """Default use_layer_norm=False should use Identity."""
        model = TeacherLSTM(input_size=10, hidden_size=32, use_layer_norm=False)
        assert isinstance(model.input_norm, torch.nn.Identity)
        assert isinstance(model.output_norm, torch.nn.Identity)

    def test_layer_norm_output_shape(self):
        """Output shape should be unchanged with LayerNorm enabled."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_layer_norm=True
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert output.shape == (4, 10)
        assert (output > 0).all()

    def test_layer_norm_gradient_flow(self):
        """LayerNorm parameters should receive gradients."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_layer_norm=True
        )
        x = torch.randn(2, 20, 10)
        loss = model(x).sum()
        loss.backward()

        # LayerNorm weight and bias should have gradients
        assert model.input_norm.weight.grad is not None
        assert model.output_norm.weight.grad is not None


# =============================================================================
# Change 5: Attention Readout
# =============================================================================

class TestAttentionReadout:
    """Tests for learned attention readout over LSTM timesteps."""

    def test_attention_output_shape(self):
        """Output shape should be unchanged with attention enabled."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_attention=True
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert output.shape == (4, 10)

    def test_attention_weights_sum_to_one(self):
        """Attention weights (softmax) should sum to 1 across time dim."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_attention=True
        )
        x = torch.randn(2, 20, 10)

        # Run forward manually to extract attention weights
        projected = model.input_norm(model.input_proj(x))
        lstm_out, _ = model.lstm(projected)
        attn_scores = model.attn_query(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weights should sum to 1 across time dimension for each batch
        weight_sums = attn_weights.sum(dim=1).squeeze(-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_attention_gradient_flow(self):
        """Attention query parameters should receive gradients."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_attention=True
        )
        x = torch.randn(2, 20, 10)
        loss = model(x).sum()
        loss.backward()

        assert model.attn_query.weight.grad is not None, (
            "Attention query weight should receive gradients"
        )
        assert model.attn_query.weight.grad.abs().sum() > 0

    def test_attention_default_off(self):
        """Default use_attention=False should use last hidden state."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_attention=False
        )
        assert model.attn_query is None

    def test_attention_positive_output(self):
        """Attention model should still produce positive rates."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_attention=True
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert (output > 0).all(), "Softplus should ensure positive outputs"


# =============================================================================
# Combined Features
# =============================================================================

class TestCombinedTeacherFeatures:
    """Tests for both LayerNorm and attention enabled."""

    def test_combined_layer_norm_and_attention(self):
        """Both features enabled should produce valid output."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32,
            use_layer_norm=True, use_attention=True,
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert output.shape == (4, 10)
        assert (output > 0).all()

    def test_from_config_new_fields(self):
        """from_config should handle new architecture keys."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "use_layer_norm": True,
                "use_attention": True,
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.hidden_size == 64
        assert model.use_attention is True
        assert isinstance(model.input_norm, torch.nn.LayerNorm)
        assert model.attn_query is not None

        # Verify forward pass
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_combined_gradient_flow(self):
        """All parameters should get gradients with both features."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32,
            use_layer_norm=True, use_attention=True,
        )
        x = torch.randn(2, 20, 10)
        loss = model(x).sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Output Distribution Tests (ADR-0009)
# =============================================================================

class TestOutputDistributions:
    """Tests for configurable output distributions (Poisson, NegBin, ZIP)."""

    def test_poisson_default(self):
        """Default output_distribution should be 'poisson'."""
        model = TeacherLSTM(input_size=10, hidden_size=32)
        assert model.output_distribution == "poisson"
        assert model.aux_proj is None

    def test_poisson_no_aux_output(self):
        """Poisson model should have no auxiliary output."""
        model = TeacherLSTM(input_size=10, hidden_size=32)
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert model.get_aux_output() is None
        assert output.shape == (2, 10)

    def test_negbin_output_shape(self):
        """NegBin model should produce rates and dispersion with correct shapes."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin"
        )
        x = torch.randn(4, 50, 10)
        rates = model(x)
        dispersion = model.get_aux_output()

        # Rate output shape
        assert rates.shape == (4, 10)
        # Dispersion output shape should match rate
        assert dispersion is not None
        assert dispersion.shape == (4, 10)

    def test_negbin_positivity(self):
        """NegBin rates and dispersion must both be positive (Softplus)."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin"
        )
        x = torch.randn(4, 50, 10)
        rates = model(x)
        dispersion = model.get_aux_output()

        assert (rates > 0).all(), "Rates must be positive"
        assert (dispersion > 0).all(), "Dispersion must be positive"

    def test_zip_output_shape(self):
        """ZIP model should produce rates and gate with correct shapes."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="zip"
        )
        x = torch.randn(4, 50, 10)
        rates = model(x)
        gate = model.get_aux_output()

        assert rates.shape == (4, 10)
        assert gate is not None
        assert gate.shape == (4, 10)

    def test_zip_gate_range(self):
        """ZIP gate should be in [0, 1] (Sigmoid)."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="zip"
        )
        x = torch.randn(4, 50, 10)
        model(x)
        gate = model.get_aux_output()

        assert (gate >= 0).all(), "Gate must be >= 0"
        assert (gate <= 1).all(), "Gate must be <= 1"

    def test_negbin_gradient_flow(self):
        """Gradients should flow through both rate and aux (dispersion) heads."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin"
        )
        x = torch.randn(2, 20, 10)
        rates = model(x)
        dispersion = model.get_aux_output()
        loss = rates.sum() + dispersion.sum()
        loss.backward()

        # aux_proj parameters should have gradients
        assert model.aux_proj.weight.grad is not None
        assert model.aux_proj.weight.grad.abs().sum() > 0

    def test_zip_gradient_flow(self):
        """Gradients should flow through both rate and aux (gate) heads."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="zip"
        )
        x = torch.randn(2, 20, 10)
        rates = model(x)
        gate = model.get_aux_output()
        loss = rates.sum() + gate.sum()
        loss.backward()

        assert model.aux_proj.weight.grad is not None
        assert model.aux_proj.weight.grad.abs().sum() > 0

    def test_negbin_more_params(self):
        """NegBin model should have more parameters than Poisson (extra aux head)."""
        poisson_model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="poisson"
        )
        negbin_model = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin"
        )
        p_params = sum(p.numel() for p in poisson_model.parameters())
        n_params = sum(p.numel() for p in negbin_model.parameters())
        # NegBin should have hidden*M + M more params (the aux_proj layer)
        assert n_params > p_params
        # Extra params = Linear(32, 10) = 32*10 + 10 = 330
        assert n_params - p_params == 32 * 10 + 10

    def test_from_config_negbin(self):
        """from_config should handle output_distribution='negbin'."""
        config = {
            "model": {
                "hidden_size": 32,
                "output_distribution": "negbin",
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.output_distribution == "negbin"
        assert model.aux_proj is not None

    def test_from_config_zip(self):
        """from_config should handle output_distribution='zip'."""
        config = {
            "model": {
                "hidden_size": 32,
                "output_distribution": "zip",
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.output_distribution == "zip"
        assert model.aux_proj is not None

    def test_from_config_default_poisson(self):
        """from_config with no output_distribution should default to poisson."""
        config = {"model": {"hidden_size": 32}}
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.output_distribution == "poisson"
        assert model.aux_proj is None

    def test_invalid_distribution_raises(self):
        """Invalid output_distribution should raise ValueError."""
        with pytest.raises(ValueError, match="output_distribution"):
            TeacherLSTM(
                input_size=10, hidden_size=32, output_distribution="gamma"
            )

    def test_negbin_combined_with_attention(self):
        """NegBin + attention should work together."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32,
            use_attention=True, output_distribution="negbin",
        )
        x = torch.randn(2, 20, 10)
        rates = model(x)
        dispersion = model.get_aux_output()
        assert rates.shape == (2, 10)
        assert dispersion.shape == (2, 10)
        assert (rates > 0).all()
        assert (dispersion > 0).all()


# =============================================================================
# Population Coupling (ADR-0009 Batch C)
# =============================================================================

class TestPopulationCoupling:
    """Tests for optional cross-neuron population coupling layer."""

    def test_coupling_default_off(self):
        """Coupling should be None when not configured."""
        model = TeacherLSTM(input_size=10, hidden_size=32)
        assert model.coupling is None

    def test_coupling_output_shape(self):
        """Output shape should be (batch, M) with coupling enabled."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_population_coupling=True,
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert output.shape == (4, 10)

    def test_coupling_positive_output(self):
        """Softplus should still enforce λ > 0 with coupling enabled."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_population_coupling=True,
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert (output > 0).all(), "Softplus should ensure positive outputs"

    def test_coupling_gradient_flow(self):
        """All coupling parameters should receive non-zero gradients."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_population_coupling=True,
        )
        x = torch.randn(2, 20, 10)
        loss = model(x).sum()
        loss.backward()

        # The coupling MLP params must receive gradients
        for name, param in model.coupling.named_parameters():
            assert param.grad is not None, f"No gradient for coupling.{name}"
            assert param.grad.abs().sum() > 0, (
                f"Zero gradient for coupling.{name}"
            )

    def test_coupling_residual_identity(self):
        """With zero-initialized coupling weights, output equals baseline."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, use_population_coupling=True,
        )
        # Zero-init the second linear so residual path dominates
        nn = model.coupling.mlp[2]  # Second Linear layer
        torch.nn.init.zeros_(nn.weight)
        torch.nn.init.zeros_(nn.bias)

        # Build identical model without coupling for comparison
        baseline = TeacherLSTM(input_size=10, hidden_size=32)
        baseline.load_state_dict(
            {k: v for k, v in model.state_dict().items()
             if not k.startswith("coupling.")},
            strict=False,
        )

        x = torch.randn(2, 20, 10)
        model.eval()
        baseline.eval()
        with torch.no_grad():
            out_coupled = model(x)
            out_baseline = baseline(x)
        assert torch.allclose(out_coupled, out_baseline, atol=1e-5), (
            "Zero-initialized coupling should produce identical output"
        )

    def test_coupling_more_params(self):
        """Coupling model should have more parameters than baseline."""
        baseline = TeacherLSTM(input_size=10, hidden_size=32)
        coupled = TeacherLSTM(
            input_size=10, hidden_size=32, use_population_coupling=True,
        )
        p_base = sum(p.numel() for p in baseline.parameters())
        p_coupled = sum(p.numel() for p in coupled.parameters())
        assert p_coupled > p_base

    def test_coupling_param_count(self):
        """Exact extra param count: M*h + h + h*M + M."""
        M, h = 10, 32
        baseline = TeacherLSTM(input_size=M, hidden_size=32)
        coupled = TeacherLSTM(
            input_size=M, hidden_size=32,
            use_population_coupling=True, coupling_hidden_size=h,
        )
        p_base = sum(p.numel() for p in baseline.parameters())
        p_coupled = sum(p.numel() for p in coupled.parameters())
        # Linear(M,h): M*h+h, Linear(h,M): h*M+M
        expected_extra = M * h + h + h * M + M
        assert p_coupled - p_base == expected_extra, (
            f"Expected {expected_extra} extra params, got {p_coupled - p_base}"
        )

    def test_coupling_from_config(self):
        """from_config should enable coupling from config dict."""
        config = {
            "model": {
                "hidden_size": 32,
                "use_population_coupling": True,
                "coupling_hidden_size": 16,
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.coupling is not None
        assert model.coupling.hidden_size == 16

        # Verify forward pass works
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_coupling_combined_with_attention_and_negbin(self):
        """Coupling should work with attention + NegBin distribution."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32,
            use_attention=True,
            use_population_coupling=True,
            output_distribution="negbin",
        )
        x = torch.randn(2, 20, 10)
        rates = model(x)
        dispersion = model.get_aux_output()
        assert rates.shape == (2, 10)
        assert dispersion.shape == (2, 10)
        assert (rates > 0).all()
        assert (dispersion > 0).all()
        assert model.coupling is not None

    def test_coupling_different_hidden_sizes(self):
        """Various coupling_hidden_size values should all work."""
        for h in [8, 16, 64]:
            model = TeacherLSTM(
                input_size=10, hidden_size=32,
                use_population_coupling=True,
                coupling_hidden_size=h,
            )
            x = torch.randn(2, 20, 10)
            output = model(x)
            assert output.shape == (2, 10), f"Failed for coupling_hidden_size={h}"
            assert model.coupling.hidden_size == h


class TestPopulationCouplingLayerStandalone:
    """Tests for the PopulationCouplingLayer module directly."""

    def test_forward_shape(self):
        """Output shape should match input shape."""
        layer = PopulationCouplingLayer(num_channels=10, hidden_size=16)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_residual_connection(self):
        """Output should equal input when MLP weights are zeroed."""
        layer = PopulationCouplingLayer(num_channels=10, hidden_size=16)
        # Zero the second linear so mlp(x) = 0
        torch.nn.init.zeros_(layer.mlp[2].weight)
        torch.nn.init.zeros_(layer.mlp[2].bias)
        x = torch.randn(4, 10)
        out = layer(x)
        assert torch.allclose(out, x, atol=1e-6)


# =============================================================================
# Covariate projection tests (ADR-0012, Option B additive fusion)
# =============================================================================

class TestCovariateProjection:
    """Tests for TeacherLSTM covariate projection layer."""

    def test_no_covariates_backward_compat(self):
        """Model with n_covariates=0 should have no covariate_proj."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=0)
        assert model.covariate_proj is None
        assert model.n_covariates == 0

    def test_covariates_creates_proj_layer(self):
        """Model with n_covariates > 0 should have a projection layer."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=5)
        assert model.covariate_proj is not None
        assert model.n_covariates == 5
        # Check layer dimensions
        assert model.covariate_proj.in_features == 5
        assert model.covariate_proj.out_features == 32

    def test_forward_with_covariates_shape(self):
        """Forward pass with covariates should produce correct output shape."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)  # (batch, T, M)
        cov = torch.randn(4, 5)     # (batch, n_covariates)
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_forward_without_covariates_when_proj_exists(self):
        """Forward pass without covariates should still work (graceful None)."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)
        out = model(x, covariates=None)
        assert out.shape == (4, 10)

    def test_forward_covariates_ignored_when_no_proj(self):
        """Covariates should be silently ignored when n_covariates=0."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=0)
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 5)
        # Should not raise — covariates ignored because proj is None
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_gradient_flow_through_covariates(self):
        """Gradients should flow through the covariate projection."""
        model = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 5, requires_grad=True)
        out = model(x, covariates=cov)
        loss = out.sum()
        loss.backward()
        assert cov.grad is not None
        assert cov.grad.shape == (4, 5)
        # Projection layer weights should also have gradients
        assert model.covariate_proj.weight.grad is not None

    def test_extra_params_from_covariates(self):
        """Covariate projection should add expected number of parameters."""
        model_no_cov = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=0)
        model_with_cov = TeacherLSTM(input_size=10, hidden_size=32, n_covariates=5)
        params_no = sum(p.numel() for p in model_no_cov.parameters())
        params_with = sum(p.numel() for p in model_with_cov.parameters())
        # Extra params = n_covariates * hidden_size + hidden_size (bias)
        expected_extra = 5 * 32 + 32  # 192
        assert params_with - params_no == expected_extra

    def test_from_config_reads_n_covariates(self):
        """from_config should pass n_covariates to the constructor."""
        config = {
            "model": {
                "hidden_size": 32,
                "num_layers": 1,
                "n_covariates": 5,
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.n_covariates == 5
        assert model.covariate_proj is not None


# =============================================================================
# Temporal Covariate Mode Tests (ADR-0012)
# =============================================================================

class TestTemporalCovariates:
    """Tests for temporal covariate feeding (input concatenation)."""

    def test_temporal_mode_forward_shape(self):
        """Temporal covariates concat to input -> correct output shape."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)     # (batch, T, M)
        cov = torch.randn(4, 20, 5)    # (batch, T, n_cov) — temporal!
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_temporal_mode_input_proj_width(self):
        """Input projection should be wider in temporal mode."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="temporal",
        )
        # input_proj should accept M + n_cov = 15
        assert model.input_proj.in_features == 15

    def test_temporal_gradient_flow(self):
        """Gradients should flow through temporal covariates."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 20, 5, requires_grad=True)
        out = model(x, covariates=cov)
        loss = out.sum()
        loss.backward()
        assert cov.grad is not None
        assert cov.grad.shape == (4, 20, 5)

    def test_temporal_without_covariates_still_works(self):
        """Model in temporal mode should work when covariates=None."""
        # This only works if n_covariates=0 (no wider input_proj)
        model = TeacherLSTM(
            input_size=10, hidden_size=32, n_covariates=0,
            covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)
        out = model(x, covariates=None)
        assert out.shape == (4, 10)

    def test_additive_mode_still_works(self):
        """Additive mode (default) should still accept (batch, n_cov)."""
        model = TeacherLSTM(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="additive",
        )
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 5)  # (batch, n_cov) — additive
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_from_config_reads_covariate_mode(self):
        """from_config should pass covariate_mode to the constructor."""
        config = {
            "model": {
                "hidden_size": 32,
                "num_layers": 1,
                "n_covariates": 5,
                "covariate_mode": "temporal",
            }
        }
        model = TeacherLSTM.from_config(config, input_size=10)
        assert model.covariate_mode == "temporal"
        assert model.input_proj.in_features == 15

    def test_invalid_covariate_mode_raises(self):
        """Invalid covariate_mode should raise ValueError."""
        with pytest.raises(ValueError, match="covariate_mode"):
            TeacherLSTM(
                input_size=10, hidden_size=32, n_covariates=5,
                covariate_mode="invalid",
            )


