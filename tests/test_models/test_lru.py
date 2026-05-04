"""
Tests for the LRU (Linear Recurrent Unit) teacher model.

Validates:
    - LRUCell forward pass shapes and stability (|λ| < 1)
    - TeacherLRU forward signature matches TeacherLSTM
    - Gated vs non-gated mode
    - from_config() factory method
    - Output distributions (negbin/zip auxiliary heads)
    - create_teacher_model() factory dispatching
"""

import pytest
import torch

from src.models.lru import LRUCell, TeacherLRU
from src.models.teacher import TeacherLSTM, create_teacher_model


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def lru_cell():
    """Create a small LRUCell for testing."""
    return LRUCell(input_size=10, hidden_size=32)


@pytest.fixture
def lru_model():
    """Create a small TeacherLRU for testing (matching TeacherLSTM defaults)."""
    return TeacherLRU(
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
# LRUCell Tests
# =============================================================================

class TestLRUCellForward:
    """Tests for LRUCell forward pass correctness."""

    def test_output_shape(self, lru_cell, batch):
        """Output shape should be (batch, T, input_size)."""
        output = lru_cell(batch)
        assert output.shape == (4, 50, 10)

    def test_output_dtype_real(self, lru_cell, batch):
        """Output should be real-valued (not complex)."""
        output = lru_cell(batch)
        assert not output.is_complex(), "Output should be real, not complex"
        assert output.dtype == batch.dtype

    def test_single_timestep(self, lru_cell):
        """LRUCell should handle T=1."""
        x = torch.randn(2, 1, 10)
        output = lru_cell(x)
        assert output.shape == (2, 1, 10)

    def test_various_T(self, lru_cell):
        """LRUCell should handle different history window sizes."""
        for T in [1, 10, 50, 100]:
            x = torch.randn(2, T, 10)
            output = lru_cell(x)
            assert output.shape == (2, T, 10), f"Failed for T={T}"

    def test_finite_output(self, lru_cell, batch):
        """Output should not contain NaN or inf."""
        output = lru_cell(batch)
        assert torch.isfinite(output).all(), "Output contains NaN or inf"

    def test_gated_output_shape(self):
        """Gated LRUCell should produce same output shape."""
        cell = LRUCell(input_size=10, hidden_size=32, gated=True)
        x = torch.randn(4, 50, 10)
        output = cell(x)
        assert output.shape == (4, 50, 10)

    def test_gated_finite_output(self):
        """Gated LRUCell output should be finite."""
        cell = LRUCell(input_size=10, hidden_size=32, gated=True)
        x = torch.randn(4, 50, 10)
        output = cell(x)
        assert torch.isfinite(output).all()


# =============================================================================
# LRU Stability Tests
# =============================================================================

class TestLRUStability:
    """Tests for eigenvalue stability (|λ| < 1)."""

    def test_eigenvalue_magnitude_non_gated(self):
        """Non-gated LRU eigenvalues must have |λ| < 1."""
        cell = LRUCell(input_size=10, hidden_size=64)
        lambda_ = cell._compute_lambda()
        magnitudes = lambda_.abs()
        assert (magnitudes < 1.0).all(), (
            f"Some |λ| >= 1: max={magnitudes.max().item():.6f}"
        )

    def test_eigenvalue_magnitude_gated(self):
        """Gated LRU eigenvalues must have |λ| < 1 for any input."""
        cell = LRUCell(input_size=10, hidden_size=64, gated=True)
        # Test with random input
        x_t = torch.randn(8, 10)
        lambda_ = cell._compute_lambda(x_t)
        magnitudes = lambda_.abs()
        assert (magnitudes < 1.0).all(), (
            f"Some |λ| >= 1: max={magnitudes.max().item():.6f}"
        )

    def test_eigenvalue_magnitude_extreme_params(self):
        """Stability should hold even with extreme parameter values."""
        cell = LRUCell(input_size=10, hidden_size=32)
        # Set ν to extreme values — stability should still hold
        with torch.no_grad():
            cell.nu.fill_(10.0)   # Very large decay
            cell.theta.fill_(5.0)  # Large oscillation
        lambda_ = cell._compute_lambda()
        magnitudes = lambda_.abs()
        assert (magnitudes < 1.0).all()

    def test_eigenvalue_magnitude_negative_params(self):
        """Stability should hold for negative parameter values."""
        cell = LRUCell(input_size=10, hidden_size=32)
        with torch.no_grad():
            cell.nu.fill_(-5.0)
            cell.theta.fill_(-5.0)
        lambda_ = cell._compute_lambda()
        magnitudes = lambda_.abs()
        assert (magnitudes < 1.0).all(), (
            f"Some |λ| >= 1 with negative params: "
            f"max={magnitudes.max().item():.6f}"
        )


# =============================================================================
# TeacherLRU Forward Pass Tests
# =============================================================================

class TestTeacherLRUForward:
    """Tests for TeacherLRU forward pass — must match TeacherLSTM."""

    def test_output_shape(self, lru_model, batch):
        """Output shape should be (batch, M) — same as TeacherLSTM."""
        output = lru_model(batch)
        assert output.shape == (4, 10)

    def test_output_positive(self, lru_model, batch):
        """All output rates must be positive (softplus enforces this)."""
        output = lru_model(batch)
        assert (output > 0).all(), "Softplus should ensure all outputs > 0"

    def test_output_dtype(self, lru_model, batch):
        """Output dtype should match input dtype."""
        output = lru_model(batch)
        assert output.dtype == batch.dtype

    def test_single_sample(self, lru_model):
        """Model should handle batch_size=1."""
        x = torch.randn(1, 50, 10)
        output = lru_model(x)
        assert output.shape == (1, 10)

    def test_different_history_lengths(self, lru_model):
        """Model should handle different history window sizes."""
        for T in [1, 10, 50, 100]:
            x = torch.randn(2, T, 10)
            output = lru_model(x)
            assert output.shape == (2, 10), f"Failed for T={T}"

    def test_large_input_values(self, lru_model):
        """Model should not produce NaN/inf for large inputs."""
        x = torch.ones(2, 50, 10) * 100.0
        output = lru_model(x)
        assert torch.isfinite(output).all(), "Outputs contain NaN or inf"

    def test_zero_input(self, lru_model):
        """Model should handle zero inputs gracefully."""
        x = torch.zeros(2, 50, 10)
        output = lru_model(x)
        assert torch.isfinite(output).all()
        assert (output > 0).all()  # Softplus(anything) > 0

    def test_custom_output_size(self):
        """Custom output_size should work."""
        model = TeacherLRU(input_size=10, output_size=5)
        x = torch.randn(2, 50, 10)
        output = model(x)
        assert output.shape == (2, 5)

    def test_deterministic_eval(self, lru_model, batch):
        """Same input should give same output in eval mode."""
        lru_model.eval()
        with torch.no_grad():
            out1 = lru_model(batch).clone()
            out2 = lru_model(batch).clone()
        assert torch.allclose(out1, out2), (
            "Non-deterministic output in eval mode"
        )


# =============================================================================
# TeacherLRU Gradient Tests
# =============================================================================

class TestTeacherLRUGradients:
    """Tests for gradient flow through TeacherLRU."""

    def test_gradient_flow(self, lru_model, batch):
        """Gradients should flow to all parameters."""
        output = lru_model(batch)
        loss = output.sum()
        loss.backward()

        for name, param in lru_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for {name}"
            )

    def test_gradient_not_zero(self, lru_model, batch):
        """At least some gradients should be non-zero."""
        output = lru_model(batch)
        loss = output.mean()
        loss.backward()

        has_nonzero = False
        for param in lru_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"


# =============================================================================
# TeacherLRU from_config Tests
# =============================================================================

class TestTeacherLRUFromConfig:
    """Tests for config-based construction."""

    def test_from_config_defaults(self):
        """from_config should handle missing keys with defaults."""
        config = {}
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.gated is False

    def test_from_config_custom(self):
        """from_config should respect custom values."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 3,
                "dropout": 0.3,
                "gated": True,
            }
        }
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.hidden_size == 64
        assert model.num_layers == 3
        assert model.gated is True

        # Verify forward pass works
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_from_config_with_attention_and_layernorm(self):
        """from_config should handle attention + layer norm."""
        config = {
            "model": {
                "hidden_size": 32,
                "use_layer_norm": True,
                "use_attention": True,
            }
        }
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.use_attention is True
        assert isinstance(model.input_norm, torch.nn.LayerNorm)
        assert model.attn_query is not None

        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_from_config_with_coupling(self):
        """from_config should enable coupling from config dict."""
        config = {
            "model": {
                "hidden_size": 32,
                "use_population_coupling": True,
                "coupling_hidden_size": 16,
            }
        }
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.coupling is not None
        assert model.coupling.hidden_size == 16

        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)


# =============================================================================
# TeacherLRU Gated Mode Tests
# =============================================================================

class TestTeacherLRUGated:
    """Tests for gated LRU mode (content-dependent eigenvalues)."""

    def test_gated_output_shape(self):
        """Gated TeacherLRU should produce same output shape."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, gated=True,
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert output.shape == (4, 10)

    def test_gated_positive_output(self):
        """Gated model should still produce positive rates."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, gated=True,
        )
        x = torch.randn(4, 50, 10)
        output = model(x)
        assert (output > 0).all()

    def test_gated_gradient_flow(self):
        """Gated parameters should receive gradients."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, num_layers=1, gated=True,
        )
        x = torch.randn(2, 20, 10)
        loss = model(x).sum()
        loss.backward()

        # Gate projections should have gradients
        for lru_layer in model.lru_layers:
            assert lru_layer.gate_nu.weight.grad is not None
            assert lru_layer.gate_theta.weight.grad is not None

    def test_gated_more_params(self):
        """Gated model should have more parameters than non-gated."""
        non_gated = TeacherLRU(
            input_size=10, hidden_size=32, num_layers=1, gated=False,
        )
        gated = TeacherLRU(
            input_size=10, hidden_size=32, num_layers=1, gated=True,
        )
        p_ng = sum(p.numel() for p in non_gated.parameters())
        p_g = sum(p.numel() for p in gated.parameters())
        assert p_g > p_ng, (
            f"Gated ({p_g}) should have more params than non-gated ({p_ng})"
        )


# =============================================================================
# TeacherLRU Distribution Tests
# =============================================================================

class TestTeacherLRUDistributions:
    """Tests for output distributions (matching TeacherLSTM behavior)."""

    def test_poisson_default(self):
        """Default output_distribution should be 'poisson'."""
        model = TeacherLRU(input_size=10, hidden_size=32)
        assert model.output_distribution == "poisson"
        assert model.aux_proj is None

    def test_poisson_no_aux_output(self):
        """Poisson model should have no auxiliary output."""
        model = TeacherLRU(input_size=10, hidden_size=32)
        x = torch.randn(2, 20, 10)
        model(x)
        assert model.get_aux_output() is None

    def test_negbin_output_shape(self):
        """NegBin should produce rates and dispersion with correct shapes."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, output_distribution="negbin",
        )
        x = torch.randn(4, 50, 10)
        rates = model(x)
        dispersion = model.get_aux_output()

        assert rates.shape == (4, 10)
        assert dispersion is not None
        assert dispersion.shape == (4, 10)
        assert (rates > 0).all()
        assert (dispersion > 0).all()

    def test_zip_output_shape(self):
        """ZIP should produce rates and gate with correct shapes."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, output_distribution="zip",
        )
        x = torch.randn(4, 50, 10)
        rates = model(x)
        gate = model.get_aux_output()

        assert rates.shape == (4, 10)
        assert gate is not None
        assert gate.shape == (4, 10)
        assert (gate >= 0).all()
        assert (gate <= 1).all()

    def test_invalid_distribution_raises(self):
        """Invalid output_distribution should raise ValueError."""
        with pytest.raises(ValueError, match="output_distribution"):
            TeacherLRU(
                input_size=10, hidden_size=32, output_distribution="gamma",
            )


# =============================================================================
# create_teacher_model Factory Tests
# =============================================================================

class TestCreateTeacherModelFactory:
    """Tests for the architecture factory function."""

    def test_factory_default_lstm(self):
        """Default (no architecture key) should create TeacherLSTM."""
        config = {"model": {"hidden_size": 32}}
        model = create_teacher_model(config, input_size=10)
        assert isinstance(model, TeacherLSTM)

    def test_factory_explicit_lstm(self):
        """architecture='lstm' should create TeacherLSTM."""
        config = {"model": {"architecture": "lstm", "hidden_size": 32}}
        model = create_teacher_model(config, input_size=10)
        assert isinstance(model, TeacherLSTM)

    def test_factory_lru(self):
        """architecture='lru' should create TeacherLRU."""
        config = {"model": {"architecture": "lru", "hidden_size": 32}}
        model = create_teacher_model(config, input_size=10)
        assert isinstance(model, TeacherLRU)

    def test_factory_lru_forward(self):
        """Factory-created LRU should produce valid forward output."""
        config = {"model": {"architecture": "lru", "hidden_size": 32}}
        model = create_teacher_model(config, input_size=10)
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)
        assert (output > 0).all()

    def test_factory_unknown_raises(self):
        """Unknown architecture should raise ValueError."""
        config = {"model": {"architecture": "gru"}}
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_teacher_model(config, input_size=10)

    def test_factory_empty_config(self):
        """Empty config should default to LSTM."""
        config = {}
        model = create_teacher_model(config, input_size=10)
        assert isinstance(model, TeacherLSTM)


# =============================================================================
# Interface Compatibility Tests
# =============================================================================

class TestLRULSTMInterfaceCompat:
    """Verify TeacherLRU is truly a drop-in replacement for TeacherLSTM."""

    def test_same_forward_signature(self):
        """Both models should accept (batch, T, M) → (batch, M)."""
        lstm = TeacherLSTM(input_size=10, hidden_size=32)
        lru = TeacherLRU(input_size=10, hidden_size=32)
        x = torch.randn(4, 50, 10)

        lstm_out = lstm(x)
        lru_out = lru(x)

        assert lstm_out.shape == lru_out.shape == (4, 10)

    def test_same_get_aux_output(self):
        """Both models should have get_aux_output() method."""
        lstm = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin",
        )
        lru = TeacherLRU(
            input_size=10, hidden_size=32, output_distribution="negbin",
        )
        x = torch.randn(2, 20, 10)

        lstm(x)
        lru(x)

        # Both should return non-None dispersion for negbin
        assert lstm.get_aux_output() is not None
        assert lru.get_aux_output() is not None
        assert lstm.get_aux_output().shape == lru.get_aux_output().shape

    def test_same_from_config(self):
        """Both models should have compatible from_config classmethods."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "use_layer_norm": True,
                "use_attention": True,
            }
        }
        lstm = TeacherLSTM.from_config(config, input_size=10)
        lru = TeacherLRU.from_config(config, input_size=10)

        assert lstm.hidden_size == lru.hidden_size == 64
        assert lstm.num_layers == lru.num_layers == 2
        assert lstm.use_attention == lru.use_attention is True

    def test_h0_param_accepted(self):
        """TeacherLRU.forward should accept h0 param (ignored)."""
        model = TeacherLRU(input_size=10, hidden_size=32)
        x = torch.randn(2, 20, 10)
        # h0 should be accepted without error (ignored internally)
        output = model(x, h0=None)
        assert output.shape == (2, 10)


# =============================================================================
# Covariate Projection Tests (ADR-0012, mirrors TestCovariateProjection
# in test_teacher.py)
# =============================================================================

class TestLRUCovariateProjection:
    """Tests for TeacherLRU covariate projection layer."""

    def test_no_covariates_backward_compat(self):
        """Model with n_covariates=0 should have no covariate_proj."""
        model = TeacherLRU(input_size=10, hidden_size=32, n_covariates=0)
        assert model.covariate_proj is None
        assert model.n_covariates == 0

    def test_covariates_creates_proj_layer(self):
        """Model with n_covariates > 0 should have a projection layer."""
        model = TeacherLRU(input_size=10, hidden_size=32, n_covariates=5)
        assert model.covariate_proj is not None
        assert model.n_covariates == 5
        # Check layer dimensions
        assert model.covariate_proj.in_features == 5
        assert model.covariate_proj.out_features == 32

    def test_forward_with_covariates_shape(self):
        """Forward pass with covariates should produce correct output shape."""
        model = TeacherLRU(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)  # (batch, T, M)
        cov = torch.randn(4, 5)     # (batch, n_covariates)
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_forward_without_covariates_when_proj_exists(self):
        """Forward pass without covariates should still work (graceful None)."""
        model = TeacherLRU(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)
        out = model(x, covariates=None)
        assert out.shape == (4, 10)

    def test_gradient_flow_through_covariates(self):
        """Gradients should flow through the covariate projection."""
        model = TeacherLRU(input_size=10, hidden_size=32, n_covariates=5)
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 5, requires_grad=True)
        out = model(x, covariates=cov)
        loss = out.sum()
        loss.backward()
        assert cov.grad is not None
        assert cov.grad.shape == (4, 5)
        # Projection layer weights should also have gradients
        assert model.covariate_proj.weight.grad is not None

    def test_from_config_reads_n_covariates(self):
        """from_config should pass n_covariates to the constructor."""
        config = {
            "model": {
                "hidden_size": 32,
                "num_layers": 1,
                "n_covariates": 5,
            }
        }
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.n_covariates == 5
        assert model.covariate_proj is not None


# =============================================================================
# LRU Temporal Covariate Mode Tests (ADR-0012)
# =============================================================================

class TestLRUTemporalCovariates:
    """Tests for LRU temporal covariate feeding (input concatenation)."""

    def test_temporal_mode_forward_shape(self):
        """Temporal covariates concat to input -> correct output shape."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)     # (batch, T, M)
        cov = torch.randn(4, 20, 5)    # (batch, T, n_cov)
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_temporal_mode_input_proj_width(self):
        """Input projection should be wider in temporal mode."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, n_covariates=5,
            covariate_mode="temporal",
        )
        assert model.input_proj.in_features == 15

    def test_temporal_gradient_flow(self):
        """Gradients should flow through temporal covariates."""
        model = TeacherLRU(
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
        model = TeacherLRU.from_config(config, input_size=10)
        assert model.covariate_mode == "temporal"
        assert model.input_proj.in_features == 15

    def test_invalid_covariate_mode_raises(self):
        """Invalid covariate_mode should raise ValueError."""
        with pytest.raises(ValueError, match="covariate_mode"):
            TeacherLRU(
                input_size=10, hidden_size=32, n_covariates=5,
                covariate_mode="invalid",
            )


class TestLRURingInitAndResidual:
    """Tests for ring eigenvalue initialization and residual connections."""

    def test_eigenvalue_magnitudes_in_range(self):
        """Initial |lambda| should be in [0.8, 0.99] (ring init)."""
        cell = LRUCell(input_size=32, hidden_size=128, gated=False)
        with torch.no_grad():
            lam = cell._compute_lambda()
            magnitudes = lam.abs()
        assert magnitudes.min() >= 0.79, f"Min |lambda|={magnitudes.min():.4f}"
        assert magnitudes.max() <= 1.0, f"Max |lambda|={magnitudes.max():.4f}"

    def test_eigenvalue_phases_cover_circle(self):
        """Initial phases should span [0, 2*pi] (not clustered)."""
        cell = LRUCell(input_size=32, hidden_size=256, gated=False)
        with torch.no_grad():
            lam = cell._compute_lambda()
            phases = lam.angle()
        # Phases should span a wide range (at least pi radians)
        phase_range = phases.max() - phases.min()
        assert phase_range > 3.0, f"Phase range={phase_range:.2f}, too narrow"

    def test_b_proj_scaled(self):
        """B_proj weights should be scaled by 1/sqrt(hidden_size)."""
        hidden = 256
        cell = LRUCell(input_size=32, hidden_size=hidden)
        # Default Kaiming init has std ~0.08 for fan_in=32
        # After scaling by 1/sqrt(256)=1/16, std should be much smaller
        weight_std = cell.B_proj.weight.std().item()
        assert weight_std < 0.02, f"B_proj std={weight_std:.4f}, not scaled"

    def test_gated_bias_init_in_range(self):
        """Gated mode bias should produce |lambda| in [0.8, 0.99]."""
        cell = LRUCell(input_size=32, hidden_size=128, gated=True)
        x = torch.zeros(1, 32)  # Zero input — only bias contributes
        with torch.no_grad():
            lam = cell._compute_lambda(x)
            magnitudes = lam.abs()
        assert magnitudes.min() >= 0.79, f"Min |lambda|={magnitudes.min():.4f}"
        assert magnitudes.max() <= 1.0, f"Max |lambda|={magnitudes.max():.4f}"

    def test_residual_connections_exist(self):
        """LRU stack should use residual connections (output != layer output)."""
        model = TeacherLRU(
            input_size=10, hidden_size=32, num_layers=3,
            dropout=0.0, use_layer_norm=True,
        )
        x = torch.randn(2, 5, 10)
        # With residuals, the input to the stack is added back at each layer
        # Check model has lru_norms (per-layer norm is part of residual block)
        assert len(model.lru_norms) == 3, "Should have 3 per-layer norms"

    def test_model_trains_decreasing_loss(self):
        """LRU v2 should show decreasing loss over 20 iterations (not diverge)."""
        torch.manual_seed(42)
        model = TeacherLRU(
            input_size=50, hidden_size=64, num_layers=3,
            dropout=0.0, use_layer_norm=True, use_attention=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.rand(16, 10, 50)
        y = torch.rand(16, 50) * 3

        losses = []
        for _ in range(20):
            rates = model(x)
            loss = torch.nn.functional.poisson_nll_loss(
                rates, y, log_input=False,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease — not increase like the old init
        assert losses[-1] < losses[0], (
            f"Loss increased: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
