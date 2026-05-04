"""
Tests for the Transformer baseline teacher model.

Validates:
    - TeacherTransformer forward pass shape and positivity
    - Causal masking (future tokens don't affect past predictions)
    - from_config() factory method
    - create_teacher_model() factory dispatching for 'transformer'
    - Interface compatibility with TeacherLSTM / TeacherLRU
    - Gradient flow
    - Covariate support (additive and temporal modes)
    - Output distributions (negbin/zip auxiliary heads)
    - Session-specific output heads
"""

import pytest
import torch

from src.models.transformer_baseline import (
    SinusoidalPositionalEncoding,
    TeacherTransformer,
)
from src.models.teacher import TeacherLSTM, create_teacher_model
from src.models.lru import TeacherLRU


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def transformer_model():
    """Create a small TeacherTransformer for testing."""
    return TeacherTransformer(
        input_size=10,
        hidden_size=32,
        num_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.1,
    )


@pytest.fixture
def batch():
    """Create a dummy batch: (batch=4, T=20, M=10)."""
    torch.manual_seed(42)
    return torch.randn(4, 20, 10)


# =============================================================================
# Positional Encoding Tests
# =============================================================================

class TestSinusoidalPositionalEncoding:
    """Tests for the sinusoidal positional encoding module."""

    def test_output_shape(self):
        """Positional encoding should preserve input shape."""
        pe = SinusoidalPositionalEncoding(d_model=32, dropout=0.0)
        x = torch.randn(4, 20, 32)
        out = pe(x)
        assert out.shape == (4, 20, 32)

    def test_different_positions_differ(self):
        """Different positions should get different encodings."""
        pe = SinusoidalPositionalEncoding(d_model=32, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Adjacent positions should not be identical
        assert not torch.allclose(out[0, 0], out[0, 1]), (
            "Adjacent positions should have different encodings"
        )

    def test_encoding_is_deterministic(self):
        """Positional encoding (without dropout) should be deterministic."""
        pe = SinusoidalPositionalEncoding(d_model=32, dropout=0.0)
        x = torch.zeros(1, 5, 32)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)

    def test_long_sequence(self):
        """Should handle sequence lengths up to max_len."""
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=1000)
        x = torch.randn(1, 500, 32)
        out = pe(x)
        assert out.shape == (1, 500, 32)


# =============================================================================
# TeacherTransformer Forward Pass Tests
# =============================================================================

class TestTeacherTransformerForward:
    """Tests for TeacherTransformer forward pass correctness."""

    def test_output_shape(self, transformer_model, batch):
        """Output shape should be (batch, M) — same as LSTM/LRU."""
        output = transformer_model(batch)
        assert output.shape == (4, 10)

    def test_output_positive(self, transformer_model, batch):
        """All output rates must be positive (softplus enforces this)."""
        output = transformer_model(batch)
        assert (output > 0).all(), "Softplus should ensure all outputs > 0"

    def test_output_finite(self, transformer_model, batch):
        """Output should not contain NaN or inf."""
        output = transformer_model(batch)
        assert torch.isfinite(output).all(), "Output contains NaN or inf"

    def test_single_sample(self, transformer_model):
        """Model should handle batch_size=1."""
        x = torch.randn(1, 20, 10)
        output = transformer_model(x)
        assert output.shape == (1, 10)

    def test_different_history_lengths(self, transformer_model):
        """Model should handle different history window sizes."""
        for T in [1, 5, 10, 50]:
            x = torch.randn(2, T, 10)
            output = transformer_model(x)
            assert output.shape == (2, 10), f"Failed for T={T}"

    def test_zero_input(self, transformer_model):
        """Model should handle zero inputs gracefully."""
        x = torch.zeros(2, 20, 10)
        output = transformer_model(x)
        assert torch.isfinite(output).all()
        assert (output > 0).all()

    def test_large_input_values(self, transformer_model):
        """Model should not produce NaN/inf for large inputs."""
        x = torch.ones(2, 20, 10) * 100.0
        output = transformer_model(x)
        assert torch.isfinite(output).all(), "Outputs contain NaN or inf"

    def test_custom_output_size(self):
        """Custom output_size should work."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, output_size=5,
        )
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 5)

    def test_deterministic_eval(self, transformer_model, batch):
        """Same input should give same output in eval mode."""
        transformer_model.eval()
        with torch.no_grad():
            out1 = transformer_model(batch).clone()
            out2 = transformer_model(batch).clone()
        assert torch.allclose(out1, out2), (
            "Non-deterministic output in eval mode"
        )


# =============================================================================
# Causal Masking Tests
# =============================================================================

class TestCausalMasking:
    """Tests to verify the causal attention mask works correctly."""

    def test_causal_mask_shape(self, transformer_model):
        """Causal mask should be (T, T)."""
        mask = transformer_model._generate_causal_mask(20, torch.device("cpu"))
        assert mask.shape == (20, 20)

    def test_causal_mask_lower_triangular(self, transformer_model):
        """Causal mask should be zero below/on diagonal, -inf above."""
        mask = transformer_model._generate_causal_mask(5, torch.device("cpu"))
        # Lower triangle + diagonal should be 0.0 (allowed)
        for i in range(5):
            for j in range(i + 1):
                assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"
        # Upper triangle should be -inf (blocked)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j] == float("-inf"), (
                    f"mask[{i},{j}] should be -inf"
                )

    def test_future_independence(self):
        """Changing future tokens should NOT change earlier predictions."""
        torch.manual_seed(42)
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4,
            num_layers=2, d_ff=64, dropout=0.0,
        )
        model.eval()

        x1 = torch.randn(1, 20, 10)
        x2 = x1.clone()
        # Modify only the LAST 5 timesteps
        x2[0, 15:, :] = torch.randn(5, 10)

        with torch.no_grad():
            # Get intermediate representations (not just final output)
            proj1 = model.input_norm(model.input_proj(x1))
            proj1 = model.pos_encoder(proj1)
            mask = model._generate_causal_mask(20, proj1.device)
            enc1 = model.transformer_encoder(proj1, mask=mask, is_causal=True)

            proj2 = model.input_norm(model.input_proj(x2))
            proj2 = model.pos_encoder(proj2)
            enc2 = model.transformer_encoder(proj2, mask=mask, is_causal=True)

        # Earlier positions (0-14) should be IDENTICAL between x1 and x2
        # because the causal mask prevents them from seeing positions 15-19
        for t in range(15):
            assert torch.allclose(enc1[0, t], enc2[0, t], atol=1e-5), (
                f"Position {t} changed when only future tokens were modified"
            )


# =============================================================================
# Gradient Tests
# =============================================================================

class TestTeacherTransformerGradients:
    """Tests for gradient flow through TeacherTransformer."""

    def test_gradient_flow(self, transformer_model, batch):
        """Gradients should flow to all parameters."""
        output = transformer_model(batch)
        loss = output.sum()
        loss.backward()

        for name, param in transformer_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for {name}"
            )

    def test_gradient_not_zero(self, transformer_model, batch):
        """At least some gradients should be non-zero."""
        output = transformer_model(batch)
        loss = output.mean()
        loss.backward()

        has_nonzero = False
        for param in transformer_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"


# =============================================================================
# from_config Tests
# =============================================================================

class TestTeacherTransformerFromConfig:
    """Tests for config-based construction."""

    def test_from_config_defaults(self):
        """from_config should handle missing keys with defaults."""
        config = {}
        model = TeacherTransformer.from_config(config, input_size=10)
        assert model.hidden_size == 256
        assert model.num_layers == 3
        assert model.n_heads == 8

    def test_from_config_custom(self):
        """from_config should respect custom values."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "n_heads": 4,
                "d_ff": 128,
                "dropout": 0.3,
            }
        }
        model = TeacherTransformer.from_config(config, input_size=10)
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.n_heads == 4
        assert model.d_ff == 128

        # Verify forward pass works
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)

    def test_from_config_with_attention_readout(self):
        """from_config should handle attention readout."""
        config = {
            "model": {
                "hidden_size": 32,
                "n_heads": 4,
                "use_layer_norm": True,
                "use_attention": True,
            }
        }
        model = TeacherTransformer.from_config(config, input_size=10)
        assert model.use_attention is True
        assert model.attn_query is not None

        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)


# =============================================================================
# Factory Dispatch Tests
# =============================================================================

class TestFactoryTransformerDispatch:
    """Tests for create_teacher_model() with architecture='transformer'."""

    def test_factory_creates_transformer(self):
        """architecture='transformer' should create TeacherTransformer."""
        config = {
            "model": {
                "architecture": "transformer",
                "hidden_size": 32,
                "n_heads": 4,
            }
        }
        model = create_teacher_model(config, input_size=10)
        assert isinstance(model, TeacherTransformer)

    def test_factory_transformer_forward(self):
        """Factory-created Transformer should produce valid output."""
        config = {
            "model": {
                "architecture": "transformer",
                "hidden_size": 32,
                "n_heads": 4,
            }
        }
        model = create_teacher_model(config, input_size=10)
        x = torch.randn(2, 20, 10)
        output = model(x)
        assert output.shape == (2, 10)
        assert (output > 0).all()


# =============================================================================
# Interface Compatibility Tests
# =============================================================================

class TestTransformerInterfaceCompat:
    """Verify TeacherTransformer is a drop-in for LSTM/LRU."""

    def test_same_forward_signature(self):
        """All three models should accept (batch, T, M) → (batch, M)."""
        lstm = TeacherLSTM(input_size=10, hidden_size=32)
        lru = TeacherLRU(input_size=10, hidden_size=32)
        tfm = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
        )
        x = torch.randn(4, 20, 10)

        lstm_out = lstm(x)
        lru_out = lru(x)
        tfm_out = tfm(x)

        assert lstm_out.shape == lru_out.shape == tfm_out.shape == (4, 10)

    def test_h0_param_accepted(self):
        """TeacherTransformer.forward should accept h0 param (ignored)."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
        )
        x = torch.randn(2, 20, 10)
        output = model(x, h0=None)
        assert output.shape == (2, 10)

    def test_has_get_aux_output(self):
        """All three models should have get_aux_output() method."""
        lstm = TeacherLSTM(
            input_size=10, hidden_size=32, output_distribution="negbin",
        )
        tfm = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            output_distribution="negbin",
        )
        x = torch.randn(2, 20, 10)

        lstm(x)
        tfm(x)

        assert lstm.get_aux_output() is not None
        assert tfm.get_aux_output() is not None
        assert lstm.get_aux_output().shape == tfm.get_aux_output().shape


# =============================================================================
# Distribution Tests
# =============================================================================

class TestTeacherTransformerDistributions:
    """Tests for output distribution variants."""

    def test_poisson_default(self):
        """Default distribution should be poisson."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
        )
        assert model.output_distribution == "poisson"
        assert model.aux_proj is None

    def test_negbin_output(self):
        """NegBin should produce rates and dispersion."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            output_distribution="negbin",
        )
        x = torch.randn(4, 20, 10)
        rates = model(x)
        dispersion = model.get_aux_output()

        assert rates.shape == (4, 10)
        assert dispersion is not None
        assert dispersion.shape == (4, 10)
        assert (rates > 0).all()
        assert (dispersion > 0).all()

    def test_zip_output(self):
        """ZIP should produce rates and gate."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            output_distribution="zip",
        )
        x = torch.randn(4, 20, 10)
        rates = model(x)
        gate = model.get_aux_output()

        assert rates.shape == (4, 10)
        assert gate is not None
        assert gate.shape == (4, 10)
        assert (gate >= 0).all()
        assert (gate <= 1).all()

    def test_invalid_distribution_raises(self):
        """Invalid distribution should raise ValueError."""
        with pytest.raises(ValueError, match="output_distribution"):
            TeacherTransformer(
                input_size=10, hidden_size=32, n_heads=4,
                output_distribution="gamma",
            )


# =============================================================================
# Covariate Tests
# =============================================================================

class TestTransformerCovariates:
    """Tests for covariate support (matching LSTM/LRU behavior)."""

    def test_additive_covariates(self):
        """Additive covariates should produce correct output shape."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            n_covariates=5, covariate_mode="additive",
        )
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 5)
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_temporal_covariates(self):
        """Temporal covariates should produce correct output shape."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            n_covariates=5, covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 20, 5)
        out = model(x, covariates=cov)
        assert out.shape == (4, 10)

    def test_temporal_gradient_flow(self):
        """Gradients should flow through temporal covariates."""
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            n_covariates=5, covariate_mode="temporal",
        )
        x = torch.randn(4, 20, 10)
        cov = torch.randn(4, 20, 5, requires_grad=True)
        out = model(x, covariates=cov)
        loss = out.sum()
        loss.backward()
        assert cov.grad is not None
        assert cov.grad.shape == (4, 20, 5)


# =============================================================================
# Session-Specific Heads Tests
# =============================================================================

class TestTransformerSessionHeads:
    """Tests for session-specific output projections."""

    def test_session_heads_output_shape(self):
        """Session heads should produce (batch, N_i) for each session."""
        session_dims = {"s0": 50, "s1": 100}
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            session_dims=session_dims,
        )

        x = torch.randn(4, 20, 10)
        out_s0 = model(x, session_id="s0")
        out_s1 = model(x, session_id="s1")

        assert out_s0.shape == (4, 50)
        assert out_s1.shape == (4, 100)

    def test_session_heads_positive(self):
        """Session head outputs should be positive (softplus)."""
        session_dims = {"s0": 50}
        model = TeacherTransformer(
            input_size=10, hidden_size=32, n_heads=4, d_ff=64,
            session_dims=session_dims,
        )
        x = torch.randn(4, 20, 10)
        out = model(x, session_id="s0")
        assert (out > 0).all()


# =============================================================================
# Validation Tests
# =============================================================================

class TestTransformerValidation:
    """Tests for input validation and error handling."""

    def test_n_heads_must_divide_hidden_size(self):
        """n_heads must evenly divide hidden_size."""
        with pytest.raises(ValueError, match="divisible"):
            TeacherTransformer(
                input_size=10, hidden_size=32, n_heads=5,
            )

    def test_invalid_covariate_mode_raises(self):
        """Invalid covariate_mode should raise ValueError."""
        with pytest.raises(ValueError, match="covariate_mode"):
            TeacherTransformer(
                input_size=10, hidden_size=32, n_heads=4,
                n_covariates=5, covariate_mode="invalid",
            )

    def test_training_decreases_loss(self):
        """Transformer should show decreasing loss over training steps."""
        torch.manual_seed(42)
        model = TeacherTransformer(
            input_size=50, hidden_size=64, num_layers=2,
            n_heads=4, d_ff=128, dropout=0.0,
            use_layer_norm=True,
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

        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Loss increased: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
