"""
Tests for src/eval/glm_baseline.py

Validates the Poisson GLM baseline with synthetic data where
expected behavior can be verified analytically.
"""

import numpy as np
import pytest

from src.eval.glm_baseline import (
    evaluate_glm,
    fit_glm_per_neuron,
    flatten_windows,
    predict_glm,
)


# =============================================================================
# Test data fixtures
# =============================================================================

@pytest.fixture
def synthetic_poisson_data():
    """
    Generate synthetic Poisson data with known rates.

    Creates a dataset where neuron 0 has a high rate (5.0) and neuron 1
    has a low rate (0.5). The GLM should learn to predict these rates
    from the history window.
    """
    rng = np.random.RandomState(42)
    n_samples = 500
    t_steps = 10
    n_neurons = 3

    # History windows: random Poisson counts
    X = rng.poisson(lam=2.0, size=(n_samples, t_steps, n_neurons)).astype(
        np.float64
    )

    # Targets: Poisson counts with rates correlated to input sums
    # Neuron i's rate = 0.1 * sum(X[:, :, i]) to create learnable signal
    y = np.zeros((n_samples, n_neurons), dtype=np.float64)
    for i in range(n_neurons):
        rates = 0.1 * X[:, :, i].sum(axis=1) + 0.5
        y[:, i] = rng.poisson(lam=rates)

    return X, y


@pytest.fixture
def simple_flat_data():
    """Pre-flattened synthetic data for direct GLM fitting."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 30  # T * M
    n_neurons = 3

    X_flat = rng.poisson(lam=2.0, size=(n_samples, n_features)).astype(
        np.float64
    )
    y = rng.poisson(lam=1.0, size=(n_samples, n_neurons)).astype(np.float64)

    return X_flat, y


# =============================================================================
# flatten_windows tests
# =============================================================================

class TestFlattenWindows:
    """Tests for flatten_windows()."""

    def test_shape_no_trim(self):
        """Flatten (N, T, M) -> (N, T*M) without trimming."""
        X = np.ones((100, 10, 5))
        y = np.ones((100, 5))
        X_flat, y_out = flatten_windows(X, y)
        assert X_flat.shape == (100, 50)
        assert y_out.shape == (100, 5)

    def test_shape_with_trim(self):
        """Flatten and trim to n_neurons_real channels."""
        X = np.ones((100, 10, 20))  # 20 padded channels
        y = np.ones((100, 20))
        X_flat, y_out = flatten_windows(X, y, n_neurons_real=8)
        assert X_flat.shape == (100, 80)  # 10 * 8
        assert y_out.shape == (100, 8)

    def test_values_preserved(self):
        """Check that flattening preserves values correctly."""
        X = np.arange(12).reshape(1, 3, 4).astype(np.float64)
        y = np.array([[10, 20, 30, 40]], dtype=np.float64)
        X_flat, y_out = flatten_windows(X, y)
        # X_flat should be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        np.testing.assert_array_equal(X_flat[0], np.arange(12))
        np.testing.assert_array_equal(y_out, y)


# =============================================================================
# fit_glm_per_neuron tests
# =============================================================================

class TestFitGLMPerNeuron:
    """Tests for fit_glm_per_neuron()."""

    def test_returns_correct_count(self, simple_flat_data):
        """Should return one model per neuron."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        assert len(models) == y.shape[1]

    def test_skips_zero_channels(self):
        """Zero-activity neurons should get None model."""
        rng = np.random.RandomState(42)
        X_flat = rng.poisson(lam=2.0, size=(100, 20)).astype(np.float64)
        # Neuron 1 has all zeros (padded channel)
        y = rng.poisson(lam=1.0, size=(100, 3)).astype(np.float64)
        y[:, 1] = 0.0

        models = fit_glm_per_neuron(X_flat, y)
        assert models[0] is not None
        assert models[1] is None  # Skipped
        assert models[2] is not None

    def test_fitted_models_have_coef(self, simple_flat_data):
        """Fitted models should have learned coefficients."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        for m in models:
            if m is not None:
                assert hasattr(m, "coef_")
                assert len(m.coef_) == X_flat.shape[1]


# =============================================================================
# predict_glm tests
# =============================================================================

class TestPredictGLM:
    """Tests for predict_glm()."""

    def test_predictions_non_negative(self, simple_flat_data):
        """GLM predictions must be non-negative (Poisson link)."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        preds = predict_glm(models, X_flat)
        assert np.all(preds >= 0), "Poisson GLM predictions must be non-negative"

    def test_output_shape(self, simple_flat_data):
        """Output shape should be (N_samples, M)."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        preds = predict_glm(models, X_flat)
        assert preds.shape == y.shape

    def test_skipped_neurons_predict_zero(self):
        """Neurons with None models should predict zero."""
        rng = np.random.RandomState(42)
        X_flat = rng.poisson(lam=2.0, size=(50, 10)).astype(np.float64)
        y = np.zeros((50, 2), dtype=np.float64)
        y[:, 0] = rng.poisson(lam=1.0, size=50)
        # Neuron 1 is all zeros → model will be None

        models = fit_glm_per_neuron(X_flat, y)
        preds = predict_glm(models, X_flat)
        np.testing.assert_array_equal(preds[:, 1], 0.0)


# =============================================================================
# evaluate_glm tests
# =============================================================================

class TestEvaluateGLM:
    """Tests for evaluate_glm()."""

    def test_returns_all_metrics(self, simple_flat_data):
        """Should return dict with all four keys."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        metrics = evaluate_glm(models, X_flat, y)
        assert "poisson_nll" in metrics
        assert "pearson_r" in metrics
        assert "mae" in metrics
        assert "mse" in metrics

    def test_metrics_are_finite(self, simple_flat_data):
        """All metric values should be finite numbers."""
        X_flat, y = simple_flat_data
        models = fit_glm_per_neuron(X_flat, y)
        metrics = evaluate_glm(models, X_flat, y)
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_train_pearson_r_positive(self, synthetic_poisson_data):
        """GLM should achieve positive Pearson r on correlated training data."""
        X, y = synthetic_poisson_data
        X_flat, y_trim = flatten_windows(X, y)
        models = fit_glm_per_neuron(X_flat, y_trim)
        metrics = evaluate_glm(models, X_flat, y_trim)
        # On correlated data, GLM should find some signal
        assert metrics["pearson_r"] > 0.0, (
            f"Expected positive Pearson r on correlated data, got {metrics['pearson_r']}"
        )

    def test_all_none_models_return_nan(self):
        """When all models are None, metrics should be NaN."""
        models = [None, None, None]
        X = np.ones((10, 5))
        y = np.ones((10, 3))
        metrics = evaluate_glm(models, X, y)
        assert np.isnan(metrics["poisson_nll"])
        assert np.isnan(metrics["pearson_r"])


# =============================================================================
# Integration test
# =============================================================================

class TestGLMIntegration:
    """End-to-end integration test for the GLM pipeline."""

    def test_full_pipeline(self, synthetic_poisson_data):
        """
        Full pipeline: flatten → fit → predict → evaluate.

        Uses synthetic data where neurons have learnable rate structure,
        so the GLM should achieve meaningful metrics.
        """
        X, y = synthetic_poisson_data

        # Split into train/test
        n_train = 400
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Flatten
        X_train_flat, y_train_trim = flatten_windows(X_train, y_train)
        X_test_flat, y_test_trim = flatten_windows(X_test, y_test)

        # Fit
        models = fit_glm_per_neuron(X_train_flat, y_train_trim)
        assert len(models) == y_train.shape[1]

        # Predict
        preds = predict_glm(models, X_test_flat)
        assert preds.shape == y_test_trim.shape
        assert np.all(preds >= 0)

        # Evaluate
        metrics = evaluate_glm(models, X_test_flat, y_test_trim)
        assert np.isfinite(metrics["poisson_nll"])
        assert np.isfinite(metrics["pearson_r"])
        assert metrics["mae"] > 0  # Not trivially perfect
        assert metrics["mse"] > 0
