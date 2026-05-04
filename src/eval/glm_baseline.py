"""
Poisson GLM baseline for spike-count forecasting.

Implements a per-neuron Poisson Generalized Linear Model (GLM) with
spike-history filters as the standard computational neuroscience baseline.
Each neuron gets its own independent GLM that predicts its next-bin firing
rate from the flattened history window of all neurons' past activity.

This is the classical benchmark that any deep recurrent model must beat
to demonstrate it captures non-trivial neural dynamics beyond what a
linear filter can explain.

Architecture:
    For each neuron i:
        log(λ_i) = w_i^T · flatten(X_t) + b_i
    where X_t is the (T, M) history window, flattened to (T*M,) features.

Uses scikit-learn's PoissonRegressor (L2-regularized GLM with log link).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def fit_glm_per_neuron(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1e-4,
    max_iter: int = 300,
) -> List[Any]:
    """
    Fit one Poisson GLM per neuron on training data.

    Args:
        X_train: Flattened history windows, shape (N_samples, T * M).
        y_train: Target spike counts, shape (N_samples, M).
        alpha: L2 regularization strength.
        max_iter: Maximum solver iterations per neuron.

    Returns:
        List of M fitted PoissonRegressor models (one per neuron).
    """
    from sklearn.linear_model import PoissonRegressor

    n_samples, n_neurons = y_train.shape
    models = []

    for i in range(n_neurons):
        # Skip neurons with zero variance (e.g., padded channels)
        if y_train[:, i].sum() == 0:
            models.append(None)
            continue

        glm = PoissonRegressor(
            alpha=alpha,
            max_iter=max_iter,
            fit_intercept=True,
        )
        glm.fit(X_train, y_train[:, i])
        models.append(glm)

    n_fitted = sum(1 for m in models if m is not None)
    logger.info(
        "Fitted %d / %d Poisson GLMs (skipped %d zero-activity channels)",
        n_fitted, n_neurons, n_neurons - n_fitted,
    )
    return models


def predict_glm(
    models: List[Any],
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Predict firing rates for all neurons using fitted GLMs.

    Args:
        models: List of M fitted PoissonRegressor models (None for skipped).
        X_test: Flattened history windows, shape (N_samples, T * M).

    Returns:
        Predicted rates, shape (N_samples, M). Skipped neurons get rate=0.
    """
    n_samples = X_test.shape[0]
    n_neurons = len(models)
    predictions = np.zeros((n_samples, n_neurons), dtype=np.float64)

    for i, model in enumerate(models):
        if model is not None:
            predictions[:, i] = model.predict(X_test)

    return predictions


def evaluate_glm(
    models: List[Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate fitted GLMs on test data.

    Computes Poisson NLL, Pearson r, MAE, and MSE — the same metrics
    used for the deep models, enabling direct comparison.

    Args:
        models: List of M fitted PoissonRegressor models.
        X_test: Flattened history windows, shape (N_samples, T * M).
        y_test: Target spike counts, shape (N_samples, M).

    Returns:
        Dict with keys: poisson_nll, pearson_r, mae, mse.
    """
    predictions = predict_glm(models, X_test)

    # Identify active channels (non-None models with nonzero predictions)
    active_mask = np.array([m is not None for m in models])
    if not active_mask.any():
        logger.warning("No active GLM channels — returning NaN metrics")
        return {
            "poisson_nll": float("nan"),
            "pearson_r": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
        }

    pred_active = predictions[:, active_mask]
    true_active = y_test[:, active_mask]

    # --- Poisson NLL ---
    # loss = lambda - y * log(lambda + eps), averaged over samples and channels
    eps = 1e-8
    poisson_nll = np.mean(
        pred_active - true_active * np.log(pred_active + eps)
    )

    # --- Pearson r (per-channel, then averaged) ---
    n_active = pred_active.shape[1]
    rs = np.zeros(n_active)
    for j in range(n_active):
        p = pred_active[:, j]
        t = true_active[:, j]
        # Skip constant channels (zero variance)
        if np.std(p) < eps or np.std(t) < eps:
            rs[j] = 0.0
        else:
            rs[j] = np.corrcoef(p, t)[0, 1]
    pearson_r = float(np.nanmean(rs))

    # --- MAE and MSE ---
    mae_val = float(np.mean(np.abs(pred_active - true_active)))
    mse_val = float(np.mean((pred_active - true_active) ** 2))

    return {
        "poisson_nll": float(poisson_nll),
        "pearson_r": pearson_r,
        "mae": mae_val,
        "mse": mse_val,
    }


def flatten_windows(
    X: np.ndarray,
    y: np.ndarray,
    n_neurons_real: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten history windows for GLM input.

    The deep models receive (batch, T, M) tensors. The GLM needs
    (batch, T*M) flattened features. This function also handles
    trimming padded channels if n_neurons_real is provided.

    Args:
        X: History windows, shape (N, T, M_padded).
        y: Targets, shape (N, M_padded).
        n_neurons_real: If set, trim to first n_neurons_real channels.

    Returns:
        (X_flat, y_trimmed) where X_flat is (N, T * M_real) and
        y_trimmed is (N, M_real).
    """
    if n_neurons_real is not None:
        X = X[:, :, :n_neurons_real]
        y = y[:, :n_neurons_real]

    n_samples, t_steps, m_channels = X.shape
    X_flat = X.reshape(n_samples, t_steps * m_channels)
    return X_flat, y
