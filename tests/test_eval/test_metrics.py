"""
Tests for src/eval/metrics.py

Tests all metric functions and naive baselines with known-answer verification.
Uses hand-crafted inputs where expected outputs can be computed analytically.
"""

import numpy as np
import pytest
import torch

from src.eval.metrics import (
    compute_all_baselines,
    mae,
    mean_rate_baseline,
    mse,
    negative_binomial_nll,
    pearson_r,
    persistence_baseline,
    poisson_nll,
    r_squared,
    zero_inflated_poisson_nll,
)


# =============================================================================
# Metric function tests
# =============================================================================

class TestPoissonNLL:
    """Tests for poisson_nll()."""

    def test_perfect_prediction(self):
        """When predicted rate == target count, NLL should be low but not zero."""
        # For Poisson NLL with log_input=False: loss = pred - target * log(pred + eps)
        target = torch.tensor([[2.0, 3.0]])
        predicted = torch.tensor([[2.0, 3.0]])
        loss = poisson_nll(predicted, target, log_input=False)
        assert loss.item() > 0  # Not zero due to Poisson formula
        assert loss.item() < 1.0  # But should be reasonably low

    def test_bad_prediction_has_higher_loss(self):
        """Bad predictions should have higher NLL than good ones."""
        target = torch.tensor([[2.0, 3.0, 1.0]])
        good_pred = torch.tensor([[2.0, 3.0, 1.0]])
        bad_pred = torch.tensor([[10.0, 0.1, 10.0]])
        good_loss = poisson_nll(good_pred, target, log_input=False)
        bad_loss = poisson_nll(bad_pred, target, log_input=False)
        assert bad_loss > good_loss

    def test_log_input_mode(self):
        """Log input mode: predicted is log(rate)."""
        target = torch.tensor([[2.0, 3.0]])
        # log(2) ≈ 0.693, log(3) ≈ 1.099
        predicted_log = torch.log(torch.tensor([[2.0, 3.0]]))
        loss = poisson_nll(predicted_log, target, log_input=True)
        assert loss.item() > 0

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        target = torch.tensor([[1.0, 2.0, 3.0]])
        predicted = torch.tensor([[1.0, 2.0, 3.0]])
        loss = poisson_nll(predicted, target, log_input=False)
        assert loss.dim() == 0


class TestPearsonR:
    """Tests for pearson_r()."""

    def test_perfect_positive_correlation(self):
        """Identical sequences should have r = 1.0."""
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        r = pearson_r(x, x)
        assert abs(r.item() - 1.0) < 1e-5

    def test_perfect_negative_correlation(self):
        """Perfectly inversely related sequences should have r = -1.0."""
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0]])
        r = pearson_r(x, y)
        assert abs(r.item() - (-1.0)) < 1e-5

    def test_uncorrelated(self):
        """Random permutations over many samples should have r ≈ 0."""
        torch.manual_seed(42)
        x = torch.randn(1000, 1)
        y = torch.randn(1000, 1)
        r = pearson_r(x, y)
        assert abs(r.item()) < 0.1  # Should be near zero

    def test_known_answer(self):
        """
        Known answer: x = [1, 2, 3], y = [1, 3, 5]
        Pearson r = 1.0 (perfectly linearly related)
        """
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([[1.0], [3.0], [5.0]])
        r = pearson_r(x, y)
        assert abs(r.item() - 1.0) < 1e-5

    def test_per_channel(self):
        """Per-channel mode should return (M,) tensor."""
        # Ch0: [1,2,3] vs [1,2,3] → r = +1.0
        # Ch1: [5,4,3] vs [1,2,3] → r = -1.0 (one decreases, other increases)
        x = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
        y = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        r = pearson_r(x, y, per_channel=True)
        assert r.shape == (2,)
        # Channel 0: perfect positive
        assert abs(r[0].item() - 1.0) < 1e-5
        # Channel 1: perfect negative
        assert abs(r[1].item() - (-1.0)) < 1e-5

    def test_1d_input(self):
        """1D inputs should work (auto-unsqueeze)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        r = pearson_r(x, y)
        assert abs(r.item() - 1.0) < 1e-5


class TestRSquared:
    """Tests for r_squared()."""

    def test_perfect_prediction(self):
        """R² of identical tensors should be 1.0."""
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        r2 = r_squared(x, x)
        assert abs(r2.item() - 1.0) < 1e-5

    def test_scaled_prediction_penalized(self):
        """R² should penalize scale mismatch, unlike Pearson R.

        If predicted = 10 * target, Pearson R = 1.0 but R² < 1.
        This is the key difference: R² catches 'right pattern, wrong scale'.
        """
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaled = target * 10.0
        r = pearson_r(scaled, target)
        r2 = r_squared(scaled, target)
        assert abs(r.item() - 1.0) < 1e-5, "Pearson R should still be 1.0"
        assert r2.item() < 0.0, "R² should be negative for 10x scale mismatch"

    def test_known_answer(self):
        """Known answer: pred=[2,4,6], target=[1,2,3].

        SS_tot = (1-2)² + (2-2)² + (3-2)² = 2
        SS_res = (1-2)² + (2-4)² + (3-6)² = 1+4+9 = 14
        R² = 1 - 14/2 = -6.0
        """
        pred = torch.tensor([[2.0], [4.0], [6.0]])
        target = torch.tensor([[1.0], [2.0], [3.0]])
        r2 = r_squared(pred, target)
        assert abs(r2.item() - (-6.0)) < 1e-5

    def test_mean_prediction_gives_zero(self):
        """Predicting the mean should give R² = 0."""
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        mean_pred = torch.full_like(target, target.mean().item())
        r2 = r_squared(mean_pred, target)
        assert abs(r2.item()) < 1e-5

    def test_per_channel(self):
        """Per-channel mode should return (M,) tensor."""
        pred = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
        target = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
        r2 = r_squared(pred, target, per_channel=True)
        assert r2.shape == (2,)
        assert abs(r2[0].item() - 1.0) < 1e-5
        assert abs(r2[1].item() - 1.0) < 1e-5

    def test_1d_input(self):
        """1D inputs should work (auto-unsqueeze)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        r2 = r_squared(x, x)
        assert abs(r2.item() - 1.0) < 1e-5


class TestMAE:
    """Tests for mae()."""

    def test_perfect_prediction(self):
        """MAE of identical tensors should be 0."""
        x = torch.tensor([1.0, 2.0, 3.0])
        assert mae(x, x).item() == 0.0

    def test_known_answer(self):
        """
        Known answer: pred=[1,3,5], target=[2,3,6]
        MAE = (|1-2| + |3-3| + |5-6|) / 3 = (1+0+1)/3 = 2/3 ≈ 0.6667
        """
        pred = torch.tensor([1.0, 3.0, 5.0])
        target = torch.tensor([2.0, 3.0, 6.0])
        result = mae(pred, target)
        assert abs(result.item() - 2 / 3) < 1e-5

    def test_symmetric(self):
        """MAE(a, b) should equal MAE(b, a)."""
        a = torch.tensor([1.0, 4.0])
        b = torch.tensor([3.0, 1.0])
        assert abs(mae(a, b).item() - mae(b, a).item()) < 1e-6


class TestMSE:
    """Tests for mse()."""

    def test_perfect_prediction(self):
        """MSE of identical tensors should be 0."""
        x = torch.tensor([1.0, 2.0, 3.0])
        assert mse(x, x).item() == 0.0

    def test_known_answer(self):
        """
        Known answer: pred=[1,3,5], target=[2,3,6]
        MSE = ((1-2)² + (3-3)² + (5-6)²) / 3 = (1+0+1)/3 = 2/3 ≈ 0.6667
        """
        pred = torch.tensor([1.0, 3.0, 5.0])
        target = torch.tensor([2.0, 3.0, 6.0])
        result = mse(pred, target)
        assert abs(result.item() - 2 / 3) < 1e-5


class TestNegBinNLL:
    """Tests for negative_binomial_nll()."""

    def test_known_answer(self):
        """NegBin NLL with known rate and dispersion should produce finite loss."""
        # rate=2.0, dispersion=5.0, observed count=2
        rate = torch.tensor([[2.0]])
        disp = torch.tensor([[5.0]])
        target = torch.tensor([[2.0]])
        loss = negative_binomial_nll(rate, disp, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_bad_prediction_higher_loss(self):
        """Bad rate predictions should have higher NLL than good ones."""
        target = torch.tensor([[2.0, 3.0]])
        disp = torch.tensor([[10.0, 10.0]])
        good_rate = torch.tensor([[2.0, 3.0]])
        bad_rate = torch.tensor([[10.0, 0.1]])
        good_loss = negative_binomial_nll(good_rate, disp, target)
        bad_loss = negative_binomial_nll(bad_rate, disp, target)
        assert bad_loss > good_loss

    def test_poisson_limit(self):
        """As dispersion r → ∞, NegBin should converge to Poisson behavior.

        Manually compute the full Poisson NLL (including lgamma(k+1))
        since F.poisson_nll_loss omits it, and compare at moderate r
        where NegBin is well-conditioned.
        """
        target = torch.tensor([[2.0, 3.0, 1.0]])
        rate = torch.tensor([[2.0, 3.0, 1.0]])

        # Manual Poisson NLL including lgamma(k+1):
        # NLL = rate - k*log(rate) + lgamma(k+1)
        manual_poisson_nll = (
            rate - target * torch.log(rate) + torch.lgamma(target + 1)
        ).mean()

        # NegBin with moderate r=100 should be close to Poisson
        r_moderate = torch.tensor([[100.0, 100.0, 100.0]])
        negbin_loss = negative_binomial_nll(rate, r_moderate, target)

        # Should be within 5% of the manual Poisson NLL
        assert abs(negbin_loss.item() - manual_poisson_nll.item()) < 0.05 * manual_poisson_nll.item()

    def test_gradient_flow(self):
        """Gradients should flow through both rate and dispersion parameters."""
        rate = torch.tensor([[2.0, 3.0]], requires_grad=True)
        disp = torch.tensor([[5.0, 5.0]], requires_grad=True)
        target = torch.tensor([[1.0, 4.0]])
        loss = negative_binomial_nll(rate, disp, target)
        loss.backward()
        assert rate.grad is not None
        assert disp.grad is not None
        assert torch.all(torch.isfinite(rate.grad))
        assert torch.all(torch.isfinite(disp.grad))

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        rate = torch.tensor([[2.0, 3.0]])
        disp = torch.tensor([[5.0, 5.0]])
        target = torch.tensor([[1.0, 4.0]])
        loss = negative_binomial_nll(rate, disp, target)
        assert loss.dim() == 0

    def test_zero_target(self):
        """NegBin NLL should handle zero spike counts correctly."""
        rate = torch.tensor([[1.0, 2.0]])
        disp = torch.tensor([[5.0, 5.0]])
        target = torch.tensor([[0.0, 0.0]])
        loss = negative_binomial_nll(rate, disp, target)
        assert torch.isfinite(loss)


class TestZIPNLL:
    """Tests for zero_inflated_poisson_nll()."""

    def test_known_answer(self):
        """ZIP NLL with known rate and gate should produce finite loss."""
        rate = torch.tensor([[2.0]])
        gate = torch.tensor([[0.1]])
        target = torch.tensor([[2.0]])
        loss = zero_inflated_poisson_nll(rate, gate, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_bad_prediction_higher_loss(self):
        """Bad rate predictions should have higher NLL than good ones."""
        target = torch.tensor([[2.0, 3.0]])
        gate = torch.tensor([[0.05, 0.05]])
        good_rate = torch.tensor([[2.0, 3.0]])
        bad_rate = torch.tensor([[10.0, 0.1]])
        good_loss = zero_inflated_poisson_nll(good_rate, gate, target)
        bad_loss = zero_inflated_poisson_nll(bad_rate, gate, target)
        assert bad_loss > good_loss

    def test_poisson_limit(self):
        """As gate π → 0, ZIP should converge to Poisson behavior.

        Note: We can't directly compare against poisson_nll() because
        F.poisson_nll_loss omits the lgamma(k+1) constant term, while
        our ZIP implementation includes it. Instead, we verify that
        small π produces nearly the same NLL as very-small π (convergence).
        """
        target = torch.tensor([[2.0, 3.0, 1.0]])
        rate = torch.tensor([[2.0, 3.0, 1.0]])
        # Two small gates — should converge
        gate_small = torch.tensor([[1e-4, 1e-4, 1e-4]])
        gate_very_small = torch.tensor([[1e-7, 1e-7, 1e-7]])
        loss_small = zero_inflated_poisson_nll(rate, gate_small, target)
        loss_very_small = zero_inflated_poisson_nll(rate, gate_very_small, target)
        # Both should be very close (converged to the Poisson limit)
        assert abs(loss_small.item() - loss_very_small.item()) < 0.01

    def test_zero_targets_with_high_gate(self):
        """
        Zero-heavy data with high gate should have lower loss than with low gate.
        High π means the model expects many zeros, which matches all-zero data.
        """
        rate = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[0.0, 0.0]])
        high_gate = torch.tensor([[0.8, 0.8]])
        low_gate = torch.tensor([[0.05, 0.05]])
        loss_high = zero_inflated_poisson_nll(rate, high_gate, target)
        loss_low = zero_inflated_poisson_nll(rate, low_gate, target)
        # Higher gate → better fit for zero targets → lower loss
        assert loss_high < loss_low

    def test_gradient_flow(self):
        """Gradients should flow through both rate and gate parameters."""
        rate = torch.tensor([[2.0, 3.0]], requires_grad=True)
        gate = torch.tensor([[0.1, 0.1]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0]])
        loss = zero_inflated_poisson_nll(rate, gate, target)
        loss.backward()
        assert rate.grad is not None
        assert gate.grad is not None
        assert torch.all(torch.isfinite(rate.grad))
        assert torch.all(torch.isfinite(gate.grad))

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        rate = torch.tensor([[2.0, 3.0]])
        gate = torch.tensor([[0.1, 0.1]])
        target = torch.tensor([[1.0, 0.0]])
        loss = zero_inflated_poisson_nll(rate, gate, target)
        assert loss.dim() == 0


# =============================================================================
# Baseline tests
# =============================================================================

class TestPersistenceBaseline:
    """Tests for persistence_baseline()."""

    def test_returns_all_metrics(self):
        """Should return a dict with all four metric keys."""
        # Constant signal: persistence is perfect
        counts = np.full((3, 100), 2, dtype=np.int32)
        results = persistence_baseline(counts, history_bins=5)
        assert "poisson_nll" in results
        assert "pearson_r" in results
        assert "mae" in results
        assert "mse" in results

    def test_constant_signal_perfect(self):
        """A constant signal should have MAE=0 for persistence baseline."""
        counts = np.full((3, 100), 5, dtype=np.int32)
        results = persistence_baseline(counts, history_bins=5)
        assert results["mae"] == 0.0
        assert results["mse"] == 0.0

    def test_alternating_signal_imperfect(self):
        """An alternating signal should have nonzero MAE for persistence."""
        # [0, 2, 0, 2, ...] — persistence always predicts the opposite
        counts = np.tile([0, 2], (1, 50)).astype(np.int32)
        results = persistence_baseline(counts, history_bins=1)
        assert results["mae"] > 0


class TestMeanRateBaseline:
    """Tests for mean_rate_baseline()."""

    def test_returns_all_metrics(self):
        """Should return a dict with all four metric keys."""
        counts = np.full((3, 100), 2, dtype=np.int32)
        results = mean_rate_baseline(counts, history_bins=5)
        assert "poisson_nll" in results
        assert "pearson_r" in results
        assert "mae" in results
        assert "mse" in results

    def test_constant_signal_perfect(self):
        """Mean-rate on constant signal should have MAE=0."""
        counts = np.full((3, 100), 5, dtype=np.int32)
        results = mean_rate_baseline(counts, history_bins=5)
        assert abs(results["mae"]) < 1e-5


class TestComputeAllBaselines:
    """Tests for compute_all_baselines()."""

    def test_returns_both_baselines(self):
        """Should return dict with persistence and mean_rate keys."""
        counts = np.full((3, 100), 2, dtype=np.int32)
        results = compute_all_baselines(counts, history_bins=5)
        assert "persistence" in results
        assert "mean_rate" in results

    def test_persistence_is_better_for_smooth_signal(self):
        """For a slowly varying signal, persistence should outperform mean-rate."""
        # Linearly increasing signal: persistence is very close, mean-rate lags
        ramp = np.arange(200, dtype=np.float64).reshape(1, 200)
        counts = ramp.astype(np.int32)
        results = compute_all_baselines(counts, history_bins=5)
        # Persistence MAE should be smaller (it only misses by ~1 per step)
        assert results["persistence"]["mae"] < results["mean_rate"]["mae"]
