"""
Tests for population-level metrics: population_rate_r, spatial_pattern_r,
population_cosine_sim.

These metrics capture system-wide dynamics rather than per-neuron prediction
quality.
"""

import pytest
import torch

from src.eval.metrics import (
    population_rate_r,
    spatial_pattern_r,
    population_cosine_sim,
)


class TestPopulationRateR:
    """Tests for population_rate_r — correlation of total firing rate."""

    def test_perfect_prediction(self):
        """Identical predictions should give r=1.0."""
        gt = torch.rand(100, 50)
        r = population_rate_r(gt, gt)
        assert abs(float(r) - 1.0) < 1e-4, f"Expected 1.0, got {r}"

    def test_scaled_prediction(self):
        """Scaled predictions should still give r=1.0 (Pearson is scale-invariant)."""
        gt = torch.rand(100, 50) + 0.1
        pred = gt * 2.5 + 3.0  # Linear transform
        r = population_rate_r(pred, gt)
        assert abs(float(r) - 1.0) < 1e-4, f"Expected 1.0, got {r}"

    def test_anticorrelated(self):
        """Anticorrelated predictions should give r close to -1.0."""
        gt = torch.rand(100, 50)
        pred = -gt + gt.mean()  # Negate the pattern
        r = population_rate_r(pred, gt)
        assert float(r) < -0.8, f"Expected r < -0.8, got {r}"

    def test_random_prediction(self):
        """Random predictions should give r near 0."""
        torch.manual_seed(42)
        gt = torch.rand(200, 50)
        pred = torch.rand(200, 50)
        r = population_rate_r(pred, gt)
        assert abs(float(r)) < 0.3, f"Expected r near 0, got {r}"

    def test_output_is_scalar(self):
        """Output should be a scalar tensor."""
        gt = torch.rand(50, 20)
        r = population_rate_r(gt, gt)
        assert r.dim() == 0, f"Expected scalar, got dim={r.dim()}"


class TestSpatialPatternR:
    """Tests for spatial_pattern_r — per-timebin neuron pattern correlation."""

    def test_perfect_prediction(self):
        """Identical predictions should give r=1.0."""
        gt = torch.rand(100, 50) + 0.1  # Avoid all-zero bins
        r = spatial_pattern_r(gt, gt)
        assert abs(float(r) - 1.0) < 1e-4, f"Expected 1.0, got {r}"

    def test_good_prediction_high_r(self):
        """Noisy copies should still give high spatial r."""
        torch.manual_seed(42)
        gt = torch.rand(100, 50) + 0.1
        pred = gt + torch.randn(100, 50) * 0.1
        r = spatial_pattern_r(pred, gt)
        assert float(r) > 0.8, f"Expected r > 0.8, got {r}"

    def test_random_prediction_low_r(self):
        """Random predictions should give low spatial r."""
        torch.manual_seed(42)
        gt = torch.rand(200, 50)
        pred = torch.rand(200, 50)
        r = spatial_pattern_r(pred, gt)
        assert abs(float(r)) < 0.3, f"Expected r near 0, got {r}"

    def test_constant_bins_excluded(self):
        """Bins where GT or pred is constant should be excluded."""
        gt = torch.zeros(10, 5)
        gt[0, :] = 1.0  # Only first bin has non-constant data
        pred = torch.zeros(10, 5)
        pred[0, :] = torch.tensor([0.5, 1.0, 0.5, 1.0, 0.5])
        # Should compute from the 1 valid bin only, not crash
        r = spatial_pattern_r(pred, gt)
        assert not torch.isnan(r), f"Got NaN"

    def test_all_constant_returns_zero(self):
        """All-constant data should return 0, not crash."""
        gt = torch.ones(10, 5)
        pred = torch.ones(10, 5) * 2
        r = spatial_pattern_r(pred, gt)
        assert float(r) == 0.0, f"Expected 0.0, got {r}"


class TestPopulationCosineSim:
    """Tests for population_cosine_sim — mean cosine similarity."""

    def test_perfect_prediction(self):
        """Identical predictions should give cosine=1.0."""
        gt = torch.rand(100, 50) + 0.01
        cos = population_cosine_sim(gt, gt)
        assert abs(float(cos) - 1.0) < 1e-4, f"Expected 1.0, got {cos}"

    def test_scaled_prediction_still_1(self):
        """Cosine similarity is magnitude-invariant."""
        gt = torch.rand(100, 50) + 0.01
        pred = gt * 5.0  # Same direction, different magnitude
        cos = population_cosine_sim(pred, gt)
        assert abs(float(cos) - 1.0) < 1e-4, f"Expected 1.0, got {cos}"

    def test_orthogonal_low_cosine(self):
        """Orthogonal vectors should give low cosine."""
        # Create orthogonal-ish predictions
        torch.manual_seed(42)
        gt = torch.rand(100, 50)
        pred = torch.rand(100, 50)
        cos = population_cosine_sim(pred, gt)
        # Random vectors in high-D are nearly orthogonal
        # but nonneg so some overlap
        assert float(cos) < 0.9, f"Expected cos < 0.9 for random, got {cos}"

    def test_output_range(self):
        """Cosine similarity should be in [-1, 1]."""
        torch.manual_seed(42)
        gt = torch.randn(100, 50)
        pred = torch.randn(100, 50)
        cos = population_cosine_sim(pred, gt)
        assert -1.0 <= float(cos) <= 1.0, f"Out of range: {cos}"

    def test_known_answer(self):
        """Known answer: vectors at 60 degrees have cosine=0.5."""
        # Simple 2D case
        gt = torch.tensor([[1.0, 0.0]])
        pred = torch.tensor([[0.5, 0.866]])  # 60 degrees
        cos = population_cosine_sim(pred, gt)
        assert abs(float(cos) - 0.5) < 0.01, f"Expected ~0.5, got {cos}"
