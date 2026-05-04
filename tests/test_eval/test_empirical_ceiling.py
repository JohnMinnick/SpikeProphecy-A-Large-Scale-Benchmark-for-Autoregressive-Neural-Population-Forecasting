"""
Tests for empirical ceiling computation.

Validates the split-half and blocked ceiling methods, plus
the ceiling efficiency calculation.
"""

import numpy as np
import pytest

from src.eval.empirical_ceiling import (
    compute_empirical_ceiling,
    compute_empirical_ceiling_blocked,
    ceiling_efficiency,
)


class TestComputeEmpiricalCeiling:
    """Tests for the simple split-half ceiling."""

    def test_output_shape(self):
        """Output has one ceiling per neuron."""
        counts = np.random.poisson(5, size=(10, 200))
        ceilings = compute_empirical_ceiling(counts)
        assert ceilings.shape == (10,)

    def test_high_rate_neurons_have_positive_ceiling(self):
        """Neurons with strong rate modulation have ceiling > 0."""
        T = 1000
        # Create a neuron with clear sinusoidal rate modulation
        rate = 3.0 + 2.0 * np.sin(2 * np.pi * np.arange(T) / 100)
        counts = np.random.poisson(rate).reshape(1, -1).astype(float)
        ceilings = compute_empirical_ceiling(counts)
        # A modulated neuron should have a positive ceiling
        assert ceilings[0] > 0.0, f"Expected ceiling > 0, got {ceilings[0]}"

    def test_silent_neurons_get_zero_ceiling(self):
        """Silent neurons (mean rate < min_rate) get ceiling = 0."""
        counts = np.zeros((3, 200))
        ceilings = compute_empirical_ceiling(counts, min_rate_hz=0.01)
        np.testing.assert_array_equal(ceilings, 0.0)

    def test_pure_noise_has_low_ceiling(self):
        """Pure Poisson noise (constant rate) has near-zero ceiling."""
        T = 2000
        # Constant rate = 5 spikes/bin → very high Fano-based ceiling
        # but split-half should give low ceiling (no temporal structure)
        counts = np.random.poisson(5, size=(1, T)).astype(float)
        ceilings = compute_empirical_ceiling(counts)
        # Should be small (near zero) since there's no real signal
        assert ceilings[0] < 0.3, f"Expected low ceiling, got {ceilings[0]}"

    def test_strong_signal_has_high_ceiling(self):
        """Strong periodic signal → high ceiling."""
        T = 1000
        # Square wave with period 100: alternates 0 and 10
        # Both halves will have variance since the pattern repeats
        rate = np.where(np.arange(T) % 100 < 50, 1.0, 10.0)
        counts = np.random.poisson(rate).reshape(1, -1).astype(float)
        ceilings = compute_empirical_ceiling(counts)
        assert ceilings[0] > 0.3, f"Expected high ceiling, got {ceilings[0]}"

    def test_ceiling_range(self):
        """All ceiling values in [0, 1] (capped by Spearman-Brown)."""
        T = 500
        rate = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(T) / 50)
        counts = np.random.poisson(rate).reshape(1, -1).astype(float)
        ceilings = compute_empirical_ceiling(counts)
        assert np.all(ceilings >= 0.0)
        assert np.all(ceilings <= 1.0)  # Spearman-Brown: 2r/(1+r) ≤ 1


class TestComputeEmpiricalCeilingBlocked:
    """Tests for the blocked split-half ceiling."""

    def test_output_shape(self):
        """Output has one ceiling per neuron."""
        counts = np.random.poisson(5, size=(10, 400))
        ceilings = compute_empirical_ceiling_blocked(counts, block_size=50)
        assert ceilings.shape == (10,)

    def test_modulated_neuron_positive_ceiling(self):
        """Neuron with rate modulation → positive blocked ceiling."""
        T = 1000
        rate = 4.0 + 3.0 * np.sin(2 * np.pi * np.arange(T) / 100)
        counts = np.random.poisson(rate).reshape(1, -1).astype(float)
        ceilings = compute_empirical_ceiling_blocked(
            counts, block_size=25,
        )
        assert ceilings[0] > 0.0

    def test_fallback_for_short_data(self):
        """With too few bins, falls back to simple split-half."""
        counts = np.random.poisson(5, size=(5, 100))
        # block_size=50 → only 2 blocks → should fall back
        ceilings = compute_empirical_ceiling_blocked(
            counts, block_size=50,
        )
        assert ceilings.shape == (5,)


class TestCeilingEfficiency:
    """Tests for the ceiling efficiency metric."""

    def test_perfect_model_gets_100_percent(self):
        """Model r == ceiling → efficiency = 1.0."""
        per_neuron_r = np.array([0.5, 0.8, 0.3])
        ceilings = np.array([0.5, 0.8, 0.3])
        mean_eff, per_eff = ceiling_efficiency(per_neuron_r, ceilings)
        np.testing.assert_almost_equal(mean_eff, 1.0, decimal=4)

    def test_half_ceiling_model(self):
        """Model at half ceiling → efficiency ≈ 0.5."""
        per_neuron_r = np.array([0.25, 0.40, 0.15])
        ceilings = np.array([0.50, 0.80, 0.30])
        mean_eff, _ = ceiling_efficiency(per_neuron_r, ceilings)
        np.testing.assert_almost_equal(mean_eff, 0.5, decimal=4)

    def test_below_min_ceiling_excluded(self):
        """Neurons with ceiling < min_ceiling get NaN efficiency."""
        per_neuron_r = np.array([0.5, 0.01])
        ceilings = np.array([0.8, 0.02])  # Second neuron below threshold
        mean_eff, per_eff = ceiling_efficiency(
            per_neuron_r, ceilings, min_ceiling=0.05,
        )
        # Only first neuron should be included
        assert np.isfinite(per_eff[0])
        assert np.isnan(per_eff[1])
        np.testing.assert_almost_equal(mean_eff, 0.5 / 0.8, decimal=4)

    def test_efficiency_capped_at_200_percent(self):
        """Efficiency is capped at 2.0 (200%) to handle noise."""
        per_neuron_r = np.array([0.9])
        ceilings = np.array([0.3])
        _, per_eff = ceiling_efficiency(per_neuron_r, ceilings)
        assert per_eff[0] <= 2.0

    def test_zero_ceiling_excluded(self):
        """Neurons with zero ceiling get NaN."""
        per_neuron_r = np.array([0.5])
        ceilings = np.array([0.0])
        mean_eff, per_eff = ceiling_efficiency(per_neuron_r, ceilings)
        assert np.isnan(per_eff[0])
