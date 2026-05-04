"""
Tests for the population shuffle ablation logic.

Validates that:
1. Shuffling preserves per-neuron marginal statistics (mean, variance).
2. Shuffling destroys cross-neuron temporal correlations.
3. The target neuron remains untouched after shuffling.
4. On a synthetic case with known cross-neuron structure, shuffling
   eliminates the predictable component.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.population_shuffle_ablation import (
    shuffle_population,
    compute_per_neuron_r,
)


class TestShufflePopulation:
    """Tests for the shuffle_population function."""

    def test_target_neuron_unchanged(self):
        """Shuffling should not modify the target neuron's time series."""
        rng = np.random.default_rng(42)
        # Create synthetic data: 10 neurons, 100 time bins
        counts = rng.poisson(5, size=(10, 100)).astype(np.float32)
        target = 3

        shuffled = shuffle_population(counts, target, rng)

        # Target neuron should be identical
        np.testing.assert_array_equal(
            shuffled[target, :], counts[target, :],
            err_msg="Target neuron was modified by shuffling!",
        )

    def test_non_target_neurons_differ(self):
        """Shuffled non-target neurons should differ from original."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(5, size=(10, 200)).astype(np.float32)
        target = 3

        shuffled = shuffle_population(counts, target, rng)

        # At least some non-target neurons should be different
        n_changed = 0
        for n in range(10):
            if n == target:
                continue
            if not np.array_equal(shuffled[n, :], counts[n, :]):
                n_changed += 1

        assert n_changed >= 8, (
            f"Expected most non-target neurons to change, "
            f"but only {n_changed}/9 changed"
        )

    def test_marginal_stats_preserved(self):
        """Shuffling should preserve per-neuron mean and variance."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(10, size=(20, 500)).astype(np.float32)
        target = 5

        shuffled = shuffle_population(counts, target, rng)

        for n in range(20):
            # Mean should be identical (same values, different order)
            np.testing.assert_allclose(
                shuffled[n, :].mean(), counts[n, :].mean(),
                atol=1e-6,
                err_msg=f"Neuron {n} mean changed after shuffling",
            )
            # Variance should be identical (allow float32 rounding)
            np.testing.assert_allclose(
                shuffled[n, :].var(), counts[n, :].var(),
                atol=1e-4,
                err_msg=f"Neuron {n} variance changed after shuffling",
            )

    def test_cross_neuron_correlation_destroyed(self):
        """Shuffling should destroy cross-neuron temporal correlations."""
        rng = np.random.default_rng(42)
        T = 1000
        # Create correlated neurons: neuron 1 = neuron 0 + noise
        base = rng.poisson(5, size=T).astype(np.float32)
        counts = np.stack([
            base,                                  # Neuron 0 (target)
            base + rng.normal(0, 0.5, T),          # Correlated with 0
            rng.poisson(5, size=T),                # Independent
        ]).astype(np.float32)

        # Original cross-correlation should be high
        orig_r = np.corrcoef(counts[0, :], counts[1, :])[0, 1]
        assert orig_r > 0.7, f"Setup error: expected high correlation, got {orig_r}"

        # After shuffling (keeping neuron 0 as target), correlation should drop
        shuffled = shuffle_population(counts, target_neuron=0, rng=rng)
        shuf_r = np.corrcoef(shuffled[0, :], shuffled[1, :])[0, 1]

        assert abs(shuf_r) < 0.15, (
            f"Shuffling should destroy correlation: "
            f"orig={orig_r:.3f}, shuffled={shuf_r:.3f}"
        )

    def test_output_shape_matches_input(self):
        """Shuffled output should have the same shape as input."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(5, size=(15, 300)).astype(np.float32)

        shuffled = shuffle_population(counts, target_neuron=7, rng=rng)
        assert shuffled.shape == counts.shape

    def test_all_values_preserved(self):
        """Shuffling should be a permutation — same set of values."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(5, size=(5, 200)).astype(np.float32)

        shuffled = shuffle_population(counts, target_neuron=2, rng=rng)

        for n in range(5):
            # Sorted values should be identical
            np.testing.assert_array_equal(
                np.sort(shuffled[n, :]), np.sort(counts[n, :]),
                err_msg=f"Neuron {n} has different values after shuffle",
            )


class TestComputePerNeuronR:
    """Tests for the compute_per_neuron_r helper."""

    def test_perfect_correlation(self):
        """Identical arrays should give r=1.0."""
        gt = np.random.randn(100, 5).astype(np.float32) + 5
        r = compute_per_neuron_r(gt, gt)
        np.testing.assert_allclose(r, 1.0, atol=1e-6)

    def test_zero_variance(self):
        """Constant neurons should give r=0.0."""
        gt = np.ones((100, 3), dtype=np.float32)
        pred = np.random.randn(100, 3).astype(np.float32)
        r = compute_per_neuron_r(gt, pred)
        np.testing.assert_array_equal(r, 0.0)

    def test_known_correlation(self):
        """Known signal + noise should give predictable r."""
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        gt = signal.reshape(-1, 1) + rng.normal(0, 0.1, (200, 1))
        pred = signal.reshape(-1, 1)
        r = compute_per_neuron_r(
            gt.astype(np.float32), pred.astype(np.float32),
        )
        # Strong correlation expected
        assert r[0] > 0.9, f"Expected high r for signal+noise, got {r[0]:.3f}"

    def test_output_shape(self):
        """Output should be (M,) array."""
        gt = np.random.randn(50, 10).astype(np.float32) + 5
        pred = np.random.randn(50, 10).astype(np.float32) + 5
        r = compute_per_neuron_r(gt, pred)
        assert r.shape == (10,)
