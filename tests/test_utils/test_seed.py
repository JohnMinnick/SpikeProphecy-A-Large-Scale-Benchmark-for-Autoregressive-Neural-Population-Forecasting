"""
Tests for src/utils/seed.py

Tests seeding reproducibility with known-answer verification:
verifies that seeding produces deterministic random outputs.
"""

import random

import numpy as np
import pytest
import torch

from src.utils.seed import get_seed_from_config, seed_everything


class TestSeedEverything:
    """Tests for seed_everything()."""

    def test_returns_seed_value(self):
        """Should return the seed that was set."""
        result = seed_everything(123)
        assert result == 123

    def test_negative_seed_raises(self):
        """Negative seeds should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-1)

    def test_python_random_deterministic(self):
        """After seeding, Python random should produce the same sequence."""
        seed_everything(42)
        seq1 = [random.random() for _ in range(5)]
        seed_everything(42)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 == seq2

    def test_numpy_deterministic(self):
        """After seeding, NumPy random should produce the same array."""
        seed_everything(42)
        arr1 = np.random.rand(10)
        seed_everything(42)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_deterministic(self):
        """After seeding, PyTorch random should produce the same tensor."""
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_different_seeds_give_different_results(self):
        """Different seeds should produce different random sequences."""
        seed_everything(42)
        arr1 = np.random.rand(100)
        seed_everything(99)
        arr2 = np.random.rand(100)
        # Extremely unlikely to be equal with different seeds
        assert not np.array_equal(arr1, arr2)

    def test_known_answer_numpy(self):
        """Known-answer test: seed=42 should produce a specific first value."""
        seed_everything(42)
        val = np.random.rand()
        # NumPy legacy RNG with seed 42, first value
        assert abs(val - 0.3745401188473625) < 1e-10

    def test_deterministic_mode(self):
        """Deterministic mode should not crash and should produce same results."""
        seed_everything(42, deterministic=True)
        t1 = torch.randn(10)
        seed_everything(42, deterministic=True)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)
        # Clean up: disable deterministic mode to not affect other tests
        torch.use_deterministic_algorithms(False)


class TestGetSeedFromConfig:
    """Tests for get_seed_from_config()."""

    def test_extracts_seed(self):
        """Should extract seed from config dict."""
        config = {"seed": 123, "other": "value"}
        assert get_seed_from_config(config) == 123

    def test_missing_seed_defaults_to_42(self):
        """Missing seed key should default to 42."""
        config = {"no_seed_here": True}
        assert get_seed_from_config(config) == 42

    def test_seed_converted_to_int(self):
        """Seed should be converted to int even if stored as float."""
        config = {"seed": 42.0}
        result = get_seed_from_config(config)
        assert result == 42
        assert isinstance(result, int)
