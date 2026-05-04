"""
Tests for ceiling-based loss weights.

Validates that build_ceiling_weights correctly loads per-neuron stats,
applies weighting strategies (binary, ceiling, softmax), and handles
edge cases (missing file, invalid params, multi-session averaging).
"""

import json
import pytest
import torch
from pathlib import Path

from src.eval.ceiling_weights import build_ceiling_weights


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_stats(tmp_path):
    """Create a minimal per_neuron_stats.json for testing."""
    stats = [
        # Session 0: 5 neurons with varying ceilings
        {"session": 0, "neuron": 0, "region": "VISp", "ceiling_analytical": 0.5},
        {"session": 0, "neuron": 1, "region": "VISp", "ceiling_analytical": 0.0},
        {"session": 0, "neuron": 2, "region": "CA1", "ceiling_analytical": 0.8},
        {"session": 0, "neuron": 3, "region": "CA1", "ceiling_analytical": 0.05},
        {"session": 0, "neuron": 4, "region": "MOp", "ceiling_analytical": 0.3},
        # Session 1: 3 neurons (overlapping indices — tests averaging)
        {"session": 1, "neuron": 0, "region": "VISp", "ceiling_analytical": 0.3},
        {"session": 1, "neuron": 1, "region": "VISp", "ceiling_analytical": 0.2},
        {"session": 1, "neuron": 2, "region": "CA1", "ceiling_analytical": 0.6},
    ]
    stats_path = tmp_path / "per_neuron_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    return stats_path


@pytest.fixture
def m_max():
    """Padded channel dimension for tests."""
    return 8


# =============================================================================
# Shape and range tests
# =============================================================================


class TestCeilingWeightsShape:
    """Tests for output shape and value ranges."""

    def test_ceiling_strategy_returns_correct_shape(self, mock_stats, m_max):
        """Weight tensor shape should match m_max."""
        weights = build_ceiling_weights(mock_stats, m_max, strategy="ceiling")
        assert weights.shape == (m_max,)

    def test_binary_strategy_returns_correct_shape(self, mock_stats, m_max):
        """Binary strategy should also produce (m_max,) tensor."""
        weights = build_ceiling_weights(mock_stats, m_max, strategy="binary")
        assert weights.shape == (m_max,)

    def test_softmax_strategy_returns_correct_shape(self, mock_stats, m_max):
        """Softmax strategy should also produce (m_max,) tensor."""
        weights = build_ceiling_weights(mock_stats, m_max, strategy="softmax")
        assert weights.shape == (m_max,)

    def test_ceiling_strategy_values_clamped(self, mock_stats, m_max):
        """All ceiling weights should be in [floor_weight, 1.0]."""
        floor = 0.1
        weights = build_ceiling_weights(
            mock_stats, m_max, strategy="ceiling", floor_weight=floor,
        )
        assert weights.min().item() >= floor - 1e-7
        assert weights.max().item() <= 1.0 + 1e-7


# =============================================================================
# Strategy-specific tests
# =============================================================================


class TestCeilingWeightsStrategies:
    """Tests for each weighting strategy."""

    def test_binary_strategy_thresholding(self, mock_stats, m_max):
        """Binary: neurons above threshold get 1.0, below get floor_weight."""
        floor = 0.1
        threshold = 0.1
        weights = build_ceiling_weights(
            mock_stats, m_max, strategy="binary",
            floor_weight=floor, threshold=threshold,
        )
        # Neuron 0: avg ceiling = (0.5 + 0.3) / 2 = 0.4 > 0.1 → 1.0
        assert weights[0].item() == pytest.approx(1.0)
        # Neuron 1: avg ceiling = (0.0 + 0.2) / 2 = 0.1 == threshold → floor
        # (threshold test is >, not >=)
        assert weights[1].item() == pytest.approx(floor)
        # Neuron 3: ceiling = 0.05 < 0.1 → floor
        assert weights[3].item() == pytest.approx(floor)

    def test_ceiling_strategy_continuous(self, mock_stats, m_max):
        """Ceiling: weights should be the raw ceiling value (clamped)."""
        floor = 0.05
        weights = build_ceiling_weights(
            mock_stats, m_max, strategy="ceiling", floor_weight=floor,
        )
        # Neuron 2: avg ceiling = (0.8 + 0.6) / 2 = 0.7
        assert weights[2].item() == pytest.approx(0.7, abs=0.01)
        # Neuron 4: ceiling = 0.3 (only session 0)
        assert weights[4].item() == pytest.approx(0.3, abs=0.01)

    def test_softmax_strategy_positive(self, mock_stats, m_max):
        """Softmax: all weights should be positive."""
        weights = build_ceiling_weights(
            mock_stats, m_max, strategy="softmax",
        )
        assert (weights > 0).all()

    def test_floor_weight_respected(self, mock_stats, m_max):
        """No weight should be below floor_weight for ceiling/binary."""
        for strategy in ("ceiling", "binary"):
            floor = 0.2
            weights = build_ceiling_weights(
                mock_stats, m_max, strategy=strategy, floor_weight=floor,
            )
            assert weights.min().item() >= floor - 1e-7


# =============================================================================
# Multi-session averaging tests
# =============================================================================


class TestMultiSessionAveraging:
    """Tests for cross-session aggregation."""

    def test_multi_session_averaging(self, mock_stats, m_max):
        """Stats from multiple sessions should be averaged per channel."""
        weights = build_ceiling_weights(
            mock_stats, m_max, strategy="ceiling", floor_weight=0.01,
        )
        # Neuron 0: sessions 0 and 1 → avg = (0.5 + 0.3) / 2 = 0.4
        assert weights[0].item() == pytest.approx(0.4, abs=0.01)
        # Neuron 2: sessions 0 and 1 → avg = (0.8 + 0.6) / 2 = 0.7
        assert weights[2].item() == pytest.approx(0.7, abs=0.01)
        # Neuron 4: only session 0 → 0.3
        assert weights[4].item() == pytest.approx(0.3, abs=0.01)


# =============================================================================
# Error handling tests
# =============================================================================


class TestCeilingWeightsErrors:
    """Tests for error handling."""

    def test_handles_missing_stats_file(self, tmp_path):
        """Should raise FileNotFoundError for missing stats file."""
        fake_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            build_ceiling_weights(fake_path, m_max=10)

    def test_invalid_strategy_raises(self, mock_stats, m_max):
        """Should raise ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="strategy"):
            build_ceiling_weights(mock_stats, m_max, strategy="invalid")

    def test_invalid_floor_weight_raises(self, mock_stats, m_max):
        """Should raise ValueError for floor_weight <= 0."""
        with pytest.raises(ValueError, match="floor_weight"):
            build_ceiling_weights(mock_stats, m_max, floor_weight=0.0)
        with pytest.raises(ValueError, match="floor_weight"):
            build_ceiling_weights(mock_stats, m_max, floor_weight=-0.1)
