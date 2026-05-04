"""Tests for behavior_loader.py — NWB behavioral data extraction."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.data.behavior_loader import (
    compute_bin_edges,
    extract_wheel_velocity,
    extract_trial_stimuli,
    extract_all_behavior,
)


class TestComputeBinEdges:
    """Tests for compute_bin_edges."""

    def test_basic_shape(self):
        """Bin edges have n_bins + 1 elements."""
        edges = compute_bin_edges(100, bin_width_ms=50.0)
        assert edges.shape == (101,)

    def test_spacing(self):
        """Bin edges are evenly spaced at bin_width_ms."""
        edges = compute_bin_edges(10, bin_width_ms=50.0)
        diffs = np.diff(edges)
        np.testing.assert_allclose(diffs, 0.05, atol=1e-10)

    def test_start_time(self):
        """Bin edges start at the specified start time."""
        edges = compute_bin_edges(5, bin_width_ms=100.0, start_time=10.0)
        assert edges[0] == 10.0
        assert edges[-1] == pytest.approx(10.5)

    def test_zero_bins(self):
        """Zero bins returns single-element array."""
        edges = compute_bin_edges(0, bin_width_ms=50.0)
        assert edges.shape == (1,)


class TestExtractWheelVelocity:
    """Tests for extract_wheel_velocity with mocked NWB data."""

    def _make_mock_nwb(self, n_samples, duration_s, position_fn=None):
        """Create mock NWB h5py file context."""
        if position_fn is None:
            # Default: sinusoidal position
            t = np.linspace(0, duration_s, n_samples)
            position = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        else:
            position = position_fn(n_samples, duration_s)

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__getitem__ = MagicMock(side_effect=lambda key: {
            "acquisition/wheel_position/data": MagicMock(
                __getitem__=lambda s, sl: position,
            ),
            "acquisition/wheel_position/starting_time": MagicMock(
                __call__=lambda: 0.0,
                __getitem__=lambda s, sl: np.float64(0.0),
            ),
        }[key])
        return mock_file

    @patch("src.data.behavior_loader.h5py.File")
    def test_output_shape(self, mock_h5):
        """Output matches number of bins."""
        n_bins = 20
        duration = 1.0  # 1 second
        bin_edges = compute_bin_edges(n_bins, bin_width_ms=50.0)

        # Mock wheel data: 500 samples at 500 Hz for 1 second
        mock_file = self._make_mock_nwb(500, duration)
        mock_h5.return_value = mock_file

        wv = extract_wheel_velocity("test.nwb", bin_edges)
        assert wv.shape == (n_bins,)
        assert wv.dtype == np.float32

    @patch("src.data.behavior_loader.h5py.File")
    def test_constant_position_zero_velocity(self, mock_h5):
        """Constant wheel position → zero velocity."""
        n_bins = 10
        bin_edges = compute_bin_edges(n_bins, bin_width_ms=50.0)

        # Constant position
        position_fn = lambda n, d: np.ones(n) * 5.0
        mock_file = self._make_mock_nwb(500, 0.5, position_fn)
        mock_h5.return_value = mock_file

        wv = extract_wheel_velocity("test.nwb", bin_edges)
        assert np.allclose(wv, 0.0, atol=1e-5)


class TestExtractTrialStimuli:
    """Tests for extract_trial_stimuli with mocked NWB data."""

    @patch("src.data.behavior_loader.h5py.File")
    def test_output_keys(self, mock_h5):
        """Returns all expected keys."""
        n_bins = 100
        bin_edges = compute_bin_edges(n_bins, bin_width_ms=50.0)

        # Mock 3 trials
        mock_trials = {
            "visual_stimulus_time": np.array([0.1, 1.0, 2.0]),
            "visual_stimulus_left_contrast": np.array([0.5, 1.0, 0.0]),
            "visual_stimulus_right_contrast": np.array([0.0, 0.5, 1.0]),
            "response_choice": np.array([1.0, -1.0, 0.0]),
            "feedback_type": np.array([1, -1, 1]),
            "start_time": np.array([0.0, 0.8, 1.8]),
            "stop_time": np.array([0.7, 1.7, 2.7]),
        }

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)

        # Set up nested dict access: f["intervals/trials"]["key"]
        trials_group = MagicMock()
        trials_group.__getitem__ = MagicMock(
            side_effect=lambda key: mock_trials[key],
        )
        mock_file.__getitem__ = MagicMock(
            side_effect=lambda key: trials_group if key == "intervals/trials" else None,
        )
        mock_h5.return_value = mock_file

        result = extract_trial_stimuli("test.nwb", bin_edges)

        expected_keys = {
            "left_contrast", "right_contrast", "response_choice",
            "feedback_type", "trial_active", "trial_index",
        }
        assert set(result.keys()) == expected_keys

        # All arrays should be shape (n_bins,)
        for key, arr in result.items():
            assert arr.shape == (n_bins,), f"{key} shape mismatch"
            if key == "trial_index":
                assert arr.dtype == np.int32, f"{key} should be int32"
            else:
                assert arr.dtype == np.float32, f"{key} should be float32"

    @patch("src.data.behavior_loader.h5py.File")
    def test_trial_active_mask(self, mock_h5):
        """trial_active is 1 within trials, 0 outside."""
        n_bins = 100
        bin_edges = compute_bin_edges(n_bins, bin_width_ms=50.0)

        mock_trials = {
            "visual_stimulus_time": np.array([0.5]),
            "visual_stimulus_left_contrast": np.array([1.0]),
            "visual_stimulus_right_contrast": np.array([0.5]),
            "response_choice": np.array([1.0]),
            "feedback_type": np.array([1]),
            "start_time": np.array([0.5]),
            "stop_time": np.array([1.5]),
        }

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        trials_group = MagicMock()
        trials_group.__getitem__ = MagicMock(
            side_effect=lambda key: mock_trials[key],
        )
        mock_file.__getitem__ = MagicMock(
            side_effect=lambda key: trials_group if key == "intervals/trials" else None,
        )
        mock_h5.return_value = mock_file

        result = extract_trial_stimuli("test.nwb", bin_edges)

        # Some bins should be active, some not
        assert result["trial_active"].sum() > 0
        assert result["trial_active"].sum() < n_bins


class TestExtractAllBehavior:
    """Tests for the convenience function."""

    @patch("src.data.behavior_loader.extract_trial_stimuli")
    @patch("src.data.behavior_loader.extract_wheel_velocity")
    def test_combines_results(self, mock_wheel, mock_stimuli):
        """Combines wheel and trial results."""
        n_bins = 50
        bin_edges = compute_bin_edges(n_bins)

        mock_wheel.return_value = np.zeros(n_bins, dtype=np.float32)
        mock_stimuli.return_value = {
            "left_contrast": np.zeros(n_bins, dtype=np.float32),
            "right_contrast": np.zeros(n_bins, dtype=np.float32),
            "response_choice": np.zeros(n_bins, dtype=np.float32),
            "feedback_type": np.zeros(n_bins, dtype=np.float32),
            "trial_active": np.zeros(n_bins, dtype=np.float32),
            "trial_index": np.full(n_bins, -1, dtype=np.int32),
        }

        result = extract_all_behavior("test.nwb", bin_edges)

        assert "wheel_velocity" in result
        assert "left_contrast" in result
        assert len(result) == 7  # wheel + 6 trial fields (incl. trial_index)
