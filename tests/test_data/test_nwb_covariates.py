"""
Tests for NWB covariate extraction (Tier 1 stimulus features).

Validates feature extraction, bin-to-trial assignment, missing column
handling, and edge cases using synthetic trial data (no real NWB files
needed for unit tests).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data.nwb_covariates import (
    extract_stimulus_features,
    _assign_bins_to_trials,
    _get_trial_column_safe,
    TIER1_FEATURES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_trials_df():
    """Create a simple trials DataFrame with 3 trials.

    Trial 0: 0.5s – 1.5s, contrast_left=0.5, contrast_right=0.0
    Trial 1: 2.0s – 3.0s, contrast_left=0.0, contrast_right=1.0
    Trial 2: 4.0s – 5.0s, contrast_left=0.25, contrast_right=0.25
    """
    return pd.DataFrame({
        "start_time": [0.5, 2.0, 4.0],
        "stop_time": [1.5, 3.0, 5.0],
        "visual_stimulus_left_contrast": [0.5, 0.0, 0.25],
        "visual_stimulus_right_contrast": [0.0, 1.0, 0.25],
    })


@pytest.fixture
def bin_params():
    """Standard bin parameters: 10ms bins, 600 bins = 6 seconds."""
    return {"num_bins": 600, "bin_width_ms": 10.0}


# =============================================================================
# _assign_bins_to_trials tests
# =============================================================================

class TestAssignBinsToTrials:
    """Tests for the bin-to-trial assignment function."""

    def test_basic_assignment(self, simple_trials_df):
        """Bins within trials should be assigned correctly."""
        # 100 bins at 10ms = 1 second of data
        bin_centers = (np.arange(100) + 0.5) * 0.01  # 0.005 to 0.995 seconds

        start_times = simple_trials_df["start_time"].values
        stop_times = simple_trials_df["stop_time"].values

        assignment = _assign_bins_to_trials(bin_centers, start_times, stop_times)

        # Bins before 0.5s should be -1 (inter-trial)
        assert np.all(assignment[:50] == -1), "Bins before trial 0 should be inter-trial"
        # Bins from 0.5s to 1.5s (indices 50-99, but trial ends at 1.5s, bin_center < 1.0s)
        # Bin center at 0.505s (index 50) should be in trial 0
        assert assignment[50] == 0, "Bin at 0.505s should be in trial 0"
        # Bin center at 0.995s (index 99) should be in trial 0
        assert assignment[99] == 0, "Bin at 0.995s should be in trial 0"

    def test_inter_trial_bins(self):
        """Bins between trials should be assigned -1."""
        bin_centers = np.array([0.75, 1.75, 2.5, 3.5])
        start_times = np.array([0.5, 2.0])
        stop_times = np.array([1.0, 3.0])

        assignment = _assign_bins_to_trials(bin_centers, start_times, stop_times)

        assert assignment[0] == 0, "0.75s should be in trial 0"
        assert assignment[1] == -1, "1.75s should be inter-trial"
        assert assignment[2] == 1, "2.5s should be in trial 1"
        assert assignment[3] == -1, "3.5s should be inter-trial"

    def test_empty_trials(self):
        """No trials → all bins are inter-trial."""
        bin_centers = np.array([0.5, 1.0, 1.5])
        start_times = np.array([])
        stop_times = np.array([])

        assignment = _assign_bins_to_trials(bin_centers, start_times, stop_times)
        assert np.all(assignment == -1), "All bins should be inter-trial"

    def test_single_trial(self):
        """Single trial should assign bins correctly."""
        bin_centers = np.array([0.05, 0.5, 0.95, 1.5])
        start_times = np.array([0.1])
        stop_times = np.array([1.0])

        assignment = _assign_bins_to_trials(bin_centers, start_times, stop_times)

        assert assignment[0] == -1, "0.05s is before trial"
        assert assignment[1] == 0, "0.5s is during trial"
        assert assignment[2] == 0, "0.95s is during trial"
        assert assignment[3] == -1, "1.5s is after trial"


# =============================================================================
# _get_trial_column_safe tests
# =============================================================================

class TestGetTrialColumnSafe:
    """Tests for safe column access from trials DataFrame."""

    def test_existing_column(self, simple_trials_df):
        """Should return column values when column exists."""
        values = _get_trial_column_safe(
            simple_trials_df, "visual_stimulus_left_contrast",
        )
        np.testing.assert_array_almost_equal(values, [0.5, 0.0, 0.25])

    def test_missing_column(self, simple_trials_df):
        """Should return defaults when column is missing."""
        values = _get_trial_column_safe(
            simple_trials_df, "nonexistent_column", default=0.0,
        )
        np.testing.assert_array_equal(values, [0.0, 0.0, 0.0])

    def test_nan_handling(self):
        """NaN values should be replaced with default."""
        df = pd.DataFrame({"data": [1.0, np.nan, 3.0]})
        values = _get_trial_column_safe(df, "data", default=-1.0)
        np.testing.assert_array_equal(values, [1.0, -1.0, 3.0])


# =============================================================================
# extract_stimulus_features tests (with mocked NWB IO)
# =============================================================================

class TestExtractStimulusFeatures:
    """Tests for the main extraction function using mocked NWB files."""

    def _mock_extract(self, trials_df, num_bins=600, bin_width_ms=10.0,
                      feature_list=None):
        """Helper to call extract_stimulus_features with a mocked NWB file.

        Patches _load_trials_dataframe to return the given DataFrame
        instead of opening a real NWB file.
        """
        with patch(
            "src.data.nwb_covariates._load_trials_dataframe",
            return_value=trials_df,
        ):
            return extract_stimulus_features(
                nwb_path="fake.nwb",
                num_bins=num_bins,
                bin_width_ms=bin_width_ms,
                feature_list=feature_list,
            )

    def test_output_shape(self, simple_trials_df, bin_params):
        """Output should have shape (n_features, num_bins)."""
        covariates, names = self._mock_extract(
            simple_trials_df, **bin_params,
        )
        assert covariates.shape == (len(TIER1_FEATURES), 600)
        assert len(names) == len(TIER1_FEATURES)

    def test_stim_on_feature(self, simple_trials_df):
        """stim_on should be 1 during trials, 0 between."""
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=600, bin_width_ms=10.0,
            feature_list=["stim_on"],
        )
        stim_on = covariates[0]

        # Bin center at 0.005s (before trial 0 at 0.5s) → 0
        assert stim_on[0] == 0.0, "Before any trial should be 0"
        # Bin center at 1.005s (during trial 0: 0.5-1.5s) → 1
        assert stim_on[100] == 1.0, "During trial should be 1"
        # Bin center at 1.705s (between trials 0 and 1) → 0
        assert stim_on[170] == 0.0, "Between trials should be 0"
        # Bin center at 2.505s (during trial 1: 2.0-3.0s) → 1
        assert stim_on[250] == 1.0, "During trial 1 should be 1"

    def test_contrast_features(self, simple_trials_df):
        """Contrast features should match trial values during trials."""
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=600, bin_width_ms=10.0,
            feature_list=["contrast_left", "contrast_right"],
        )
        cl = covariates[0]  # contrast_left
        cr = covariates[1]  # contrast_right

        # During trial 0 (0.5-1.5s): left=0.5, right=0.0
        # Bin 75 → center 0.755s
        assert cl[75] == pytest.approx(0.5)
        assert cr[75] == pytest.approx(0.0)

        # During trial 1 (2.0-3.0s): left=0.0, right=1.0
        # Bin 250 → center 2.505s
        assert cl[250] == pytest.approx(0.0)
        assert cr[250] == pytest.approx(1.0)

        # Inter-trial → 0.0
        assert cl[0] == pytest.approx(0.0)
        assert cr[0] == pytest.approx(0.0)

    def test_trial_phase_feature(self, simple_trials_df):
        """trial_phase should go from 0 to 1 within each trial."""
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=600, bin_width_ms=10.0,
            feature_list=["trial_phase"],
        )
        phase = covariates[0]

        # Inter-trial bins should have phase 0
        assert phase[0] == 0.0, "Inter-trial phase should be 0"

        # Near start of trial 0 (0.5s): bin 50, center 0.505s
        # phase = (0.505 - 0.5) / (1.5 - 0.5) = 0.005
        assert phase[50] == pytest.approx(0.005, abs=0.01)

        # Near end of trial 0: bin 149, center 1.495s
        # phase = (1.495 - 0.5) / 1.0 = 0.995
        assert phase[149] == pytest.approx(0.995, abs=0.01)

    def test_inter_trial_feature(self, simple_trials_df):
        """inter_trial should be the complement of stim_on."""
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=600, bin_width_ms=10.0,
            feature_list=["stim_on", "inter_trial"],
        )
        stim_on = covariates[0]
        inter_trial = covariates[1]

        # stim_on + inter_trial should sum to 1 everywhere
        np.testing.assert_array_almost_equal(
            stim_on + inter_trial,
            np.ones(600),
            err_msg="stim_on + inter_trial should always equal 1",
        )

    def test_missing_contrast_columns(self):
        """Missing contrast columns should produce zero features."""
        # DataFrame with only start/stop times, no contrast columns
        df = pd.DataFrame({
            "start_time": [0.5, 2.0],
            "stop_time": [1.5, 3.0],
        })
        covariates, names = self._mock_extract(
            df, num_bins=400, bin_width_ms=10.0,
            feature_list=["contrast_left", "contrast_right"],
        )
        # All zeros since columns don't exist
        np.testing.assert_array_equal(covariates[0], np.zeros(400))
        np.testing.assert_array_equal(covariates[1], np.zeros(400))

    def test_no_trials_returns_zeros(self):
        """When trials table is None, all covariates should be zero."""
        covariates, names = self._mock_extract(
            None, num_bins=100, bin_width_ms=10.0,
        )
        assert covariates.shape == (len(TIER1_FEATURES), 100)
        np.testing.assert_array_equal(
            covariates, np.zeros_like(covariates),
        )

    def test_empty_trials_returns_zeros(self):
        """Empty trials DataFrame should produce all-zero covariates."""
        empty_df = pd.DataFrame(columns=["start_time", "stop_time"])
        covariates, names = self._mock_extract(
            empty_df, num_bins=100, bin_width_ms=10.0,
        )
        np.testing.assert_array_equal(
            covariates, np.zeros_like(covariates),
        )

    def test_selective_features(self, simple_trials_df):
        """Should only extract requested features."""
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=100, bin_width_ms=10.0,
            feature_list=["stim_on"],
        )
        assert covariates.shape == (1, 100)
        assert names == ["stim_on"]

    def test_invalid_feature_raises(self, simple_trials_df):
        """Invalid feature name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown feature"):
            self._mock_extract(
                simple_trials_df, num_bins=100, bin_width_ms=10.0,
                feature_list=["invalid_feature"],
            )

    def test_dtype_is_float32(self, simple_trials_df):
        """Output covariates should be float32."""
        covariates, _ = self._mock_extract(
            simple_trials_df, num_bins=100, bin_width_ms=10.0,
        )
        assert covariates.dtype == np.float32

    def test_feature_names_match_requested(self, simple_trials_df):
        """Returned feature names should match the requested list."""
        features = ["contrast_left", "trial_phase"]
        covariates, names = self._mock_extract(
            simple_trials_df, num_bins=100, bin_width_ms=10.0,
            feature_list=features,
        )
        assert names == features
