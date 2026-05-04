"""
Tests for src/data/real_data_loader.py

Tests the NWB real data loader with mock objects (no real NWB file required).
Validates unit filtering logic, spike-time-to-sample conversion, MockSorting
construction, metadata completeness, error handling, and integration with
bin_spike_trains().
"""

import numpy as np
import pytest

from src.data.modulated_generator import MockSorting
from src.data.real_data_loader import (
    filter_units,
    _spike_times_to_samples,
)
from src.data.binning import bin_spike_trains


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_spike_times():
    """
    5 units with varying firing rates over a 10-second recording.

    Unit 0: 50 spikes (5.0 Hz) — "good" quality, VISp
    Unit 1: 5 spikes (0.5 Hz) — "good" quality, VISp (below 1 Hz filter)
    Unit 2: 30 spikes (3.0 Hz) — "noise" quality, VISp
    Unit 3: 40 spikes (4.0 Hz) — "good" quality, CA1
    Unit 4: 20 spikes (2.0 Hz) — "good" quality, VISp
    """
    rng = np.random.default_rng(42)
    duration_s = 10.0
    return {
        "spike_times_list": [
            np.sort(rng.uniform(0, duration_s, size=50)),   # Unit 0
            np.sort(rng.uniform(0, duration_s, size=5)),    # Unit 1
            np.sort(rng.uniform(0, duration_s, size=30)),   # Unit 2
            np.sort(rng.uniform(0, duration_s, size=40)),   # Unit 3
            np.sort(rng.uniform(0, duration_s, size=20)),   # Unit 4
        ],
        "unit_indices": [0, 1, 2, 3, 4],
        "quality_list": ["good", "good", "noise", "good", "good"],
        "brain_region_list": ["VISp", "VISp", "VISp", "CA1", "VISp"],
        "duration_s": duration_s,
        "sampling_frequency": 30000.0,
    }


# ---------------------------------------------------------------------------
# Spike-time-to-sample conversion tests
# ---------------------------------------------------------------------------

class TestSpikeTimesToSamples:
    """Tests for _spike_times_to_samples()."""

    def test_known_answer(self):
        """Spike at exactly 1.0s should map to sample 30000 at 30kHz."""
        spike_times = np.array([1.0])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        assert samples[0] == 30000

    def test_multiple_spikes(self):
        """Multiple spikes should be converted and sorted."""
        spike_times = np.array([0.5, 0.1, 0.9])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        # Should be sorted after np.unique
        assert np.all(np.diff(samples) > 0)
        assert len(samples) == 3

    def test_clips_to_max_sample(self):
        """Spikes beyond max_sample should be excluded."""
        spike_times = np.array([0.5, 1.5, 2.5])
        # max_sample at 1.0s → only first spike should survive
        samples = _spike_times_to_samples(spike_times, 30000.0, 30000)
        assert len(samples) == 1
        assert samples[0] == 15000  # 0.5s * 30000 Hz

    def test_removes_negative_times(self):
        """Negative spike times should be excluded."""
        spike_times = np.array([-0.5, 0.1, 0.5])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        # -0.5s → -15000 (filtered out)
        assert len(samples) == 2

    def test_empty_input(self):
        """Empty spike array should return empty."""
        spike_times = np.array([])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        assert len(samples) == 0

    def test_output_is_int64(self):
        """Sample indices should be int64."""
        spike_times = np.array([0.1, 0.2])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        assert samples.dtype == np.int64

    def test_removes_duplicates(self):
        """Very close spike times that round to the same sample should be deduplicated."""
        # Two spike times that differ by less than 1/fs
        spike_times = np.array([0.100000, 0.100001])
        samples = _spike_times_to_samples(spike_times, 30000.0, 300000)
        # At 30kHz, 0.100000 → 3000, 0.100001 → 3000 (same sample)
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# Unit filtering tests
# ---------------------------------------------------------------------------

class TestFilterUnits:
    """Tests for the filter_units() function."""

    def test_no_filters_keeps_all(self, sample_spike_times):
        """With no filters, all units should be kept."""
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            min_firing_rate_hz=0.0,
            max_units=None,
            quality_labels=None,
            quality_list=None,
        )
        assert len(filtered_st) == 5
        assert stats["final_units"] == 5

    def test_quality_filter(self, sample_spike_times):
        """Quality filter should keep only 'good' units (0, 1, 3, 4)."""
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            quality_labels=["good"],
            quality_list=sample_spike_times["quality_list"],
        )
        assert len(filtered_st) == 4  # Units 0, 1, 3, 4
        assert 2 not in filtered_idx  # Unit 2 was "noise"
        assert stats["after_quality_filter"] == 4

    def test_min_rate_filter(self, sample_spike_times):
        """Min rate filter at 1.0 Hz should exclude unit 1 (0.5 Hz)."""
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            min_firing_rate_hz=1.0,
        )
        # Unit 1 has 5 spikes / 10s = 0.5 Hz → excluded
        assert len(filtered_st) == 4
        assert 1 not in filtered_idx

    def test_brain_region_filter(self, sample_spike_times):
        """Region filter for 'VISp' should exclude unit 3 (CA1)."""
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            brain_region="VISp",
            brain_region_list=sample_spike_times["brain_region_list"],
        )
        assert len(filtered_st) == 4  # Units 0, 1, 2, 4
        assert 3 not in filtered_idx

    def test_max_units_cap(self, sample_spike_times):
        """Max units cap should reduce unit count via random subsample."""
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            max_units=2,
            rng=np.random.default_rng(42),
        )
        assert len(filtered_st) == 2
        assert stats["final_units"] == 2

    def test_combined_filters(self, sample_spike_times):
        """
        Combined filters: quality='good' + min_rate=1.0 + region='VISp'.

        Expected survivors:
          - Unit 0: good, VISp, 5.0 Hz ✓
          - Unit 1: good, VISp, 0.5 Hz ✗ (rate too low)
          - Unit 2: noise ✗ (quality)
          - Unit 3: good, CA1 ✗ (wrong region)
          - Unit 4: good, VISp, 2.0 Hz ✓
        """
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            min_firing_rate_hz=1.0,
            quality_labels=["good"],
            quality_list=sample_spike_times["quality_list"],
            brain_region="VISp",
            brain_region_list=sample_spike_times["brain_region_list"],
        )
        assert len(filtered_st) == 2
        assert sorted(filtered_idx) == [0, 4]

    def test_all_filtered_out_returns_empty(self):
        """If all units are filtered out, should return empty lists."""
        spike_times_list = [np.array([0.1])]  # 1 spike in 10s = 0.1 Hz
        filtered_st, filtered_idx, stats = filter_units(
            spike_times_list=spike_times_list,
            unit_indices=[0],
            sampling_frequency=30000.0,
            duration_s=10.0,
            min_firing_rate_hz=5.0,  # 0.1 Hz < 5.0 Hz
        )
        assert len(filtered_st) == 0
        assert stats["final_units"] == 0

    def test_filter_stats_keys(self, sample_spike_times):
        """Filter stats dict should always contain initial and final counts."""
        _, _, stats = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
        )
        assert "initial_units" in stats
        assert "final_units" in stats

    def test_max_units_preserves_order(self, sample_spike_times):
        """After subsampling, indices should remain in original order."""
        _, filtered_idx, _ = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            max_units=3,
            rng=np.random.default_rng(42),
        )
        assert filtered_idx == sorted(filtered_idx)


# ---------------------------------------------------------------------------
# MockSorting construction from filtered data
# ---------------------------------------------------------------------------

class TestMockSortingFromFilteredData:
    """Tests for constructing MockSorting from filtered spike data."""

    def test_mock_sorting_round_trip(self, sample_spike_times):
        """
        Filtered data → MockSorting → bin_spike_trains should produce
        valid spike-count matrix.
        """
        # Filter to 'good' quality, >=1 Hz
        filtered_st, filtered_idx, _ = filter_units(
            spike_times_list=sample_spike_times["spike_times_list"],
            unit_indices=sample_spike_times["unit_indices"],
            sampling_frequency=sample_spike_times["sampling_frequency"],
            duration_s=sample_spike_times["duration_s"],
            min_firing_rate_hz=1.0,
            quality_labels=["good"],
            quality_list=sample_spike_times["quality_list"],
        )

        # Convert to sample indices and build MockSorting
        fs = sample_spike_times["sampling_frequency"]
        max_sample = int(sample_spike_times["duration_s"] * fs)
        spike_trains = {}
        for new_id, st in enumerate(filtered_st):
            samples = _spike_times_to_samples(st, fs, max_sample)
            spike_trains[new_id] = samples

        sorting = MockSorting(spike_trains, fs)

        # Verify MockSorting interface
        assert len(sorting.get_unit_ids()) == len(filtered_st)
        assert sorting.get_sampling_frequency() == fs

        # Run through bin_spike_trains
        spike_counts, metadata = bin_spike_trains(sorting, bin_width_ms=10.0)

        # Validate output shape and types
        assert spike_counts.shape[0] == len(filtered_st)
        assert spike_counts.shape[1] > 0
        assert spike_counts.dtype == np.int32
        assert np.all(spike_counts >= 0)
        assert metadata["num_units"] == len(filtered_st)

    def test_binned_total_spikes_reasonable(self, sample_spike_times):
        """
        Total binned spikes should be close to input spike count
        (some spikes may fall outside the last partial bin).
        """
        fs = sample_spike_times["sampling_frequency"]
        duration_s = sample_spike_times["duration_s"]
        max_sample = int(duration_s * fs)

        # Use unit 0 only (50 spikes)
        st = sample_spike_times["spike_times_list"][0]
        samples = _spike_times_to_samples(st, fs, max_sample)
        sorting = MockSorting({0: samples}, fs)

        counts, meta = bin_spike_trains(sorting, bin_width_ms=10.0)

        # All 50 spikes should be binned (10s / 10ms = 1000 bins, all fit)
        assert counts.sum() == len(samples)


# ---------------------------------------------------------------------------
# load_nwb_spikes integration tests (mocked pynwb)
# ---------------------------------------------------------------------------

class TestLoadNwbSpikesErrors:
    """Tests for error handling in load_nwb_spikes."""

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing NWB file."""
        from src.data.real_data_loader import load_nwb_spikes

        config = {
            "source": {"path": "nonexistent/file.nwb"},
            "seed": 42,
        }
        # The function will try to import pynwb first, then check file
        # If pynwb is not installed, ImportError is raised instead
        with pytest.raises((FileNotFoundError, ImportError)):
            load_nwb_spikes(config)

    def test_import_error_message(self):
        """ImportError should mention pynwb install instructions."""
        # This test is meaningful only if pynwb is NOT installed.
        # If it IS installed, we skip.
        try:
            import pynwb  # noqa: F401
            pytest.skip("pynwb is installed; cannot test ImportError path")
        except ImportError:
            pass

        from src.data.real_data_loader import load_nwb_spikes

        config = {
            "source": {"path": "some_file.nwb"},
            "seed": 42,
        }
        with pytest.raises(ImportError, match="pynwb"):
            load_nwb_spikes(config)
