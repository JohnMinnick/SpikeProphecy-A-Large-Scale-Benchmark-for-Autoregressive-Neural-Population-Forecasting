"""
Tests for src/data/binning.py

Tests spike-train binning with known-answer verification, edge case handling,
shape validation, and save/load round-trip consistency.
"""

import json

import numpy as np
import pytest

from src.data.binning import (
    bin_spike_trains,
    load_binned_data,
    save_binned_data,
    validate_spike_counts,
)


class FakeSorting:
    """
    Minimal mock that matches SpikeInterface SortingExtractor API.

    Unlike a generic mock, this actually stores and returns real spike data
    so tests can verify real binning logic.
    """

    def __init__(self, unit_spike_trains, sampling_frequency=30000.0):
        """
        Args:
            unit_spike_trains: dict mapping unit_id -> np.ndarray of sample indices.
            sampling_frequency: Sampling rate in Hz.
        """
        self._trains = unit_spike_trains
        self._fs = sampling_frequency

    def get_unit_ids(self):
        """Return unit IDs as a numpy array (matches SpikeInterface behavior)."""
        return np.array(list(self._trains.keys()))

    def get_sampling_frequency(self):
        """Return sampling frequency."""
        return self._fs

    def get_unit_spike_train(self, unit_id):
        """Return spike train (sample indices) for a unit."""
        return self._trains[unit_id]


class FakeRecording:
    """Minimal mock matching SpikeInterface RecordingExtractor API."""

    def __init__(self, num_samples, sampling_frequency=30000.0):
        self._num_samples = num_samples
        self._fs = sampling_frequency

    def get_num_samples(self):
        """Return total number of samples."""
        return self._num_samples

    def get_sampling_frequency(self):
        """Return sampling frequency."""
        return self._fs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_sorting():
    """
    3 units, 1 second at 30 kHz, known spike positions.

    Unit 0: spikes at 5ms, 15ms, 25ms  (samples: 150, 450, 750)
    Unit 1: spikes at 10ms, 10.5ms     (samples: 300, 315)  ← two in same bin
    Unit 2: no spikes (silent unit)
    """
    trains = {
        "0": np.array([150, 450, 750]),
        "1": np.array([300, 315]),
        "2": np.array([], dtype=np.int64),
    }
    return FakeSorting(trains, sampling_frequency=30000.0)


@pytest.fixture
def simple_recording():
    """1-second recording at 30 kHz."""
    return FakeRecording(num_samples=30000, sampling_frequency=30000.0)


# ---------------------------------------------------------------------------
# bin_spike_trains tests
# ---------------------------------------------------------------------------

class TestBinSpikeTrains:
    """Tests for bin_spike_trains()."""

    def test_output_shape(self, simple_sorting, simple_recording):
        """Output should be (M=3, T=100) for 1s at 10ms bins."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert counts.shape == (3, 100)
        assert meta["num_units"] == 3
        assert meta["num_bins"] == 100

    def test_output_dtype_is_int32(self, simple_sorting, simple_recording):
        """Spike counts should be int32."""
        counts, _ = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert counts.dtype == np.int32

    def test_known_answer_unit0(self, simple_sorting, simple_recording):
        """
        Unit 0 has spikes at samples 150, 450, 750.
        With 10ms bins (300 samples/bin):
          - Sample 150 → bin 0
          - Sample 450 → bin 1
          - Sample 750 → bin 2
        """
        counts, _ = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        # Bin 0 should have 1 spike, bin 1 should have 1, bin 2 should have 1
        assert counts[0, 0] == 1
        assert counts[0, 1] == 1
        assert counts[0, 2] == 1
        # All other bins for unit 0 should be 0
        assert counts[0, 3:].sum() == 0
        # Total for unit 0
        assert counts[0, :].sum() == 3

    def test_known_answer_unit1_overlap(self, simple_sorting, simple_recording):
        """
        Unit 1 has spikes at samples 300, 315.
        Both fall in bin 1 (samples 300-599).
        """
        counts, _ = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert counts[1, 1] == 2  # Two spikes in the same bin
        assert counts[1, :].sum() == 2  # Only 2 spikes total

    def test_silent_unit(self, simple_sorting, simple_recording):
        """Unit 2 (no spikes) should have all zeros."""
        counts, _ = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert counts[2, :].sum() == 0

    def test_no_negative_counts(self, simple_sorting, simple_recording):
        """Spike counts should never be negative."""
        counts, _ = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert np.all(counts >= 0)

    def test_total_spikes_in_metadata(self, simple_sorting, simple_recording):
        """Metadata should report correct total spike count."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        assert meta["total_spikes"] == 5  # 3 + 2 + 0

    def test_different_bin_widths(self, simple_sorting, simple_recording):
        """Wider bins should produce fewer bins but same total spikes."""
        counts_10, meta_10 = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        counts_20, meta_20 = bin_spike_trains(
            simple_sorting, bin_width_ms=20.0, recording=simple_recording,
        )
        # 20ms bins → 50 bins for 1s
        assert counts_20.shape == (3, 50)
        # Total spikes should be the same regardless of bin width
        assert counts_10.sum() == counts_20.sum()

    def test_metadata_keys(self, simple_sorting, simple_recording):
        """Metadata should contain all required keys from ADR-0003."""
        _, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        required_keys = [
            "source", "bin_width_ms", "sampling_rate",
            "unit_ids", "duration_s", "num_bins", "num_units",
            "total_spikes", "firing_rates_hz",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"

    def test_duration_matches_bins(self, simple_sorting, simple_recording):
        """Duration should equal num_bins × bin_width."""
        _, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        expected_duration = meta["num_bins"] * meta["bin_width_ms"] / 1000.0
        assert abs(meta["duration_s"] - expected_duration) < 1e-6

    def test_without_recording_infers_duration(self, simple_sorting):
        """Without recording, duration should be inferred from spikes."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=None,
        )
        # Last spike is at sample 750; + bin_width (300) = 1050 samples
        # 1050 // 300 = 3 bins
        assert meta["num_bins"] == 3
        assert counts.shape == (3, 3)


# ---------------------------------------------------------------------------
# validate_spike_counts tests
# ---------------------------------------------------------------------------

class TestValidateSpikeCountts:
    """Tests for validate_spike_counts()."""

    def test_valid_data_passes(self, simple_sorting, simple_recording):
        """Normal data should pass validation."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        result = validate_spike_counts(counts, meta)
        assert result["stats"]["total_spikes"] == 5
        assert result["stats"]["shape"] == (3, 100)

    def test_detects_silent_units(self, simple_sorting, simple_recording):
        """Validation should warn about silent units."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        result = validate_spike_counts(counts, meta, min_rate_hz=0.1)
        # Unit 2 has 0 Hz, which is < 0.1 Hz
        has_silent_warning = any("firing rate" in w.lower() for w in result["warnings"])
        assert has_silent_warning

    def test_statistics_are_correct(self, simple_sorting, simple_recording):
        """Stats dict should have accurate values."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        result = validate_spike_counts(counts, meta)
        stats = result["stats"]
        assert stats["total_spikes"] == 5
        assert stats["max_count_per_bin"] == 2  # Unit 1 has 2 in one bin
        assert stats["dtype"] == "int32"


# ---------------------------------------------------------------------------
# save/load round-trip tests
# ---------------------------------------------------------------------------

class TestSaveLoadBinnedData:
    """Tests for save_binned_data() and load_binned_data()."""

    def test_round_trip(self, simple_sorting, simple_recording, tmp_path):
        """Save and reload should produce identical data."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        # Save
        save_binned_data(counts, meta, tmp_path, name="test")

        # Load
        loaded_counts, loaded_meta = load_binned_data(tmp_path, name="test")

        # Verify exact match
        np.testing.assert_array_equal(counts, loaded_counts)
        assert loaded_meta["num_units"] == meta["num_units"]
        assert loaded_meta["num_bins"] == meta["num_bins"]
        assert loaded_meta["total_spikes"] == meta["total_spikes"]
        assert loaded_meta["bin_width_ms"] == meta["bin_width_ms"]

    def test_creates_output_directory(self, simple_sorting, simple_recording, tmp_path):
        """Should create nested directories if they don't exist."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        nested_dir = tmp_path / "nested" / "dir"
        save_binned_data(counts, meta, nested_dir)
        assert (nested_dir / "binned_counts.npy").is_file()
        assert (nested_dir / "binned_metadata.json").is_file()

    def test_saved_metadata_is_valid_json(self, simple_sorting, simple_recording, tmp_path):
        """Metadata file should be valid, parseable JSON."""
        counts, meta = bin_spike_trains(
            simple_sorting, bin_width_ms=10.0, recording=simple_recording,
        )
        save_binned_data(counts, meta, tmp_path)
        with open(tmp_path / "binned_metadata.json", "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)
        assert parsed["num_units"] == 3
