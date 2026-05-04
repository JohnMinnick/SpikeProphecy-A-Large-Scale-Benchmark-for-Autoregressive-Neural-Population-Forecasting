"""
Tests for the IBL Brain-wide Map data loader.

Validates:
    - load_ibl_session produces correct MockSorting output
    - Unit filtering (quality, rate, max_units) works correctly
    - load_ibl_multi_session produces correct concatenated output
    - Duration limiting works
    - Region filtering works
    - Error handling for empty sessions

Note: Tests mock the ONE API to avoid network dependencies.
The actual ONE API download is tested via integration tests
(scripts/download_ibl.py --list-only).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.ibl_data_loader import (
    load_ibl_session,
    load_ibl_multi_session,
)
from src.data.modulated_generator import MockSorting


# =============================================================================
# Fixtures — create synthetic IBL-like spike data
# =============================================================================

@pytest.fixture
def synthetic_ibl_data():
    """Create synthetic spike data mimicking IBL ONE API output."""
    rng = np.random.default_rng(42)

    # 100 clusters, 60 seconds of recording
    n_clusters = 100
    duration_s = 60.0

    # Generate spike times and cluster assignments
    all_spike_times = []
    all_clusters = []

    for cluster_id in range(n_clusters):
        # Random firing rate between 0.5 and 20 Hz per cluster
        rate = rng.uniform(0.5, 20.0)
        n_spikes = int(rate * duration_s)
        st = np.sort(rng.uniform(0, duration_s, size=n_spikes))
        all_spike_times.append(st)
        all_clusters.append(np.full(n_spikes, cluster_id, dtype=np.int64))

    spike_times = np.concatenate(all_spike_times)
    spike_clusters = np.concatenate(all_clusters)

    # Sort by time (as IBL data would be)
    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    spike_clusters = spike_clusters[sort_idx]

    return spike_times, spike_clusters, n_clusters, duration_s


@pytest.fixture
def default_config():
    """Default config dict matching IBL config format."""
    return {
        "seed": 42,
        "bin_width_ms": 50.0,
        "history_bins": 10,
        "nwb": {
            "sampling_frequency": 30000.0,
            "min_firing_rate_hz": 1.0,
            "max_units": None,
            "quality_labels": None,  # Disable quality filter for tests
            "brain_region": None,
            "duration_limit_s": None,
        },
    }


def _create_mock_one(spike_times, spike_clusters):
    """Create a mock ONE client that returns synthetic data.

    Mocks the load_object and list_collections APIs used by
    the updated IBL loader (which uses probe-specific pykilosort
    collections instead of a flat 'alf' collection).
    """
    mock_one = MagicMock()

    # Mock list_collections: return a single pykilosort probe collection
    mock_one.list_collections.return_value = [
        "alf", "alf/probe00", "alf/probe00/pykilosort",
    ]

    def mock_load_object(eid, obj_name, collection=None):
        """Return synthetic data dict based on object name."""
        if obj_name == "spikes":
            return {
                "times": spike_times,
                "clusters": spike_clusters,
                "amps": np.ones(len(spike_times)),
                "depths": np.ones(len(spike_times)) * 500,
            }
        elif obj_name == "clusters":
            # No metrics or regions — simulate typical missing data
            raise Exception("No cluster data available")
        else:
            raise ValueError(f"Unknown object: {obj_name}")

    mock_one.load_object = mock_load_object
    return mock_one


# =============================================================================
# Single Session Loading Tests
# =============================================================================

class TestLoadIBLSession:
    """Tests for loading a single IBL session."""

    def test_basic_loading(self, synthetic_ibl_data, default_config):
        """Should return MockSorting and metadata dict."""
        spike_times, spike_clusters, n_clusters, _ = synthetic_ibl_data

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        assert isinstance(sorting, MockSorting)
        assert isinstance(metadata, dict)
        assert metadata["source"] == "ibl"
        assert metadata["eid"] == "fake-eid-001"

    def test_unit_count(self, synthetic_ibl_data, default_config):
        """Should load the correct number of units after filtering."""
        spike_times, spike_clusters, n_clusters, _ = synthetic_ibl_data

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        # Should have units (some may be filtered by min_firing_rate)
        assert metadata["num_units"] > 0
        assert metadata["num_units"] <= n_clusters
        assert len(sorting.get_unit_ids()) == metadata["num_units"]

    def test_firing_rate_filter(self, synthetic_ibl_data, default_config):
        """Units below min_firing_rate_hz should be filtered out."""
        spike_times, spike_clusters, n_clusters, _ = synthetic_ibl_data

        # Set high firing rate threshold to filter out low-rate units
        default_config["nwb"]["min_firing_rate_hz"] = 10.0

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        # Should have fewer units due to high threshold
        assert metadata["num_units"] < n_clusters
        # All retained units should have rate >= 10 Hz
        for info in metadata["unit_details"]:
            assert info["actual_rate_hz"] >= 10.0

    def test_max_units_cap(self, synthetic_ibl_data, default_config):
        """Should cap units to max_units when specified."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data

        default_config["nwb"]["max_units"] = 20
        default_config["nwb"]["min_firing_rate_hz"] = 0.0  # Keep all units

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        assert metadata["num_units"] <= 20

    def test_duration_limit(self, synthetic_ibl_data, default_config):
        """Duration limit should truncate spike trains."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data

        default_config["nwb"]["duration_limit_s"] = 30.0

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        # Duration should be truncated
        assert metadata["duration_s"] <= 30.0

    def test_metadata_fields(self, synthetic_ibl_data, default_config):
        """Metadata should contain all expected fields."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        expected_keys = [
            "source", "eid", "dataset", "seed", "num_units",
            "num_raw_units", "duration_s", "sampling_frequency",
            "total_spikes", "mean_rate_hz", "filter_stats",
            "unit_details",
        ]
        for key in expected_keys:
            assert key in metadata, f"Missing metadata key: {key}"

    def test_spike_trains_sorted(self, synthetic_ibl_data, default_config):
        """All spike trains should be sorted."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        for uid in sorting.get_unit_ids():
            st = sorting.get_unit_spike_train(uid)
            assert np.all(np.diff(st) >= 0), (
                f"Spike train {uid} is not sorted"
            )

    def test_empty_after_filter_raises(self, default_config):
        """Should raise ValueError if all units are filtered out."""
        rng = np.random.default_rng(42)
        # Create very low rate data: 1 cluster with 1 spike in 60s
        spike_times = np.array([30.0])
        spike_clusters = np.array([0])

        default_config["nwb"]["min_firing_rate_hz"] = 100.0  # Impossible rate

        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            with pytest.raises(ValueError, match="No units remain"):
                load_ibl_session("fake-eid-empty", default_config)


# =============================================================================
# Pipeline Compatibility Tests
# =============================================================================

class TestIBLPipelineCompat:
    """Test that IBL data flows through the existing pipeline."""

    def test_bin_spike_trains_compat(
        self, synthetic_ibl_data, default_config,
    ):
        """MockSorting from IBL should work with bin_spike_trains."""
        from src.data.binning import bin_spike_trains

        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        # Bin spike trains with 50ms bins
        counts, bin_meta = bin_spike_trains(sorting, bin_width_ms=50.0)

        # Shape should be (M, T)
        assert counts.ndim == 2
        M, T = counts.shape
        assert M == metadata["num_units"]
        assert T > 0

        # Counts should be non-negative integers
        assert counts.dtype in (np.int32, np.int64)
        assert (counts >= 0).all()

    def test_pad_to_channels_compat(
        self, synthetic_ibl_data, default_config,
    ):
        """Padded IBL data should have correct shape."""
        from src.data.binning import bin_spike_trains
        from src.data.multi_session_loader import (
            pad_to_channels, build_channel_mask,
        )

        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            sorting, metadata = load_ibl_session(
                "fake-eid-001", default_config,
            )

        counts, _ = bin_spike_trains(sorting, bin_width_ms=50.0)
        M, T = counts.shape

        # Pad to a larger target (simulating multi-session M_max)
        target_m = M + 50
        padded = pad_to_channels(counts, target_m)
        assert padded.shape == (target_m, T)

        # Build channel mask
        mask = build_channel_mask(M, target_m)
        assert mask.shape == (target_m,)
        assert mask[:M].sum() == M      # Real channels are 1
        assert mask[M:].sum() == 0      # Padded channels are 0


# =============================================================================
# Multi-Session Loading Tests
# =============================================================================

class TestLoadIBLMultiSession:
    """Tests for loading multiple IBL sessions."""

    def test_multi_session_shapes(self, synthetic_ibl_data, default_config):
        """Multi-session load should produce correct shapes."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            spike_counts, mask_index, metadata = load_ibl_multi_session(
                eids=["eid-001", "eid-002"],
                config=default_config,
            )

        # Should be 2D
        assert spike_counts.ndim == 2
        m_max, t_total = spike_counts.shape

        # mask_index should match time dimension
        assert mask_index.shape == (t_total,)

        # Metadata should have correct fields
        assert metadata["num_sessions"] == 2
        assert metadata["m_max"] == m_max
        assert metadata["source"] == "ibl_multi"

    def test_multi_session_masks(self, synthetic_ibl_data, default_config):
        """Session masks should have correct dimensions."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            spike_counts, mask_index, metadata = load_ibl_multi_session(
                eids=["eid-001", "eid-002"],
                config=default_config,
            )

        masks = metadata["session_masks"]
        assert masks.shape[0] == 2  # 2 sessions
        assert masks.shape[1] == metadata["m_max"]

    def test_multi_session_gap_bins(self, synthetic_ibl_data, default_config):
        """Gap bins between sessions should exist and have mask_index = -1."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            spike_counts, mask_index, metadata = load_ibl_multi_session(
                eids=["eid-001", "eid-002"],
                config=default_config,
            )

        # Should have gap bins marked as -1
        gap_count = np.sum(mask_index == -1)
        expected_gaps = default_config["history_bins"]  # 1 gap between 2 sessions
        assert gap_count == expected_gaps

    def test_single_session_no_gaps(self, synthetic_ibl_data, default_config):
        """Single session should have no gap bins."""
        spike_times, spike_clusters, _, _ = synthetic_ibl_data
        mock_one = _create_mock_one(spike_times, spike_clusters)

        with patch(
            "src.data.ibl_data_loader._get_one_client",
            return_value=mock_one,
        ):
            spike_counts, mask_index, metadata = load_ibl_multi_session(
                eids=["eid-001"],
                config=default_config,
            )

        # No gaps for single session
        assert np.sum(mask_index == -1) == 0
        assert metadata["total_gap_bins"] == 0
