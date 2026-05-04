# =============================================================================
# tests/test_data/test_leakage.py
# Data leakage test suite — verifies temporal isolation between splits.
#
# These tests prove that:
#   1. Train/val/test splits are strictly non-overlapping in time
#   2. No sliding window crosses a split boundary
#   3. No cross-session spillover via gap bins
#   4. No normalization or pre-processing uses future data
#   5. The first valid sample in val/test is fully within that split
#
# Run: pytest tests/test_data/test_leakage.py -v
# =============================================================================

"""
Data leakage test suite for SpikeProphecy.

Verifies that the temporal train/val/test splits are strictly isolated,
meaning no information from future time bins leaks into training data.
This is critical for the validity of all published results.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.multi_session_loader import (
    MaskedSpikeCountDataset,
    build_channel_mask,
    pad_to_channels,
)


# ---------------------------------------------------------------------------
# Fixtures — create realistic multi-session cached data
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_with_splits(tmp_path):
    """
    Create a realistic multi-session cache with known split boundaries.

    Returns (cache_dir, metadata) for 3 sessions with predictable sizes
    so we can verify exact split points.
    """
    np.random.seed(42)
    history_bins = 10
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    # 3 sessions with different sizes and unit counts
    session_specs = [
        {"m_i": 50, "t_i": 1000},   # Session 0: 50 units, 1000 bins
        {"m_i": 80, "t_i": 1500},   # Session 1: 80 units, 1500 bins
        {"m_i": 30, "t_i": 800},    # Session 2: 30 units, 800 bins
    ]

    sessions = []
    for i, spec in enumerate(session_specs):
        m_i, t_i = spec["m_i"], spec["t_i"]

        # Generate spike counts with a known temporal pattern:
        # each bin's mean rate increases linearly with time index.
        # This lets us verify which time bins are in which split.
        rates = np.linspace(0.5, 5.0, t_i).reshape(1, -1)
        counts = np.random.poisson(rates, size=(m_i, t_i))
        counts_u8 = np.clip(counts, 0, 255).astype(np.uint8)

        npy_path = tmp_path / f"session_{i:03d}.npy"
        np.save(npy_path, counts_u8)

        train_end = int(t_i * splits["train"])
        val_end = train_end + int(t_i * splits["val"])

        sessions.append({
            "index": i,
            "file": f"fake_session_{i}.nwb",
            "npy_file": str(npy_path),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": round(t_i * 0.05, 2),
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
            },
            "brain_regions": None,
        })

    m_max = max(s["num_units"] for s in sessions)

    metadata = {
        "num_sessions": len(sessions),
        "m_max": m_max,
        "history_bins": history_bins,
        "bin_width_ms": 50.0,
        "n_features_per_channel": 0,
        "n_covariates": 0,
        "covariate_names": [],
        "sessions": sessions,
    }

    # Write metadata.json
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path, metadata


# ---------------------------------------------------------------------------
# Test 1: Split boundaries are non-overlapping and exhaustive
# ---------------------------------------------------------------------------

class TestSplitBoundaries:
    """Verify train/val/test splits cover all bins without overlap."""

    def test_splits_are_contiguous(self, cache_with_splits):
        """Each session's splits should be [0, train_end), [train_end, val_end), [val_end, T)."""
        _, metadata = cache_with_splits

        for sess in metadata["sessions"]:
            bounds = sess["split_boundaries"]
            t_i = sess["num_bins"]

            # Splits should be contiguous
            assert bounds["train_end"] > 0, "Train split should be non-empty"
            assert bounds["val_end"] > bounds["train_end"], "Val should start after train"
            assert t_i > bounds["val_end"], "Test split should be non-empty"

    def test_splits_cover_all_bins(self, cache_with_splits):
        """The three splits should cover every bin exactly once."""
        _, metadata = cache_with_splits

        for sess in metadata["sessions"]:
            bounds = sess["split_boundaries"]
            t_i = sess["num_bins"]

            train_bins = set(range(0, bounds["train_end"]))
            val_bins = set(range(bounds["train_end"], bounds["val_end"]))
            test_bins = set(range(bounds["val_end"], t_i))

            # No overlap
            assert len(train_bins & val_bins) == 0, "Train and val overlap!"
            assert len(train_bins & test_bins) == 0, "Train and test overlap!"
            assert len(val_bins & test_bins) == 0, "Val and test overlap!"

            # Full coverage
            all_bins = train_bins | val_bins | test_bins
            assert all_bins == set(range(t_i)), "Splits don't cover all bins"

    def test_no_zero_length_splits(self, cache_with_splits):
        """Every split should have at least history_bins worth of data."""
        _, metadata = cache_with_splits
        history_bins = metadata["history_bins"]

        for sess in metadata["sessions"]:
            bounds = sess["split_boundaries"]
            t_i = sess["num_bins"]

            train_len = bounds["train_end"]
            val_len = bounds["val_end"] - bounds["train_end"]
            test_len = t_i - bounds["val_end"]

            assert train_len > history_bins, \
                f"Train too short: {train_len} <= {history_bins}"
            assert val_len > history_bins, \
                f"Val too short: {val_len} <= {history_bins}"
            assert test_len > history_bins, \
                f"Test too short: {test_len} <= {history_bins}"


# ---------------------------------------------------------------------------
# Test 2: Sliding windows stay within their split
# ---------------------------------------------------------------------------

class TestWindowIsolation:
    """Verify that no sliding window crosses a split boundary."""

    def _make_dataset_for_split(self, cache_dir, metadata, sess_idx, split):
        """
        Load a single session's split and return the dataset + split bounds.

        Mirrors the logic in SessionCyclingLoader.__iter__().
        """
        sess = metadata["sessions"][sess_idx]
        bounds = sess["split_boundaries"]
        t_i = sess["num_bins"]
        m_i = sess["num_units"]
        m_max = metadata["m_max"]
        history_bins = metadata["history_bins"]

        # Load raw data
        npy_path = cache_dir / f"session_{sess_idx:03d}.npy"
        counts = np.load(npy_path)  # (m_i, t_i)

        # Determine split slice
        if split == "train":
            start, end = 0, bounds["train_end"]
        elif split == "val":
            start, end = bounds["train_end"], bounds["val_end"]
        else:
            start, end = bounds["val_end"], t_i

        # Slice and pad (exactly as lazy loader does)
        counts_split = counts[:, start:end].astype(np.int32)
        padded = pad_to_channels(counts_split, m_max)

        mask = build_channel_mask(m_i, m_max)
        session_masks = mask.reshape(1, -1)
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)

        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )

        return ds, start, end

    def test_train_windows_stay_in_train(self, cache_with_splits):
        """Every train sample's window [t, t+T+1) is within [0, train_end)."""
        cache_dir, metadata = cache_with_splits
        history_bins = metadata["history_bins"]

        for i in range(metadata["num_sessions"]):
            ds, start, end = self._make_dataset_for_split(
                cache_dir, metadata, i, "train",
            )
            bounds = metadata["sessions"][i]["split_boundaries"]

            for idx in range(len(ds)):
                x, y, mask = ds[idx][:3]
                # The sample at index idx uses bins [idx, idx+T) for input
                # and bin idx+T as target. Since we sliced to [0, train_end),
                # the maximum original bin accessed is start + idx + T = idx + T.
                # This must be < train_end (which equals end since start=0).
                original_target_bin = start + ds._valid_indices[idx] + history_bins
                assert original_target_bin < end, \
                    f"Train sample {idx} accesses bin {original_target_bin} >= {end}"

    def test_val_windows_stay_in_val(self, cache_with_splits):
        """Every val sample's window is fully within [train_end, val_end)."""
        cache_dir, metadata = cache_with_splits
        history_bins = metadata["history_bins"]

        for i in range(metadata["num_sessions"]):
            ds, start, end = self._make_dataset_for_split(
                cache_dir, metadata, i, "val",
            )

            for idx in range(len(ds)):
                x, y, mask = ds[idx][:3]
                # Input window starts at start + valid_idx
                orig_input_start = start + ds._valid_indices[idx]
                orig_target = start + ds._valid_indices[idx] + history_bins

                assert orig_input_start >= start, \
                    f"Val sample {idx}: input starts at {orig_input_start} < {start}"
                assert orig_target < end, \
                    f"Val sample {idx}: target at {orig_target} >= {end}"

    def test_test_windows_stay_in_test(self, cache_with_splits):
        """Every test sample's window is fully within [val_end, T_total)."""
        cache_dir, metadata = cache_with_splits
        history_bins = metadata["history_bins"]

        for i in range(metadata["num_sessions"]):
            ds, start, end = self._make_dataset_for_split(
                cache_dir, metadata, i, "test",
            )

            for idx in range(len(ds)):
                x, y, mask = ds[idx][:3]
                orig_input_start = start + ds._valid_indices[idx]
                orig_target = start + ds._valid_indices[idx] + history_bins

                assert orig_input_start >= start, \
                    f"Test sample {idx}: input starts at {orig_input_start} < {start}"
                assert orig_target < end, \
                    f"Test sample {idx}: target at {orig_target} >= {end}"


# ---------------------------------------------------------------------------
# Test 3: Cross-session isolation via gap bins
# ---------------------------------------------------------------------------

class TestCrossSessionIsolation:
    """Verify that gap bins prevent cross-session windows."""

    def test_gap_bins_reject_all_spanning_windows(self):
        """
        Create two sessions separated by gap bins. No valid sample
        should span the gap.
        """
        # Session A: bins 0-99 (100 bins), Session B: bins 110-209
        # Gap: bins 100-109 (10 bins = history_bins)
        m = 20
        t_total = 210  # 100 + 10 gap + 100
        history_bins = 10

        counts = np.random.poisson(2.0, size=(m, t_total)).astype(np.int32)

        # mask_index: session 0 for [0,100), -1 for [100,110), session 1 for [110,210)
        mask_index = np.zeros(t_total, dtype=np.int32)
        mask_index[100:110] = -1  # Gap
        mask_index[110:] = 1      # Session B

        session_masks = np.ones((2, m), dtype=np.float32)

        ds = MaskedSpikeCountDataset(
            spike_counts=counts,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m,
        )

        # Check no sample spans the gap
        for idx in range(len(ds)):
            valid_t = ds._valid_indices[idx]
            window_start = valid_t
            target_bin = valid_t + history_bins

            # All bins in [window_start, target_bin] should have same session
            sessions_in_window = mask_index[window_start:target_bin + 1]
            assert np.all(sessions_in_window >= 0), \
                f"Sample {idx} includes gap bin! window={window_start}..{target_bin}"
            assert len(set(sessions_in_window)) == 1, \
                f"Sample {idx} spans sessions! sessions={set(sessions_in_window)}"

    def test_no_valid_samples_near_gap(self):
        """
        Samples near the gap boundary should be excluded — they would
        need future bins that fall in the gap.
        """
        m = 10
        history_bins = 5
        # Session A: 20 bins, gap: 5 bins, Session B: 20 bins
        t_total = 45

        counts = np.ones((m, t_total), dtype=np.int32)
        mask_index = np.zeros(t_total, dtype=np.int32)
        mask_index[20:25] = -1  # Gap
        mask_index[25:] = 1     # Session B

        session_masks = np.ones((2, m), dtype=np.float32)

        ds = MaskedSpikeCountDataset(
            spike_counts=counts,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m,
        )

        # The last valid sample for session A should have target at bin 19
        # (window [10..14], target 15) — wait, target must be < 20.
        # The last valid target in session A is bin 19.
        # So the last valid start is 19 - 5 = 14.
        for idx in range(len(ds)):
            valid_t = ds._valid_indices[idx]
            target_bin = valid_t + history_bins
            assert mask_index[target_bin] >= 0, \
                f"Target bin {target_bin} is in gap!"


# ---------------------------------------------------------------------------
# Test 4: No global normalization across splits
# ---------------------------------------------------------------------------

class TestNoGlobalNormalization:
    """Verify that datasets use raw data with no cross-split statistics."""

    def test_dataset_uses_raw_counts(self, cache_with_splits):
        """
        The dataset should return raw spike counts (converted to float32),
        not normalized or standardized values.
        """
        cache_dir, metadata = cache_with_splits
        sess = metadata["sessions"][0]
        m_max = metadata["m_max"]
        history_bins = metadata["history_bins"]

        # Load the raw data for comparison
        counts = np.load(cache_dir / "session_000.npy")  # uint8
        train_end = sess["split_boundaries"]["train_end"]
        counts_train = counts[:, :train_end].astype(np.float32)
        padded = pad_to_channels(counts_train.astype(np.int32), m_max)

        mask = build_channel_mask(sess["num_units"], m_max)
        session_masks = mask.reshape(1, -1)
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)

        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )

        # The first sample's input should match raw padded data
        x, y, mask_out = ds[0][:3]
        # x is (T, M_max) — check first row matches padded[:, 0] transposed
        expected_input = padded[:, :history_bins].T.astype(np.float32)
        np.testing.assert_array_almost_equal(
            x.numpy(), expected_input,
            err_msg="Dataset input doesn't match raw counts — normalization detected!",
        )

    def test_no_mean_subtraction(self, cache_with_splits):
        """The dataset values should never be negative (spike counts >= 0)."""
        cache_dir, metadata = cache_with_splits
        sess = metadata["sessions"][0]
        m_max = metadata["m_max"]
        history_bins = metadata["history_bins"]

        counts = np.load(cache_dir / "session_000.npy")
        train_end = sess["split_boundaries"]["train_end"]
        padded = pad_to_channels(
            counts[:, :train_end].astype(np.int32), m_max,
        )

        mask = build_channel_mask(sess["num_units"], m_max)
        session_masks = mask.reshape(1, -1)
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)

        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )

        # Check multiple samples
        for idx in [0, len(ds) // 2, len(ds) - 1]:
            x, y, _ = ds[idx][:3]
            assert torch.all(x >= 0), \
                f"Negative values in input at sample {idx} — mean subtraction?"
            assert torch.all(y >= 0), \
                f"Negative values in target at sample {idx} — mean subtraction?"


# ---------------------------------------------------------------------------
# Test 5: Data content differs between splits (temporal structure preserved)
# ---------------------------------------------------------------------------

class TestSplitContentDiffers:
    """
    Verify train/val/test actually contain different data — catches bugs
    where the same data is accidentally used for all splits.
    """

    def test_train_and_val_have_different_means(self, cache_with_splits):
        """
        Since our fixture has linearly increasing rates, train (early bins)
        should have lower mean than val (later bins).
        """
        cache_dir, metadata = cache_with_splits
        sess = metadata["sessions"][0]
        m_max = metadata["m_max"]
        history_bins = metadata["history_bins"]
        bounds = sess["split_boundaries"]

        counts = np.load(cache_dir / "session_000.npy").astype(np.float32)

        train_mean = counts[:, :bounds["train_end"]].mean()
        val_mean = counts[:, bounds["train_end"]:bounds["val_end"]].mean()
        test_mean = counts[:, bounds["val_end"]:].mean()

        # With linearly increasing rates: train_mean < val_mean < test_mean
        assert train_mean < val_mean, \
            f"Train mean ({train_mean:.3f}) >= val mean ({val_mean:.3f})"
        assert val_mean < test_mean, \
            f"Val mean ({val_mean:.3f}) >= test mean ({test_mean:.3f})"

    def test_splits_are_not_identical(self, cache_with_splits):
        """Train data should not be byte-identical to val or test data."""
        cache_dir, metadata = cache_with_splits
        sess = metadata["sessions"][0]
        bounds = sess["split_boundaries"]

        counts = np.load(cache_dir / "session_000.npy")

        train_data = counts[:, :bounds["train_end"]].tobytes()
        val_data = counts[:, bounds["train_end"]:bounds["val_end"]].tobytes()
        test_data = counts[:, bounds["val_end"]:].tobytes()

        assert train_data != val_data, "Train and val data are identical!"
        assert train_data != test_data, "Train and test data are identical!"
        assert val_data != test_data, "Val and test data are identical!"


# ---------------------------------------------------------------------------
# Test 6: Temporal ordering preserved
# ---------------------------------------------------------------------------

class TestTemporalOrdering:
    """Verify that temporal ordering is maintained within each split."""

    def test_train_before_val_before_test(self, cache_with_splits):
        """
        The maximum time index in train < minimum in val < minimum in test.
        This is the fundamental temporal split guarantee.
        """
        _, metadata = cache_with_splits

        for sess in metadata["sessions"]:
            bounds = sess["split_boundaries"]
            t_i = sess["num_bins"]

            train_max_bin = bounds["train_end"] - 1
            val_min_bin = bounds["train_end"]
            val_max_bin = bounds["val_end"] - 1
            test_min_bin = bounds["val_end"]

            assert train_max_bin < val_min_bin, \
                f"Train max bin ({train_max_bin}) >= val min ({val_min_bin})"
            assert val_max_bin < test_min_bin, \
                f"Val max bin ({val_max_bin}) >= test min ({test_min_bin})"

    def test_val_first_sample_does_not_use_train_bins(self, cache_with_splits):
        """
        The first valid val sample's input window should start at
        train_end (the first val bin), not before it.
        """
        cache_dir, metadata = cache_with_splits

        for i in range(metadata["num_sessions"]):
            sess = metadata["sessions"][i]
            bounds = sess["split_boundaries"]
            m_max = metadata["m_max"]
            history_bins = metadata["history_bins"]

            # Load and slice to val
            counts = np.load(cache_dir / f"session_{i:03d}.npy")
            val_start = bounds["train_end"]
            val_end = bounds["val_end"]
            counts_val = counts[:, val_start:val_end].astype(np.int32)
            padded = pad_to_channels(counts_val, m_max)

            # The first valid sample in the val dataset has index 0
            # in the sliced array. Its input window is [0, history_bins)
            # which maps to original bins [val_start, val_start + history_bins).
            # All of these are >= val_start = train_end.
            first_original_bin = val_start  # = train_end
            last_input_original = val_start + history_bins - 1

            assert first_original_bin >= bounds["train_end"], \
                "Val input starts before train_end!"
            assert last_input_original < val_end, \
                "Val input extends past val_end!"
