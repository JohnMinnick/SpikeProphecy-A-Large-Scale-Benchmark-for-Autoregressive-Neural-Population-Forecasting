"""
Tests for src/data/spike_dataset.py

Tests the SpikeCountDataset, temporal splitting, and DataLoader creation
with known-answer verification for sliding-window indexing.
"""

import numpy as np
import pytest
import torch

from src.data.spike_dataset import (
    SpikeCountDataset,
    create_dataloaders,
    temporal_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_counts():
    """
    Small spike-count matrix for testing: 3 channels, 20 bins.

    Known values make assertions easy to verify manually:
        Channel 0: [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0]
        Channel 1: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        Channel 2: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    """
    counts = np.array([
        [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ], dtype=np.int32)
    return counts


# ---------------------------------------------------------------------------
# SpikeCountDataset tests
# ---------------------------------------------------------------------------

class TestSpikeCountDataset:
    """Tests for SpikeCountDataset."""

    def test_num_samples(self, small_counts):
        """With T=5 history and 20 bins, should have 15 samples."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        assert len(ds) == 15  # 20 - 5

    def test_output_shapes(self, small_counts):
        """Input should be (T, M)=(5, 3), target should be (M,)=(3,)."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        x, y = ds[0]
        assert x.shape == (5, 3)
        assert y.shape == (3,)

    def test_known_answer_first_sample(self, small_counts):
        """
        First sample (idx=0):
          Input: bins 0-4, transposed to (5, 3)
          Target: bin 5
        """
        ds = SpikeCountDataset(small_counts, history_bins=5)
        x, y = ds[0]

        # Input: columns 0-4 of each channel, stacked as rows
        # Row 0 = [ch0[0], ch1[0], ch2[0]] = [1, 0, 2]
        # Row 1 = [ch0[1], ch1[1], ch2[1]] = [0, 1, 2]
        # Row 2 = [ch0[2], ch1[2], ch2[2]] = [2, 0, 2]
        # Row 3 = [ch0[3], ch1[3], ch2[3]] = [0, 1, 2]
        # Row 4 = [ch0[4], ch1[4], ch2[4]] = [1, 0, 2]
        expected_x = torch.tensor([
            [1, 0, 2],
            [0, 1, 2],
            [2, 0, 2],
            [0, 1, 2],
            [1, 0, 2],
        ], dtype=torch.float32)
        assert torch.equal(x, expected_x)

        # Target: bin 5 = [ch0[5], ch1[5], ch2[5]] = [0, 1, 2]
        expected_y = torch.tensor([0, 1, 2], dtype=torch.float32)
        assert torch.equal(y, expected_y)

    def test_known_answer_last_sample(self, small_counts):
        """
        Last sample (idx=14):
          Input: bins 14-18
          Target: bin 19
        """
        ds = SpikeCountDataset(small_counts, history_bins=5)
        x, y = ds[14]

        # Target: bin 19 = [ch0[19], ch1[19], ch2[19]] = [0, 1, 2]
        expected_y = torch.tensor([0, 1, 2], dtype=torch.float32)
        assert torch.equal(y, expected_y)

    def test_sliding_window_continuity(self, small_counts):
        """Consecutive samples should overlap by T-1 bins."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        x0, _ = ds[0]
        x1, _ = ds[1]
        # x0[1:] should equal x1[:-1] (shifted by one time step)
        assert torch.equal(x0[1:, :], x1[:-1, :])

    def test_dtype_is_float32(self, small_counts):
        """Default dtype should be float32."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_invalid_history_bins_raises(self, small_counts):
        """history_bins >= total_bins should raise ValueError."""
        with pytest.raises(ValueError, match="must be < total_bins"):
            SpikeCountDataset(small_counts, history_bins=20)

    def test_zero_history_raises(self, small_counts):
        """history_bins=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            SpikeCountDataset(small_counts, history_bins=0)

    def test_1d_input_raises(self):
        """1D array should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            SpikeCountDataset(np.array([1, 2, 3]), history_bins=1)

    def test_index_out_of_range_raises(self, small_counts):
        """Accessing beyond num_samples should raise IndexError."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        with pytest.raises(IndexError):
            ds[15]

    def test_all_samples_have_correct_shapes(self, small_counts):
        """Every sample should have consistent shapes."""
        ds = SpikeCountDataset(small_counts, history_bins=5)
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (5, 3), f"Sample {i} x shape mismatch"
            assert y.shape == (3,), f"Sample {i} y shape mismatch"


# ---------------------------------------------------------------------------
# temporal_split tests
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    """Tests for temporal_split()."""

    def test_split_sizes(self, small_counts):
        """70/15/15 split of 20 bins should give 14/3/3."""
        train, val, test = temporal_split(small_counts)
        assert train.shape == (3, 14)
        assert val.shape == (3, 3)
        assert test.shape == (3, 3)

    def test_no_data_leakage(self, small_counts):
        """Train/val/test should be contiguous, non-overlapping time slices."""
        train, val, test = temporal_split(small_counts)
        # Reconstruct original
        reconstructed = np.concatenate([train, val, test], axis=1)
        np.testing.assert_array_equal(reconstructed, small_counts)

    def test_custom_ratios(self, small_counts):
        """Custom ratios should produce proportional splits."""
        train, val, test = temporal_split(
            small_counts, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        )
        assert train.shape[1] == 10  # 50% of 20
        assert val.shape[1] == 5     # 25% of 20
        assert test.shape[1] == 5    # 25% of 20

    def test_invalid_ratios_raise(self, small_counts):
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            temporal_split(small_counts, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_preserves_channel_order(self, small_counts):
        """Channel ordering should be maintained across splits."""
        train, _, _ = temporal_split(small_counts)
        # Channel 2 should still be all 2s in the training portion
        assert np.all(train[2, :] == 2)


# ---------------------------------------------------------------------------
# create_dataloaders tests
# ---------------------------------------------------------------------------

class TestCreateDataloaders:
    """Tests for create_dataloaders()."""

    @pytest.fixture
    def larger_counts(self):
        """100-bin dataset: large enough for splits + history window."""
        rng = np.random.default_rng(42)
        return rng.poisson(lam=3, size=(3, 100)).astype(np.int32)

    def test_returns_three_loaders(self, larger_counts):
        """Should return dict with train, val, test keys."""
        config = {
            "history_bins": 5,
            "forecast_horizon": 1,
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 4,
            "compute": {"num_workers": 0, "pin_memory": False},
        }
        loaders = create_dataloaders(larger_counts, config)
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_batch_shapes(self, larger_counts):
        """Batches from loader should have correct shapes."""
        config = {
            "history_bins": 5,
            "forecast_horizon": 1,
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 4,
            "compute": {"num_workers": 0, "pin_memory": False},
        }
        loaders = create_dataloaders(larger_counts, config)
        # Get one batch from train loader
        batch_x, batch_y = next(iter(loaders["train"]))
        # batch_x: (batch, T=5, M=3), batch_y: (batch, M=3)
        assert batch_x.shape[1] == 5   # history_bins
        assert batch_x.shape[2] == 3   # channels
        assert batch_y.shape[1] == 3   # channels

    def test_all_targets_are_nonnegative(self, larger_counts):
        """All target values should be >= 0 (spike counts)."""
        config = {
            "history_bins": 5,
            "forecast_horizon": 1,
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 4,
            "compute": {"num_workers": 0, "pin_memory": False},
        }
        loaders = create_dataloaders(larger_counts, config)
        for batch_x, batch_y in loaders["train"]:
            assert torch.all(batch_y >= 0), "Negative target values found"


# ---------------------------------------------------------------------------
# History feature integration tests
# ---------------------------------------------------------------------------

class TestHistoryFeatureIntegration:
    """Tests for SpikeCountDataset with history features."""

    @pytest.fixture
    def counts_and_features(self):
        """3 channels, 20 bins, with 2 feature sets (6 rows)."""
        counts = np.ones((3, 20), dtype=np.int32)
        # Fake feature matrix: 2 features × 3 channels = 6 rows
        features = np.random.default_rng(42).random(
            (6, 20)
        ).astype(np.float32)
        return counts, features

    def test_input_shape_with_features(self, counts_and_features):
        """Input should be (T, M + N*M) when features are provided."""
        counts, features = counts_and_features
        ds = SpikeCountDataset(
            counts, history_bins=5, history_feature_matrix=features,
        )
        x, y = ds[0]
        # 3 channels + 6 feature rows = 9 input dims
        assert x.shape == (5, 9)
        # Target is still only the original 3 channels
        assert y.shape == (3,)

    def test_target_unchanged_with_features(self, counts_and_features):
        """Targets should be original spike counts, not features."""
        counts, features = counts_and_features
        ds_with = SpikeCountDataset(
            counts, history_bins=5, history_feature_matrix=features,
        )
        ds_without = SpikeCountDataset(counts, history_bins=5)
        # Targets should be identical
        _, y_with = ds_with[0]
        _, y_without = ds_without[0]
        assert torch.equal(y_with, y_without)

    def test_backward_compat_no_features(self):
        """Without features, behavior should be identical to original."""
        counts = np.array([
            [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ], dtype=np.int32)
        ds = SpikeCountDataset(counts, history_bins=5)
        x, y = ds[0]
        assert x.shape == (5, 3)
        assert y.shape == (3,)
        # num_history_features should be 0
        assert ds.num_history_features == 0
        assert ds.input_size == 3

    def test_input_size_attribute(self, counts_and_features):
        """input_size should reflect M + N*M."""
        counts, features = counts_and_features
        ds = SpikeCountDataset(
            counts, history_bins=5, history_feature_matrix=features,
        )
        assert ds.input_size == 9  # 3 + 6
        assert ds.num_history_features == 6

    def test_feature_matrix_shape_mismatch_raises(self):
        """Feature matrix with wrong T_total should raise ValueError."""
        counts = np.ones((3, 20), dtype=np.int32)
        bad_features = np.ones((3, 15), dtype=np.float32)
        with pytest.raises(ValueError, match="must match"):
            SpikeCountDataset(
                counts, history_bins=5, history_feature_matrix=bad_features,
            )

    def test_create_dataloaders_with_features(self):
        """create_dataloaders should handle history features in config."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(lam=3, size=(3, 100)).astype(np.int32)
        config = {
            "history_bins": 5,
            "forecast_horizon": 1,
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 4,
            "compute": {"num_workers": 0, "pin_memory": False},
            "bin_width_ms": 10.0,
            "history_features": {
                "enabled": True,
                "isi": {"enabled": True, "max_isi_ms": 500.0},
                "ema_rate": {"enabled": True, "alpha": 0.1},
                "refractory": {"enabled": True, "refractory_bins": 1},
            },
        }
        loaders = create_dataloaders(counts, config)
        batch_x, batch_y = next(iter(loaders["train"]))
        # Input: M + 3*M = 3 + 9 = 12
        assert batch_x.shape[2] == 12
        # Target: M = 3
        assert batch_y.shape[1] == 3

    def test_create_dataloaders_features_disabled(self):
        """create_dataloaders with features disabled → original shapes."""
        rng = np.random.default_rng(42)
        counts = rng.poisson(lam=3, size=(3, 100)).astype(np.int32)
        config = {
            "history_bins": 5,
            "forecast_horizon": 1,
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 4,
            "compute": {"num_workers": 0, "pin_memory": False},
            "history_features": {"enabled": False},
        }
        loaders = create_dataloaders(counts, config)
        batch_x, batch_y = next(iter(loaders["train"]))
        assert batch_x.shape[2] == 3
        assert batch_y.shape[1] == 3

