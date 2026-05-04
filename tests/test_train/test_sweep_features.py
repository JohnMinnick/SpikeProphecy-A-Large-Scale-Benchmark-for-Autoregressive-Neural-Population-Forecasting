"""
Tests for the feature/ablation sweep code changes.

Tests three new capabilities added for the NRP sweep:
1. MaskedSpikeCountDataset with output_channels (history features)
2. Masked NegBin/ZIP loss in Trainer._compute_masked_loss()
3. Per-region Pearson r aggregation in Trainer.evaluate()

These tests are designed to catch crashes before deploying to NRP,
where debugging is expensive and GPU time is limited.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.data.multi_session_loader import (
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)
from src.models.teacher import TeacherLSTM
from src.train.trainer import Trainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def masked_data_with_features():
    """
    Simulates a padded multi-session panel with history features appended.

    Session 0: 3 real channels, padded to M_max=5
    2 history features (EMA, ISI), each adding M_max channels
    Total input width: 5 + 2*5 = 15
    Total bins: 100

    Returns:
        spike_counts: (15, 100) padded panel (counts + features)
        mask_index: (100,) array of zeros (single session)
        session_masks: (1, 5) mask [1,1,1,0,0]
        m_max: 5
        n_feat: 2
    """
    np.random.seed(42)
    m_i = 3          # Real channels
    m_max = 5        # Padded channel count
    n_feat = 2       # Number of history features per channel
    t_bins = 100     # Total time bins

    # Spike counts: (m_i, T) -> pad to (m_max, T)
    counts = np.random.randint(0, 10, (m_i, t_bins), dtype=np.int32)
    padded_counts = pad_to_channels(counts, m_max)  # (5, 100)

    # Simulate 2 history features, each (m_max, T) after padding
    feat1 = np.zeros((m_max, t_bins), dtype=np.float32)
    feat1[:m_i, :] = np.random.randn(m_i, t_bins).astype(np.float32)

    feat2 = np.zeros((m_max, t_bins), dtype=np.float32)
    feat2[:m_i, :] = np.random.randn(m_i, t_bins).astype(np.float32)

    # Concatenate: counts + feat1 + feat2 = (15, 100)
    panel = np.concatenate(
        [padded_counts.astype(np.float32), feat1, feat2], axis=0,
    )

    # Mask: session 0 has 3 real channels out of 5
    mask = build_channel_mask(m_i, m_max)
    session_masks = mask.reshape(1, -1)  # (1, 5)
    mask_index = np.zeros(t_bins, dtype=np.int32)

    return panel, mask_index, session_masks, m_max, n_feat


@pytest.fixture
def small_masked_data():
    """
    Create small masked data (x, y, mask triples) for Trainer tests.

    Simulates multi-session batches where some channels are masked.
    """
    torch.manual_seed(42)
    m = 5  # M_max
    m_real = 3  # Real channels in this session
    n_samples = 60
    T = 10

    # Input: (n_samples, T, m) — model sees all channels
    x = torch.randn(n_samples, T, m).abs()
    # Target: (n_samples, m) — non-negative
    y = torch.abs(torch.randn(n_samples, m))
    # Mask: channels 0-2 are real, 3-4 are padded
    mask = torch.zeros(n_samples, m)
    mask[:, :m_real] = 1.0

    train_ds = TensorDataset(x[:40], y[:40], mask[:40])
    val_ds = TensorDataset(x[40:], y[40:], mask[40:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    return train_loader, val_loader, m, m_real


# =============================================================================
# MaskedSpikeCountDataset with output_channels tests
# =============================================================================

class TestOutputChannels:
    """Tests for MaskedSpikeCountDataset with output_channels parameter."""

    def test_x_has_full_input_width(self, masked_data_with_features):
        """Input x should span full panel width (counts + features)."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        history_bins = 10
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )
        x, y, mask = ds[0]
        # x shape: (T=history_bins, full_width)
        expected_width = m_max + n_feat * m_max  # 5 + 2*5 = 15
        assert x.shape == (history_bins, expected_width), (
            f"Expected x shape ({history_bins}, {expected_width}), "
            f"got {x.shape}"
        )

    def test_y_has_output_channels_width(self, masked_data_with_features):
        """Target y should have width = output_channels (M_max only)."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=10,
            output_channels=m_max,
        )
        x, y, mask = ds[0]
        assert y.shape == (m_max,), (
            f"Expected y shape ({m_max},), got {y.shape}"
        )

    def test_mask_has_output_channels_width(self, masked_data_with_features):
        """Mask should have width = output_channels (M_max), not full input."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=10,
            output_channels=m_max,
        )
        x, y, mask = ds[0]
        assert mask.shape == (m_max,), (
            f"Expected mask shape ({m_max},), got {mask.shape}"
        )

    def test_mask_values_correct(self, masked_data_with_features):
        """Mask should be [1,1,1,0,0] for 3 real channels padded to 5."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=10,
            output_channels=m_max,
        )
        _, _, mask = ds[0]
        expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(mask, expected)

    def test_y_values_from_counts_only(self, masked_data_with_features):
        """Target y should come from the spike count channels, not features."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        history_bins = 10
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )
        # Check first sample: y should be panel[:m_max, history_bins]
        _, y, _ = ds[0]
        expected_y = torch.from_numpy(
            panel[:m_max, history_bins].astype(np.float32)
        )
        torch.testing.assert_close(y, expected_y)

    def test_x_includes_features(self, masked_data_with_features):
        """Input x should include feature channels beyond M_max."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        history_bins = 10
        ds = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )
        x, _, _ = ds[0]
        # x[:, m_max:] should contain feature data (not all zeros)
        feature_part = x[:, m_max:]
        assert feature_part.shape[1] == n_feat * m_max
        # Feature data for real channels should not be all zeros
        assert feature_part.abs().sum() > 0, (
            "Feature channels in x are all zeros — features not included"
        )

    def test_default_output_channels_equals_input(self):
        """Without output_channels, y should match full input width."""
        np.random.seed(42)
        m = 5
        t = 50
        counts = np.random.randint(0, 5, (m, t), dtype=np.int32)
        mask_index = np.zeros(t, dtype=np.int32)
        session_masks = np.ones((1, m), dtype=np.float32)

        ds = MaskedSpikeCountDataset(
            spike_counts=counts,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=10,
            # No output_channels specified
        )
        x, y, mask = ds[0]
        assert y.shape == (m,)
        assert mask.shape == (m,)

    def test_sample_count_unchanged_with_features(
        self, masked_data_with_features
    ):
        """Adding features should not change the number of valid samples."""
        panel, mask_index, session_masks, m_max, n_feat = (
            masked_data_with_features
        )
        history_bins = 10
        t_bins = panel.shape[1]

        # With features
        ds_feat = MaskedSpikeCountDataset(
            spike_counts=panel,
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
            output_channels=m_max,
        )

        # Without features (counts only)
        ds_plain = MaskedSpikeCountDataset(
            spike_counts=panel[:m_max],
            mask_index=mask_index,
            session_masks=session_masks,
            history_bins=history_bins,
        )

        assert len(ds_feat) == len(ds_plain), (
            f"Feature dataset has {len(ds_feat)} samples, "
            f"plain has {len(ds_plain)}"
        )
        # Both should have t_bins - history_bins samples
        expected = t_bins - history_bins
        assert len(ds_feat) == expected


# =============================================================================
# Masked NegBin/ZIP loss tests
# =============================================================================

class TestMaskedLoss:
    """Tests for _compute_masked_loss with NegBin and ZIP loss types."""

    def _make_masked_trainer(self, small_masked_data, loss_type, dist="poisson"):
        """Helper to create a Trainer with masked data and specific loss."""
        train_loader, val_loader, m, m_real = small_masked_data
        config = {
            "model": {"hidden_size": 16, "output_distribution": dist},
            "training": {
                "epochs": 3,
                "learning_rate": 0.01,
                "warmup_steps": 0,
                "patience": 10,
                "grad_clip_norm": 1.0,
                "weight_decay": 0.0,
            },
            "loss": {"type": loss_type, "log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1,
            dropout=0.0, output_distribution=dist,
        )
        return Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )

    def test_poisson_masked_loss_returns_finite(self, small_masked_data):
        """Poisson masked loss should produce finite values."""
        trainer = self._make_masked_trainer(
            small_masked_data, "poisson_nll", "poisson"
        )
        loss = trainer._train_one_epoch()
        assert np.isfinite(loss), f"Poisson masked loss is not finite: {loss}"

    def test_negbin_masked_loss_returns_finite(self, small_masked_data):
        """NegBin masked loss should produce finite values."""
        trainer = self._make_masked_trainer(
            small_masked_data, "negbin_nll", "negbin"
        )
        loss = trainer._train_one_epoch()
        assert np.isfinite(loss), f"NegBin masked loss is not finite: {loss}"

    def test_zip_masked_loss_returns_finite(self, small_masked_data):
        """ZIP masked loss should produce finite values."""
        trainer = self._make_masked_trainer(
            small_masked_data, "zip_nll", "zip"
        )
        loss = trainer._train_one_epoch()
        assert np.isfinite(loss), f"ZIP masked loss is not finite: {loss}"

    def test_negbin_masked_training_runs(self, small_masked_data):
        """NegBin should complete full training with masked data."""
        trainer = self._make_masked_trainer(
            small_masked_data, "negbin_nll", "negbin"
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        # All losses should be finite
        for loss in history["train_loss"]:
            assert np.isfinite(loss), f"NaN/Inf in NegBin masked training"

    def test_zip_masked_training_runs(self, small_masked_data):
        """ZIP should complete full training with masked data."""
        trainer = self._make_masked_trainer(
            small_masked_data, "zip_nll", "zip"
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        for loss in history["train_loss"]:
            assert np.isfinite(loss), f"NaN/Inf in ZIP masked training"

    def test_negbin_masked_val_metrics_finite(self, small_masked_data):
        """All NegBin val metrics should be finite with masked data."""
        trainer = self._make_masked_trainer(
            small_masked_data, "negbin_nll", "negbin"
        )
        metrics = trainer._validate()
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_zip_masked_val_metrics_finite(self, small_masked_data):
        """All ZIP val metrics should be finite with masked data."""
        trainer = self._make_masked_trainer(
            small_masked_data, "zip_nll", "zip"
        )
        metrics = trainer._validate()
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_masked_loss_ignores_padded_channels(self, small_masked_data):
        """Masked loss should produce different values than unmasked loss.

        If the mask is correctly applied, training with masked data (where
        padded channels are zeroed out) should produce different loss values
        than training without masks (where padded channels contribute).
        """
        # Create masked trainer
        train_loader, val_loader, m, m_real = small_masked_data
        config = {
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "warmup_steps": 0,
                "patience": 10,
                "grad_clip_norm": 1.0,
                "weight_decay": 0.0,
            },
            "loss": {"log_input": False},
        }

        # Trainer with masked data
        torch.manual_seed(42)
        model_masked = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        trainer_masked = Trainer(
            model_masked, train_loader, val_loader, config,
            torch.device("cpu"),
        )

        # Compute masked loss
        mode = trainer_masked._train_one_epoch()

        # Now create unmasked trainer with same data but no masks
        torch.manual_seed(42)
        # Strip masks from data — access underlying tensor storage
        x_all = train_loader.dataset.tensors[0]
        y_all = train_loader.dataset.tensors[1]
        unmasked_ds = TensorDataset(x_all, y_all)
        unmasked_loader = DataLoader(unmasked_ds, batch_size=16, shuffle=True)

        model_unmasked = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        trainer_unmasked = Trainer(
            model_unmasked, unmasked_loader, val_loader, config,
            torch.device("cpu"),
        )
        unmasked_loss = trainer_unmasked._train_one_epoch()

        # Losses should be different (masking out 2/5 channels changes the avg)
        assert mode != unmasked_loss, (
            "Masked and unmasked losses are identical — mask may not be applied"
        )


# =============================================================================
# Per-region Pearson r tests
# =============================================================================

class TestPerRegionPearsonR:
    """Tests for per-region Pearson r aggregation in evaluate()."""

    def _make_trainer_with_regions(self, region_map=None):
        """Create a Trainer with synthetic data and optional region_map."""
        torch.manual_seed(42)
        m = 6  # channels
        n_train, n_val = 80, 20
        T = 10

        x_train = torch.randn(n_train, T, m).abs()
        y_train = torch.abs(torch.randn(n_train, m))
        x_val = torch.randn(n_val, T, m).abs()
        y_val = torch.abs(torch.randn(n_val, m))

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        config = {
            "training": {
                "epochs": 3,
                "learning_rate": 0.01,
                "warmup_steps": 0,
                "patience": 10,
                "grad_clip_norm": 1.0,
                "weight_decay": 0.0,
            },
            "loss": {"log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )

        if region_map is not None:
            trainer.region_map = region_map

        return trainer, val_loader

    def test_no_region_map_returns_base_metrics(self):
        """Without region_map, evaluate() should return base metrics (no per-region keys)."""
        trainer, val_loader = self._make_trainer_with_regions(region_map=None)
        metrics = trainer.evaluate(val_loader, prefix="val")
        # Must have core metrics
        required_keys = {
            "val_loss", "val_poisson_nll", "val_pearson_r",
            "val_mae", "val_mse",
        }
        assert required_keys.issubset(set(metrics.keys()))
        # Must NOT have any per-region metrics
        region_keys = [k for k in metrics if "_pearson_r_" in k]
        assert len(region_keys) == 0

    def test_region_map_adds_per_region_metrics(self):
        """With region_map, evaluate() should include per-region Pearson r."""
        # 3 channels in VISp, 2 in CA1, 1 in LP
        region_map = {
            0: "VISp", 1: "VISp", 2: "VISp",
            3: "CA1", 4: "CA1",
            5: "LP",
        }
        trainer, val_loader = self._make_trainer_with_regions(
            region_map=region_map
        )
        metrics = trainer.evaluate(val_loader, prefix="val")

        # Should have base metrics + 3 per-region entries
        assert "val_pearson_r_VISp" in metrics
        assert "val_pearson_r_CA1" in metrics
        assert "val_pearson_r_LP" in metrics

    def test_per_region_metrics_are_finite(self):
        """All per-region Pearson r values should be finite."""
        region_map = {
            0: "VISp", 1: "VISp", 2: "VISp",
            3: "CA1", 4: "CA1",
            5: "LP",
        }
        trainer, val_loader = self._make_trainer_with_regions(
            region_map=region_map
        )
        metrics = trainer.evaluate(val_loader, prefix="val")

        for key in ["val_pearson_r_VISp", "val_pearson_r_CA1", "val_pearson_r_LP"]:
            assert np.isfinite(metrics[key]), (
                f"{key} is not finite: {metrics[key]}"
            )

    def test_per_region_r_in_valid_range(self):
        """Per-region Pearson r should be in [-1, 1]."""
        region_map = {
            0: "VISp", 1: "VISp", 2: "VISp",
            3: "CA1", 4: "CA1",
            5: "LP",
        }
        trainer, val_loader = self._make_trainer_with_regions(
            region_map=region_map
        )
        # Train a few epochs first for non-trivial predictions
        trainer.train()
        metrics = trainer.evaluate(val_loader, prefix="val")

        for key in ["val_pearson_r_VISp", "val_pearson_r_CA1", "val_pearson_r_LP"]:
            r = metrics[key]
            assert -1.0 <= r <= 1.0 or np.isclose(abs(r), 1.0, atol=0.01), (
                f"{key}={r} is outside [-1, 1]"
            )

    def test_region_name_sanitization(self):
        """Region names with slashes/spaces should be sanitized in keys."""
        region_map = {
            0: "VIS/p", 1: "VIS/p",
            2: "CA 1", 3: "CA 1",
            4: "LP", 5: "LP",
        }
        trainer, val_loader = self._make_trainer_with_regions(
            region_map=region_map
        )
        metrics = trainer.evaluate(val_loader, prefix="val")

        # Slashes → underscores, spaces → underscores
        assert "val_pearson_r_VIS_p" in metrics
        assert "val_pearson_r_CA_1" in metrics
        assert "val_pearson_r_LP" in metrics

    def test_region_map_with_masked_data(self):
        """Per-region Pearson r should work with masked (multi-session) data."""
        torch.manual_seed(42)
        m = 5
        n_val = 30
        T = 10

        x_val = torch.randn(n_val, T, m).abs()
        y_val = torch.abs(torch.randn(n_val, m))
        # Mask: only first 3 channels are real
        mask_val = torch.zeros(n_val, m)
        mask_val[:, :3] = 1.0

        val_ds = TensorDataset(x_val, y_val, mask_val)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        # Need a train loader too
        x_train = torch.randn(60, T, m).abs()
        y_train = torch.abs(torch.randn(60, m))
        mask_train = torch.zeros(60, m)
        mask_train[:, :3] = 1.0
        train_ds = TensorDataset(x_train, y_train, mask_train)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        config = {
            "training": {
                "epochs": 1,
                "learning_rate": 0.01,
                "warmup_steps": 0,
                "patience": 10,
                "grad_clip_norm": 1.0,
                "weight_decay": 0.0,
            },
            "loss": {"log_input": False},
        }
        model = TeacherLSTM(
            input_size=m, hidden_size=16, num_layers=1, dropout=0.0,
        )
        trainer = Trainer(
            model, train_loader, val_loader, config, torch.device("cpu"),
        )
        # Region map: only covers the 3 real channels
        trainer.region_map = {0: "VISp", 1: "VISp", 2: "CA1"}

        metrics = trainer.evaluate(val_loader, prefix="val")

        assert "val_pearson_r_VISp" in metrics
        assert "val_pearson_r_CA1" in metrics
        assert np.isfinite(metrics["val_pearson_r_VISp"])
        assert np.isfinite(metrics["val_pearson_r_CA1"])

    def test_overall_r_unchanged_with_region_map(self):
        """Adding a region_map should not change the overall Pearson r."""
        # Without region map
        trainer_no_region, val_loader = self._make_trainer_with_regions(
            region_map=None
        )
        metrics_no_region = trainer_no_region.evaluate(val_loader, prefix="val")

        # With region map (same model state — new instance from same seed)
        region_map = {
            0: "VISp", 1: "VISp", 2: "VISp",
            3: "CA1", 4: "CA1",
            5: "LP",
        }
        trainer_with_region, _ = self._make_trainer_with_regions(
            region_map=region_map
        )
        # Copy model weights to ensure identical predictions
        trainer_with_region.model.load_state_dict(
            trainer_no_region.model.state_dict()
        )
        metrics_with_region = trainer_with_region.evaluate(
            val_loader, prefix="val"
        )

        # Overall Pearson r should be identical regardless of region_map
        assert abs(
            metrics_no_region["val_pearson_r"]
            - metrics_with_region["val_pearson_r"]
        ) < 1e-6, (
            f"Overall r changed: {metrics_no_region['val_pearson_r']:.8f} vs "
            f"{metrics_with_region['val_pearson_r']:.8f}"
        )


# =============================================================================
# Filter units brain_region_list propagation tests
# =============================================================================

class TestFilterUnitsBrainRegions:
    """Tests that filter_units correctly propagates brain_region_list."""

    def test_region_list_survives_quality_filter(self):
        """Brain region list should be filtered alongside spike times."""
        from src.data.real_data_loader import filter_units

        spike_times = [
            np.array([0.1, 0.2, 0.5]),   # good
            np.array([0.1, 0.3]),          # bad
            np.array([0.1, 0.4, 0.8]),     # good
        ]
        unit_indices = [0, 1, 2]
        quality_list = ["good", "bad", "good"]
        brain_region_list = ["VISp", "CA1", "LP"]

        _, _, stats = filter_units(
            spike_times_list=spike_times,
            unit_indices=unit_indices,
            sampling_frequency=30000.0,
            duration_s=1.0,
            quality_labels=["good"],
            quality_list=quality_list,
            brain_region_list=brain_region_list,
        )

        # After filtering, only VISp and LP should remain
        assert stats["brain_region_list"] == ["VISp", "LP"]

    def test_region_list_survives_rate_filter(self):
        """Brain region list should survive firing rate filtering."""
        from src.data.real_data_loader import filter_units

        # Unit 0: 10 spikes in 1s = 10Hz (kept)
        # Unit 1: 0 spikes = 0Hz (removed)
        # Unit 2: 5 spikes in 1s = 5Hz (kept)
        spike_times = [
            np.arange(10, dtype=np.float64) * 0.1,
            np.array([], dtype=np.float64),
            np.arange(5, dtype=np.float64) * 0.2,
        ]
        unit_indices = [0, 1, 2]
        brain_region_list = ["VISp", "CA1", "LP"]

        _, _, stats = filter_units(
            spike_times_list=spike_times,
            unit_indices=unit_indices,
            sampling_frequency=30000.0,
            duration_s=1.0,
            min_firing_rate_hz=1.0,
            brain_region_list=brain_region_list,
        )

        # Unit 1 removed (0Hz < 1Hz threshold)
        assert stats["brain_region_list"] == ["VISp", "LP"]

    def test_region_list_survives_max_units_cap(self):
        """Brain region list should survive max_units cap."""
        from src.data.real_data_loader import filter_units

        spike_times = [np.array([0.1]) for _ in range(5)]
        unit_indices = list(range(5))
        brain_region_list = ["VISp", "CA1", "LP", "SC", "MOs"]

        _, _, stats = filter_units(
            spike_times_list=spike_times,
            unit_indices=unit_indices,
            sampling_frequency=30000.0,
            duration_s=1.0,
            max_units=3,
            brain_region_list=brain_region_list,
            rng=np.random.default_rng(42),
        )

        # Should have exactly 3 regions remaining
        assert len(stats["brain_region_list"]) == 3
        # Each remaining region should be one of the originals
        for r in stats["brain_region_list"]:
            assert r in brain_region_list

    def test_region_list_none_when_absent(self):
        """When no brain_region_list is provided, it should remain None."""
        from src.data.real_data_loader import filter_units

        spike_times = [np.array([0.1, 0.2, 0.5])]
        unit_indices = [0]

        _, _, stats = filter_units(
            spike_times_list=spike_times,
            unit_indices=unit_indices,
            sampling_frequency=30000.0,
            duration_s=1.0,
            brain_region_list=None,
        )

        assert stats["brain_region_list"] is None
