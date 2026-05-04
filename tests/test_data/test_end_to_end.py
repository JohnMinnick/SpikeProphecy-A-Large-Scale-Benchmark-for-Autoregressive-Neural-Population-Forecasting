"""
End-to-end smoke test for the full data pipeline.

Tests the complete flow: generate → bin → validate → split → dataset → baseline.
This is the critical integration test that ensures all components work together.
"""

import logging

import numpy as np
import pytest
import torch

from src.data.binning import bin_spike_trains, save_binned_data, validate_spike_counts
from src.data.spike_dataset import SpikeCountDataset, create_dataloaders, temporal_split
from src.data.spikeinterface_generator import generate_synthetic_recording
from src.eval.metrics import compute_all_baselines, mae, mse, pearson_r, poisson_nll


# Use a small recording for speed (5 units, 10s)
E2E_CONFIG = {
    "seed": 42,
    "bin_width_ms": 10,
    "history_bins": 20,
    "forecast_horizon": 1,
    "batch_size": 16,
    "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
    "compute": {"num_workers": 0, "pin_memory": False},
    "spikeinterface": {
        "probe": "Neuropixels1-128",
        "num_neurons": 5,
        "duration_s": 10.0,
        "sampling_frequency": 30000.0,
        "noise": {
            "levels": [12.0, 15.0],
            "spatial_decay": 25.0,
        },
        "drift": {
            "mode": "zigzag",
            "amplitude_factor": 0.5,
            "period_s": 200,
            "non_rigid_gradient": None,
        },
        "firing": {
            "rates": [2.0, 8.0],
            "refractory_period_ms": 4.0,
        },
        "templates": {
            "ms_before": 1.5,
            "ms_after": 3.0,
            "alpha": [150.0, 500.0],
            "spatial_decay": [10, 45],
        },
    },
}


@pytest.fixture(scope="module")
def generated_data():
    """Generate a synthetic recording (cached across all tests in this module)."""
    rec, sort, extra = generate_synthetic_recording(E2E_CONFIG)
    sort.register_recording(rec)
    return rec, sort, extra


@pytest.fixture(scope="module")
def binned_data(generated_data):
    """Bin the generated recording (cached across all tests in this module)."""
    rec, sort, _ = generated_data
    counts, metadata = bin_spike_trains(
        sort, bin_width_ms=E2E_CONFIG["bin_width_ms"], recording=rec,
    )
    return counts, metadata


class TestEndToEndPipeline:
    """Full pipeline integration test."""

    # --- Step 1: Generation ---

    def test_generation_produces_valid_data(self, generated_data):
        """Generator should produce recording + sorting with spikes."""
        rec, sort, _ = generated_data
        assert rec.get_num_channels() == 128
        assert sort.get_num_units() == 5
        total_spikes = sum(
            len(sort.get_unit_spike_train(u)) for u in sort.get_unit_ids()
        )
        assert total_spikes > 0, "No spikes generated"

    # --- Step 2: Binning ---

    def test_binning_produces_correct_shape(self, binned_data):
        """Binned matrix should be (M=5, T=1000) for 10s at 10ms bins."""
        counts, meta = binned_data
        assert counts.shape[0] == 5  # 5 units
        assert counts.shape[1] == 1000  # 10s / 10ms = 1000 bins
        assert counts.dtype == np.int32

    def test_binning_preserves_total_spikes(self, generated_data, binned_data):
        """Total binned spikes should match total raw spikes."""
        _, sort, _ = generated_data
        counts, meta = binned_data

        raw_total = sum(
            len(sort.get_unit_spike_train(u)) for u in sort.get_unit_ids()
        )
        binned_total = int(counts.sum())

        # Should match exactly (all spikes within 10s should be binned)
        assert binned_total == raw_total, (
            f"Spike count mismatch: raw={raw_total}, binned={binned_total}"
        )

    def test_validation_passes(self, binned_data):
        """Validation should pass for generated data."""
        counts, meta = binned_data
        result = validate_spike_counts(counts, meta)
        assert result["stats"]["total_spikes"] > 0
        assert result["stats"]["max_count_per_bin"] >= 0
        # No shape/dtype warnings
        shape_warnings = [w for w in result["warnings"]
                          if "mismatch" in w.lower() or "negative" in w.lower()]
        assert len(shape_warnings) == 0

    # --- Step 3: Split ---

    def test_temporal_split_no_leakage(self, binned_data):
        """Reconstruction from splits should exactly match original."""
        counts, _ = binned_data
        train, val, test = temporal_split(counts)
        reconstructed = np.concatenate([train, val, test], axis=1)
        np.testing.assert_array_equal(reconstructed, counts)

    # --- Step 4: Dataset ---

    def test_dataset_creates_valid_samples(self, binned_data):
        """Dataset should produce (T, M) input and (M,) target."""
        counts, _ = binned_data
        train, _, _ = temporal_split(counts)
        ds = SpikeCountDataset(train, history_bins=E2E_CONFIG["history_bins"])

        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (20, 5)  # (history_bins, num_units)
        assert y.shape == (5,)

    def test_dataloader_iterates(self, binned_data):
        """DataLoader should produce correctly shaped batches."""
        counts, _ = binned_data
        loaders = create_dataloaders(counts, E2E_CONFIG)

        batch_count = 0
        for batch_x, batch_y in loaders["train"]:
            assert batch_x.dim() == 3  # (batch, T, M)
            assert batch_y.dim() == 2  # (batch, M)
            assert batch_x.shape[1] == 20  # history_bins
            assert batch_x.shape[2] == 5   # num_units
            assert batch_y.shape[1] == 5   # num_units
            batch_count += 1

        assert batch_count > 0, "No batches produced by training DataLoader"

    # --- Step 5: Baselines ---

    def test_baselines_produce_metrics(self, binned_data):
        """Both baselines should produce all four metrics."""
        counts, _ = binned_data
        results = compute_all_baselines(counts, E2E_CONFIG["history_bins"])

        for baseline_name in ["persistence", "mean_rate"]:
            metrics = results[baseline_name]
            assert "poisson_nll" in metrics
            assert "pearson_r" in metrics
            assert "mae" in metrics
            assert "mse" in metrics
            # MAE and MSE should be non-negative
            assert metrics["mae"] >= 0
            assert metrics["mse"] >= 0

    def test_persistence_baseline_values_reasonable(self, binned_data):
        """Persistence baseline MAE should be in a reasonable range."""
        counts, _ = binned_data
        results = compute_all_baselines(counts, E2E_CONFIG["history_bins"])
        pers_mae = results["persistence"]["mae"]
        # For sparse spike counts (mostly 0s/1s), MAE should be small
        assert 0 <= pers_mae < 5.0, f"Persistence MAE={pers_mae} seems unreasonable"

    # --- Step 6: Save/Load round-trip ---

    def test_save_and_reload(self, binned_data, tmp_path):
        """Saved binned data should reload identically."""
        counts, meta = binned_data
        save_binned_data(counts, meta, tmp_path / "e2e_test")

        from src.data.binning import load_binned_data
        loaded_counts, loaded_meta = load_binned_data(tmp_path / "e2e_test")

        np.testing.assert_array_equal(counts, loaded_counts)
        assert loaded_meta["total_spikes"] == meta["total_spikes"]
        assert loaded_meta["num_bins"] == meta["num_bins"]
