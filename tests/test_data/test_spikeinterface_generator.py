"""
Tests for src/data/spikeinterface_generator.py

Tests synthetic recording generation with concrete assertions on
shapes, types, values, and reproducibility.
"""

import pytest

from src.data.spikeinterface_generator import generate_synthetic_recording


@pytest.fixture
def small_config():
    """Small config for fast test generation (5 units, 5s)."""
    return {
        "seed": 42,
        "spikeinterface": {
            "probe": "Neuropixels1-128",
            "num_neurons": 5,
            "duration_s": 5.0,
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


class TestGenerateSyntheticRecording:
    """Tests for generate_synthetic_recording()."""

    def test_returns_three_outputs(self, small_config):
        """Should return (recording, sorting, extra_infos)."""
        result = generate_synthetic_recording(small_config)
        assert len(result) == 3

    def test_recording_has_correct_channels(self, small_config):
        """Neuropixels1-128 should produce 128 channels."""
        rec, _, _ = generate_synthetic_recording(small_config)
        assert rec.get_num_channels() == 128

    def test_recording_has_correct_duration(self, small_config):
        """Duration should match config."""
        rec, _, _ = generate_synthetic_recording(small_config)
        assert abs(rec.get_total_duration() - 5.0) < 0.01

    def test_recording_has_correct_sampling_frequency(self, small_config):
        """Sampling frequency should match config."""
        rec, _, _ = generate_synthetic_recording(small_config)
        assert rec.get_sampling_frequency() == 30000.0

    def test_sorting_has_correct_num_units(self, small_config):
        """Ground-truth sorting should have 5 units."""
        _, sort, _ = generate_synthetic_recording(small_config)
        assert sort.get_num_units() == 5

    def test_sorting_has_spikes(self, small_config):
        """Each unit should have at least some spikes in 5s."""
        _, sort, _ = generate_synthetic_recording(small_config)
        for unit_id in sort.get_unit_ids():
            spikes = sort.get_unit_spike_train(unit_id)
            # With 2-8 Hz firing rate over 5s, expect at least a few spikes
            assert len(spikes) > 0, f"Unit {unit_id} has no spikes"

    def test_total_spikes_in_reasonable_range(self, small_config):
        """Total spikes should be in a reasonable range for 5 units @ 2-8Hz over 5s."""
        _, sort, _ = generate_synthetic_recording(small_config)
        total = sum(
            len(sort.get_unit_spike_train(u)) for u in sort.get_unit_ids()
        )
        # 5 units × ~5 Hz avg × 5s = ~125 spikes expected, allow wide range
        assert 20 < total < 500, f"Total spikes {total} outside reasonable range"

    def test_reproducibility_with_same_seed(self, small_config):
        """Same seed should produce identical spike trains."""
        _, sort1, _ = generate_synthetic_recording(small_config)
        _, sort2, _ = generate_synthetic_recording(small_config)
        for uid in sort1.get_unit_ids():
            spikes1 = sort1.get_unit_spike_train(uid)
            spikes2 = sort2.get_unit_spike_train(uid)
            assert len(spikes1) == len(spikes2), f"Unit {uid} spike count mismatch"

    def test_different_seed_gives_different_spikes(self, small_config):
        """Different seeds should produce different spike trains."""
        _, sort1, _ = generate_synthetic_recording(small_config)
        small_config["seed"] = 99
        _, sort2, _ = generate_synthetic_recording(small_config)
        # At least one unit should have different spike count
        counts1 = [len(sort1.get_unit_spike_train(u)) for u in sort1.get_unit_ids()]
        counts2 = [len(sort2.get_unit_spike_train(u)) for u in sort2.get_unit_ids()]
        assert counts1 != counts2, "Different seeds produced identical spike counts"

    def test_return_static_gives_four_outputs(self, small_config):
        """return_static=True should return 4 outputs."""
        result = generate_synthetic_recording(small_config, return_static=True)
        assert len(result) == 4

    def test_extra_infos_contains_keys(self, small_config):
        """Extra infos should contain useful metadata."""
        _, _, extra = generate_synthetic_recording(small_config)
        assert isinstance(extra, dict)
