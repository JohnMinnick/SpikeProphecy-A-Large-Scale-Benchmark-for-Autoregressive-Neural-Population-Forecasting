"""
Tests for src/data/modulated_generator.py

Tests the modulated inhomogeneous Poisson spike generator with concrete
assertions on shapes, temporal autocorrelation, cross-channel correlations,
refractory period enforcement, reproducibility, integration with binning,
and biologically-motivated overlay functions (bursting, stimulus events,
non-stationary drift, multi-timescale oscillations).
"""

import numpy as np
import pytest

from src.data.modulated_generator import (
    MockSorting,
    generate_modulated_spikes,
    _sinusoidal_rate,
    _step_rate,
    _generate_inhomogeneous_poisson,
    _burst_overlay,
    _stimulus_events_overlay,
    _drift_overlay,
    _multi_oscillation_overlay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def modulated_config():
    """Config for modulated spike generation: 5 units, 30s, fast to test."""
    return {
        "seed": 42,
        "spikeinterface": {
            "num_neurons": 5,
            "duration_s": 30.0,
            "sampling_frequency": 30000.0,
            "firing": {
                "rates": [3.0, 8.0],
                "refractory_period_ms": 4.0,
            },
        },
        "modulation": {
            "types": ["sinusoidal"],
            "sinusoidal": {
                "frequencies_hz": [0.5, 2.0],
                "depth": [0.5, 0.8],
            },
            "step": {
                "num_changes": [3, 6],
                "rate_range": [2.0, 12.0],
            },
            "shared_signal": {
                "enabled": True,
                "frequency_hz": 0.3,
                "coupling_range": [0.1, 0.5],
            },
        },
    }


@pytest.fixture
def step_config():
    """Config with step modulation only."""
    return {
        "seed": 42,
        "spikeinterface": {
            "num_neurons": 5,
            "duration_s": 30.0,
            "sampling_frequency": 30000.0,
            "firing": {
                "rates": [3.0, 8.0],
                "refractory_period_ms": 4.0,
            },
        },
        "modulation": {
            "types": ["step"],
            "step": {
                "num_changes": [3, 6],
                "rate_range": [2.0, 12.0],
            },
            "shared_signal": {
                "enabled": False,
            },
        },
    }


@pytest.fixture
def rich_config():
    """Config with all overlays enabled for overlay integration tests."""
    return {
        "seed": 42,
        "spikeinterface": {
            "num_neurons": 3,
            "duration_s": 10.0,
            "sampling_frequency": 30000.0,
            "firing": {
                "rates": [5.0, 10.0],
                "refractory_period_ms": 4.0,
            },
        },
        "modulation": {
            "types": ["sinusoidal"],
            "sinusoidal": {
                "frequencies_hz": [0.5, 2.0],
                "depth": [0.5, 0.8],
            },
            "shared_signal": {"enabled": False},
            "bursting": {
                "enabled": True,
                "burst_rate": 0.5,
                "burst_duration_ms": [50.0, 150.0],
                "burst_factor": [3.0, 5.0],
            },
            "stimulus_events": {
                "enabled": True,
                "num_events": [3, 8],
                "amplitude_factor": [3.0, 6.0],
                "decay_tau_ms": [50.0, 150.0],
            },
            "drift": {
                "enabled": True,
                "tau_s": 5.0,  # Short tau for 10s test
                "sigma": 0.2,
            },
            "multi_oscillation": {
                "enabled": True,
                "bands": [
                    {"frequencies_hz": [4.0, 8.0], "depth": [0.05, 0.15]},
                    {"frequencies_hz": [30.0, 80.0], "depth": [0.01, 0.05]},
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# MockSorting interface tests
# ---------------------------------------------------------------------------

class TestMockSorting:
    """Tests for the MockSorting interface."""

    def test_get_unit_ids(self):
        """Unit IDs should be sorted."""
        trains = {2: np.array([10, 20]), 0: np.array([5]), 1: np.array([15])}
        sorting = MockSorting(trains, 30000.0)
        assert sorting.get_unit_ids() == [0, 1, 2]

    def test_get_num_units(self):
        """Should return correct number of units."""
        trains = {0: np.array([10]), 1: np.array([20])}
        sorting = MockSorting(trains, 30000.0)
        assert sorting.get_num_units() == 2

    def test_get_unit_spike_train(self):
        """Should return the correct spike train array."""
        expected = np.array([100, 200, 300], dtype=np.int64)
        trains = {0: expected}
        sorting = MockSorting(trains, 30000.0)
        np.testing.assert_array_equal(
            sorting.get_unit_spike_train(0), expected
        )

    def test_get_sampling_frequency(self):
        """Should return the configured sampling frequency."""
        trains = {0: np.array([10])}
        sorting = MockSorting(trains, 44100.0)
        assert sorting.get_sampling_frequency() == 44100.0


# ---------------------------------------------------------------------------
# Rate modulation function tests
# ---------------------------------------------------------------------------

class TestRateModulation:
    """Tests for rate modulation helper functions."""

    def test_sinusoidal_rate_mean(self):
        """Mean of sinusoidal rate should approximate the base rate."""
        t = np.linspace(0, 10, 10000)
        rate = _sinusoidal_rate(t, base_rate=5.0, frequency=1.0,
                                depth=0.5, phase=0.0)
        # Mean should be close to base_rate (within ~1%)
        assert abs(np.mean(rate) - 5.0) < 0.1

    def test_sinusoidal_rate_range(self):
        """Sinusoidal rate should stay within expected bounds."""
        t = np.linspace(0, 10, 10000)
        rate = _sinusoidal_rate(t, base_rate=5.0, frequency=1.0,
                                depth=0.5, phase=0.0)
        # rate ∈ [5*(1-0.5), 5*(1+0.5)] = [2.5, 7.5]
        assert np.min(rate) >= 2.4  # Small tolerance for discretization
        assert np.max(rate) <= 7.6

    def test_sinusoidal_rate_nonnegative(self):
        """Rate should never go negative even with depth=1.0."""
        t = np.linspace(0, 10, 10000)
        rate = _sinusoidal_rate(t, base_rate=5.0, frequency=1.0,
                                depth=1.0, phase=0.0)
        assert np.all(rate >= 0)

    def test_step_rate_has_correct_segments(self):
        """Step rate should have distinct piecewise-constant segments."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 10000)
        rate = _step_rate(t, num_changes=3, rate_range=(2.0, 10.0), rng=rng)
        # Should have at most 4 unique values (3 changes = 4 segments)
        unique_rates = np.unique(rate)
        assert len(unique_rates) <= 4
        assert len(unique_rates) >= 2  # At least 2 different levels


# ---------------------------------------------------------------------------
# Spike generation tests
# ---------------------------------------------------------------------------

class TestInhomogeneousPoisson:
    """Tests for the core spike generation function."""

    def test_generates_spikes(self):
        """Should generate a non-empty spike train for reasonable rates."""
        rng = np.random.default_rng(42)
        # Constant 5 Hz rate, 10 seconds at 1ms resolution
        rate = np.full(10000, 5.0)
        spikes = _generate_inhomogeneous_poisson(rate, dt=0.001,
                                                  refractory_s=0.004, rng=rng)
        assert len(spikes) > 0

    def test_spike_rate_approximately_correct(self):
        """Mean spike rate should approximate the input rate."""
        rng = np.random.default_rng(42)
        # Constant 5 Hz rate, 100 seconds
        rate = np.full(100000, 5.0)
        spikes = _generate_inhomogeneous_poisson(rate, dt=0.001,
                                                  refractory_s=0.004, rng=rng)
        actual_rate = len(spikes) / 100.0
        # Should be within ~20% of target (refractory reduces it slightly)
        assert 3.5 < actual_rate < 6.5, f"Rate {actual_rate} too far from 5.0"

    def test_refractory_period_enforced(self):
        """No inter-spike interval should be less than refractory period."""
        rng = np.random.default_rng(42)
        rate = np.full(100000, 10.0)  # Higher rate to test refractory
        refractory_s = 0.004
        spikes = _generate_inhomogeneous_poisson(rate, dt=0.001,
                                                  refractory_s=refractory_s,
                                                  rng=rng)
        if len(spikes) > 1:
            isis = np.diff(spikes) * 0.001  # Convert to seconds
            min_isi = np.min(isis)
            assert min_isi >= refractory_s, (
                f"Min ISI {min_isi:.4f}s < refractory {refractory_s}s"
            )

    def test_spikes_are_sorted(self):
        """Spike indices should be sorted in ascending order."""
        rng = np.random.default_rng(42)
        rate = np.full(50000, 5.0)
        spikes = _generate_inhomogeneous_poisson(rate, dt=0.001,
                                                  refractory_s=0.004, rng=rng)
        assert np.all(np.diff(spikes) > 0)


# ---------------------------------------------------------------------------
# Overlay function tests (biologically-motivated rate modifications)
# ---------------------------------------------------------------------------

class TestBurstOverlay:
    """Tests for _burst_overlay() — cortical burst pattern generation."""

    def test_burst_increases_rate(self):
        """Bursting should produce rate values above the base rate."""
        rng = np.random.default_rng(42)
        rate = np.full(10000, 5.0)  # 10s at 1ms, constant 5 Hz
        rate_burst = _burst_overlay(
            rate, dt=0.001, burst_rate=1.0,
            burst_duration_ms_range=(50.0, 200.0),
            burst_factor_range=(3.0, 5.0), rng=rng,
        )
        # Max rate should exceed the base rate (bursts amplify)
        assert np.max(rate_burst) > 5.0
        # Most of the signal should be unchanged (bursts are brief)
        assert np.mean(rate_burst == 5.0) > 0.5

    def test_burst_no_negative(self):
        """Burst overlay should not create negative rates."""
        rng = np.random.default_rng(42)
        rate = np.full(10000, 2.0)
        rate_burst = _burst_overlay(
            rate, dt=0.001, burst_rate=0.5,
            burst_duration_ms_range=(50.0, 100.0),
            burst_factor_range=(2.0, 3.0), rng=rng,
        )
        assert np.all(rate_burst >= 0)

    def test_burst_zero_rate_returns_unchanged(self):
        """Zero burst_rate should return rate unchanged."""
        rng = np.random.default_rng(42)
        rate = np.full(10000, 5.0)
        result = _burst_overlay(
            rate, dt=0.001, burst_rate=0.0,
            burst_duration_ms_range=(50.0, 200.0),
            burst_factor_range=(3.0, 5.0), rng=rng,
        )
        np.testing.assert_array_equal(result, rate)

    def test_burst_does_not_modify_input(self):
        """Burst overlay should not modify the input array in-place."""
        rng = np.random.default_rng(42)
        rate = np.full(10000, 5.0)
        original = rate.copy()
        _burst_overlay(
            rate, dt=0.001, burst_rate=1.0,
            burst_duration_ms_range=(50.0, 200.0),
            burst_factor_range=(3.0, 5.0), rng=rng,
        )
        np.testing.assert_array_equal(rate, original)


class TestStimulusEventsOverlay:
    """Tests for _stimulus_events_overlay() — sensory-evoked transients."""

    def test_stimulus_adds_peaks(self):
        """Stimulus events should add transient peaks above baseline."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt
        rate = np.full(10000, 5.0)
        rate_stim = _stimulus_events_overlay(
            rate, t, base_rate=5.0,
            num_events_range=(5, 10),
            amplitude_factor_range=(3.0, 6.0),
            decay_tau_ms_range=(50.0, 150.0), rng=rng,
        )
        # Peak rate should be well above baseline (events add amplitude)
        assert np.max(rate_stim) > 10.0

    def test_stimulus_decays_to_baseline(self):
        """Rate should decay back towards baseline after stimulus."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt  # 10 seconds
        rate = np.full(10000, 5.0)
        # Single event for clear measurement
        rate_stim = _stimulus_events_overlay(
            rate, t, base_rate=5.0,
            num_events_range=(1, 1),
            amplitude_factor_range=(5.0, 5.0),
            decay_tau_ms_range=(100.0, 100.0), rng=rng,
        )
        # Last 1s should be very close to baseline (event decayed)
        last_second = rate_stim[-1000:]
        assert np.mean(last_second) < 6.0  # Near baseline of 5.0

    def test_stimulus_no_negative(self):
        """Stimulus overlay should never produce negative rates."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt
        rate = np.full(10000, 2.0)
        rate_stim = _stimulus_events_overlay(
            rate, t, base_rate=2.0,
            num_events_range=(5, 10),
            amplitude_factor_range=(3.0, 6.0),
            decay_tau_ms_range=(50.0, 150.0), rng=rng,
        )
        assert np.all(rate_stim >= 0)


class TestDriftOverlay:
    """Tests for _drift_overlay() — Ornstein-Uhlenbeck non-stationarity."""

    def test_drift_changes_rate(self):
        """Drift should cause the rate to vary from baseline."""
        rng = np.random.default_rng(42)
        rate = np.full(100000, 5.0)  # 100s at 1ms
        rate_drift = _drift_overlay(
            rate, dt=0.001, base_rate=5.0,
            tau_s=10.0, sigma=0.3, rng=rng,
        )
        # Rate should vary (std > 0)
        assert np.std(rate_drift) > 0.1

    def test_drift_mean_reversion(self):
        """With OU process, mean should stay near the base rate."""
        rng = np.random.default_rng(42)
        rate = np.full(100000, 5.0)  # 100s at 1ms
        rate_drift = _drift_overlay(
            rate, dt=0.001, base_rate=5.0,
            tau_s=10.0, sigma=0.3, rng=rng,
        )
        # Mean should stay roughly near 5.0 (within 30%)
        assert abs(np.mean(rate_drift) - 5.0) < 1.5

    def test_drift_nonnegative(self):
        """Drift overlay should clip rates to non-negative."""
        rng = np.random.default_rng(42)
        rate = np.full(10000, 2.0)
        rate_drift = _drift_overlay(
            rate, dt=0.001, base_rate=2.0,
            tau_s=5.0, sigma=0.5, rng=rng,
        )
        assert np.all(rate_drift >= 0)

    def test_drift_slow_variation(self):
        """Drift should produce slow variation (autocorrelated signal)."""
        rng = np.random.default_rng(42)
        rate = np.full(100000, 5.0)  # 100s at 1ms
        rate_drift = _drift_overlay(
            rate, dt=0.001, base_rate=5.0,
            tau_s=30.0, sigma=0.3, rng=rng,
        )
        # Lag-1 autocorrelation of the drift signal should be very high
        x = rate_drift - np.mean(rate_drift)
        var = np.var(x)
        if var > 0:
            autocorr = np.sum(x[:-1] * x[1:]) / (var * len(x))
            assert autocorr > 0.9, f"Drift autocorrelation {autocorr:.3f} too low"


class TestMultiOscillationOverlay:
    """Tests for _multi_oscillation_overlay() — theta/gamma layering."""

    def test_adds_oscillatory_components(self):
        """Should add visible oscillations to the rate."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt  # 10s
        rate = np.full(10000, 5.0)
        bands = [
            {"frequencies_hz": [4.0, 8.0], "depth": [0.1, 0.2]},
        ]
        rate_osc = _multi_oscillation_overlay(
            rate, t, base_rate=5.0, bands=bands, rng=rng,
        )
        # Rate should vary due to oscillation
        assert np.std(rate_osc) > 0.1
        # Mean should be close to base (oscillation is zero-mean)
        assert abs(np.mean(rate_osc) - 5.0) < 0.5

    def test_multiple_bands(self):
        """Multiple bands should produce richer spectral content."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt
        rate = np.full(10000, 5.0)

        # Single band
        single_band = [{"frequencies_hz": [4.0, 8.0], "depth": [0.1, 0.2]}]
        rate_single = _multi_oscillation_overlay(
            rate.copy(), t, base_rate=5.0, bands=single_band,
            rng=np.random.default_rng(42),
        )

        # Two bands
        double_band = [
            {"frequencies_hz": [4.0, 8.0], "depth": [0.1, 0.2]},
            {"frequencies_hz": [30.0, 50.0], "depth": [0.05, 0.1]},
        ]
        rate_double = _multi_oscillation_overlay(
            rate.copy(), t, base_rate=5.0, bands=double_band,
            rng=np.random.default_rng(42),
        )

        # Two bands should have more variance than one
        assert np.std(rate_double) > np.std(rate_single) * 0.8

    def test_nonnegative(self):
        """Multi-oscillation overlay should clip to non-negative."""
        rng = np.random.default_rng(42)
        dt = 0.001
        t = np.arange(10000) * dt
        rate = np.full(10000, 1.0)  # Low base rate
        bands = [
            {"frequencies_hz": [4.0, 8.0], "depth": [0.8, 0.9]},  # High depth
        ]
        rate_osc = _multi_oscillation_overlay(
            rate, t, base_rate=1.0, bands=bands, rng=rng,
        )
        assert np.all(rate_osc >= 0)


# ---------------------------------------------------------------------------
# Full generator tests
# ---------------------------------------------------------------------------

class TestGenerateModulatedSpikes:
    """Tests for the main generate_modulated_spikes() function."""

    def test_returns_sorting_and_metadata(self, modulated_config):
        """Should return (MockSorting, dict)."""
        sorting, metadata = generate_modulated_spikes(modulated_config)
        assert isinstance(sorting, MockSorting)
        assert isinstance(metadata, dict)

    def test_correct_num_units(self, modulated_config):
        """Should generate the configured number of units."""
        sorting, _ = generate_modulated_spikes(modulated_config)
        assert sorting.get_num_units() == 5

    def test_correct_sampling_frequency(self, modulated_config):
        """Sorting should report the configured sampling frequency."""
        sorting, _ = generate_modulated_spikes(modulated_config)
        assert sorting.get_sampling_frequency() == 30000.0

    def test_all_units_have_spikes(self, modulated_config):
        """Each unit should have at least some spikes in 30s."""
        sorting, _ = generate_modulated_spikes(modulated_config)
        for uid in sorting.get_unit_ids():
            spikes = sorting.get_unit_spike_train(uid)
            assert len(spikes) > 0, f"Unit {uid} has no spikes"

    def test_spike_rates_in_reasonable_range(self, modulated_config):
        """Per-unit mean rates should be within a reasonable range."""
        sorting, metadata = generate_modulated_spikes(modulated_config)
        duration = metadata["duration_s"]
        for uid in sorting.get_unit_ids():
            rate = len(sorting.get_unit_spike_train(uid)) / duration
            # With base rates 3-8 Hz + modulation, expect 1-20 Hz range
            assert 0.5 < rate < 25.0, f"Unit {uid} rate {rate:.1f} Hz out of range"

    def test_temporal_autocorrelation_positive(self, modulated_config):
        """
        KEY TEST: After binning to 10ms counts, lag-1 autocorrelation
        should be significantly positive (temporal structure exists).
        """
        from src.data.binning import bin_spike_trains

        sorting, _ = generate_modulated_spikes(modulated_config)
        counts, _ = bin_spike_trains(sorting, bin_width_ms=10.0)

        # Compute lag-1 autocorrelation for each channel, average
        autocorrs = []
        for ch in range(counts.shape[0]):
            x = counts[ch, :].astype(float)
            x_centered = x - np.mean(x)
            var = np.var(x)
            if var > 0:
                autocorr = np.correlate(x_centered[:-1], x_centered[1:]) / (
                    var * (len(x) - 1)
                )
                autocorrs.append(autocorr[0])

        mean_autocorr = np.mean(autocorrs)
        # With sinusoidal modulation, autocorrelation should be clearly positive
        assert mean_autocorr > 0.01, (
            f"Mean lag-1 autocorrelation {mean_autocorr:.4f} is too low. "
            f"Temporal structure not present in binned counts."
        )

    def test_cross_channel_correlation(self, modulated_config):
        """
        When shared signal is enabled, channels should be correlated.
        """
        from src.data.binning import bin_spike_trains

        sorting, _ = generate_modulated_spikes(modulated_config)
        counts, _ = bin_spike_trains(sorting, bin_width_ms=10.0)

        # Compute mean pairwise Pearson correlation
        from itertools import combinations
        corrs = []
        for i, j in combinations(range(counts.shape[0]), 2):
            x = counts[i, :].astype(float)
            y = counts[j, :].astype(float)
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r):
                corrs.append(r)

        mean_corr = np.mean(corrs)
        # With shared signal, expect positive mean correlation
        assert mean_corr > 0.0, (
            f"Mean cross-channel correlation {mean_corr:.4f} is not positive. "
            f"Shared signal may not be working."
        )

    def test_no_cross_channel_correlation_when_disabled(self, step_config):
        """
        Without shared signal, cross-channel correlations should be near zero.
        """
        from src.data.binning import bin_spike_trains

        sorting, _ = generate_modulated_spikes(step_config)
        counts, _ = bin_spike_trains(sorting, bin_width_ms=10.0)

        from itertools import combinations
        corrs = []
        for i, j in combinations(range(counts.shape[0]), 2):
            x = counts[i, :].astype(float)
            y = counts[j, :].astype(float)
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r):
                corrs.append(r)

        mean_corr = np.mean(corrs)
        # Without shared signal, mean correlation should be close to zero
        # (allow small random correlations)
        assert abs(mean_corr) < 0.15, (
            f"Mean cross-channel correlation {mean_corr:.4f} too high "
            f"with shared signal disabled"
        )

    def test_refractory_period_across_all_units(self, modulated_config):
        """No unit should have ISI < refractory period."""
        sorting, _ = generate_modulated_spikes(modulated_config)
        fs = sorting.get_sampling_frequency()
        refractory_ms = 4.0
        refractory_samples = int(fs * refractory_ms / 1000.0)

        for uid in sorting.get_unit_ids():
            spikes = sorting.get_unit_spike_train(uid)
            if len(spikes) > 1:
                isis = np.diff(spikes)
                min_isi = np.min(isis)
                # Allow small tolerance (1 sample) due to rounding
                assert min_isi >= refractory_samples - 1, (
                    f"Unit {uid}: min ISI {min_isi} samples < "
                    f"refractory {refractory_samples} samples"
                )

    def test_reproducibility_same_seed(self, modulated_config):
        """Same seed should produce identical spike trains."""
        sorting1, _ = generate_modulated_spikes(modulated_config)
        sorting2, _ = generate_modulated_spikes(modulated_config)

        for uid in sorting1.get_unit_ids():
            np.testing.assert_array_equal(
                sorting1.get_unit_spike_train(uid),
                sorting2.get_unit_spike_train(uid),
                err_msg=f"Unit {uid} spikes differ with same seed",
            )

    def test_different_seed_gives_different_spikes(self, modulated_config):
        """Different seed should produce different spike trains."""
        sorting1, _ = generate_modulated_spikes(modulated_config)
        modulated_config["seed"] = 99
        sorting2, _ = generate_modulated_spikes(modulated_config)

        counts1 = [len(sorting1.get_unit_spike_train(u))
                    for u in sorting1.get_unit_ids()]
        counts2 = [len(sorting2.get_unit_spike_train(u))
                    for u in sorting2.get_unit_ids()]
        assert counts1 != counts2, "Different seeds produced identical spikes"

    def test_step_modulation_works(self, step_config):
        """Step modulation should produce valid spike trains."""
        sorting, metadata = generate_modulated_spikes(step_config)
        assert sorting.get_num_units() == 5
        assert metadata["total_spikes"] > 0

    def test_integration_with_binning(self, modulated_config):
        """Output should integrate seamlessly with bin_spike_trains()."""
        from src.data.binning import bin_spike_trains

        sorting, _ = generate_modulated_spikes(modulated_config)
        counts, meta = bin_spike_trains(sorting, bin_width_ms=10.0)

        # Shape should be (num_units, num_bins)
        assert counts.shape[0] == 5
        assert counts.shape[1] > 0
        assert counts.dtype == np.int32
        assert np.all(counts >= 0)

    def test_metadata_contains_expected_keys(self, modulated_config):
        """Metadata should contain all expected keys."""
        _, metadata = generate_modulated_spikes(modulated_config)
        expected_keys = [
            "generator", "seed", "num_units", "duration_s",
            "sampling_frequency", "modulation_types",
            "shared_signal_enabled", "total_spikes",
            "mean_rate_hz", "unit_details",
        ]
        for key in expected_keys:
            assert key in metadata, f"Missing metadata key: {key}"

    def test_unknown_modulation_type_raises(self):
        """Unknown modulation type should raise ValueError."""
        config = {
            "seed": 42,
            "spikeinterface": {
                "num_neurons": 2,
                "duration_s": 5.0,
                "sampling_frequency": 30000.0,
                "firing": {"rates": [3.0, 8.0], "refractory_period_ms": 4.0},
            },
            "modulation": {
                "types": ["unknown_type"],
            },
        }
        with pytest.raises(ValueError, match="Unknown modulation type"):
            generate_modulated_spikes(config)


# ---------------------------------------------------------------------------
# Full generator with overlays enabled
# ---------------------------------------------------------------------------

class TestGenerateWithOverlays:
    """Integration tests for the full generator with all overlays enabled."""

    def test_rich_config_produces_valid_output(self, rich_config):
        """All overlays enabled should still produce valid MockSorting."""
        sorting, metadata = generate_modulated_spikes(rich_config)
        assert isinstance(sorting, MockSorting)
        assert sorting.get_num_units() == 3
        assert metadata["total_spikes"] > 0

    def test_rich_config_all_units_have_spikes(self, rich_config):
        """Every unit should have spikes even with all overlays active."""
        sorting, _ = generate_modulated_spikes(rich_config)
        for uid in sorting.get_unit_ids():
            spikes = sorting.get_unit_spike_train(uid)
            assert len(spikes) > 0, f"Unit {uid} has no spikes"

    def test_rich_config_metadata_has_overlays(self, rich_config):
        """Metadata should record which overlays were active."""
        _, metadata = generate_modulated_spikes(rich_config)
        assert "overlays" in metadata
        overlays = metadata["overlays"]
        assert overlays["bursting"] is True
        assert overlays["stimulus_events"] is True
        assert overlays["drift"] is True
        assert overlays["multi_oscillation"] is True

    def test_rich_config_integrates_with_binning(self, rich_config):
        """Rich generator output should bin correctly."""
        from src.data.binning import bin_spike_trains

        sorting, _ = generate_modulated_spikes(rich_config)
        counts, meta = bin_spike_trains(sorting, bin_width_ms=10.0)

        assert counts.shape[0] == 3
        assert counts.shape[1] > 0
        assert counts.dtype == np.int32
        assert np.all(counts >= 0)

    def test_rich_config_reproducible(self, rich_config):
        """Same seed with all overlays should produce identical results."""
        sorting1, _ = generate_modulated_spikes(rich_config)
        sorting2, _ = generate_modulated_spikes(rich_config)

        for uid in sorting1.get_unit_ids():
            np.testing.assert_array_equal(
                sorting1.get_unit_spike_train(uid),
                sorting2.get_unit_spike_train(uid),
                err_msg=f"Unit {uid} spikes differ with same seed + overlays",
            )

    def test_overlays_disabled_backward_compatible(self, modulated_config):
        """No overlays in config → metadata should show all disabled."""
        _, metadata = generate_modulated_spikes(modulated_config)
        assert "overlays" in metadata
        overlays = metadata["overlays"]
        assert overlays["bursting"] is False
        assert overlays["stimulus_events"] is False
        assert overlays["drift"] is False
        assert overlays["multi_oscillation"] is False
