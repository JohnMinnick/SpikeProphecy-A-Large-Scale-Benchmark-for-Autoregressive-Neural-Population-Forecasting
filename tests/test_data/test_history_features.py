"""
Tests for src/data/history_features.py

Tests the ISI, EMA rate, refractory period, and dispatcher functions
with known-answer verification and causality checks.
"""

import numpy as np
import pytest

from src.data.history_features import (
    compute_ema_rate,
    compute_history_features,
    compute_isi_features,
    compute_refractory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_counts():
    """
    Small spike-count matrix for testing: 3 channels, 10 bins.

    Known values for hand-computation:
        Channel 0: [1, 0, 0, 2, 0, 0, 0, 1, 0, 0]   (spikes at t=0,3,7)
        Channel 1: [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]   (spikes at t=2,5,9)
        Channel 2: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   (spike every bin)
    """
    counts = np.array([
        [1, 0, 0, 2, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.int32)
    return counts


# ---------------------------------------------------------------------------
# ISI tests
# ---------------------------------------------------------------------------

class TestComputeISI:
    """Tests for compute_isi_features()."""

    def test_shape(self, small_counts):
        """Output shape should match input."""
        isi = compute_isi_features(small_counts, bin_width_ms=10.0)
        assert isi.shape == small_counts.shape

    def test_dtype_float32(self, small_counts):
        """Output should be float32."""
        isi = compute_isi_features(small_counts, bin_width_ms=10.0)
        assert isi.dtype == np.float32

    def test_values_in_unit_range(self, small_counts):
        """All ISI values should be in [0, 1]."""
        isi = compute_isi_features(small_counts, bin_width_ms=10.0)
        assert np.all(isi >= 0.0)
        assert np.all(isi <= 1.0)

    def test_first_bin_is_max(self, small_counts):
        """First bin should have max ISI (no prior spikes) = 1.0."""
        isi = compute_isi_features(small_counts, bin_width_ms=10.0, max_isi_ms=500.0)
        # All channels at t=0: no prior bins exist → ISI = max = 1.0
        np.testing.assert_allclose(isi[:, 0], 1.0)

    def test_known_answer_channel0(self, small_counts):
        """
        Channel 0: spikes at t=0,3,7. bin_width=10ms, max_isi=500ms (50 bins).
        ISI at each bin (bins since last spike, causal):
          t=0: no prior bins → 50/50 = 1.0
          t=1: spike at t=0 reset counter to 0, then +1 → ISI reads 0 → 0/50 = 0.0
               Wait — the loop: ISI[t] = bins_since (pre-update), spike resets,
               non-spike increments. So:
          Let's trace bins_since_spike for channel 0 (init=50):
            t=0: isi=50/50=1.0, spike → reset to 0
            t=1: isi=0/50=0.0, no spike → inc to 1
            t=2: isi=1/50=0.02, no spike → inc to 2
            t=3: isi=2/50=0.04, spike → reset to 0
            t=4: isi=0/50=0.0, no spike → inc to 1
        """
        isi = compute_isi_features(small_counts, bin_width_ms=10.0, max_isi_ms=500.0)
        np.testing.assert_allclose(isi[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(isi[0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(isi[0, 2], 1.0 / 50.0, atol=1e-6)
        np.testing.assert_allclose(isi[0, 3], 2.0 / 50.0, atol=1e-6)
        np.testing.assert_allclose(isi[0, 4], 0.0, atol=1e-6)

    def test_always_spiking_channel(self, small_counts):
        """Channel 2 spikes every bin; ISI after t=0 should be 0.0.

        Every bin has a spike, so bins_since_spike resets to 0 each time.
        The ISI reads the counter before the update, so it's always 0.
        """
        isi = compute_isi_features(small_counts, bin_width_ms=10.0, max_isi_ms=500.0)
        for t in range(1, 10):
            np.testing.assert_allclose(
                isi[2, t], 0.0, atol=1e-6,
                err_msg=f"Channel 2 ISI wrong at t={t}",
            )

    def test_causality(self, small_counts):
        """
        ISI at time t should only depend on counts at t-1 and earlier.
        Modifying counts at time t should not change ISI at time t.
        """
        isi_original = compute_isi_features(small_counts, bin_width_ms=10.0)
        # Modify count at t=5
        modified = small_counts.copy()
        modified[0, 5] = 10
        isi_modified = compute_isi_features(modified, bin_width_ms=10.0)
        # ISI at t=5 should be the same (depends only on past)
        np.testing.assert_allclose(isi_original[:, 5], isi_modified[:, 5])


# ---------------------------------------------------------------------------
# EMA rate tests
# ---------------------------------------------------------------------------

class TestComputeEMA:
    """Tests for compute_ema_rate()."""

    def test_shape(self, small_counts):
        """Output shape should match input."""
        ema = compute_ema_rate(small_counts, alpha=0.1)
        assert ema.shape == small_counts.shape

    def test_first_bin_is_zero(self, small_counts):
        """EMA at t=0 should be zero (no past data)."""
        ema = compute_ema_rate(small_counts, alpha=0.1)
        np.testing.assert_allclose(ema[:, 0], 0.0)

    def test_known_answer(self, small_counts):
        """
        Channel 0: [1, 0, 0, 2, 0, ...]
        With alpha=0.5:
          ema[0] = 0
          ema[1] = 0.5 * count[0] + 0.5 * ema[0] = 0.5 * 1 + 0 = 0.5
          ema[2] = 0.5 * count[1] + 0.5 * ema[1] = 0.5 * 0 + 0.5 * 0.5 = 0.25
          ema[3] = 0.5 * count[2] + 0.5 * ema[2] = 0.5 * 0 + 0.5 * 0.25 = 0.125
          ema[4] = 0.5 * count[3] + 0.5 * ema[3] = 0.5 * 2 + 0.5 * 0.125 = 1.0625
        """
        ema = compute_ema_rate(small_counts, alpha=0.5)
        np.testing.assert_allclose(ema[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(ema[0, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(ema[0, 2], 0.25, atol=1e-6)
        np.testing.assert_allclose(ema[0, 3], 0.125, atol=1e-6)
        np.testing.assert_allclose(ema[0, 4], 1.0625, atol=1e-6)

    def test_non_negative(self, small_counts):
        """EMA should always be non-negative for non-negative counts."""
        ema = compute_ema_rate(small_counts, alpha=0.1)
        assert np.all(ema >= 0.0)

    def test_causality(self, small_counts):
        """Modifying counts at time t should not change EMA at time t."""
        ema_original = compute_ema_rate(small_counts, alpha=0.2)
        modified = small_counts.copy()
        modified[0, 5] = 10
        ema_modified = compute_ema_rate(modified, alpha=0.2)
        # EMA at t=5 uses count[4], not count[5]
        np.testing.assert_allclose(ema_original[:, 5], ema_modified[:, 5])

    def test_invalid_alpha_raises(self, small_counts):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_ema_rate(small_counts, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_ema_rate(small_counts, alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_ema_rate(small_counts, alpha=-0.1)


# ---------------------------------------------------------------------------
# Refractory tests
# ---------------------------------------------------------------------------

class TestComputeRefractory:
    """Tests for compute_refractory()."""

    def test_shape(self, small_counts):
        """Output shape should match input."""
        ref = compute_refractory(small_counts, refractory_bins=1)
        assert ref.shape == small_counts.shape

    def test_first_bin_is_zero(self, small_counts):
        """Refractory at t=0 should be zero (no prior bins)."""
        ref = compute_refractory(small_counts, refractory_bins=1)
        np.testing.assert_allclose(ref[:, 0], 0.0)

    def test_binary_values(self, small_counts):
        """All values should be 0.0 or 1.0."""
        ref = compute_refractory(small_counts, refractory_bins=1)
        assert np.all((ref == 0.0) | (ref == 1.0))

    def test_known_answer_refractory_1(self, small_counts):
        """
        Channel 0: [1, 0, 0, 2, 0, 0, 0, 1, 0, 0]
        refractory_bins=1: flag is 1 if count[t-1] >= 1
          t=0: no prior → 0
          t=1: count[0]=1 → 1
          t=2: count[1]=0 → 0
          t=3: count[2]=0 → 0
          t=4: count[3]=2 → 1
        """
        ref = compute_refractory(small_counts, refractory_bins=1)
        expected_ch0 = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        np.testing.assert_array_equal(ref[0], expected_ch0)

    def test_known_answer_refractory_2(self, small_counts):
        """
        Channel 0: [1, 0, 0, 2, 0, ...] with refractory_bins=2.
        Flag is 1 if any of count[t-2:t] >= 1:
          t=0: 0
          t=1: count[0]=1 → 1
          t=2: count[0:2]=[1,0] → 1 (t=0 had spike, still within 2-bin window)
          t=3: count[1:3]=[0,0] → 0
          t=4: count[2:4]=[0,2] → 1
          t=5: count[3:5]=[2,0] → 1
        """
        ref = compute_refractory(small_counts, refractory_bins=2)
        expected_ch0 = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
        np.testing.assert_array_equal(ref[0], expected_ch0)

    def test_always_spiking(self, small_counts):
        """Channel 2 spikes every bin — refractory should be 1 for t>0."""
        ref = compute_refractory(small_counts, refractory_bins=1)
        np.testing.assert_allclose(ref[2, 0], 0.0)
        for t in range(1, 10):
            np.testing.assert_allclose(ref[2, t], 1.0)

    def test_invalid_refractory_bins_raises(self, small_counts):
        """refractory_bins < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="refractory_bins must be >= 1"):
            compute_refractory(small_counts, refractory_bins=0)


# ---------------------------------------------------------------------------
# Dispatcher (compute_history_features) tests
# ---------------------------------------------------------------------------

class TestComputeHistoryFeatures:
    """Tests for compute_history_features() dispatcher."""

    def test_disabled_returns_empty(self, small_counts):
        """Master toggle OFF → empty feature matrix, 0 per channel."""
        config = {"history_features": {"enabled": False}}
        features, n = compute_history_features(small_counts, config)
        assert features.shape == (0, small_counts.shape[1])
        assert n == 0

    def test_no_config_returns_empty(self, small_counts):
        """Missing history_features key → empty."""
        features, n = compute_history_features(small_counts, {})
        assert features.shape == (0, small_counts.shape[1])
        assert n == 0

    def test_enabled_but_no_sub_features(self, small_counts):
        """Master ON but all subs OFF → empty."""
        config = {
            "history_features": {
                "enabled": True,
                "isi": {"enabled": False},
                "ema_rate": {"enabled": False},
                "refractory": {"enabled": False},
            }
        }
        features, n = compute_history_features(small_counts, config)
        assert features.shape == (0, small_counts.shape[1])
        assert n == 0

    def test_single_feature_shape(self, small_counts):
        """One feature enabled → (M, T_total), n=1."""
        config = {
            "history_features": {
                "enabled": True,
                "isi": {"enabled": True, "max_isi_ms": 500.0},
            }
        }
        features, n = compute_history_features(small_counts, config)
        m, t = small_counts.shape
        assert features.shape == (m, t)
        assert n == 1

    def test_all_features_shape(self, small_counts):
        """All three features enabled → (3*M, T_total), n=3."""
        config = {
            "bin_width_ms": 10.0,
            "history_features": {
                "enabled": True,
                "isi": {"enabled": True, "max_isi_ms": 500.0},
                "ema_rate": {"enabled": True, "alpha": 0.1},
                "refractory": {"enabled": True, "refractory_bins": 1},
            },
        }
        features, n = compute_history_features(small_counts, config)
        m, t = small_counts.shape
        assert features.shape == (3 * m, t)
        assert n == 3

    def test_feature_order_is_isi_ema_refractory(self, small_counts):
        """Features should be stacked in order: ISI, EMA, refractory."""
        config = {
            "bin_width_ms": 10.0,
            "history_features": {
                "enabled": True,
                "isi": {"enabled": True, "max_isi_ms": 500.0},
                "ema_rate": {"enabled": True, "alpha": 0.1},
                "refractory": {"enabled": True, "refractory_bins": 1},
            },
        }
        features, _ = compute_history_features(small_counts, config)
        m = small_counts.shape[0]

        # Compare blocks against individual functions
        isi = compute_isi_features(small_counts, bin_width_ms=10.0, max_isi_ms=500.0)
        ema = compute_ema_rate(small_counts, alpha=0.1)
        ref = compute_refractory(small_counts, refractory_bins=1)

        np.testing.assert_allclose(features[0:m], isi)
        np.testing.assert_allclose(features[m:2*m], ema)
        np.testing.assert_allclose(features[2*m:3*m], ref)
