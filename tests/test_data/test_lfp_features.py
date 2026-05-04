"""
Tests for LFP band-power feature extraction.

Uses synthetic sinusoidal signals with known frequency content to
validate band-power extraction, log transformation, shape contracts,
and edge cases.  No real NWB files are needed.
"""

import numpy as np
import pytest

from src.data.lfp_features import (
    DEFAULT_BANDS,
    _bandpower_welch,
    compute_lfp_band_power,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_sine(freq_hz: float, duration_s: float, fs: float) -> np.ndarray:
    """Generate a pure sine wave at a given frequency."""
    t = np.arange(0, duration_s, 1.0 / fs)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _make_bin_edges(duration_s: float, bin_width_s: float) -> np.ndarray:
    """Generate evenly-spaced bin edges in seconds."""
    n_bins = int(duration_s / bin_width_s)
    return np.linspace(0.0, duration_s, n_bins + 1)


# =============================================================================
# _bandpower_welch tests
# =============================================================================


class TestBandpowerWelch:
    """Tests for the internal Welch band-power helper."""

    def test_known_frequency_in_band(self):
        """A 40 Hz sine should have high power in gamma band (30-80 Hz)."""
        fs = 1000.0
        signal = _make_sine(40.0, duration_s=1.0, fs=fs)

        gamma_power = _bandpower_welch(signal, fs, band=(30.0, 80.0))
        theta_power = _bandpower_welch(signal, fs, band=(4.0, 8.0))

        # Gamma should be much larger than theta for a 40 Hz signal
        assert gamma_power > 0.0, "Gamma power should be positive"
        assert gamma_power > 100 * theta_power, (
            f"40 Hz signal should have >> gamma than theta power, "
            f"got gamma={gamma_power:.6e}, theta={theta_power:.6e}"
        )

    def test_dc_signal_has_no_ac_power(self):
        """A constant (DC) signal should have ~zero power in all AC bands."""
        fs = 1000.0
        signal = np.ones(1000, dtype=np.float32) * 5.0  # DC offset

        for band_name, band in DEFAULT_BANDS.items():
            power = _bandpower_welch(signal, fs, band)
            assert power < 1e-6, (
                f"DC signal should have ~zero {band_name} power, "
                f"got {power:.6e}"
            )

    def test_multi_frequency_peaks_in_correct_bands(self):
        """Signal with 6 Hz + 50 Hz components should show in theta + gamma."""
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        # 6 Hz (theta) + 50 Hz (gamma), equal amplitude
        signal = (
            np.sin(2 * np.pi * 6.0 * t) + np.sin(2 * np.pi * 50.0 * t)
        ).astype(np.float32)

        theta_power = _bandpower_welch(signal, fs, band=(4.0, 8.0))
        gamma_power = _bandpower_welch(signal, fs, band=(30.0, 80.0))
        beta_power = _bandpower_welch(signal, fs, band=(12.0, 30.0))

        # Theta and gamma should both be larger than beta
        assert theta_power > 10 * beta_power, (
            f"6 Hz component should dominate theta, "
            f"theta={theta_power:.6e}, beta={beta_power:.6e}"
        )
        assert gamma_power > 10 * beta_power, (
            f"50 Hz component should dominate gamma, "
            f"gamma={gamma_power:.6e}, beta={beta_power:.6e}"
        )

    def test_very_short_segment_returns_zero(self):
        """Segments shorter than 4 samples should return 0.0."""
        fs = 1000.0
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        power = _bandpower_welch(signal, fs, band=(4.0, 8.0))
        assert power == 0.0, f"Expected 0.0 for 3-sample segment, got {power}"


# =============================================================================
# compute_lfp_band_power tests
# =============================================================================


class TestComputeLfpBandPower:
    """Tests for the main LFP band-power extraction function."""

    def test_output_shape_default_bands(self):
        """Output shape should be (n_bands * n_channels, n_bins)."""
        fs = 1000.0
        n_channels = 3
        duration_s = 2.0
        bin_width_s = 0.05  # 50 ms bins

        # Multi-channel noise
        rng = np.random.default_rng(42)
        lfp = rng.standard_normal((n_channels, int(duration_s * fs)))
        lfp = lfp.astype(np.float32)

        bin_edges = _make_bin_edges(duration_s, bin_width_s)
        n_bins = len(bin_edges) - 1

        features = compute_lfp_band_power(lfp, fs, bin_edges)

        n_bands = len(DEFAULT_BANDS)
        expected_shape = (n_bands * n_channels, n_bins)
        assert features.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {features.shape}"
        )

    def test_output_shape_single_channel_1d(self):
        """1-D input (single channel) should be handled correctly."""
        fs = 1000.0
        duration_s = 1.0
        signal = _make_sine(20.0, duration_s, fs)  # 1-D
        bin_edges = _make_bin_edges(duration_s, 0.05)
        n_bins = len(bin_edges) - 1

        features = compute_lfp_band_power(signal, fs, bin_edges)

        n_bands = len(DEFAULT_BANDS)
        assert features.shape == (n_bands, n_bins)

    def test_custom_bands(self):
        """Custom band dict should control output row count."""
        fs = 1000.0
        duration_s = 1.0
        signal = _make_sine(20.0, duration_s, fs)
        bin_edges = _make_bin_edges(duration_s, 0.1)
        n_bins = len(bin_edges) - 1

        custom_bands = {"low": (1.0, 10.0), "high": (10.0, 100.0)}
        features = compute_lfp_band_power(
            signal, fs, bin_edges, bands=custom_bands,
        )

        assert features.shape == (2, n_bins), (
            f"Expected 2 band rows, got {features.shape[0]}"
        )

    def test_log_transform_range(self):
        """Output should be log10-transformed (negative values expected for small power)."""
        fs = 1000.0
        duration_s = 2.0
        signal = _make_sine(40.0, duration_s, fs)
        bin_edges = _make_bin_edges(duration_s, 0.05)

        features = compute_lfp_band_power(signal, fs, bin_edges)

        # Log10 of small power values should be negative
        # Log10 of _LOG_EPS (1e-20) = -20
        # Actual power should produce values > -20
        assert features.dtype == np.float32, (
            f"Expected float32, got {features.dtype}"
        )
        # At least some values should be > -20 (not all zero power)
        assert np.any(features > -19.0), (
            f"Expected some log-power > -19, max is {features.max():.2f}"
        )

    def test_multichannel_stacking_order(self):
        """Bands should be stacked channel-first: ch0 bands, then ch1 bands."""
        fs = 1000.0
        duration_s = 4.0
        # 0.5s bins = 500 samples each — enough for Welch to resolve 6 Hz
        bin_edges = _make_bin_edges(duration_s, 0.5)

        # Channel 0: 6 Hz (theta), Channel 1: 50 Hz (gamma)
        ch0 = _make_sine(6.0, duration_s, fs)
        ch1 = _make_sine(50.0, duration_s, fs)
        lfp = np.stack([ch0, ch1], axis=0)

        bands = {"theta": (4.0, 8.0), "gamma": (30.0, 80.0)}
        features = compute_lfp_band_power(lfp, fs, bin_edges, bands=bands)

        # Shape: (2 bands * 2 channels, n_bins) = (4, n_bins)
        # Row 0: ch0 theta (should be high — 6 Hz signal)
        # Row 1: ch0 gamma (should be low — no gamma in ch0)
        # Row 2: ch1 theta (should be low — no theta in ch1)
        # Row 3: ch1 gamma (should be high — 50 Hz signal)
        ch0_theta = features[0, :].mean()
        ch0_gamma = features[1, :].mean()
        ch1_theta = features[2, :].mean()
        ch1_gamma = features[3, :].mean()

        # ch0 theta > ch0 gamma (6 Hz signal in ch0)
        assert ch0_theta > ch0_gamma, (
            f"ch0 should have more theta than gamma, "
            f"theta={ch0_theta:.2f}, gamma={ch0_gamma:.2f}"
        )
        # ch1 gamma > ch1 theta (50 Hz signal in ch1)
        assert ch1_gamma > ch1_theta, (
            f"ch1 should have more gamma than theta, "
            f"gamma={ch1_gamma:.2f}, theta={ch1_theta:.2f}"
        )

    def test_invalid_ndim_raises(self):
        """3-D input should raise ValueError."""
        fs = 1000.0
        lfp_3d = np.zeros((2, 3, 100), dtype=np.float32)
        bin_edges = _make_bin_edges(1.0, 0.1)

        with pytest.raises(ValueError, match="1-D or 2-D"):
            compute_lfp_band_power(lfp_3d, fs, bin_edges)

    def test_insufficient_bin_edges_raises(self):
        """Fewer than 2 bin edges should raise ValueError."""
        fs = 1000.0
        signal = _make_sine(10.0, 1.0, fs)

        with pytest.raises(ValueError, match=">=.*2 elements"):
            compute_lfp_band_power(signal, fs, np.array([0.0]))

    def test_short_bins_graceful(self):
        """Very short bins (few samples per bin) should not crash."""
        fs = 500.0
        duration_s = 0.1  # Only 50 samples total
        signal = _make_sine(10.0, duration_s, fs)
        # 10 bins of 10ms each = 5 samples per bin
        bin_edges = _make_bin_edges(duration_s, 0.01)

        # Should not raise — bins with <4 samples get zero power
        features = compute_lfp_band_power(signal, fs, bin_edges)
        assert features.shape[1] == len(bin_edges) - 1


# =============================================================================
# load_lfp_from_nwb tests (mocked)
# =============================================================================


class TestLoadLfpFromNwb:
    """Tests for the NWB LFP reader using mocked NWB structures."""

    def test_missing_ecephys_returns_none(self, tmp_path):
        """NWB file without ecephys processing module should return None."""
        from unittest.mock import MagicMock, patch

        from src.data.lfp_nwb_reader import load_lfp_from_nwb

        # Create a dummy file so FileNotFoundError isn't raised
        dummy_path = tmp_path / "test.nwb"
        dummy_path.touch()

        # Mock NWB file with no ecephys module
        mock_nwbfile = MagicMock()
        mock_nwbfile.processing = {}

        mock_io = MagicMock()
        mock_io.__enter__ = MagicMock(return_value=mock_io)
        mock_io.__exit__ = MagicMock(return_value=False)
        mock_io.read.return_value = mock_nwbfile

        mock_io_cls = MagicMock(return_value=mock_io)

        with patch.dict(
            "sys.modules",
            {"pynwb": MagicMock(NWBHDF5IO=mock_io_cls)},
        ):
            # Re-import to pick up the patched pynwb
            import importlib
            import src.data.lfp_nwb_reader as reader_mod
            importlib.reload(reader_mod)
            result = reader_mod.load_lfp_from_nwb(dummy_path)

        assert result is None

    def test_valid_lfp_returns_signal_and_rate(self, tmp_path):
        """Mock NWB with valid LFP should return (signal, rate) tuple."""
        from unittest.mock import MagicMock, patch

        from src.data.lfp_nwb_reader import load_lfp_from_nwb

        dummy_path = tmp_path / "test.nwb"
        dummy_path.touch()

        # Create mock LFP data: (n_samples, n_channels) as stored in NWB
        n_samples, n_channels = 5000, 3
        mock_data = np.random.randn(n_samples, n_channels).astype(np.float32)

        # Build mock NWB hierarchy:
        # nwbfile.processing["ecephys"].data_interfaces["LFP"]
        #   .electrical_series["lfp_series"].data / .rate
        mock_series = MagicMock()
        mock_series.data.__getitem__ = MagicMock(return_value=mock_data)
        mock_series.rate = 2500.0

        mock_lfp_container = MagicMock()
        mock_lfp_container.electrical_series = {"lfp_series": mock_series}

        mock_ecephys = MagicMock()
        mock_ecephys.data_interfaces = {"LFP": mock_lfp_container}

        mock_nwbfile = MagicMock()
        mock_nwbfile.processing = {"ecephys": mock_ecephys}

        mock_io = MagicMock()
        mock_io.__enter__ = MagicMock(return_value=mock_io)
        mock_io.__exit__ = MagicMock(return_value=False)
        mock_io.read.return_value = mock_nwbfile

        mock_io_cls = MagicMock(return_value=mock_io)

        with patch.dict(
            "sys.modules",
            {"pynwb": MagicMock(NWBHDF5IO=mock_io_cls)},
        ):
            # Re-import to pick up the patched pynwb
            import importlib
            import src.data.lfp_nwb_reader as reader_mod
            importlib.reload(reader_mod)
            result = reader_mod.load_lfp_from_nwb(dummy_path)

        assert result is not None
        lfp_signal, sampling_rate = result

        # Should be transposed to (n_channels, n_samples)
        assert lfp_signal.shape == (n_channels, n_samples)
        assert lfp_signal.dtype == np.float32
        assert sampling_rate == 2500.0

    def test_file_not_found_raises(self):
        """Non-existent path should raise FileNotFoundError."""
        from src.data.lfp_nwb_reader import load_lfp_from_nwb

        with pytest.raises(FileNotFoundError):
            load_lfp_from_nwb("/nonexistent/path/to/file.nwb")
