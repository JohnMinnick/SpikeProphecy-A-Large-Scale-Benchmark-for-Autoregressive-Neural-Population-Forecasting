"""
Format-agnostic LFP band-power feature extraction.

Computes spectral band power from continuous voltage (LFP) traces and
aligns output to spike-count bins.  Uses Welch's method for robust
spectral estimation.  Output is log-transformed to compress the wide
dynamic range of power spectral density.

This module has NO hardware- or file-format-specific imports.  NWB
reading is handled by ``lfp_nwb_reader.py``; future Maxwell MEA
readers will provide the same ``(lfp_signal, sampling_rate)`` tuple.

Design decisions documented in ADR-0015 (Tier A+).

Usage:
    from src.data.lfp_features import compute_lfp_band_power, DEFAULT_BANDS

    # lfp_signal: (n_channels, n_samples) float array from any source
    # bin_edges_s: (n_bins + 1,) array of bin boundary times in seconds
    features = compute_lfp_band_power(
        lfp_signal, sampling_rate=2500.0, bin_edges_s=bin_edges_s,
    )
    # features: (n_bands * n_channels, n_bins) float32 array
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Default frequency bands (Hz)
# -------------------------------------------------------------------------

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 80.0),
    "high_gamma": (80.0, 200.0),
}

# Small constant added before log transform to avoid log(0)
_LOG_EPS = 1e-20


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------


def _bandpower_welch(
    signal_segment: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    nperseg: Optional[int] = None,
) -> float:
    """
    Compute average power in a frequency band for a 1-D signal segment.

    Uses Welch's method (overlapping Hann-windowed FFT segments) for
    robust spectral estimation.  The power in the target band is the
    mean of the PSD values whose frequencies fall within [f_low, f_high].

    Args:
        signal_segment: 1-D array of voltage samples.
        fs: Sampling rate in Hz.
        band: (f_low, f_high) frequency range in Hz.
        nperseg: Samples per Welch segment.  Defaults to min(256, len(signal)).
            Smaller values give noisier but faster estimates.

    Returns:
        Average spectral power (µV²/Hz) in the band.  Returns 0.0 if
        the segment is too short for any spectral estimate.
    """
    n = len(signal_segment)

    # Cannot compute spectrum from fewer than 4 samples
    if n < 4:
        return 0.0

    # Default segment length: 256 samples or signal length, whichever is
    # smaller.  Ensures Welch can always run without error.
    if nperseg is None:
        nperseg = min(256, n)

    freqs, psd = welch(signal_segment, fs=fs, nperseg=nperseg)

    # Select frequencies within the band
    f_low, f_high = band
    band_mask = (freqs >= f_low) & (freqs <= f_high)

    if not np.any(band_mask):
        # Band falls outside the resolvable frequency range
        return 0.0

    return float(np.mean(psd[band_mask]))


# -------------------------------------------------------------------------
# Main extraction function
# -------------------------------------------------------------------------


def compute_lfp_band_power(
    lfp_signal: np.ndarray,
    sampling_rate: float,
    bin_edges_s: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Compute LFP band power aligned to spike-count bins.

    For each bin (defined by consecutive edges in ``bin_edges_s``), extracts
    the corresponding LFP segment and computes spectral power in each
    frequency band using Welch's method.  Output is log10-transformed
    to compress dynamic range.

    Args:
        lfp_signal: Continuous LFP voltage, shape (n_channels, n_samples).
            Can also be 1-D (n_samples,) for single-channel data — will
            be reshaped to (1, n_samples).
        sampling_rate: LFP sampling rate in Hz (e.g. 2500.0).
        bin_edges_s: Bin edge times in seconds, shape (n_bins + 1,).
            Consecutive pairs define each bin: [edge[i], edge[i+1]).
        bands: Dict mapping band name → (f_low, f_high) in Hz.
            Defaults to ``DEFAULT_BANDS`` (theta, beta, gamma, high_gamma).

    Returns:
        Log band-power features, shape (n_bands * n_channels, n_bins),
        dtype float32.  Bands are stacked channel-first: all bands for
        channel 0, then all bands for channel 1, etc.

    Raises:
        ValueError: If lfp_signal is not 1-D or 2-D, or if bin_edges_s
            has fewer than 2 elements.
    """
    # Default bands
    if bands is None:
        bands = DEFAULT_BANDS

    # Handle 1-D input (single channel)
    if lfp_signal.ndim == 1:
        lfp_signal = lfp_signal[np.newaxis, :]
    elif lfp_signal.ndim != 2:
        raise ValueError(
            f"lfp_signal must be 1-D or 2-D, got {lfp_signal.ndim}-D"
        )

    if len(bin_edges_s) < 2:
        raise ValueError(
            f"bin_edges_s must have >= 2 elements, got {len(bin_edges_s)}"
        )

    n_channels, n_samples = lfp_signal.shape
    n_bins = len(bin_edges_s) - 1
    band_names = list(bands.keys())
    n_bands = len(band_names)

    # Output: (n_bands * n_channels, n_bins)
    features = np.zeros(
        (n_bands * n_channels, n_bins), dtype=np.float32,
    )

    logger.info(
        "Computing LFP band power: %d channels × %d bands × %d bins "
        "(fs=%.0f Hz)",
        n_channels, n_bands, n_bins, sampling_rate,
    )

    for bin_idx in range(n_bins):
        # Convert bin edges (seconds) to sample indices
        t_start = bin_edges_s[bin_idx]
        t_end = bin_edges_s[bin_idx + 1]
        i_start = int(t_start * sampling_rate)
        i_end = int(t_end * sampling_rate)

        # Clamp to valid sample range
        i_start = max(0, min(i_start, n_samples))
        i_end = max(i_start, min(i_end, n_samples))

        for ch_idx in range(n_channels):
            segment = lfp_signal[ch_idx, i_start:i_end]

            for band_idx, band_name in enumerate(band_names):
                power = _bandpower_welch(
                    segment, sampling_rate, bands[band_name],
                )
                # Feature row index: channel-first stacking
                row = ch_idx * n_bands + band_idx
                features[row, bin_idx] = power

    # Log-transform to compress dynamic range
    features = np.log10(features + _LOG_EPS).astype(np.float32)

    logger.info(
        "LFP band power computed: shape %s, range [%.2f, %.2f]",
        features.shape, features.min(), features.max(),
    )

    return features
