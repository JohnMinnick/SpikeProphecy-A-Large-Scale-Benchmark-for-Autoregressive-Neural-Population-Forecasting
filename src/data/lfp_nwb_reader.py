"""
NWB-specific LFP reader for Steinmetz 2019 Neuropixels data.

Thin adapter that loads continuous LFP voltage from NWB files and
returns a format-agnostic ``(lfp_signal, sampling_rate)`` tuple for
use with ``lfp_features.compute_lfp_band_power()``.

This is the ONLY module that imports ``pynwb`` for LFP access.
Future hardware adapters (e.g., Maxwell MEA) will provide the same
interface without touching this file.

Design decisions documented in ADR-0015 (Tier A+).

Usage:
    from src.data.lfp_nwb_reader import load_lfp_from_nwb

    result = load_lfp_from_nwb("path/to/session.nwb")
    if result is not None:
        lfp_signal, sampling_rate = result
        # lfp_signal: (n_channels, n_samples) float array
        # sampling_rate: float (Hz)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_lfp_from_nwb(
    nwb_path: str | Path,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load LFP data from a Steinmetz 2019 NWB file.

    Opens the NWB file, reads the LFP ``ElectricalSeries`` from the
    ``ecephys`` processing module, extracts the voltage data and
    sampling rate, and closes the file handle promptly.

    Args:
        nwb_path: Path to the NWB file.

    Returns:
        Tuple of ``(lfp_signal, sampling_rate)`` where:
            - lfp_signal: shape (n_channels, n_samples), float32
            - sampling_rate: float, in Hz
        Returns ``None`` if the NWB file has no LFP data in its
        ``processing["ecephys"]`` module.

    Raises:
        ImportError: If ``pynwb`` is not installed.
        FileNotFoundError: If ``nwb_path`` does not exist.
    """
    nwb_path = Path(nwb_path)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    try:
        from pynwb import NWBHDF5IO
    except ImportError as e:
        raise ImportError(
            "pynwb is required for NWB LFP loading. "
            "Install it with: pip install pynwb"
        ) from e

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        # Check for ecephys processing module
        if "ecephys" not in nwbfile.processing:
            logger.warning(
                "No 'ecephys' processing module in %s — skipping LFP",
                nwb_path.name,
            )
            return None

        ecephys = nwbfile.processing["ecephys"]

        # Look for LFP container, then find the first ElectricalSeries
        lfp_container = ecephys.data_interfaces.get("LFP", None)
        if lfp_container is None:
            logger.warning(
                "No 'LFP' interface in ecephys for %s — skipping",
                nwb_path.name,
            )
            return None

        # Get the first (and typically only) electrical series in LFP
        series_names = list(lfp_container.electrical_series.keys())
        if not series_names:
            logger.warning(
                "LFP container is empty in %s — skipping",
                nwb_path.name,
            )
            return None

        lfp_series = lfp_container.electrical_series[series_names[0]]

        # Extract data: typically (n_samples, n_channels) in NWB
        lfp_data = lfp_series.data[:]
        sampling_rate = float(lfp_series.rate)

        # Ensure float32 and transpose to (n_channels, n_samples)
        if lfp_data.ndim == 1:
            # Single channel
            lfp_signal = lfp_data[np.newaxis, :].astype(np.float32)
        elif lfp_data.ndim == 2:
            # (n_samples, n_channels) → (n_channels, n_samples)
            lfp_signal = lfp_data.T.astype(np.float32)
        else:
            logger.warning(
                "Unexpected LFP shape %s in %s — skipping",
                lfp_data.shape, nwb_path.name,
            )
            return None

    logger.info(
        "Loaded LFP from %s: %d channels × %d samples (%.0f Hz)",
        nwb_path.name,
        lfp_signal.shape[0],
        lfp_signal.shape[1],
        sampling_rate,
    )

    return lfp_signal, sampling_rate
