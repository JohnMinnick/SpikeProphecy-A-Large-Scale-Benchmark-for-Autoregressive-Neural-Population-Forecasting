"""
Spike-train binning pipeline.

Converts ground-truth spike trains from a SpikeInterface SortingExtractor
into binned spike-count matrices suitable for model training.

The output is the common representation defined in ADR-0003:
    spike_counts: np.ndarray of shape (M, T_total) — binned spike counts
    metadata: dict with source info, bin width, sampling rate, etc.

Usage:
    from src.data.binning import bin_spike_trains, save_binned_data

    counts, metadata = bin_spike_trains(sorting, bin_width_ms=10)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def bin_spike_trains(
    sorting,
    bin_width_ms: float = 10.0,
    recording=None,
    source_type: str = "spikeinterface",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Bin spike trains into a spike-count matrix.

    Converts spike sample indices from a SortingExtractor into a matrix of
    spike counts per unit per time bin.

    Args:
        sorting: SpikeInterface SortingExtractor with ground-truth spike trains.
        bin_width_ms: Temporal bin width in milliseconds.
        recording: Optional SpikeInterface RecordingExtractor. If provided,
            used to determine total duration. Otherwise inferred from spikes.
        source_type: Data source identifier for metadata.

    Returns:
        Tuple of:
            - spike_counts: np.ndarray of shape (M, T_total), dtype int32.
              M = number of units, T_total = number of time bins.
            - metadata: dict with keys:
              source, bin_width_ms, sampling_rate, unit_ids,
              duration_s, num_bins, num_units, total_spikes
    """
    fs = sorting.get_sampling_frequency()
    unit_ids = sorting.get_unit_ids()
    num_units = len(unit_ids)

    # Convert bin width from ms to samples
    bin_width_samples = int(fs * bin_width_ms / 1000.0)

    # Determine total number of samples (and thus bins)
    if recording is not None:
        total_samples = recording.get_num_samples()
    else:
        # Infer from the latest spike across all units
        max_sample = 0
        for uid in unit_ids:
            spikes = sorting.get_unit_spike_train(uid)
            if len(spikes) > 0:
                max_sample = max(max_sample, int(spikes[-1]))
        # Add one bin width beyond the last spike
        total_samples = max_sample + bin_width_samples

    # Compute number of complete bins
    num_bins = total_samples // bin_width_samples
    duration_s = num_bins * bin_width_ms / 1000.0

    logger.info(
        "Binning %d units: bin_width=%.1fms, fs=%.0fHz, "
        "total_samples=%d, num_bins=%d, duration=%.1fs",
        num_units, bin_width_ms, fs, total_samples, num_bins, duration_s,
    )

    # Allocate count matrix
    spike_counts = np.zeros((num_units, num_bins), dtype=np.int32)

    # Bin each unit's spike train
    total_spikes = 0
    spikes_outside = 0  # Track spikes that fall outside the binning range
    for i, uid in enumerate(unit_ids):
        spike_samples = sorting.get_unit_spike_train(uid)

        if len(spike_samples) == 0:
            logger.debug("Unit %s has zero spikes", uid)
            continue

        # Convert sample indices to bin indices
        bin_indices = spike_samples // bin_width_samples

        # Filter to valid range [0, num_bins)
        valid_mask = bin_indices < num_bins
        n_outside = int(np.sum(~valid_mask))
        spikes_outside += n_outside
        bin_indices = bin_indices[valid_mask]

        # Count spikes per bin using np.bincount for efficiency
        if len(bin_indices) > 0:
            counts = np.bincount(bin_indices, minlength=num_bins)
            spike_counts[i, :] = counts[:num_bins]

        total_spikes += len(bin_indices)

    if spikes_outside > 0:
        logger.warning(
            "%d spikes fell outside the binning range and were excluded",
            spikes_outside,
        )

    # Compute summary statistics for validation
    firing_rates = spike_counts.sum(axis=1) / duration_s
    logger.info(
        "Binning complete: total_spikes=%d, mean_rate=%.2f Hz, "
        "min_rate=%.2f Hz, max_rate=%.2f Hz",
        total_spikes, firing_rates.mean(), firing_rates.min(), firing_rates.max(),
    )

    # Build metadata
    metadata = {
        "source": source_type,
        "bin_width_ms": bin_width_ms,
        "sampling_rate": float(fs),
        "unit_ids": [str(uid) for uid in unit_ids],
        "duration_s": duration_s,
        "num_bins": num_bins,
        "num_units": num_units,
        "total_spikes": int(total_spikes),
        "firing_rates_hz": firing_rates.tolist(),
    }

    return spike_counts, metadata


def validate_spike_counts(
    spike_counts: np.ndarray,
    metadata: Dict[str, Any],
    min_rate_hz: float = 0.1,
    max_rate_hz: float = 100.0,
) -> Dict[str, Any]:
    """
    Validate a binned spike-count matrix for data integrity.

    Checks shape, dtype, value ranges, and per-unit firing rates.

    Args:
        spike_counts: Binned count matrix (M, T_total).
        metadata: Metadata dict from bin_spike_trains().
        min_rate_hz: Minimum expected firing rate (warn if below).
        max_rate_hz: Maximum expected firing rate (warn if above).

    Returns:
        Dict with validation results:
            - valid: bool (True if all checks pass)
            - warnings: list of warning strings
            - stats: dict with summary statistics
    """
    warnings = []

    # Check shape consistency
    m, t = spike_counts.shape
    if m != metadata["num_units"]:
        warnings.append(
            f"Unit count mismatch: matrix has {m} rows, "
            f"metadata says {metadata['num_units']}"
        )
    if t != metadata["num_bins"]:
        warnings.append(
            f"Bin count mismatch: matrix has {t} cols, "
            f"metadata says {metadata['num_bins']}"
        )

    # Check dtype
    if spike_counts.dtype != np.int32:
        warnings.append(f"Expected int32 dtype, got {spike_counts.dtype}")

    # Check for negative values (should never happen)
    if np.any(spike_counts < 0):
        warnings.append("Negative spike counts detected!")

    # Check firing rates
    duration_s = metadata["duration_s"]
    firing_rates = spike_counts.sum(axis=1) / duration_s

    # Warn about silent units
    silent_units = np.where(firing_rates < min_rate_hz)[0]
    if len(silent_units) > 0:
        warnings.append(
            f"{len(silent_units)} units have firing rate < {min_rate_hz} Hz: "
            f"indices {silent_units.tolist()}"
        )

    # Warn about suspiciously high firing rates
    fast_units = np.where(firing_rates > max_rate_hz)[0]
    if len(fast_units) > 0:
        warnings.append(
            f"{len(fast_units)} units have firing rate > {max_rate_hz} Hz: "
            f"indices {fast_units.tolist()}"
        )

    # Summary statistics
    stats = {
        "shape": spike_counts.shape,
        "dtype": str(spike_counts.dtype),
        "total_spikes": int(spike_counts.sum()),
        "mean_rate_hz": float(firing_rates.mean()),
        "min_rate_hz": float(firing_rates.min()),
        "max_rate_hz": float(firing_rates.max()),
        "max_count_per_bin": int(spike_counts.max()),
        "fraction_nonzero": float(np.count_nonzero(spike_counts) / spike_counts.size),
    }

    for w in warnings:
        logger.warning("Validation: %s", w)

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "stats": stats,
    }


def save_binned_data(
    spike_counts: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Union[str, Path],
    name: str = "binned",
) -> Path:
    """
    Save binned spike counts and metadata to disk.

    Saves the count matrix as a .npy file and metadata as a .json file.

    Args:
        spike_counts: Binned count matrix (M, T_total).
        metadata: Metadata dict from bin_spike_trains().
        output_dir: Directory to save files.
        name: Base name for the output files.

    Returns:
        Path to the output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save count matrix
    counts_path = out_path / f"{name}_counts.npy"
    np.save(counts_path, spike_counts)
    logger.info("Saved spike counts (%s) to %s", spike_counts.shape, counts_path)

    # Save metadata
    meta_path = out_path / f"{name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    return out_path


def load_binned_data(
    input_dir: Union[str, Path],
    name: str = "binned",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load previously saved binned spike counts and metadata.

    Args:
        input_dir: Directory containing saved files.
        name: Base name used when saving.

    Returns:
        Tuple of (spike_counts, metadata).
    """
    in_path = Path(input_dir)

    counts_path = in_path / f"{name}_counts.npy"
    meta_path = in_path / f"{name}_metadata.json"

    spike_counts = np.load(counts_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        "Loaded binned data: shape=%s, %d spikes, %.1fs duration",
        spike_counts.shape, metadata["total_spikes"], metadata["duration_s"],
    )

    return spike_counts, metadata
