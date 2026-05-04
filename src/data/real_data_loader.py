"""
Real neural data loader for NWB files.

Loads sorted spike trains from NWB (Neurodata Without Borders) files and
wraps them into a MockSorting-compatible object for downstream use with
bin_spike_trains() and the rest of the pipeline.

Designed for datasets like Steinmetz et al. 2019 (Neuropixels), which store
spike times in the NWB ``units`` table. See ADR-0008 for design rationale.

Usage:
    from src.data.real_data_loader import load_nwb_spikes

    sorting, metadata = load_nwb_spikes(config)
    counts, meta = bin_spike_trains(sorting, bin_width_ms=10)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.modulated_generator import MockSorting

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit filtering helpers
# ---------------------------------------------------------------------------

def _get_column_safe(units_table, column_name: str) -> Optional[list]:
    """
    Safely retrieve a column from the NWB units table.

    Returns None if the column does not exist, rather than raising.

    Args:
        units_table: NWB DynamicTable (nwbfile.units).
        column_name: Name of the column to retrieve.

    Returns:
        List of column values, or None if the column is absent.
    """
    if column_name in units_table.colnames:
        return list(units_table[column_name].data[:])
    return None


def filter_units(
    spike_times_list: List[np.ndarray],
    unit_indices: List[int],
    sampling_frequency: float,
    duration_s: float,
    min_firing_rate_hz: float = 0.0,
    max_units: Optional[int] = None,
    quality_labels: Optional[List[str]] = None,
    quality_list: Optional[List[str]] = None,
    brain_region: Optional[str] = None,
    brain_region_list: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """
    Filter units based on configurable quality and rate criteria.

    Applies filters in order:
      1. Quality label (e.g., "good") if quality data is available
      2. Brain region (e.g., "VISp") if region data is available
      3. Minimum firing rate threshold
      4. Maximum unit count (random subsample if exceeded)

    Args:
        spike_times_list: List of spike-time arrays (seconds) per unit.
        unit_indices: Original unit indices (for metadata tracking).
        sampling_frequency: Sampling frequency in Hz.
        duration_s: Total recording duration in seconds.
        min_firing_rate_hz: Minimum firing rate to keep a unit.
        max_units: Maximum number of units to retain (random subset).
        quality_labels: List of quality labels to accept (e.g., ["good"]).
        quality_list: Per-unit quality labels from NWB (same length as
            spike_times_list). None if not available.
        brain_region: Brain region string to filter on (e.g., "VISp").
        brain_region_list: Per-unit brain region labels from NWB. None if
            not available.
        rng: NumPy random generator for reproducible subsampling.

    Returns:
        Tuple of:
            - Filtered spike_times_list
            - Filtered unit_indices
            - filter_stats dict with counts at each filter stage
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_initial = len(spike_times_list)
    filter_stats = {"initial_units": n_initial}

    # --- 1. Filter by quality label ---
    if quality_labels is not None and quality_list is not None:
        keep = [
            i for i, q in enumerate(quality_list)
            if str(q) in quality_labels
        ]
        spike_times_list = [spike_times_list[i] for i in keep]
        unit_indices = [unit_indices[i] for i in keep]
        # Also filter the region list if present
        if brain_region_list is not None:
            brain_region_list = [brain_region_list[i] for i in keep]
        filter_stats["after_quality_filter"] = len(spike_times_list)
        logger.info(
            "Quality filter (%s): %d → %d units",
            quality_labels, n_initial, len(spike_times_list),
        )

    # --- 2. Filter by brain region ---
    if brain_region is not None and brain_region_list is not None:
        keep = [
            i for i, r in enumerate(brain_region_list)
            if brain_region in str(r)
        ]
        spike_times_list = [spike_times_list[i] for i in keep]
        unit_indices = [unit_indices[i] for i in keep]
        filter_stats["after_region_filter"] = len(spike_times_list)
        logger.info(
            "Region filter (%s): → %d units",
            brain_region, len(spike_times_list),
        )

    # --- 3. Filter by minimum firing rate ---
    if min_firing_rate_hz > 0 and duration_s > 0:
        keep = [
            i for i, st in enumerate(spike_times_list)
            if len(st) / duration_s >= min_firing_rate_hz
        ]
        spike_times_list = [spike_times_list[i] for i in keep]
        unit_indices = [unit_indices[i] for i in keep]
        # Keep brain_region_list in sync
        if brain_region_list is not None:
            brain_region_list = [brain_region_list[i] for i in keep]
        filter_stats["after_rate_filter"] = len(spike_times_list)
        logger.info(
            "Min rate filter (>=%.1f Hz): → %d units",
            min_firing_rate_hz, len(spike_times_list),
        )

    # --- 4. Cap to max_units (random subsample) ---
    if max_units is not None and len(spike_times_list) > max_units:
        indices = rng.choice(
            len(spike_times_list), size=max_units, replace=False
        )
        indices = sorted(indices)  # Keep original ordering
        spike_times_list = [spike_times_list[i] for i in indices]
        unit_indices = [unit_indices[i] for i in indices]
        # Keep brain_region_list in sync
        if brain_region_list is not None:
            brain_region_list = [brain_region_list[i] for i in indices]
        filter_stats["after_max_units_cap"] = len(spike_times_list)
        logger.info(
            "Max units cap (%d): → %d units",
            max_units, len(spike_times_list),
        )

    filter_stats["final_units"] = len(spike_times_list)
    # Include filtered brain_region_list in stats for downstream use
    filter_stats["brain_region_list"] = brain_region_list
    return spike_times_list, unit_indices, filter_stats


def _spike_times_to_samples(
    spike_times_s: np.ndarray,
    sampling_frequency: float,
    max_sample: int,
) -> np.ndarray:
    """
    Convert spike times in seconds to sample indices.

    Args:
        spike_times_s: Array of spike times in seconds.
        sampling_frequency: Sampling rate in Hz.
        max_sample: Maximum valid sample index (exclusive).

    Returns:
        Sorted, unique int64 array of sample indices within [0, max_sample).
    """
    # Convert to sample indices
    samples = (spike_times_s * sampling_frequency).astype(np.int64)

    # Remove duplicates from rounding and ensure sorted
    samples = np.unique(samples)

    # Clip to valid range [0, max_sample)
    samples = samples[(samples >= 0) & (samples < max_sample)]

    return samples


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_nwb_spikes(
    config: Dict[str, Any],
) -> Tuple[MockSorting, Dict[str, Any]]:
    """
    Load sorted spike trains from an NWB file.

    Reads the ``units`` table from an NWB file, filters units by quality
    criteria, converts spike times to sample indices, and wraps the result
    in a MockSorting object compatible with ``bin_spike_trains()``.

    Args:
        config: Configuration dictionary. Expected keys:
            - source.path (str): Path to the NWB file.
            - seed (int): Random seed for reproducibility.
            - nwb.sampling_frequency (float): Target sampling frequency (Hz).
                Default 30000.0.
            - nwb.min_firing_rate_hz (float): Minimum unit firing rate.
                Default 1.0.
            - nwb.max_units (int|null): Maximum number of units. Default null.
            - nwb.quality_labels (list|null): Accepted quality labels.
                Default ["good"].
            - nwb.brain_region (str|null): Brain region filter. Default null.
            - nwb.duration_limit_s (float|null): If set, truncate to this
                duration. Default null.

    Returns:
        Tuple of:
            - MockSorting: SortingExtractor-like object with spike trains.
            - metadata: Dict with loading parameters and summary statistics.

    Raises:
        FileNotFoundError: If the NWB file does not exist.
        ImportError: If pynwb is not installed.
        ValueError: If no units remain after filtering.
    """
    # --- Lazy import of pynwb (optional dependency) ---
    try:
        from pynwb import NWBHDF5IO  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "pynwb is required for NWB data loading. "
            "Install it with: pip install pynwb\n"
            "Or: pip install spike-prophecy[nwb]"
        ) from e

    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # --- Extract NWB-specific config ---
    source_config = config.get("source", {})
    nwb_path = Path(source_config.get("path", ""))

    if not nwb_path.exists():
        raise FileNotFoundError(
            f"NWB file not found: {nwb_path}\n"
            f"Download from Figshare and place in data/raw/."
        )

    nwb_config = config.get("nwb", {})
    sampling_frequency = nwb_config.get("sampling_frequency", 30000.0)
    min_firing_rate_hz = nwb_config.get("min_firing_rate_hz", 1.0)
    max_units = nwb_config.get("max_units", None)
    quality_labels = nwb_config.get("quality_labels", ["good"])
    brain_region = nwb_config.get("brain_region", None)
    duration_limit_s = nwb_config.get("duration_limit_s", None)

    logger.info("Loading NWB file: %s", nwb_path)

    # --- Read NWB file ---
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        # Check that units table exists
        if nwbfile.units is None or len(nwbfile.units) == 0:
            raise ValueError(
                f"NWB file has no units table or it is empty: {nwb_path}"
            )

        num_raw_units = len(nwbfile.units)
        logger.info("NWB file contains %d units", num_raw_units)

        # --- Extract spike times for each unit ---
        spike_times_list = []
        unit_indices = list(range(num_raw_units))

        for i in range(num_raw_units):
            # NWB units table spike_times is a ragged array
            st = np.array(nwbfile.units["spike_times"][i], dtype=np.float64)
            spike_times_list.append(st)

        # --- Determine recording duration ---
        # Use the maximum spike time across all units
        all_max_times = [
            st[-1] for st in spike_times_list if len(st) > 0
        ]
        recording_duration_s = max(all_max_times) if all_max_times else 0.0

        if duration_limit_s is not None and duration_limit_s < recording_duration_s:
            # Truncate spike trains to duration limit
            logger.info(
                "Truncating from %.1fs to %.1fs",
                recording_duration_s, duration_limit_s,
            )
            recording_duration_s = duration_limit_s
            spike_times_list = [
                st[st <= duration_limit_s] for st in spike_times_list
            ]

        # --- Extract quality and region labels (if available) ---
        quality_list = _get_column_safe(nwbfile.units, "quality")
        brain_region_list = _get_column_safe(
            nwbfile.units, "location"
        ) or _get_column_safe(nwbfile.units, "brain_region")

        # Fallback: if units table lacks region, derive from peak_channel
        # → electrodes table location (common in Steinmetz NWB files)
        if brain_region_list is None:
            peak_channels = _get_column_safe(nwbfile.units, "peak_channel")
            if peak_channels is not None:
                try:
                    elec_locs = list(
                        nwbfile.electrodes["location"].data[:]
                    )
                    brain_region_list = []
                    for pc in peak_channels:
                        # peak_channel is (1,) array or scalar
                        ch_idx = int(pc[0]) if hasattr(pc, '__len__') else int(pc)
                        if 0 <= ch_idx < len(elec_locs):
                            loc = elec_locs[ch_idx]
                            # Decode bytes if needed
                            if isinstance(loc, bytes):
                                loc = loc.decode("utf-8")
                            brain_region_list.append(str(loc))
                        else:
                            brain_region_list.append("unknown")
                    logger.info(
                        "Brain regions from electrodes: %d units → %d regions",
                        len(brain_region_list),
                        len(set(brain_region_list)),
                    )
                except Exception as e:
                    logger.warning(
                        "Could not extract regions from electrodes: %s", e,
                    )
                    brain_region_list = None

    # --- Filter units ---
    spike_times_list, unit_indices, filter_stats = filter_units(
        spike_times_list=spike_times_list,
        unit_indices=unit_indices,
        sampling_frequency=sampling_frequency,
        duration_s=recording_duration_s,
        min_firing_rate_hz=min_firing_rate_hz,
        max_units=max_units,
        quality_labels=quality_labels if quality_list is not None else None,
        quality_list=quality_list,
        brain_region=brain_region,
        brain_region_list=brain_region_list,
        rng=rng,
    )

    if len(spike_times_list) == 0:
        raise ValueError(
            "No units remain after filtering. "
            "Try relaxing quality_labels, min_firing_rate_hz, or brain_region."
        )

    # --- Convert spike times to sample indices and build MockSorting ---
    max_sample = int(recording_duration_s * sampling_frequency)
    spike_trains = {}
    unit_rates_info = []

    for new_id, (st, orig_idx) in enumerate(
        zip(spike_times_list, unit_indices)
    ):
        samples = _spike_times_to_samples(st, sampling_frequency, max_sample)
        spike_trains[new_id] = samples

        actual_rate = len(samples) / recording_duration_s if recording_duration_s > 0 else 0.0
        unit_rates_info.append({
            "unit_id": new_id,
            "original_nwb_index": orig_idx,
            "actual_rate_hz": float(actual_rate),
            "num_spikes": len(samples),
        })

    sorting = MockSorting(spike_trains, sampling_frequency)

    # --- Summary statistics ---
    total_spikes = sum(info["num_spikes"] for info in unit_rates_info)
    mean_rate = np.mean([info["actual_rate_hz"] for info in unit_rates_info])
    num_units = len(spike_trains)

    logger.info(
        "Loaded %d units, %d total spikes, mean rate=%.2f Hz, duration=%.1fs",
        num_units, total_spikes, mean_rate, recording_duration_s,
    )

    # --- Build metadata ---
    # Extract post-filter brain region labels (may be None if NWB lacks them)
    brain_regions = filter_stats.get("brain_region_list", None)

    metadata = {
        "source": "nwb",
        "nwb_path": str(nwb_path),
        "seed": seed,
        "num_units": num_units,
        "num_raw_units": num_raw_units,
        "duration_s": recording_duration_s,
        "sampling_frequency": sampling_frequency,
        "total_spikes": total_spikes,
        "mean_rate_hz": float(mean_rate),
        "filter_stats": filter_stats,
        "unit_details": unit_rates_info,
        "brain_regions": brain_regions,  # Per-unit region labels (post-filter)
    }

    return sorting, metadata
