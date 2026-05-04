"""
NWB covariate extraction for Tier 1 stimulus features.

Extracts trial-level stimulus and context signals from NWB files and
aligns them to spike-count bins.  These covariates are used as auxiliary
inputs to the teacher model via an additive projection layer (Option B,
see ADR-0012 / ADR-0013).

Tier 1 features (stimulus timing + trial context):
    - stim_on: binary, 1 during stimulus presentation
    - contrast_left: left visual stimulus contrast (0–1)
    - contrast_right: right visual stimulus contrast (0–1)
    - trial_phase: normalized position within current trial (0→1)
    - inter_trial: binary, 1 between trials

Usage:
    from src.data.nwb_covariates import extract_stimulus_features

    covariates, names = extract_stimulus_features(
        nwb_path="data/raw/Steinmetz2019_Forssmann_2017-11-01.nwb",
        num_bins=50000,
        bin_width_ms=10.0,
    )
    # covariates: (n_covariates, T), float32
    # names: ["stim_on", "contrast_left", ...]
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# All available Tier 1 features
TIER1_FEATURES = [
    "stim_on",
    "contrast_left",
    "contrast_right",
    "trial_phase",
    "inter_trial",
]


def _get_trial_column_safe(
    trials_df: "pd.DataFrame",  # noqa: F821 — lazy import
    column_name: str,
    default: float = 0.0,
) -> np.ndarray:
    """
    Safely retrieve a column from the trials DataFrame.

    Returns a numpy array of the column values if the column exists,
    otherwise returns an array filled with the default value.  This
    handles NWB file variation across sessions where some columns may
    be absent.

    Args:
        trials_df: Pandas DataFrame from nwbfile.trials.to_dataframe().
        column_name: Name of the column to retrieve.
        default: Value to fill when the column is missing.

    Returns:
        Numpy array of shape (n_trials,) with column values or defaults.
    """
    if column_name in trials_df.columns:
        values = trials_df[column_name].values.astype(np.float64)
        # Replace NaN with default
        values = np.where(np.isnan(values), default, values)
        return values
    else:
        logger.debug(
            "Trials column '%s' not found, filling with %.2f",
            column_name, default,
        )
        return np.full(len(trials_df), default, dtype=np.float64)


def extract_stimulus_features(
    nwb_path: Union[str, Path],
    num_bins: int,
    bin_width_ms: float = 10.0,
    feature_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract Tier 1 stimulus features from an NWB file aligned to bins.

    Opens the NWB file, reads the trials table, and produces a
    covariate matrix where each row is a feature and each column is
    a time bin.  Features are aligned by determining which trial (if
    any) each bin's center falls within.

    Args:
        nwb_path: Path to the NWB file.
        num_bins: Number of time bins in the spike-count matrix.
        bin_width_ms: Width of each bin in milliseconds.
        feature_list: Which features to extract.  Default: all Tier 1.
            Must be a subset of TIER1_FEATURES.

    Returns:
        Tuple of:
            - covariates: Shape (n_covariates, num_bins), float32.
            - feature_names: List of feature name strings.

    Raises:
        ImportError: If pynwb is not installed.
        FileNotFoundError: If the NWB file does not exist.
    """
    # Validate feature list
    if feature_list is None:
        feature_list = list(TIER1_FEATURES)
    else:
        for f in feature_list:
            if f not in TIER1_FEATURES:
                raise ValueError(
                    f"Unknown feature '{f}'. Must be one of {TIER1_FEATURES}"
                )

    nwb_path = Path(nwb_path)
    n_features = len(feature_list)

    # Compute bin centers in seconds
    bin_width_s = bin_width_ms / 1000.0
    bin_centers_s = (np.arange(num_bins) + 0.5) * bin_width_s

    # Allocate output (all zeros by default = safe fallback)
    covariates = np.zeros((n_features, num_bins), dtype=np.float32)

    # Extract trial data from NWB
    trials_df = _load_trials_dataframe(nwb_path)

    if trials_df is None or len(trials_df) == 0:
        logger.warning(
            "No trials table in NWB file %s — returning all-zero covariates",
            nwb_path.name,
        )
        return covariates, feature_list

    # Extract trial timing columns (required for alignment)
    start_times = _get_trial_column_safe(trials_df, "start_time", 0.0)
    stop_times = _get_trial_column_safe(trials_df, "stop_time", 0.0)

    # Validate trial times
    valid_trials = (stop_times > start_times) & np.isfinite(start_times)
    if not valid_trials.any():
        logger.warning(
            "All trials have invalid timing — returning all-zero covariates"
        )
        return covariates, feature_list

    # Pre-extract optional columns
    contrast_left = _get_trial_column_safe(
        trials_df, "visual_stimulus_left_contrast", 0.0,
    )
    contrast_right = _get_trial_column_safe(
        trials_df, "visual_stimulus_right_contrast", 0.0,
    )

    # Assign each bin to a trial (or mark as inter-trial)
    # Use searchsorted on start_times for efficient trial lookup
    trial_assignment = _assign_bins_to_trials(
        bin_centers_s, start_times, stop_times,
    )

    # Build each feature
    for feat_idx, feat_name in enumerate(feature_list):
        if feat_name == "stim_on":
            # 1 if bin center falls within any trial, 0 otherwise
            covariates[feat_idx] = (trial_assignment >= 0).astype(np.float32)

        elif feat_name == "contrast_left":
            # Contrast of the trial's left stimulus (0 if inter-trial)
            for t in range(num_bins):
                trial_idx = trial_assignment[t]
                if trial_idx >= 0:
                    covariates[feat_idx, t] = contrast_left[trial_idx]

        elif feat_name == "contrast_right":
            # Contrast of the trial's right stimulus (0 if inter-trial)
            for t in range(num_bins):
                trial_idx = trial_assignment[t]
                if trial_idx >= 0:
                    covariates[feat_idx, t] = contrast_right[trial_idx]

        elif feat_name == "trial_phase":
            # Normalized position within current trial: 0 at start, 1 at end
            for t in range(num_bins):
                trial_idx = trial_assignment[t]
                if trial_idx >= 0:
                    duration = stop_times[trial_idx] - start_times[trial_idx]
                    if duration > 0:
                        elapsed = bin_centers_s[t] - start_times[trial_idx]
                        covariates[feat_idx, t] = np.clip(
                            elapsed / duration, 0.0, 1.0,
                        )

        elif feat_name == "inter_trial":
            # 1 if bin center is between trials, 0 during a trial
            covariates[feat_idx] = (trial_assignment < 0).astype(np.float32)

    n_stim_bins = int((trial_assignment >= 0).sum())
    logger.info(
        "Extracted %d stimulus features from %s: "
        "%d/%d bins during trials (%.1f%%)",
        n_features, nwb_path.name,
        n_stim_bins, num_bins,
        100.0 * n_stim_bins / max(num_bins, 1),
    )

    return covariates, feature_list


def _load_trials_dataframe(
    nwb_path: Path,
) -> Optional["pd.DataFrame"]:  # noqa: F821
    """
    Load the trials table from an NWB file as a Pandas DataFrame.

    Opens the NWB file, reads the trials table, converts it to a
    DataFrame, and closes the file handle before returning.  This
    ensures the HDF5 file is released promptly.

    Args:
        nwb_path: Path to the NWB file.

    Returns:
        Pandas DataFrame of trials, or None if no trials table exists.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as e:
        raise ImportError(
            "pynwb is required for NWB covariate extraction. "
            "Install it with: pip install pynwb"
        ) from e

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        if nwbfile.trials is None or len(nwbfile.trials) == 0:
            return None

        # Convert to DataFrame while the IO context is still open
        trials_df = nwbfile.trials.to_dataframe()

    return trials_df


def _assign_bins_to_trials(
    bin_centers_s: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
) -> np.ndarray:
    """
    Assign each bin to the trial it falls within, or -1 for inter-trial.

    Uses binary search on sorted start_times for O(T log N) efficiency
    where T = number of bins and N = number of trials.

    Args:
        bin_centers_s: Center times of each bin in seconds, shape (T,).
        start_times: Trial start times in seconds, shape (N,).
        stop_times: Trial stop times in seconds, shape (N,).

    Returns:
        Array of shape (T,) with trial indices (0-based) or -1 for
        bins that fall between trials.
    """
    T = len(bin_centers_s)
    assignment = np.full(T, -1, dtype=np.int32)

    # Sort trials by start time (they should already be sorted, but be safe)
    sorted_idx = np.argsort(start_times)
    sorted_starts = start_times[sorted_idx]
    sorted_stops = stop_times[sorted_idx]

    # For each bin center, find the candidate trial via searchsorted
    # np.searchsorted(sorted_starts, center, side='right') - 1 gives
    # the index of the last trial that started at or before center
    candidate_idx = np.searchsorted(sorted_starts, bin_centers_s, side="right") - 1

    # Vectorized check: is the bin center within the candidate trial?
    valid = candidate_idx >= 0
    # Check that center < stop_time of the candidate trial
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        candidates = candidate_idx[valid_idx]
        within_trial = bin_centers_s[valid_idx] < sorted_stops[candidates]
        # Map back to original (unsorted) trial indices
        original_idx = sorted_idx[candidates[within_trial]]
        assignment[valid_idx[within_trial]] = original_idx

    return assignment
