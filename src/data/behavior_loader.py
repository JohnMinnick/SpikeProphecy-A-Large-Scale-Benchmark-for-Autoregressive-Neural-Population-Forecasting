"""
Behavioral data extraction from Steinmetz 2019 NWB files.

Extracts and bin-aligns stimulus (visual contrast) and motor (wheel velocity)
variables to match spike count time bins, enabling sensory-motor probing
of neural dynamics models.

Available NWB behavioral fields:
  - intervals/trials/visual_stimulus_left_contrast: per-trial float
  - intervals/trials/visual_stimulus_right_contrast: per-trial float
  - intervals/trials/visual_stimulus_time: per-trial stimulus onset
  - intervals/trials/go_cue: per-trial go cue time
  - intervals/trials/response_choice: per-trial response (-1, 0, 1)
  - intervals/trials/feedback_type: per-trial feedback (-1 or 1)
  - acquisition/wheel_position/data: continuous wheel position
  - acquisition/wheel_position/starting_time: start time (seconds)

KOSMOS Sensory-Motor Probe — Phase 1.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def extract_wheel_velocity(
    nwb_path: str,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Extract wheel velocity from NWB and bin-align to spike count time bins.

    Differentiates wheel position to get velocity, then averages velocity
    within each time bin to produce a (T,) array matching spike bins.

    Args:
        nwb_path: Path to .nwb file.
        bin_edges: Time bin edges in seconds, shape (T+1,).

    Returns:
        wheel_velocity: Binned wheel velocity, shape (T,).
    """
    with h5py.File(nwb_path, "r") as f:
        wheel_data = f["acquisition/wheel_position/data"][:]
        wheel_start = float(f["acquisition/wheel_position/starting_time"][()])

    # Infer sampling rate from data length and session duration
    # Steinmetz data: 500 Hz sampling for wheel
    n_samples = len(wheel_data)
    session_duration = bin_edges[-1] - bin_edges[0]

    # Estimate sampling rate from the data
    # Wheel is typically sampled at 100 Hz or higher
    fs = n_samples / session_duration if session_duration > 0 else 100.0
    logger.info(
        "Wheel: %d samples, estimated fs=%.1f Hz, duration=%.1f s",
        n_samples, fs, session_duration,
    )

    # Create timestamps for wheel samples
    wheel_times = wheel_start + np.arange(n_samples) / fs

    # Differentiate to get velocity (position → velocity)
    wheel_vel = np.diff(wheel_data) * fs  # Scale by sampling rate
    wheel_vel = np.append(wheel_vel, wheel_vel[-1])  # Pad to same length

    # Bin-average velocity into spike count time bins using np.digitize
    # This is O(n_samples) instead of O(n_samples × n_bins)
    n_bins = len(bin_edges) - 1
    binned_velocity = np.zeros(n_bins, dtype=np.float32)

    # Assign each wheel sample to a bin (0-indexed, clipped to valid range)
    bin_indices = np.digitize(wheel_times, bin_edges) - 1  # 0-indexed
    valid = (bin_indices >= 0) & (bin_indices < n_bins)

    # Accumulate velocity sum and count per bin
    if valid.any():
        vel_valid = wheel_vel[valid]
        idx_valid = bin_indices[valid]
        bin_sums = np.bincount(idx_valid, weights=vel_valid, minlength=n_bins)
        bin_counts = np.bincount(idx_valid, minlength=n_bins)
        # Average where count > 0
        nonzero = bin_counts > 0
        binned_velocity[nonzero] = (bin_sums[nonzero] / bin_counts[nonzero]).astype(np.float32)

    logger.info(
        "Wheel velocity: binned to %d bins, range [%.3f, %.3f]",
        n_bins, binned_velocity.min(), binned_velocity.max(),
    )
    return binned_velocity


def extract_trial_stimuli(
    nwb_path: str,
    bin_edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract per-trial stimulus and response variables, expanded to time bins.

    For each time bin, assigns the contrast/response from the trial that
    contains that time point. Bins outside any trial get value 0.

    Args:
        nwb_path: Path to .nwb file.
        bin_edges: Time bin edges in seconds, shape (T+1,).

    Returns:
        Dict with keys:
            - 'left_contrast': (T,) left visual contrast per bin
            - 'right_contrast': (T,) right visual contrast per bin
            - 'response_choice': (T,) response choice per bin (-1, 0, 1)
            - 'feedback_type': (T,) feedback per bin (-1 or 1)
            - 'trial_active': (T,) binary mask — 1 if bin is within a trial
            - 'trial_index': (T,) int32 — trial index for each bin (-1 if
              not in any trial). Used for trial-level holdout splitting.
    """
    with h5py.File(nwb_path, "r") as f:
        trials = f["intervals/trials"]
        stim_time = trials["visual_stimulus_time"][:]
        left_contrast = trials["visual_stimulus_left_contrast"][:]
        right_contrast = trials["visual_stimulus_right_contrast"][:]
        response_choice = trials["response_choice"][:]
        feedback_type = trials["feedback_type"][:]
        trial_start = trials["start_time"][:]
        trial_stop = trials["stop_time"][:]

    n_trials = len(stim_time)
    n_bins = len(bin_edges) - 1
    logger.info("Trials: %d trials, %d time bins", n_trials, n_bins)

    # Bin center times
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Initialize output arrays
    result = {
        "left_contrast": np.zeros(n_bins, dtype=np.float32),
        "right_contrast": np.zeros(n_bins, dtype=np.float32),
        "response_choice": np.zeros(n_bins, dtype=np.float32),
        "feedback_type": np.zeros(n_bins, dtype=np.float32),
        "trial_active": np.zeros(n_bins, dtype=np.float32),
        # Track which trial each bin belongs to (-1 = not in any trial)
        "trial_index": np.full(n_bins, -1, dtype=np.int32),
    }

    # For each trial, fill bins that fall within trial window
    for t_idx in range(n_trials):
        t_start = trial_start[t_idx]
        t_stop = trial_stop[t_idx]

        # Find bins within this trial
        trial_mask = (bin_centers >= t_start) & (bin_centers < t_stop)

        if not trial_mask.any():
            continue

        result["left_contrast"][trial_mask] = left_contrast[t_idx]
        result["right_contrast"][trial_mask] = right_contrast[t_idx]
        result["response_choice"][trial_mask] = response_choice[t_idx]
        result["feedback_type"][trial_mask] = feedback_type[t_idx]
        result["trial_active"][trial_mask] = 1.0
        result["trial_index"][trial_mask] = t_idx

    n_active = int(result["trial_active"].sum())
    logger.info(
        "Trial data: %d/%d bins active (%.1f%%), %d unique trials",
        n_active, n_bins, 100.0 * n_active / max(n_bins, 1), n_trials,
    )
    return result


def extract_all_behavior(
    nwb_path: str,
    bin_edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract all behavioral variables from NWB, aligned to spike bins.

    Convenience function that combines wheel velocity and trial stimuli
    into a single dict.

    Args:
        nwb_path: Path to .nwb file.
        bin_edges: Time bin edges in seconds, shape (T+1,).

    Returns:
        Dict with all behavioral arrays, each shape (T,).
    """
    behavior = {}

    # Wheel velocity (continuous motor output)
    try:
        behavior["wheel_velocity"] = extract_wheel_velocity(nwb_path, bin_edges)
    except Exception as e:
        logger.warning("Could not extract wheel velocity: %s", e)

    # Trial stimuli (event-locked sensory and decision)
    try:
        trial_data = extract_trial_stimuli(nwb_path, bin_edges)
        behavior.update(trial_data)
    except Exception as e:
        logger.warning("Could not extract trial stimuli: %s", e)

    return behavior


def compute_bin_edges(
    n_bins: int,
    bin_width_ms: float = 50.0,
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Compute time bin edges matching spike count binning.

    Args:
        n_bins: Number of time bins (T dimension of spike counts).
        bin_width_ms: Bin width in milliseconds.
        start_time: Session start time in seconds.

    Returns:
        bin_edges: Array of shape (n_bins + 1,) in seconds.
    """
    bin_width_s = bin_width_ms / 1000.0
    return start_time + np.arange(n_bins + 1) * bin_width_s
