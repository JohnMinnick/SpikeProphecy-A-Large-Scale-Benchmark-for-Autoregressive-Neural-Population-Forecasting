"""
Behavioral data extraction from IBL sessions via ONE API.

Extracts and bin-aligns stimulus (visual contrast) and motor (choice, wheel)
variables from IBL Repeated Site recordings to match spike count time bins.
This enables multi-head behavioral decoding from the shared SNN backbone.

IBL behavioral fields (via one.load_object(eid, 'trials')):
    - contrastLeft: per-trial left visual contrast (0, 0.0625, 0.125, 0.25, 1.0)
    - contrastRight: per-trial right visual contrast (same scale)
    - choice: per-trial response (-1 left, +1 right; no no-go in IBL)
    - feedbackType: per-trial feedback (-1 incorrect, +1 correct)
    - stimOn_times: per-trial stimulus onset (seconds)
    - response_times: per-trial response time (seconds)
    - feedback_times: per-trial feedback time (seconds)
    - goCue_times: per-trial go cue time (seconds)

IBL wheel data (via one.load_object(eid, 'wheel')):
    - position: continuous wheel position array
    - timestamps: corresponding timestamps

Output format matches behavior_loader.py for Steinmetz compatibility:
    - left_contrast, right_contrast, response_choice, feedback_type,
      trial_active, trial_index — all shape (T,) aligned to spike bins.

Note: IBL uses binary choice {-1, +1} (no no-go). The 3-class response
head still works — class index 1 (no-go) simply never appears in IBL data.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_ibl_one_client(cache_dir: Optional[str] = None):
    """
    Create and return an IBL ONE API client.

    Reuses the same pattern as ibl_data_loader._get_one_client to avoid
    duplicate client creation logic.

    Args:
        cache_dir: Local cache directory for downloaded data.

    Returns:
        ONE client instance.
    """
    try:
        from one.api import ONE
    except ImportError as e:
        raise ImportError(
            "ONE-api is required for IBL behavior loading. "
            "Install it with: pip install ONE-api"
        ) from e

    if cache_dir is None:
        cache_dir = "data/raw/ibl"

    from pathlib import Path
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        cache_dir=cache_dir,
        silent=True,
        password="international",
    )
    return one


def extract_ibl_trial_stimuli(
    eid: str,
    bin_edges: np.ndarray,
    cache_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract per-trial stimulus and response variables from IBL, expanded to time bins.

    For each time bin, assigns the contrast/response from the trial that
    contains that time point. Bins outside any trial get value 0.
    Output format is identical to behavior_loader.extract_trial_stimuli()
    for Steinmetz compatibility.

    Args:
        eid: IBL session Experiment ID (UUID string).
        bin_edges: Time bin edges in seconds, shape (T+1,).
        cache_dir: Local ONE API cache directory.

    Returns:
        Dict with keys:
            - 'left_contrast': (T,) left visual contrast per bin
            - 'right_contrast': (T,) right visual contrast per bin
            - 'response_choice': (T,) response choice per bin (-1, 0, +1)
            - 'feedback_type': (T,) feedback per bin (-1 or +1)
            - 'trial_active': (T,) binary mask — 1 if bin is within a trial
            - 'trial_index': (T,) int32 — trial index for each bin (-1 if
              not in any trial). Used for trial-level holdout splitting.
    """
    one = _get_ibl_one_client(cache_dir)

    # ------------------------------------------------------------------
    # Load trials object from IBL ONE API
    # ------------------------------------------------------------------
    logger.info("Loading IBL trials for session: %s", eid)
    trials = one.load_object(eid, "trials")

    # Extract trial fields — handle potential NaN values
    contrast_left = np.array(trials.contrastLeft, dtype=np.float64)
    contrast_right = np.array(trials.contrastRight, dtype=np.float64)
    choice = np.array(trials.choice, dtype=np.float64)
    feedback = np.array(trials.feedbackType, dtype=np.float64)

    # Replace NaN contrasts with 0.0 (IBL uses NaN for absent stimuli)
    contrast_left = np.nan_to_num(contrast_left, nan=0.0)
    contrast_right = np.nan_to_num(contrast_right, nan=0.0)

    # Trial timing — use stimOn_times as trial start, feedback_times as trial end
    # This captures the behaviorally relevant window for each trial.
    stim_on = np.array(trials.stimOn_times, dtype=np.float64)
    feedback_times = np.array(trials.feedback_times, dtype=np.float64)

    # Some trials may have NaN timing — filter those out
    valid_trials = (
        ~np.isnan(stim_on) & ~np.isnan(feedback_times)
        & ~np.isnan(choice) & ~np.isnan(feedback)
    )

    n_trials_total = len(stim_on)
    n_valid = int(valid_trials.sum())
    logger.info(
        "IBL trials: %d total, %d valid (%.1f%%)",
        n_trials_total, n_valid, 100.0 * n_valid / max(n_trials_total, 1),
    )

    # ------------------------------------------------------------------
    # Expand trial data to time bins (same logic as behavior_loader)
    # ------------------------------------------------------------------
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    result = {
        "left_contrast": np.zeros(n_bins, dtype=np.float32),
        "right_contrast": np.zeros(n_bins, dtype=np.float32),
        "response_choice": np.zeros(n_bins, dtype=np.float32),
        "feedback_type": np.zeros(n_bins, dtype=np.float32),
        "trial_active": np.zeros(n_bins, dtype=np.float32),
        "trial_index": np.full(n_bins, -1, dtype=np.int32),
    }

    # For each valid trial, fill bins within the trial window
    for t_idx in range(n_trials_total):
        if not valid_trials[t_idx]:
            continue

        t_start = stim_on[t_idx]
        # Add a small buffer after feedback to capture post-feedback activity
        t_stop = feedback_times[t_idx] + 0.5  # 500ms post-feedback buffer

        # Find bins within this trial window
        trial_mask = (bin_centers >= t_start) & (bin_centers < t_stop)

        if not trial_mask.any():
            continue

        result["left_contrast"][trial_mask] = float(contrast_left[t_idx])
        result["right_contrast"][trial_mask] = float(contrast_right[t_idx])
        # IBL choice: -1 (left), +1 (right) — matches Steinmetz encoding
        # No-go (0) never appears in IBL, but the 3-class head handles it
        result["response_choice"][trial_mask] = float(choice[t_idx])
        result["feedback_type"][trial_mask] = float(feedback[t_idx])
        result["trial_active"][trial_mask] = 1.0
        result["trial_index"][trial_mask] = t_idx

    n_active = int(result["trial_active"].sum())
    logger.info(
        "IBL trial data: %d/%d bins active (%.1f%%), %d valid trials",
        n_active, n_bins, 100.0 * n_active / max(n_bins, 1), n_valid,
    )

    return result


def extract_ibl_wheel_velocity(
    eid: str,
    bin_edges: np.ndarray,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Extract wheel velocity from IBL session and bin-align to spike count time bins.

    Differentiates wheel position to get velocity, then averages velocity
    within each time bin to produce a (T,) array matching spike bins.

    Args:
        eid: IBL session Experiment ID (UUID string).
        bin_edges: Time bin edges in seconds, shape (T+1,).
        cache_dir: Local ONE API cache directory.

    Returns:
        wheel_velocity: Binned wheel velocity, shape (T,).
    """
    one = _get_ibl_one_client(cache_dir)

    # Load wheel data from IBL
    logger.info("Loading IBL wheel data for session: %s", eid)
    wheel = one.load_object(eid, "wheel")

    wheel_pos = np.array(wheel.position, dtype=np.float64)
    wheel_times = np.array(wheel.timestamps, dtype=np.float64)

    # Compute velocity via differentiation
    dt = np.diff(wheel_times)
    dp = np.diff(wheel_pos)

    # Avoid division by zero for duplicate timestamps
    dt[dt == 0] = 1e-6
    wheel_vel = dp / dt
    # Pad to same length as position
    wheel_vel = np.append(wheel_vel, wheel_vel[-1] if len(wheel_vel) > 0 else 0.0)

    logger.info(
        "IBL wheel: %d samples, duration=%.1fs, vel range [%.3f, %.3f]",
        len(wheel_pos), wheel_times[-1] - wheel_times[0],
        wheel_vel.min(), wheel_vel.max(),
    )

    # Bin-average velocity into spike count time bins using np.digitize
    n_bins = len(bin_edges) - 1
    binned_velocity = np.zeros(n_bins, dtype=np.float32)

    bin_indices = np.digitize(wheel_times, bin_edges) - 1  # 0-indexed
    valid = (bin_indices >= 0) & (bin_indices < n_bins)

    if valid.any():
        vel_valid = wheel_vel[valid]
        idx_valid = bin_indices[valid]
        bin_sums = np.bincount(idx_valid, weights=vel_valid, minlength=n_bins)
        bin_counts = np.bincount(idx_valid, minlength=n_bins)
        nonzero = bin_counts > 0
        binned_velocity[nonzero] = (
            bin_sums[nonzero] / bin_counts[nonzero]
        ).astype(np.float32)

    logger.info(
        "IBL wheel velocity: binned to %d bins, range [%.3f, %.3f]",
        n_bins, binned_velocity.min(), binned_velocity.max(),
    )
    return binned_velocity


def extract_ibl_all_behavior(
    eid: str,
    bin_edges: np.ndarray,
    cache_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract all behavioral variables from an IBL session, aligned to spike bins.

    Convenience function combining trial stimuli and wheel velocity.
    Output format matches behavior_loader.extract_all_behavior() for
    Steinmetz compatibility.

    Args:
        eid: IBL session Experiment ID (UUID string).
        bin_edges: Time bin edges in seconds, shape (T+1,).
        cache_dir: Local ONE API cache directory.

    Returns:
        Dict with all behavioral arrays, each shape (T,).
    """
    behavior = {}

    # Trial stimuli (sensory, decision, reward variables)
    try:
        trial_data = extract_ibl_trial_stimuli(eid, bin_edges, cache_dir)
        behavior.update(trial_data)
    except Exception as e:
        logger.warning("Could not extract IBL trial stimuli for %s: %s", eid, e)

    # Wheel velocity (continuous motor output)
    try:
        behavior["wheel_velocity"] = extract_ibl_wheel_velocity(
            eid, bin_edges, cache_dir,
        )
    except Exception as e:
        logger.warning("Could not extract IBL wheel velocity for %s: %s", eid, e)

    return behavior


# ---------------------------------------------------------------------------
# Local-only path (no ONE-API): read trials directly from a cached parquet.
# Used by the NeurIPS behavioral eval pipeline when ONE is not available
# in-process (e.g. running inside the Windows venv without onlinealyx creds).
# ---------------------------------------------------------------------------
def _find_trials_pqt(eid: str, ibl_root: str) -> Optional[str]:
    """Locate the trials.table.pqt for an IBL eid by joining sessions.pqt
    metadata with the on-disk ALF tree.

    Returns the absolute path to the most-recent revision of the trials
    table, or None if no parquet is found for that eid.
    """
    from pathlib import Path
    import pandas as pd
    sessions_pqt = Path(ibl_root) / "2022_Q2_IBL_et_al_RepeatedSite" / "sessions.pqt"
    if not sessions_pqt.exists():
        return None
    df = pd.read_parquet(sessions_pqt)
    if eid not in df.index:
        return None
    row = df.loc[eid]
    lab, subject, date, number = (
        str(row["lab"]), str(row["subject"]), str(row["date"]), int(row["number"])
    )
    sess_dir = Path(ibl_root) / lab / "Subjects" / subject / date / f"{number:03d}" / "alf"
    if not sess_dir.exists():
        return None
    # Prefer the latest revision dir (#YYYY-MM-DD#) if present, else flat alf/.
    candidates = list(sess_dir.glob("_ibl_trials.table.pqt"))
    rev_dirs = sorted(
        [d for d in sess_dir.iterdir() if d.is_dir() and d.name.startswith("#")],
        reverse=True,
    )
    for rd in rev_dirs:
        candidates += list(rd.glob("_ibl_trials.table.pqt"))
    return str(candidates[0]) if candidates else None


def extract_ibl_trial_stimuli_local(
    eid: str,
    bin_edges: np.ndarray,
    ibl_root: str = "data/raw/ibl",
) -> Dict[str, np.ndarray]:
    """Local-only counterpart to ``extract_ibl_trial_stimuli``.

    Reads the cached ``_ibl_trials.table.pqt`` for the given eid directly
    instead of going through the ONE API.  Output schema matches
    ``extract_ibl_trial_stimuli`` so downstream code can dispatch by
    session source without further changes.
    """
    import pandas as pd
    pqt_path = _find_trials_pqt(eid, ibl_root)
    if pqt_path is None:
        raise FileNotFoundError(
            f"No _ibl_trials.table.pqt for eid={eid} under {ibl_root}"
        )
    df = pd.read_parquet(pqt_path)

    contrast_left = np.nan_to_num(
        np.array(df["contrastLeft"], dtype=np.float64), nan=0.0,
    )
    contrast_right = np.nan_to_num(
        np.array(df["contrastRight"], dtype=np.float64), nan=0.0,
    )
    choice = np.array(df["choice"], dtype=np.float64)
    feedback = np.array(df["feedbackType"], dtype=np.float64)
    stim_on = np.array(df["stimOn_times"], dtype=np.float64)
    feedback_times = np.array(df["feedback_times"], dtype=np.float64)

    valid_trials = (
        ~np.isnan(stim_on) & ~np.isnan(feedback_times)
        & ~np.isnan(choice) & ~np.isnan(feedback)
    )

    n_trials_total = len(stim_on)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    result = {
        "left_contrast": np.zeros(n_bins, dtype=np.float32),
        "right_contrast": np.zeros(n_bins, dtype=np.float32),
        "response_choice": np.zeros(n_bins, dtype=np.float32),
        "feedback_type": np.zeros(n_bins, dtype=np.float32),
        "trial_active": np.zeros(n_bins, dtype=np.float32),
        "trial_index": np.full(n_bins, -1, dtype=np.int32),
    }

    for t_idx in range(n_trials_total):
        if not valid_trials[t_idx]:
            continue
        t_start = stim_on[t_idx]
        t_stop = feedback_times[t_idx] + 0.5
        trial_mask = (bin_centers >= t_start) & (bin_centers < t_stop)
        if not trial_mask.any():
            continue
        result["left_contrast"][trial_mask] = float(contrast_left[t_idx])
        result["right_contrast"][trial_mask] = float(contrast_right[t_idx])
        result["response_choice"][trial_mask] = float(choice[t_idx])
        result["feedback_type"][trial_mask] = float(feedback[t_idx])
        result["trial_active"][trial_mask] = 1.0
        result["trial_index"][trial_mask] = t_idx

    return result
