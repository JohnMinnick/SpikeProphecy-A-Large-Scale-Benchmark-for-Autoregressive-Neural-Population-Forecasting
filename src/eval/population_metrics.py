"""
Population-level evaluation metrics for spike-count forecasting.

Goes beyond per-neuron marginal metrics to test whether the model
captures shared population dynamics and condition-averaged responses.

Metrics:
    1. Co-BPS (co-smoothing bits per spike) — Neural Latents Benchmark standard.
       Holds out neurons and measures prediction quality from remaining population.
    2. PSTH R² — Trial-averaged comparison against empirical PSTHs.
       Tests if the model captures consistent stimulus-evoked responses.
    3. Calibration curve — Predicted rate vs observed count agreement.
       Tests if the model is systematically biased.
    4. Population Rate Correlation (PRC) — Correlation of population-summed rates.
    5. Population Vector Cosine Similarity — Frame-by-frame population state alignment.

All functions accept (T, M) numpy arrays matching the interface in
comparison_metrics.py. Return values are dicts suitable for direct
W&B logging and S3 metrics.json serialization.

References:
    - Co-BPS: Pei et al. (2021), "Neural Latents Benchmark '21", NeurIPS
    - PSTH R²: Pandarinath et al. (2018), "LFADS", Nature Methods
    - Calibration: Niculescu-Mizil & Caruana (2005), ICML

Usage:
    from src.eval.population_metrics import (
        co_bps, psth_r_squared, calibration_curve,
        population_rate_correlation, population_cosine_similarity,
        population_dtw
    )
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Co-BPS (Co-smoothing Bits Per Spike)
# ---------------------------------------------------------------------------

def co_bps(
    gt: np.ndarray,
    pred_rates: np.ndarray,
    held_out_fraction: float = 0.2,
    seed: int = 42,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Co-smoothing bits per spike (Neural Latents Benchmark standard).

    Tests whether the model captures shared population dynamics by:
    1. Splitting neurons into held-in (80%) and held-out (20%)
    2. Training a Ridge regression from held-in predicted rates → held-out GT
    3. Computing BPS on held-out neurons using the linear readout predictions

    If the model truly learns population-level dynamics, a linear projection
    from 80% of neurons should predict the remaining 20%.

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates from the model.
        held_out_fraction: Fraction of neurons to hold out (default 0.2).
        seed: Random seed for neuron split reproducibility.
        eps: Small constant for numerical stability.

    Returns:
        Dict with:
            co_bps: Mean co-smoothing BPS across held-out neurons
            co_bps_std: Std of per-neuron co-BPS
            n_held_out: Number of held-out neurons
            n_held_in: Number of held-in neurons
    """
    T, M = gt.shape
    assert pred_rates.shape == (T, M), (
        f"Shape mismatch: gt={gt.shape}, pred={pred_rates.shape}"
    )

    # --- Split neurons into held-in and held-out ---
    rng = np.random.RandomState(seed)
    n_held_out = max(1, int(M * held_out_fraction))
    all_indices = np.arange(M)
    rng.shuffle(all_indices)
    held_out_idx = all_indices[:n_held_out]
    held_in_idx = all_indices[n_held_out:]

    # --- Split time into train/test for the linear readout ---
    # Use first 80% of time bins to fit readout, last 20% to evaluate
    t_split = int(T * 0.8)
    if t_split < 10 or (T - t_split) < 10:
        logger.warning(
            "co_bps: too few time bins (%d) for reliable split", T
        )
        return {
            "co_bps": 0.0, "co_bps_std": 0.0,
            "n_held_out": n_held_out, "n_held_in": len(held_in_idx),
        }

    # Held-in model predictions → features for linear readout
    X_train = pred_rates[:t_split, :][:, held_in_idx]
    X_test = pred_rates[t_split:, :][:, held_in_idx]

    # Held-out ground truth → targets for linear readout
    y_train = gt[:t_split, :][:, held_out_idx]
    y_test = gt[t_split:, :][:, held_out_idx]

    # --- Fit Ridge regression (held-in rates → held-out counts) ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Predict held-out neuron rates from held-in population
    y_pred = ridge.predict(X_test)
    y_pred = np.clip(y_pred, eps, None)  # Rates must be positive

    # --- Compute BPS on held-out neurons ---
    bps_per_neuron = []
    for i in range(n_held_out):
        k = y_test[:, i]  # Observed spikes (held-out, test time)
        total_spikes = k.sum()
        if total_spikes < 1:
            continue  # Skip silent neurons

        mu = k.mean()  # Mean-rate baseline
        lam = y_pred[:, i]  # Linear readout predictions

        # BPS = (1/n_spikes) * sum(k * log2(lambda / mu))
        log_ratio = np.log2(lam / max(mu, eps))
        bps = np.sum(k * log_ratio) / total_spikes
        bps_per_neuron.append(bps)

    bps_arr = np.array(bps_per_neuron) if bps_per_neuron else np.array([0.0])

    result = {
        "co_bps": float(np.mean(bps_arr)),
        "co_bps_std": float(np.std(bps_arr)),
        "n_held_out": n_held_out,
        "n_held_in": len(held_in_idx),
    }
    logger.info(
        "Co-BPS: %.4f ± %.4f (%d held-out, %d held-in)",
        result["co_bps"], result["co_bps_std"],
        result["n_held_out"], result["n_held_in"],
    )
    return result


# ---------------------------------------------------------------------------
# 1b. Population Rate Correlation (PRC)
# ---------------------------------------------------------------------------

def population_rate_correlation(
    gt: np.ndarray,
    pred_rates: np.ndarray,
) -> Dict[str, float]:
    """
    Population Rate Correlation (PRC).

    Tests whether the model captures macroscopic bulk dynamics.
    Sums spikes across all M neurons for each time bin, then computes the
    Pearson correlation between the empirical population sum and the model's
    predicted population sum sequence.

    Useful for exposing whether the model has learned the overall excitation/
    inhibition rhythm of the circuit, even if per-neuron single-bin precision
    is buried in Poisson noise.

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates from the model.

    Returns:
        Dict with:
            population_rate_corr: The Pearson r correlation [0, 1]
    """
    T, M = gt.shape
    assert pred_rates.shape == (T, M)

    pop_gt = gt.sum(axis=1)           # (T,)
    pop_pred = pred_rates.sum(axis=1) # (T,)

    # If the network is entirely silent, correlation is undefined
    if np.std(pop_gt) < 1e-10 or np.std(pop_pred) < 1e-10:
        logger.warning("population_rate_correlation: zero variance in pop rate")
        return {"population_rate_corr": 0.0}

    prc = np.corrcoef(pop_gt, pop_pred)[0, 1]
    
    # Handle NaNs just in case
    if not np.isfinite(prc):
        prc = 0.0

    logger.info("Population Rate Correlation: %.4f", prc)
    return {"population_rate_corr": float(prc)}


# ---------------------------------------------------------------------------
# 1c. Population Vector Cosine Similarity
# ---------------------------------------------------------------------------

def population_cosine_similarity(
    gt: np.ndarray,
    pred_rates: np.ndarray,
) -> Dict[str, float]:
    """
    Mean Cosine Similarity of Population Vectors.

    At each time step t, treats the network state as a vector in R^M.
    Computes the cosine similarity between the ground truth state and the
    predicted state, then averages across time.

    Answers: "In any given 50ms window, how accurately does the model guess
    the relative distribution of spikes across the population (regardless of
    absolute amplitude)?"

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates from the model.

    Returns:
        Dict with:
            pop_cosine_sim_mean: Mean cosine similarity [0, 1] across time bins
            pop_cosine_sim_std: Standard deviation across time bins
    """
    from numpy.linalg import norm
    
    T, M = gt.shape
    assert pred_rates.shape == (T, M)

    similarities = []
    
    for t in range(T):
        g = gt[t]
        p = pred_rates[t]
        
        ng = norm(g)
        np_norm = norm(p)
        
        if ng > 1e-8 and np_norm > 1e-8:
            sim = np.dot(g, p) / (ng * np_norm)
            similarities.append(sim)

    if not similarities:
        logger.warning("population_cosine_similarity: all bins empty")
        return {
            "pop_cosine_sim_mean": 0.0,
            "pop_cosine_sim_std": 0.0
        }

    sim_arr = np.array(similarities)
    mean_sim = float(np.mean(sim_arr))
    std_sim = float(np.std(sim_arr))

    logger.info("Population Cosine Sim: %.4f ± %.4f", mean_sim, std_sim)
    return {
        "pop_cosine_sim_mean": mean_sim,
        "pop_cosine_sim_std": std_sim
    }


# ---------------------------------------------------------------------------
# 1d. Population Dynamic Time Warping (DTW) Sequence Alignment
# ---------------------------------------------------------------------------

def population_dtw(
    gt: np.ndarray,
    pred_rates: np.ndarray,
) -> Dict[str, float]:
    """
    Population Dynamic Time Warping (DTW) Distance.

    Tests whether the model captures the overall topology and rhythms of the 
    population rate (excitation/inhibition cycles) without penalizing strict 
    bin-to-bin temporal jitter. Uses FastDTW to find the lowest-cost warping 
    path between the 1D population ground truth and predicted sequences.

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates from the model.

    Returns:
        Dict with:
            pop_dtw_distance: Total DTW cost/distance
            pop_dtw_avg_error: Average DTW cost per warping step (spikes/bin)
    """
    try:
        from fastdtw import fastdtw
    except ImportError:
        logger.warning("population_dtw requires fastdtw. Returning 0.0")
        return {
            "pop_dtw_distance": 0.0,
            "pop_dtw_avg_error": 0.0
        }

    T, M = gt.shape
    assert pred_rates.shape == (T, M)

    pop_gt = gt.sum(axis=1).flatten()
    pop_pred = pred_rates.sum(axis=1).flatten()

    if T < 2:
        return {"pop_dtw_distance": 0.0, "pop_dtw_avg_error": 0.0}

    # FastDTW handles 1D arrays to avoid scipy distance scalar errors
    dist, path = fastdtw(pop_gt, pop_pred)
    avg_error = dist / max(len(path), 1)

    logger.info("Population DTW Error: %.4f spikes/bin", avg_error)
    return {
        "pop_dtw_distance": float(dist),
        "pop_dtw_avg_error": float(avg_error)
    }

# ---------------------------------------------------------------------------
# 2. PSTH R² (Trial-Averaged Evaluation)
# ---------------------------------------------------------------------------

def psth_r_squared(
    gt: np.ndarray,
    pred_rates: np.ndarray,
    trial_ids: np.ndarray,
    condition_labels: np.ndarray,
) -> Dict[str, float]:
    """
    PSTH R² — trial-averaged prediction quality (LFADS standard).

    For each experimental condition (e.g., stimulus × response), averages
    model predictions across trials to form a model PSTH, then compares
    against the empirical PSTH (trial-averaged ground truth).

    Tests whether the model captures *consistent* stimulus-evoked responses
    that replicate across trials, not just single-trial noise.

    Args:
        gt: (T, M) ground truth spike counts for the full time series.
        pred_rates: (T, M) predicted rates for the full time series.
        trial_ids: (T,) integer array mapping each time bin to a trial
            (e.g., 0, 0, 0, 1, 1, 1, 2, ...). Bins with trial_id == -1
            are inter-trial intervals and are excluded.
        condition_labels: (n_trials,) array mapping each trial_id to a
            condition label (e.g., "left", "right", "no-go"). Trials
            with the same label are averaged together.

    Returns:
        Dict with:
            psth_r2_mean: Mean PSTH R² across neurons
            psth_r2_std: Std PSTH R² across neurons
            n_conditions: Number of unique conditions
            n_trials: Total number of trials
            psth_r2_per_neuron: List of per-neuron R² values
    """
    T, M = gt.shape
    assert pred_rates.shape == (T, M)
    assert trial_ids.shape == (T,)

    # --- Get unique conditions and map trials to conditions ---
    unique_conditions = np.unique(condition_labels)
    n_conditions = len(unique_conditions)

    if n_conditions < 2:
        logger.warning("PSTH R²: need ≥2 conditions, got %d", n_conditions)
        return {
            "psth_r2_mean": 0.0, "psth_r2_std": 0.0,
            "n_conditions": n_conditions, "n_trials": len(condition_labels),
            "psth_r2_per_neuron": [],
        }

    # --- Find the minimum trial length across all trials ---
    unique_trials = np.unique(trial_ids)
    unique_trials = unique_trials[unique_trials >= 0]  # Exclude inter-trial
    trial_lengths = []
    for tid in unique_trials:
        trial_lengths.append(np.sum(trial_ids == tid))
    min_trial_len = min(trial_lengths) if trial_lengths else 0

    if min_trial_len < 2:
        logger.warning("PSTH R²: trials too short (min=%d bins)", min_trial_len)
        return {
            "psth_r2_mean": 0.0, "psth_r2_std": 0.0,
            "n_conditions": n_conditions, "n_trials": len(unique_trials),
            "psth_r2_per_neuron": [],
        }

    # --- Build condition-averaged PSTHs ---
    # For each condition: collect all trials, truncate to min_trial_len,
    # average across trials → (min_trial_len, M) PSTH
    gt_psths = []   # List of (min_trial_len, M) arrays
    pred_psths = []

    for cond in unique_conditions:
        # Find all trials belonging to this condition
        cond_trial_ids = [
            tid for tid in unique_trials
            if condition_labels[tid] == cond
        ]

        if len(cond_trial_ids) < 2:
            continue  # Need ≥2 trials to form a meaningful average

        gt_trials = []
        pred_trials = []
        for tid in cond_trial_ids:
            mask = trial_ids == tid
            gt_trial = gt[mask][:min_trial_len]
            pred_trial = pred_rates[mask][:min_trial_len]
            gt_trials.append(gt_trial)
            pred_trials.append(pred_trial)

        # Average across trials → PSTH for this condition
        gt_psth = np.mean(gt_trials, axis=0)    # (min_trial_len, M)
        pred_psth = np.mean(pred_trials, axis=0)

        gt_psths.append(gt_psth)
        pred_psths.append(pred_psth)

    if not gt_psths:
        logger.warning("PSTH R²: no conditions with ≥2 trials")
        return {
            "psth_r2_mean": 0.0, "psth_r2_std": 0.0,
            "n_conditions": 0, "n_trials": len(unique_trials),
            "psth_r2_per_neuron": [],
        }

    # Concatenate all condition PSTHs → (n_conditions * min_trial_len, M)
    gt_concat = np.concatenate(gt_psths, axis=0)
    pred_concat = np.concatenate(pred_psths, axis=0)

    # --- Compute R² per neuron ---
    r2_per_neuron = []
    for n in range(M):
        g = gt_concat[:, n]
        p = pred_concat[:, n]
        ss_tot = np.sum((g - g.mean()) ** 2)
        if ss_tot < 1e-10:
            r2_per_neuron.append(0.0)  # Constant across conditions
            continue
        ss_res = np.sum((g - p) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        r2_per_neuron.append(float(r2))

    r2_arr = np.array(r2_per_neuron)

    result = {
        "psth_r2_mean": float(np.mean(r2_arr)),
        "psth_r2_std": float(np.std(r2_arr)),
        "n_conditions": len(gt_psths),
        "n_trials": int(len(unique_trials)),
        "psth_r2_per_neuron": [float(v) for v in r2_arr],
    }
    logger.info(
        "PSTH R²: %.4f ± %.4f (%d conditions, %d trials)",
        result["psth_r2_mean"], result["psth_r2_std"],
        result["n_conditions"], result["n_trials"],
    )
    return result


# ---------------------------------------------------------------------------
# 3. Calibration Curve
# ---------------------------------------------------------------------------

def calibration_curve(
    gt: np.ndarray,
    pred_rates: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, object]:
    """
    Calibration curve for Poisson rate predictions.

    Bins all (neuron, time) predictions by predicted rate, then compares
    the mean predicted rate vs mean observed count in each bin.
    Perfect calibration = 45° line (predicted == observed).

    Computable during training validation and suitable for W&B/S3 logging.

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates (non-negative).
        n_bins: Number of bins for the calibration curve (default 20).

    Returns:
        Dict with:
            calibration_error: Mean absolute calibration error (scalar)
            calibration_error_max: Max per-bin calibration error
            calibration_slope: Linear fit slope (perfect = 1.0)
            calibration_intercept: Linear fit intercept (perfect = 0.0)
            bin_pred_means: List of mean predicted rate per bin (for plotting)
            bin_obs_means: List of mean observed count per bin (for plotting)
            bin_counts: List of sample counts per bin
    """
    T, M = gt.shape
    assert pred_rates.shape == (T, M)

    # Flatten to 1D for binning
    gt_flat = gt.ravel()
    pred_flat = pred_rates.ravel()

    # Sort by predicted rate and create equal-count bins
    sort_idx = np.argsort(pred_flat)
    gt_sorted = gt_flat[sort_idx]
    pred_sorted = pred_flat[sort_idx]

    # Split into n_bins equal-count groups
    bin_size = len(pred_sorted) // n_bins
    if bin_size < 10:
        logger.warning(
            "calibration_curve: too few samples per bin (%d)", bin_size
        )

    bin_pred_means = []
    bin_obs_means = []
    bin_counts = []

    for i in range(n_bins):
        start = i * bin_size
        # Last bin gets all remaining samples
        end = (i + 1) * bin_size if i < n_bins - 1 else len(pred_sorted)

        bin_pred = pred_sorted[start:end]
        bin_obs = gt_sorted[start:end]

        bin_pred_means.append(float(np.mean(bin_pred)))
        bin_obs_means.append(float(np.mean(bin_obs)))
        bin_counts.append(int(end - start))

    pred_arr = np.array(bin_pred_means)
    obs_arr = np.array(bin_obs_means)

    # Calibration error: mean absolute deviation from perfect calibration
    cal_error = float(np.mean(np.abs(pred_arr - obs_arr)))
    cal_error_max = float(np.max(np.abs(pred_arr - obs_arr)))

    # Linear fit to calibration curve (perfect: slope=1, intercept=0)
    if len(pred_arr) >= 2 and pred_arr.std() > 0:
        coeffs = np.polyfit(pred_arr, obs_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    else:
        slope, intercept = 1.0, 0.0

    result = {
        "calibration_error": cal_error,
        "calibration_error_max": cal_error_max,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "bin_pred_means": bin_pred_means,
        "bin_obs_means": bin_obs_means,
        "bin_counts": bin_counts,
    }
    logger.info(
        "Calibration: error=%.4f, slope=%.3f, intercept=%.4f",
        cal_error, slope, intercept,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: compute all population metrics at once
# ---------------------------------------------------------------------------

def compute_population_metrics(
    gt: np.ndarray,
    pred_rates: np.ndarray,
    trial_ids: Optional[np.ndarray] = None,
    condition_labels: Optional[np.ndarray] = None,
    held_out_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Compute all population-level metrics in one call.

    Co-BPS and calibration are always computed. PSTH R² requires
    trial_ids and condition_labels (skipped if not provided).

    Return dict is flat and JSON-serializable — ready for W&B logging
    and S3 metrics.json upload.

    Args:
        gt: (T, M) ground truth spike counts.
        pred_rates: (T, M) predicted firing rates.
        trial_ids: Optional (T,) trial ID per time bin (-1 = inter-trial).
        condition_labels: Optional (n_trials,) condition label per trial.
        held_out_fraction: Fraction of neurons to hold out for co-BPS.
        seed: Random seed for neuron split.

    Returns:
        Dict with all metric results, prefixed by metric name.
    """
    results = {}

    # Always compute regular population density metrics
    co_bps_result = co_bps(
        gt, pred_rates,
        held_out_fraction=held_out_fraction,
        seed=seed,
    )
    for k, v in co_bps_result.items():
        results[k] = v

    cal_result = calibration_curve(gt, pred_rates)
    for k, v in cal_result.items():
        results[k] = v

    prc_result = population_rate_correlation(gt, pred_rates)
    for k, v in prc_result.items():
        results[k] = v

    cos_result = population_cosine_similarity(gt, pred_rates)
    for k, v in cos_result.items():
        results[k] = v

    dtw_result = population_dtw(gt, pred_rates)
    for k, v in dtw_result.items():
        results[k] = v

    # PSTH R² only if trial data provided
    if trial_ids is not None and condition_labels is not None:
        psth_result = psth_r_squared(
            gt, pred_rates, trial_ids, condition_labels,
        )
        for k, v in psth_result.items():
            # Exclude per-neuron list from top-level (too large for W&B)
            if k != "psth_r2_per_neuron":
                results[k] = v
    else:
        results["psth_r2_mean"] = None
        results["psth_r2_std"] = None

    return results
