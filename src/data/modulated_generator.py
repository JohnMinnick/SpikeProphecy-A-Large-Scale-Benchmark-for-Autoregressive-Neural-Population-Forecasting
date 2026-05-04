"""
Modulated inhomogeneous Poisson spike generator.

Generates synthetic spike trains with time-varying firing rates to create
temporal structure that the teacher LSTM can learn. Replaces the SpikeInterface
homogeneous Poisson generator for the purposes of spike-count forecasting.

Modulation types:
    - Sinusoidal: Oscillating rates at configurable frequencies/depths
    - Step: Piecewise-constant rates with random change-points
    - Ramp: Linear rate sweeps over configurable windows

Optionally adds cross-channel correlations via a shared latent signal
that modulates all neurons' rates simultaneously.

The output is a MockSorting object that implements the SpikeInterface
SortingExtractor interface used by binning.py (get_unit_ids,
get_unit_spike_train, get_sampling_frequency).

Usage:
    from src.data.modulated_generator import generate_modulated_spikes

    sorting, metadata = generate_modulated_spikes(config)
    counts, meta = bin_spike_trains(sorting, bin_width_ms=10)
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MockSorting:
    """
    Minimal SortingExtractor-like object for downstream compatibility.

    Implements the subset of the SpikeInterface SortingExtractor interface
    that binning.py requires: get_unit_ids(), get_unit_spike_train(),
    and get_sampling_frequency().

    Attributes:
        _unit_ids: List of unit IDs.
        _spike_trains: Dict mapping unit_id -> spike sample indices (int64).
        _sampling_frequency: Sampling frequency in Hz.
    """

    def __init__(
        self,
        spike_trains: Dict[int, np.ndarray],
        sampling_frequency: float,
    ):
        """
        Args:
            spike_trains: Dict mapping unit_id -> array of spike sample indices.
            sampling_frequency: Sampling rate in Hz.
        """
        self._unit_ids = sorted(spike_trains.keys())
        self._spike_trains = spike_trains
        self._sampling_frequency = sampling_frequency

    def get_unit_ids(self) -> List[int]:
        """Return list of unit IDs."""
        return self._unit_ids

    def get_num_units(self) -> int:
        """Return number of units."""
        return len(self._unit_ids)

    def get_unit_spike_train(self, unit_id: int) -> np.ndarray:
        """Return spike sample indices for a given unit."""
        return self._spike_trains[unit_id]

    def get_sampling_frequency(self) -> float:
        """Return sampling frequency in Hz."""
        return self._sampling_frequency


# ---------------------------------------------------------------------------
# Rate modulation functions
# ---------------------------------------------------------------------------

def _sinusoidal_rate(
    t: np.ndarray,
    base_rate: float,
    frequency: float,
    depth: float,
    phase: float,
) -> np.ndarray:
    """
    Compute sinusoidal rate modulation.

    rate(t) = base_rate * (1 + depth * sin(2π * freq * t + phase))

    Args:
        t: Time array in seconds.
        base_rate: Mean firing rate in Hz.
        frequency: Modulation frequency in Hz.
        depth: Modulation depth in [0, 1]. 0 = no modulation, 1 = full.
        phase: Phase offset in radians.

    Returns:
        Instantaneous rate array (Hz), same shape as t. Always >= 0.
    """
    rate = base_rate * (1.0 + depth * np.sin(2.0 * np.pi * frequency * t + phase))
    # Clip to non-negative (depth near 1.0 could cause negative dips)
    return np.maximum(rate, 0.0)


def _step_rate(
    t: np.ndarray,
    num_changes: int,
    rate_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute step-function rate modulation with random change-points.

    Args:
        t: Time array in seconds.
        num_changes: Number of step transitions.
        rate_range: (min_rate, max_rate) range for step levels.
        rng: NumPy random generator for reproducibility.

    Returns:
        Piecewise-constant rate array (Hz), same shape as t.
    """
    # Random change-points sorted in time
    change_times = np.sort(rng.uniform(t[0], t[-1], size=num_changes))
    # Random rate levels for each segment (num_changes + 1 segments)
    levels = rng.uniform(rate_range[0], rate_range[1], size=num_changes + 1)

    # Build the rate array
    rate = np.full_like(t, levels[0])
    for i, ct in enumerate(change_times):
        rate[t >= ct] = levels[i + 1]

    return rate


def _ramp_rate(
    t: np.ndarray,
    base_rate: float,
    slope: float,
    rate_range: Tuple[float, float],
) -> np.ndarray:
    """
    Compute linear ramp rate modulation.

    rate(t) = base_rate + slope * (t - t_start), clipped to rate_range.

    Args:
        t: Time array in seconds.
        base_rate: Starting rate in Hz.
        slope: Rate of change in Hz/s.
        rate_range: (min_rate, max_rate) to clip to.

    Returns:
        Linearly ramping rate array (Hz), same shape as t.
    """
    rate = base_rate + slope * (t - t[0])
    return np.clip(rate, rate_range[0], rate_range[1])


def _shared_latent_signal(
    t: np.ndarray,
    frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a shared latent signal for cross-channel correlations.

    A low-frequency sinusoid with a random phase, normalized to [0, 1].

    Args:
        t: Time array in seconds.
        frequency: Signal frequency in Hz.
        rng: NumPy random generator.

    Returns:
        Shared signal array in [0, 1], same shape as t.
    """
    phase = rng.uniform(0, 2.0 * np.pi)
    signal = 0.5 * (1.0 + np.sin(2.0 * np.pi * frequency * t + phase))
    return signal


# ---------------------------------------------------------------------------
# Rate overlay functions (applied after primary modulation)
# ---------------------------------------------------------------------------

def _burst_overlay(
    rate: np.ndarray,
    dt: float,
    burst_rate: float,
    burst_duration_ms_range: Tuple[float, float],
    burst_factor_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply bursting overlay: multiply rate by a burst factor during random epochs.

    Simulates cortical bursting by inserting brief windows where the firing
    rate is amplified. Burst timing follows a Poisson process.

    Args:
        rate: Base rate array (Hz), modified in-place.
        dt: Time step in seconds.
        burst_rate: Mean number of bursts per second.
        burst_duration_ms_range: (min, max) burst duration in ms.
        burst_factor_range: (min, max) rate multiplier during bursts.
        rng: NumPy random generator.

    Returns:
        Modified rate array with bursting applied.
    """
    duration_s = len(rate) * dt
    n_expected = int(burst_rate * duration_s)
    if n_expected <= 0:
        return rate

    # Generate burst onset times (Poisson process)
    n_bursts = rng.poisson(n_expected)
    if n_bursts == 0:
        return rate

    burst_onsets_s = np.sort(rng.uniform(0, duration_s, size=n_bursts))

    rate_out = rate.copy()
    for onset in burst_onsets_s:
        # Random burst duration and factor for this burst
        dur_ms = rng.uniform(burst_duration_ms_range[0], burst_duration_ms_range[1])
        factor = rng.uniform(burst_factor_range[0], burst_factor_range[1])

        # Convert to sample indices
        start_idx = int(onset / dt)
        end_idx = min(start_idx + int(dur_ms / 1000.0 / dt), len(rate))

        # Multiply rate during burst window
        rate_out[start_idx:end_idx] *= factor

    return rate_out


def _stimulus_events_overlay(
    rate: np.ndarray,
    t: np.ndarray,
    base_rate: float,
    num_events_range: Tuple[int, int],
    amplitude_factor_range: Tuple[float, float],
    decay_tau_ms_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply stimulus-evoked transient events: sharp peaks with exponential decay.

    Simulates sensory-evoked neural responses — a sudden rate increase at
    "stimulus" onset that decays exponentially back to baseline.

    Args:
        rate: Base rate array (Hz).
        t: Time array in seconds, same length as rate.
        base_rate: Base firing rate (Hz) used to scale amplitude.
        num_events_range: (min, max) number of stimulus events.
        amplitude_factor_range: (min, max) peak rate multiplier.
        decay_tau_ms_range: (min, max) decay time constant in ms.
        rng: NumPy random generator.

    Returns:
        Rate array with stimulus transients added.
    """
    num_events = rng.integers(num_events_range[0], num_events_range[1] + 1)
    if num_events <= 0:
        return rate

    rate_out = rate.copy()
    duration_s = t[-1] - t[0]

    # Generate random event times
    event_times = np.sort(rng.uniform(t[0], t[-1], size=num_events))

    for event_t in event_times:
        # Random amplitude and decay for this event
        amplitude = base_rate * rng.uniform(
            amplitude_factor_range[0], amplitude_factor_range[1]
        )
        tau_s = rng.uniform(
            decay_tau_ms_range[0], decay_tau_ms_range[1]
        ) / 1000.0

        # Exponential decay kernel: A * exp(-(t - t_event) / tau)
        # Only apply for t >= event_t
        mask = t >= event_t
        delta_t = t[mask] - event_t
        transient = amplitude * np.exp(-delta_t / tau_s)

        # Add transient to rate
        rate_out[mask] += transient

    return rate_out


def _drift_overlay(
    rate: np.ndarray,
    dt: float,
    base_rate: float,
    tau_s: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply non-stationary drift via an Ornstein-Uhlenbeck (OU) process.

    Adds slow, mean-reverting fluctuations to the firing rate that mimic
    recording drift over minutes-long sessions. The OU process ensures
    rates stay bounded and statistically stationary.

    The drift signal is: dx = -x/tau * dt + sigma * sqrt(2*dt/tau) * dW
    Applied multiplicatively: rate_out = rate * (1 + x)

    Args:
        rate: Base rate array (Hz).
        dt: Time step in seconds.
        base_rate: Base rate (Hz) — used for scaling (currently unused but
            kept for future amplitude-dependent drift).
        tau_s: Mean-reversion timescale in seconds.
        sigma: Noise strength (std of the stationary distribution).
        rng: NumPy random generator.

    Returns:
        Rate array with drift applied. Clipped to non-negative.
    """
    n_steps = len(rate)
    # OU process parameters
    # dx = -(x / tau) * dt + sigma * sqrt(2 * dt / tau) * N(0,1)
    noise_scale = sigma * np.sqrt(2.0 * dt / tau_s)

    # Simulate OU process
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        x[i] = x[i - 1] - (x[i - 1] / tau_s) * dt + noise_scale * rng.standard_normal()

    # Apply multiplicatively: rate * (1 + x), clipped to non-negative
    rate_out = rate * (1.0 + x)
    return np.maximum(rate_out, 0.0)


def _multi_oscillation_overlay(
    rate: np.ndarray,
    t: np.ndarray,
    base_rate: float,
    bands: list,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add multiple oscillatory components at biologically relevant frequencies.

    Layers sinusoids at different frequency bands (e.g., theta 4-8 Hz,
    gamma 30-80 Hz) to create spectral richness beyond a single modulation
    frequency.

    Args:
        rate: Base rate array (Hz).
        t: Time array in seconds.
        base_rate: Base firing rate (Hz) for scaling depths.
        bands: List of dicts, each with:
            - frequencies_hz: [min, max] frequency range
            - depth: [min, max] modulation depth range
        rng: NumPy random generator.

    Returns:
        Rate array with multi-band oscillations added. Clipped to non-negative.
    """
    rate_out = rate.copy()

    for band in bands:
        freq_range = band.get("frequencies_hz", [4.0, 8.0])
        depth_range = band.get("depth", [0.05, 0.15])

        # Random frequency, depth, and phase for this band
        freq = rng.uniform(freq_range[0], freq_range[1])
        depth = rng.uniform(depth_range[0], depth_range[1])
        phase = rng.uniform(0, 2.0 * np.pi)

        # Additive oscillation: base_rate * depth * sin(...)
        oscillation = base_rate * depth * np.sin(
            2.0 * np.pi * freq * t + phase
        )
        rate_out += oscillation

    return np.maximum(rate_out, 0.0)



def _generate_inhomogeneous_poisson(
    rate_func: np.ndarray,
    dt: float,
    refractory_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spike times from an inhomogeneous Poisson process via thinning.

    For each time step dt, the probability of a spike is rate(t) * dt.
    We use direct Bernoulli sampling (accurate for small dt) and enforce
    a refractory period post-hoc.

    Args:
        rate_func: Instantaneous firing rate at each time step (Hz).
        dt: Time step duration in seconds.
        refractory_s: Refractory period in seconds.
        rng: NumPy random generator.

    Returns:
        Array of spike times as sample indices (int64), sorted.
    """
    # Probability of spike at each time step: p = rate * dt
    # Clip to [0, 1] for safety (high rates + large dt)
    p_spike = np.clip(rate_func * dt, 0.0, 1.0)

    # Draw Bernoulli trials
    spikes_mask = rng.random(len(p_spike)) < p_spike
    spike_indices = np.where(spikes_mask)[0].astype(np.int64)

    # Enforce refractory period
    if len(spike_indices) > 1 and refractory_s > 0:
        refractory_samples = int(refractory_s / dt)
        filtered = [spike_indices[0]]
        for idx in spike_indices[1:]:
            if idx - filtered[-1] >= refractory_samples:
                filtered.append(idx)
        spike_indices = np.array(filtered, dtype=np.int64)

    return spike_indices


# ---------------------------------------------------------------------------
# Main generator function
# ---------------------------------------------------------------------------

def generate_modulated_spikes(
    config: Dict[str, Any],
) -> Tuple[MockSorting, Dict[str, Any]]:
    """
    Generate spike trains with time-varying (modulated) firing rates.

    Each neuron's firing rate is modulated by configurable temporal patterns
    (sinusoidal, step, ramp). Optionally, a shared latent signal creates
    cross-channel correlations.

    The spike generation uses the Bernoulli approximation to an
    inhomogeneous Poisson process at 1 ms resolution (dt=0.001s),
    which is accurate for typical firing rates (1-20 Hz).

    Args:
        config: Configuration dictionary. Expected keys:
            - seed (int): Random seed for reproducibility.
            - modulation (dict): Modulation parameters.
            - spikeinterface.num_neurons (int): Number of units.
            - spikeinterface.duration_s (float): Duration in seconds.
            - spikeinterface.sampling_frequency (float): Sampling rate.
            - spikeinterface.firing.rates (list): [min_rate, max_rate] Hz.
            - spikeinterface.firing.refractory_period_ms (float): Refractory.

    Returns:
        Tuple of:
            - MockSorting: SortingExtractor-like object with spike trains.
            - metadata: Dict with generation parameters and summary stats.
    """
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # --- Extract parameters from config ---
    si_config = config.get("spikeinterface", {})
    num_units = si_config.get("num_neurons", 20)
    duration_s = si_config.get("duration_s", 600.0)
    sampling_frequency = si_config.get("sampling_frequency", 30000.0)

    firing_config = si_config.get("firing", {})
    firing_rates = firing_config.get("rates", [2.0, 8.0])
    refractory_ms = firing_config.get("refractory_period_ms", 4.0)
    refractory_s = refractory_ms / 1000.0

    # Modulation config
    mod_config = config.get("modulation", {})
    mod_types = mod_config.get("types", ["sinusoidal"])

    # Sinusoidal modulation params
    sin_config = mod_config.get("sinusoidal", {})
    sin_freq_range = sin_config.get("frequencies_hz", [0.5, 2.0])
    sin_depth_range = sin_config.get("depth", [0.3, 0.8])

    # Step modulation params
    step_config = mod_config.get("step", {})
    step_changes_range = step_config.get("num_changes", [3, 8])
    step_rate_range = step_config.get("rate_range", [1.0, 15.0])

    # Ramp modulation params
    ramp_config = mod_config.get("ramp", {})
    ramp_slope_range = ramp_config.get("slope_range", [-0.5, 0.5])

    # Shared signal (cross-channel correlation)
    shared_config = mod_config.get("shared_signal", {})
    shared_enabled = shared_config.get("enabled", True)
    shared_freq = shared_config.get("frequency_hz", 0.3)
    shared_coupling_range = shared_config.get("coupling_range", [0.1, 0.5])

    # --- Overlay configs (biologically-motivated rate modifications) ---
    # Bursting: cortical neurons fire in 50-200ms clusters
    burst_config = mod_config.get("bursting", {})
    burst_enabled = burst_config.get("enabled", False)
    burst_rate = burst_config.get("burst_rate", 0.3)
    burst_dur_range = burst_config.get("burst_duration_ms", [50.0, 200.0])
    burst_factor_range = burst_config.get("burst_factor", [3.0, 5.0])

    # Stimulus events: sensory-evoked transients with exponential decay
    stim_config = mod_config.get("stimulus_events", {})
    stim_enabled = stim_config.get("enabled", False)
    stim_num_range = stim_config.get("num_events", [5, 15])
    stim_amp_range = stim_config.get("amplitude_factor", [3.0, 8.0])
    stim_tau_range = stim_config.get("decay_tau_ms", [50.0, 200.0])

    # Drift: slow non-stationarity via Ornstein-Uhlenbeck process
    drift_config = mod_config.get("drift", {})
    drift_enabled = drift_config.get("enabled", False)
    drift_tau_s = drift_config.get("tau_s", 60.0)
    drift_sigma = drift_config.get("sigma", 0.3)

    # Multi-oscillation: theta (4-8 Hz) + gamma (30-80 Hz) layering
    osc_config = mod_config.get("multi_oscillation", {})
    osc_enabled = osc_config.get("enabled", False)
    osc_bands = osc_config.get("bands", [
        {"frequencies_hz": [4.0, 8.0], "depth": [0.05, 0.15]},
        {"frequencies_hz": [30.0, 80.0], "depth": [0.01, 0.05]},
    ])

    # --- Time array at 1 ms resolution for rate computation ---
    dt = 0.001  # 1 ms time steps for Bernoulli approximation
    num_steps = int(duration_s / dt)
    t = np.arange(num_steps) * dt

    logger.info(
        "Generating modulated spikes: %d units, %.0fs, seed=%d, "
        "modulation=%s, shared_signal=%s",
        num_units, duration_s, seed, mod_types, shared_enabled,
    )

    # --- Generate shared latent signal if enabled ---
    shared_signal = None
    if shared_enabled:
        shared_signal = _shared_latent_signal(t, shared_freq, rng)
        logger.info(
            "Shared latent signal: freq=%.2f Hz", shared_freq,
        )

    # --- Generate spike trains for each unit ---
    spike_trains = {}
    unit_rates_info = []  # For metadata

    for unit_id in range(num_units):
        # Random base rate for this unit
        base_rate = rng.uniform(firing_rates[0], firing_rates[1])

        # Randomly assign a modulation type from the configured list
        mod_type = rng.choice(mod_types)

        # Build the rate function based on modulation type
        if mod_type == "sinusoidal":
            freq = rng.uniform(sin_freq_range[0], sin_freq_range[1])
            depth = rng.uniform(sin_depth_range[0], sin_depth_range[1])
            phase = rng.uniform(0, 2.0 * np.pi)
            rate = _sinusoidal_rate(t, base_rate, freq, depth, phase)
            mod_params = {
                "type": "sinusoidal",
                "frequency_hz": float(freq),
                "depth": float(depth),
                "phase_rad": float(phase),
            }
        elif mod_type == "step":
            num_changes = rng.integers(
                step_changes_range[0], step_changes_range[1] + 1
            )
            rate = _step_rate(
                t, int(num_changes), tuple(step_rate_range), rng
            )
            mod_params = {
                "type": "step",
                "num_changes": int(num_changes),
            }
        elif mod_type == "ramp":
            slope = rng.uniform(ramp_slope_range[0], ramp_slope_range[1])
            rate = _ramp_rate(
                t, base_rate, slope,
                rate_range=(firing_rates[0] * 0.5, firing_rates[1] * 2.0),
            )
            mod_params = {
                "type": "ramp",
                "slope_hz_per_s": float(slope),
            }
        else:
            raise ValueError(
                f"Unknown modulation type: {mod_type}. "
                f"Supported: sinusoidal, step, ramp"
            )

        # Add shared signal contribution if enabled
        if shared_signal is not None:
            coupling = rng.uniform(
                shared_coupling_range[0], shared_coupling_range[1]
            )
            # Shared signal adds a fraction of the base rate
            rate = rate + coupling * base_rate * shared_signal
            mod_params["shared_coupling"] = float(coupling)

        # --- Apply overlays (biologically-grounded rate modifications) ---

        # Bursting: insert brief high-rate epochs (cortical burst patterns)
        if burst_enabled:
            rate = _burst_overlay(
                rate, dt, burst_rate,
                tuple(burst_dur_range), tuple(burst_factor_range), rng,
            )
            mod_params["bursting"] = True

        # Stimulus events: transient peaks with exponential decay
        if stim_enabled:
            rate = _stimulus_events_overlay(
                rate, t, base_rate,
                tuple(stim_num_range), tuple(stim_amp_range),
                tuple(stim_tau_range), rng,
            )
            mod_params["stimulus_events"] = True

        # Drift: slow Ornstein-Uhlenbeck non-stationarity
        if drift_enabled:
            rate = _drift_overlay(
                rate, dt, base_rate, drift_tau_s, drift_sigma, rng,
            )
            mod_params["drift"] = True

        # Multi-oscillation: theta + gamma band layering
        if osc_enabled:
            rate = _multi_oscillation_overlay(
                rate, t, base_rate, osc_bands, rng,
            )
            mod_params["multi_oscillation"] = True

        # Generate spikes via inhomogeneous Poisson (Bernoulli approx)
        spike_samples_1ms = _generate_inhomogeneous_poisson(
            rate, dt, refractory_s, rng
        )

        # Convert from 1ms resolution indices to sampling_frequency indices
        # spike time in seconds = spike_sample_1ms * dt
        # spike sample at fs = spike_time * fs
        spike_times_s = spike_samples_1ms * dt
        spike_samples = (spike_times_s * sampling_frequency).astype(np.int64)

        # Remove any duplicates that might arise from rounding
        spike_samples = np.unique(spike_samples)

        # Ensure all spikes are within recording duration
        max_sample = int(duration_s * sampling_frequency)
        spike_samples = spike_samples[spike_samples < max_sample]

        spike_trains[unit_id] = spike_samples

        # Track metadata
        actual_rate = len(spike_samples) / duration_s
        unit_rates_info.append({
            "unit_id": unit_id,
            "base_rate_hz": float(base_rate),
            "actual_rate_hz": float(actual_rate),
            "modulation": mod_params,
            "num_spikes": len(spike_samples),
        })

        logger.debug(
            "Unit %d: base=%.1f Hz, actual=%.1f Hz, mod=%s, %d spikes",
            unit_id, base_rate, actual_rate, mod_type, len(spike_samples),
        )

    # --- Build MockSorting object ---
    sorting = MockSorting(spike_trains, sampling_frequency)

    # --- Summary statistics ---
    total_spikes = sum(info["num_spikes"] for info in unit_rates_info)
    mean_rate = np.mean([info["actual_rate_hz"] for info in unit_rates_info])

    logger.info(
        "Generated %d total spikes across %d units, mean rate=%.2f Hz",
        total_spikes, num_units, mean_rate,
    )

    # --- Metadata ---
    metadata = {
        "generator": "modulated_poisson",
        "seed": seed,
        "num_units": num_units,
        "duration_s": duration_s,
        "sampling_frequency": sampling_frequency,
        "dt_s": dt,
        "overlays": {
            "bursting": burst_enabled,
            "stimulus_events": stim_enabled,
            "drift": drift_enabled,
            "multi_oscillation": osc_enabled,
        },
        "modulation_types": mod_types,
        "shared_signal_enabled": shared_enabled,
        "total_spikes": total_spikes,
        "mean_rate_hz": float(mean_rate),
        "unit_details": unit_rates_info,
    }

    return sorting, metadata
