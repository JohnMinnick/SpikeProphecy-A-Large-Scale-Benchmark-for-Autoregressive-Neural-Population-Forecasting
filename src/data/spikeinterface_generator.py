"""
SpikeInterface-based synthetic data generator.

Generates synthetic extracellular recordings with controllable drift, noise,
and spike statistics using SpikeInterface's generation module. This is the
primary generator for local development (no NEURON required).

Usage:
    from src.data.spikeinterface_generator import generate_synthetic_recording

    recording, sorting, extra = generate_synthetic_recording(config)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import spikeinterface.full as si

logger = logging.getLogger(__name__)


def generate_synthetic_recording(
    config: Dict[str, Any],
    return_static: bool = False,
) -> Tuple:
    """
    Generate a synthetic drifting recording using SpikeInterface.

    Reads parameters from a config dictionary (matching configs/data/default.yaml)
    and produces a drifting recording with ground-truth spike sorting.

    Args:
        config: Configuration dictionary with keys:
            - seed (int): Random seed
            - spikeinterface (dict): Generator-specific parameters including
              probe, num_neurons, duration_s, noise, drift, and firing settings
        return_static: If True, also return the static (no-drift) recording.

    Returns:
        Tuple of:
            - recording: SpikeInterface RecordingExtractor (drifting)
            - sorting: SpikeInterface SortingExtractor (ground-truth)
            - extra_infos: dict with unit locations, templates, motion, etc.
            If return_static=True, returns (rec_static, rec_drifting, sorting, extra).
    """
    seed = config.get("seed", 42)

    # Extract SpikeInterface-specific generation parameters
    si_config = config.get("spikeinterface", {})

    # Probe configuration
    probe_name = si_config.get("probe", "Neuropixels1-128")
    num_units = si_config.get("num_neurons", 20)
    duration_s = si_config.get("duration_s", 600.0)
    sampling_frequency = si_config.get("sampling_frequency", 30000.0)

    # Drift configuration
    drift_config = si_config.get("drift", {})
    drift_mode = drift_config.get("mode", "zigzag")
    drift_amplitude = drift_config.get("amplitude_factor", 0.5)
    drift_period_s = drift_config.get("period_s", 200)
    non_rigid_gradient = drift_config.get("non_rigid_gradient", None)

    # Noise configuration
    noise_config = si_config.get("noise", {})
    noise_levels = noise_config.get("levels", (12.0, 15.0))
    noise_spatial_decay = noise_config.get("spatial_decay", 25.0)
    # Ensure noise_levels is a tuple
    if isinstance(noise_levels, list):
        noise_levels = tuple(noise_levels)

    # Firing rate configuration
    firing_config = si_config.get("firing", {})
    firing_rates = firing_config.get("rates", (2.0, 8.0))
    refractory_period_ms = firing_config.get("refractory_period_ms", 4.0)
    # Ensure firing_rates is a tuple
    if isinstance(firing_rates, list):
        firing_rates = tuple(firing_rates)

    # Template configuration
    template_config = si_config.get("templates", {})
    ms_before = template_config.get("ms_before", 1.5)
    ms_after = template_config.get("ms_after", 3.0)
    alpha_range = template_config.get("alpha", (150.0, 500.0))
    spatial_decay_range = template_config.get("spatial_decay", (10, 45))
    # Ensure ranges are tuples
    if isinstance(alpha_range, list):
        alpha_range = tuple(alpha_range)
    if isinstance(spatial_decay_range, list):
        spatial_decay_range = tuple(spatial_decay_range)

    # Build displacement vector kwargs for drift
    displacement_kwargs = dict(
        displacement_sampling_frequency=5.0,
        drift_step_um=1,
        motion_list=[
            dict(
                drift_mode=drift_mode,
                non_rigid_gradient=non_rigid_gradient,
                t_start_drift=0.0,
                t_end_drift=None,
                period_s=drift_period_s,
                amplitude_factor=drift_amplitude,
            ),
        ],
    )

    logger.info(
        "Generating synthetic recording: %d units, %.0fs, probe=%s, seed=%d",
        num_units, duration_s, probe_name, seed,
    )
    logger.info(
        "Drift: mode=%s, amplitude=%.2f, period=%ds",
        drift_mode, drift_amplitude, drift_period_s,
    )
    logger.info(
        "Noise levels: %s, spatial_decay=%.1f",
        noise_levels, noise_spatial_decay,
    )

    # Generate the recording
    rec_static, rec_drifting, gt_sorting, extra_infos = (
        si.generate_drifting_recording(
            probe_name=probe_name,
            num_units=num_units,
            duration=duration_s,
            sampling_frequency=sampling_frequency,
            generate_displacement_vector_kwargs=displacement_kwargs,
            generate_templates_kwargs=dict(
                ms_before=ms_before,
                ms_after=ms_after,
                mode="ellipsoid",
                unit_params=dict(
                    alpha=alpha_range,
                    spatial_decay=spatial_decay_range,
                ),
            ),
            generate_sorting_kwargs=dict(
                firing_rates=firing_rates,
                refractory_period_ms=refractory_period_ms,
            ),
            generate_noise_kwargs=dict(
                noise_levels=noise_levels,
                spatial_decay=noise_spatial_decay,
            ),
            extra_outputs=True,
            seed=seed,
        )
    )

    # Log summary statistics
    n_units = gt_sorting.get_num_units()
    n_channels = rec_drifting.get_num_channels()
    total_spikes = sum(
        len(gt_sorting.get_unit_spike_train(u))
        for u in gt_sorting.get_unit_ids()
    )
    logger.info(
        "Generated recording: %d channels, %d units, %d total spikes, %.1fs duration",
        n_channels, n_units, total_spikes, rec_drifting.get_total_duration(),
    )

    if return_static:
        return rec_static, rec_drifting, gt_sorting, extra_infos
    else:
        return rec_drifting, gt_sorting, extra_infos


def save_recording(
    recording,
    sorting,
    output_dir: str,
    format: str = "binary",
    n_jobs: int = 1,
) -> Path:
    """
    Save a generated recording and sorting to disk.

    Args:
        recording: SpikeInterface RecordingExtractor.
        sorting: SpikeInterface SortingExtractor (ground truth).
        output_dir: Directory to save the recording.
        format: Output format ('binary' or 'zarr').
        n_jobs: Number of parallel jobs for saving.

    Returns:
        Path to the output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save the recording traces
    rec_path = out_path / "recording"
    logger.info("Saving recording to %s (format=%s)...", rec_path, format)
    if format == "zarr":
        recording.save(folder=rec_path, format="zarr", n_jobs=n_jobs)
    else:
        recording.save(folder=rec_path, format="binary", n_jobs=n_jobs)

    # Save the ground-truth sorting
    sort_path = out_path / "sorting"
    logger.info("Saving ground-truth sorting to %s...", sort_path)
    sorting.save(folder=sort_path)

    logger.info("Saved recording and sorting to %s", out_path)
    return out_path
