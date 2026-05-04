"""
Tier 1 data visualizations for SpikeProphecy.

Produces publication-quality plots of raw recordings, spike trains,
and binned data. All functions accept data arrays as input — they
never import from src/models/ or src/train/.

Available plots:
    - plot_raster: spike raster (dots per unit over time)
    - plot_voltage_heatmap: multi-channel voltage trace heatmap
    - plot_binned_counts_heatmap: binned spike-count matrix
    - plot_firing_rate_histogram: per-unit firing rate distribution
    - plot_isi_distribution: inter-spike interval histogram
    - plot_autocorrelation: spike autocorrelation (ACF)
    - plot_cross_correlogram: pairwise cross-correlation
    - plot_firing_rate_traces: smoothed instantaneous rate time series
    - plot_psd: power spectral density of binned counts
    - plot_data_summary: combined 4-panel overview
    - plot_neuro_summary: comprehensive 8-panel neuroscience-grade figure

Usage:
    from src.viz.data_plots import plot_raster, plot_data_summary
    from src.viz.style import apply_style

    apply_style()
    fig = plot_raster(spike_trains, unit_ids, sampling_freq=30000)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import (
    CMAP_SEQUENTIAL,
    CMAP_SPIKE,
    COLORS,
    figure_full_width,
    figure_single_column,
    save_figure,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Raster Plot
# =============================================================================

def plot_raster(
    spike_trains: Dict[str, np.ndarray],
    sampling_frequency: float = 30000.0,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Spike Raster",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a spike raster (classic neuroscience visualization).

    Each unit gets a horizontal row; spikes are shown as vertical ticks.

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices (np.ndarray).
        sampling_frequency: Sampling rate in Hz (to convert samples → seconds).
        time_range: Optional (start_s, end_s) to zoom into a time window.
        title: Plot title.
        ax: Optional existing axes to plot into.

    Returns:
        Matplotlib Figure containing the raster plot.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    unit_ids = list(spike_trains.keys())

    for i, uid in enumerate(unit_ids):
        spike_times_s = spike_trains[uid] / sampling_frequency

        # Apply time range filter if specified
        if time_range is not None:
            mask = (spike_times_s >= time_range[0]) & (spike_times_s <= time_range[1])
            spike_times_s = spike_times_s[mask]

        ax.scatter(
            spike_times_s,
            np.full_like(spike_times_s, i),
            marker="|",
            s=10,
            linewidths=0.5,
            color=COLORS["blue"],
            alpha=0.8,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    ax.set_yticks(range(len(unit_ids)))
    ax.set_yticklabels([str(uid) for uid in unit_ids])
    ax.set_title(title)
    ax.invert_yaxis()  # Unit 0 at top

    if time_range is not None:
        ax.set_xlim(time_range)

    return fig


# =============================================================================
# Voltage Heatmap
# =============================================================================

def plot_voltage_heatmap(
    traces: np.ndarray,
    sampling_frequency: float = 30000.0,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Extracellular Voltage Traces",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a multi-channel voltage heatmap.

    Shows channels on the y-axis and time on the x-axis, with voltage
    encoded as color intensity. Useful for visualizing drift and noise.

    Args:
        traces: Voltage array, shape (n_samples, n_channels).
        sampling_frequency: Sampling rate in Hz.
        time_range: Optional (start_s, end_s) to show a subset.
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    n_samples, n_channels = traces.shape

    # Determine time window
    if time_range is not None:
        start_samp = int(time_range[0] * sampling_frequency)
        end_samp = int(time_range[1] * sampling_frequency)
        traces = traces[start_samp:end_samp, :]
        t_start = time_range[0]
    else:
        t_start = 0.0

    n_samples_plot = traces.shape[0]
    t_end = t_start + n_samples_plot / sampling_frequency

    if ax is None:
        fig, ax = figure_full_width()
    else:
        fig = ax.get_figure()

    # Normalize per channel for visibility
    vmax = np.percentile(np.abs(traces), 99)

    im = ax.imshow(
        traces.T,
        aspect="auto",
        cmap=CMAP_SEQUENTIAL,
        vmin=-vmax,
        vmax=vmax,
        extent=[t_start, t_end, n_channels - 0.5, -0.5],
        interpolation="none",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, label="Voltage (µV)", shrink=0.8)

    return fig


# =============================================================================
# Binned Count Heatmap
# =============================================================================

def plot_binned_counts_heatmap(
    spike_counts: np.ndarray,
    bin_width_ms: float = 10.0,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Binned Spike Counts",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the binned spike-count matrix as a heatmap.

    Shows the (M × T) matrix that feeds into the model, with units on
    the y-axis and time bins on the x-axis.

    Args:
        spike_counts: Count matrix, shape (M, T_total).
        bin_width_ms: Bin width in ms (for x-axis scaling).
        time_range: Optional (start_s, end_s).
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    m, t_total = spike_counts.shape

    # Convert bin indices to seconds
    bin_width_s = bin_width_ms / 1000.0

    if time_range is not None:
        start_bin = int(time_range[0] / bin_width_s)
        end_bin = int(time_range[1] / bin_width_s)
        subset = spike_counts[:, start_bin:end_bin]
        t_start = time_range[0]
    else:
        subset = spike_counts
        start_bin = 0
        t_start = 0.0

    t_end = t_start + subset.shape[1] * bin_width_s

    if ax is None:
        fig, ax = figure_full_width()
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        subset,
        aspect="auto",
        cmap=CMAP_SPIKE,
        extent=[t_start, t_end, m - 0.5, -0.5],
        interpolation="none",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, label="Spike count", shrink=0.8)

    return fig


# =============================================================================
# Firing Rate Histogram
# =============================================================================

def plot_firing_rate_histogram(
    firing_rates: np.ndarray,
    title: str = "Firing Rate Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot histogram of per-unit firing rates.

    Args:
        firing_rates: Array of firing rates in Hz, shape (M,).
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    ax.hist(
        firing_rates,
        bins=max(10, len(firing_rates) // 2),
        color=COLORS["blue"],
        edgecolor="white",
        alpha=0.85,
    )

    # Add mean line
    mean_rate = np.mean(firing_rates)
    ax.axvline(
        mean_rate, color=COLORS["red"], linestyle="--", linewidth=1.5,
        label=f"Mean: {mean_rate:.1f} Hz",
    )

    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Number of Units")
    ax.set_title(title)
    ax.legend()

    return fig


# =============================================================================
# ISI Distribution
# =============================================================================

def plot_isi_distribution(
    spike_trains: Dict[str, np.ndarray],
    sampling_frequency: float = 30000.0,
    max_isi_ms: float = 100.0,
    title: str = "Inter-Spike Interval Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the inter-spike interval (ISI) distribution across all units.

    ISI distributions reveal burstiness, refractory period violations,
    and overall firing regularity.

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices.
        sampling_frequency: Sampling rate in Hz.
        max_isi_ms: Maximum ISI to show (clips x-axis).
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    # Collect all ISIs
    all_isis_ms = []
    for uid, spikes in spike_trains.items():
        if len(spikes) > 1:
            isis_samples = np.diff(spikes)
            isis_ms = isis_samples / sampling_frequency * 1000.0
            all_isis_ms.extend(isis_ms)

    all_isis_ms = np.array(all_isis_ms)

    if len(all_isis_ms) == 0:
        ax.text(0.5, 0.5, "No ISIs to plot", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    # Filter to max ISI
    plotted = all_isis_ms[all_isis_ms <= max_isi_ms]

    ax.hist(
        plotted,
        bins=50,
        color=COLORS["green"],
        edgecolor="white",
        alpha=0.85,
    )

    # Mark refractory period (typical: 1-2 ms)
    ax.axvline(
        2.0, color=COLORS["red"], linestyle="--", linewidth=1,
        label="Refractory (2 ms)",
    )

    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    return fig


# =============================================================================
# Autocorrelation Function (ACF) Plot
# =============================================================================

def plot_autocorrelation(
    spike_trains: Dict[str, np.ndarray],
    sampling_frequency: float = 30000.0,
    max_lag_ms: float = 50.0,
    bin_width_ms: float = 0.5,
    title: str = "Spike Autocorrelation",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot average spike autocorrelation across units.

    Shows the probability of observing a spike at time lag τ given a spike
    at time 0. Reveals rhythmicity, bursting, and refractory gaps.

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices.
        sampling_frequency: Sampling rate in Hz.
        max_lag_ms: Maximum lag to compute, in ms.
        bin_width_ms: Histogram bin width in ms.
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    max_lag_samples = int(max_lag_ms * sampling_frequency / 1000.0)
    n_bins = int(max_lag_ms / bin_width_ms)
    bin_edges_ms = np.linspace(0, max_lag_ms, n_bins + 1)

    # Accumulate per-unit autocorrelation histograms
    all_counts = np.zeros(n_bins)
    n_units_used = 0

    for uid, spikes in spike_trains.items():
        if len(spikes) < 10:
            continue

        # Compute ISIs up to max lag
        isis = []
        for i, s in enumerate(spikes):
            # Look forward from each spike
            for j in range(i + 1, len(spikes)):
                diff = spikes[j] - s
                if diff > max_lag_samples:
                    break
                isis.append(diff / sampling_frequency * 1000.0)

        if len(isis) > 0:
            counts, _ = np.histogram(isis, bins=bin_edges_ms)
            # Normalize by number of reference spikes
            all_counts += counts / max(len(spikes), 1)
            n_units_used += 1

    if n_units_used > 0:
        all_counts /= n_units_used

    # Plot as bar chart
    bin_centers = (bin_edges_ms[:-1] + bin_edges_ms[1:]) / 2
    ax.bar(
        bin_centers, all_counts,
        width=bin_width_ms * 0.9,
        color=COLORS["blue"],
        edgecolor="none",
        alpha=0.85,
    )

    # Mark refractory period
    ax.axvspan(0, 2.0, alpha=0.15, color=COLORS["red"], label="Refractory")

    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Correlation (a.u.)")
    ax.set_title(title)
    ax.legend(fontsize=7)

    return fig


# =============================================================================
# Cross-Correlogram
# =============================================================================

def plot_cross_correlogram(
    spike_trains: Dict[str, np.ndarray],
    sampling_frequency: float = 30000.0,
    max_lag_ms: float = 50.0,
    bin_width_ms: float = 1.0,
    num_pairs: int = 5,
    title: str = "Cross-Correlogram",
    ax: Optional[plt.Axes] = None,
    rng: Optional[np.random.Generator] = None,
) -> plt.Figure:
    """
    Plot cross-correlograms for random unit pairs.

    Shows the timing relationship between spikes on different units.
    Peaks at short lags indicate functional connectivity or shared input.

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices.
        sampling_frequency: Sampling rate in Hz.
        max_lag_ms: Maximum lag in ms (symmetric).
        bin_width_ms: Histogram bin width in ms.
        num_pairs: Number of random pairs to average.
        title: Plot title.
        ax: Optional axes.
        rng: Optional random generator for pair selection.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    if rng is None:
        rng = np.random.default_rng(42)

    unit_ids = list(spike_trains.keys())
    if len(unit_ids) < 2:
        ax.text(0.5, 0.5, "Need ≥ 2 units", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    max_lag_samples = int(max_lag_ms * sampling_frequency / 1000.0)
    n_bins = int(2 * max_lag_ms / bin_width_ms)
    bin_edges_ms = np.linspace(-max_lag_ms, max_lag_ms, n_bins + 1)

    # Select random pairs
    from itertools import combinations
    all_pairs = list(combinations(unit_ids, 2))
    num_pairs = min(num_pairs, len(all_pairs))
    selected_pairs = rng.choice(len(all_pairs), size=num_pairs, replace=False)

    # Accumulate cross-correlation histograms
    all_counts = np.zeros(n_bins)
    n_valid = 0

    for pair_idx in selected_pairs:
        uid_a, uid_b = all_pairs[pair_idx]
        spikes_a = spike_trains[uid_a]
        spikes_b = spike_trains[uid_b]

        if len(spikes_a) < 5 or len(spikes_b) < 5:
            continue

        # Compute cross-correlation via np.searchsorted for efficiency
        diffs = []
        for s in spikes_a:
            # Find spikes in B within range of this A spike
            lo = np.searchsorted(spikes_b, s - max_lag_samples)
            hi = np.searchsorted(spikes_b, s + max_lag_samples)
            for b in spikes_b[lo:hi]:
                diff_ms = (b - s) / sampling_frequency * 1000.0
                diffs.append(diff_ms)

        if len(diffs) > 0:
            counts, _ = np.histogram(diffs, bins=bin_edges_ms)
            all_counts += counts / max(len(spikes_a), 1)
            n_valid += 1

    if n_valid > 0:
        all_counts /= n_valid

    bin_centers = (bin_edges_ms[:-1] + bin_edges_ms[1:]) / 2
    ax.bar(
        bin_centers, all_counts,
        width=bin_width_ms * 0.9,
        color=COLORS["green"],
        edgecolor="none",
        alpha=0.85,
    )

    ax.axvline(0, color=COLORS["black"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Correlation (a.u.)")
    ax.set_title(f"{title} ({num_pairs} pairs)")

    return fig


# =============================================================================
# Firing Rate Traces (smoothed instantaneous rates)
# =============================================================================

def plot_firing_rate_traces(
    spike_counts: np.ndarray,
    bin_width_ms: float = 10.0,
    smooth_sigma_bins: int = 10,
    num_units: int = 5,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Firing Rate Traces",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot smoothed firing rate traces for a subset of units.

    Shows instantaneous firing rates as continuous traces, revealing
    temporal modulation patterns (drift, oscillations, bursts).

    Args:
        spike_counts: Binned count matrix, shape (M, T).
        bin_width_ms: Bin width in ms.
        smooth_sigma_bins: Gaussian smoothing kernel sigma (in bins).
        num_units: Number of units to display.
        time_range: Optional (start_s, end_s).
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    from scipy.ndimage import gaussian_filter1d

    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    m, t_total = spike_counts.shape
    num_units = min(num_units, m)
    bin_width_s = bin_width_ms / 1000.0

    # Time axis
    t = np.arange(t_total) * bin_width_s

    # Apply time range filter
    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[mask]
    else:
        t_plot = t
        mask = np.ones(t_total, dtype=bool)

    # Plot smoothed rates for selected units
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
              COLORS["red"], COLORS["purple"], COLORS["cyan"]]

    for i in range(num_units):
        # Convert counts to Hz
        rate_hz = spike_counts[i, :].astype(float) / bin_width_s
        # Smooth with Gaussian filter
        rate_smooth = gaussian_filter1d(rate_hz, sigma=smooth_sigma_bins)
        rate_plot = rate_smooth[mask]

        ax.plot(
            t_plot, rate_plot,
            color=colors[i % len(colors)],
            linewidth=1.0,
            alpha=0.8,
            label=f"Unit {i}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    return fig


# =============================================================================
# Power Spectral Density
# =============================================================================

def plot_psd(
    spike_counts: np.ndarray,
    bin_width_ms: float = 10.0,
    max_freq_hz: float = 50.0,
    title: str = "Power Spectral Density",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the average power spectral density of binned spike counts.

    Reveals dominant oscillatory frequencies in the population activity.
    Peaks at theta (~5 Hz), alpha (~10 Hz), or gamma (~40 Hz) indicate
    rhythmic firing patterns.

    Args:
        spike_counts: Binned count matrix, shape (M, T).
        bin_width_ms: Bin width in ms.
        max_freq_hz: Maximum frequency to display.
        title: Plot title.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = figure_single_column()
    else:
        fig = ax.get_figure()

    m, t_total = spike_counts.shape
    fs = 1000.0 / bin_width_ms  # Sampling rate in Hz for binned data

    # Compute PSD for each unit, then average
    from scipy.signal import welch
    psds = []
    for i in range(m):
        x = spike_counts[i, :].astype(float)
        x = x - np.mean(x)  # Remove DC component
        freqs, pxx = welch(x, fs=fs, nperseg=min(1024, t_total // 2))
        psds.append(pxx)

    mean_psd = np.mean(psds, axis=0)

    # Filter to max frequency
    freq_mask = freqs <= max_freq_hz
    freqs_plot = freqs[freq_mask]
    psd_plot = mean_psd[freq_mask]

    ax.semilogy(
        freqs_plot, psd_plot,
        color=COLORS["purple"],
        linewidth=1.5,
    )

    # Shade typical neural oscillation bands
    ax.axvspan(4, 8, alpha=0.08, color=COLORS["blue"], label="θ (4-8 Hz)")
    ax.axvspan(8, 13, alpha=0.08, color=COLORS["green"], label="α (8-13 Hz)")
    ax.axvspan(30, 50, alpha=0.08, color=COLORS["orange"], label="γ (30-50 Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right")

    return fig


# =============================================================================
# Combined Data Summary (original 4-panel)
# =============================================================================

def plot_data_summary(
    spike_trains: Dict[str, np.ndarray],
    spike_counts: np.ndarray,
    metadata: Dict[str, Any],
    sampling_frequency: float = 30000.0,
    time_window: Tuple[float, float] = (0.0, 5.0),
    output_dir: Optional[Union[str, "Path"]] = None,
) -> plt.Figure:
    """
    Create a 4-panel data summary figure.

    Panels:
        1. Spike raster (zoomed to time_window)
        2. Binned count heatmap (zoomed to time_window)
        3. Firing rate histogram
        4. ISI distribution

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices.
        spike_counts: Binned count matrix, shape (M, T_total).
        metadata: Metadata dict from binning (needs firing_rates_hz, bin_width_ms).
        sampling_frequency: Sampling rate in Hz.
        time_window: (start_s, end_s) for zoomed views.
        output_dir: If provided, save the figure there.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Raster
    plot_raster(
        spike_trains, sampling_frequency,
        time_range=time_window,
        title="Spike Raster",
        ax=axes[0, 0],
    )

    # Panel 2: Binned counts
    plot_binned_counts_heatmap(
        spike_counts,
        bin_width_ms=metadata.get("bin_width_ms", 10.0),
        time_range=time_window,
        title="Binned Spike Counts",
        ax=axes[0, 1],
    )

    # Panel 3: Firing rates
    firing_rates = np.array(metadata.get("firing_rates_hz", []))
    if len(firing_rates) > 0:
        plot_firing_rate_histogram(
            firing_rates,
            title="Firing Rate Distribution",
            ax=axes[1, 0],
        )
    else:
        axes[1, 0].text(
            0.5, 0.5, "No firing rate data",
            transform=axes[1, 0].transAxes, ha="center",
        )

    # Panel 4: ISI distribution
    plot_isi_distribution(
        spike_trains, sampling_frequency,
        title="ISI Distribution",
        ax=axes[1, 1],
    )

    fig.suptitle("Data Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Save if output directory provided
    if output_dir is not None:
        save_figure(fig, "data_summary", output_dir, close=False)

    return fig


# =============================================================================
# Comprehensive Neuroscience-Grade Summary (8 panels)
# =============================================================================

def plot_neuro_summary(
    spike_trains: Dict[str, np.ndarray],
    spike_counts: np.ndarray,
    metadata: Dict[str, Any],
    sampling_frequency: float = 30000.0,
    time_window: Tuple[float, float] = (0.0, 5.0),
    output_dir: Optional[Union[str, "Path"]] = None,
) -> plt.Figure:
    """
    Create a comprehensive 8-panel neuroscience-grade data summary.

    Designed for "Turing test" evaluation: a neuroscientist should find
    these plots indistinguishable from real recording data.

    Panels (4×2 grid):
        Row 1: Spike raster (zoomed) | Binned count heatmap (zoomed)
        Row 2: Firing rate traces    | Firing rate histogram
        Row 3: Autocorrelation       | Cross-correlogram
        Row 4: ISI distribution      | Power spectral density

    Args:
        spike_trains: Dict mapping unit_id → spike sample indices.
        spike_counts: Binned count matrix, shape (M, T_total).
        metadata: Metadata dict (needs bin_width_ms, firing_rates_hz).
        sampling_frequency: Sampling rate in Hz.
        time_window: (start_s, end_s) for zoomed raster/heatmap views.
        output_dir: If provided, save the figure there.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    bin_width_ms = metadata.get("bin_width_ms", 10.0)

    # ------ Row 1: Raster + Heatmap (zoomed) ------
    plot_raster(
        spike_trains, sampling_frequency,
        time_range=time_window,
        title="Spike Raster",
        ax=axes[0, 0],
    )

    plot_binned_counts_heatmap(
        spike_counts,
        bin_width_ms=bin_width_ms,
        time_range=time_window,
        title="Binned Spike Counts",
        ax=axes[0, 1],
    )

    # ------ Row 2: Firing rate traces + histogram ------
    plot_firing_rate_traces(
        spike_counts,
        bin_width_ms=bin_width_ms,
        smooth_sigma_bins=10,
        num_units=5,
        time_range=(0.0, 30.0),  # Show first 30s for temporal trends
        title="Smoothed Firing Rates (30s)",
        ax=axes[1, 0],
    )

    firing_rates = np.array(metadata.get("firing_rates_hz", []))
    if len(firing_rates) > 0:
        plot_firing_rate_histogram(
            firing_rates,
            title="Firing Rate Distribution",
            ax=axes[1, 1],
        )
    else:
        # Compute from spike_counts if metadata lacks it
        duration_s = spike_counts.shape[1] * bin_width_ms / 1000.0
        rates = spike_counts.sum(axis=1) / duration_s
        plot_firing_rate_histogram(
            rates,
            title="Firing Rate Distribution",
            ax=axes[1, 1],
        )

    # ------ Row 3: Autocorrelation + Cross-correlogram ------
    plot_autocorrelation(
        spike_trains, sampling_frequency,
        max_lag_ms=50.0,
        bin_width_ms=0.5,
        title="Autocorrelation",
        ax=axes[2, 0],
    )

    plot_cross_correlogram(
        spike_trains, sampling_frequency,
        max_lag_ms=50.0,
        bin_width_ms=1.0,
        num_pairs=5,
        title="Cross-Correlogram",
        ax=axes[2, 1],
    )

    # ------ Row 4: ISI + PSD ------
    plot_isi_distribution(
        spike_trains, sampling_frequency,
        max_isi_ms=100.0,
        title="ISI Distribution",
        ax=axes[3, 0],
    )

    plot_psd(
        spike_counts,
        bin_width_ms=bin_width_ms,
        max_freq_hz=50.0,
        title="Power Spectral Density",
        ax=axes[3, 1],
    )

    # Title and layout
    fig.suptitle(
        "Neural Data Quality Assessment",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save if output directory provided
    if output_dir is not None:
        save_figure(fig, "neuro_summary", output_dir, close=False)

    return fig

