"""
Tests for src/viz/style.py and src/viz/data_plots.py

Validates that visualization functions produce correct figure objects,
save files in the expected formats, and handle edge cases.
Uses 'Agg' backend so tests run headlessly without a display.
"""

import json

import matplotlib
# Force non-interactive backend BEFORE any pyplot import
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz.style import (
    COLORS,
    COLOR_CYCLE,
    apply_style,
    reset_style,
    save_figure,
    figure_single_column,
    figure_full_width,
    figure_square,
)
from src.viz.data_plots import (
    plot_raster,
    plot_voltage_heatmap,
    plot_binned_counts_heatmap,
    plot_firing_rate_histogram,
    plot_isi_distribution,
    plot_data_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def sample_spike_trains():
    """Simple spike trains for 3 units, sampling at 30kHz."""
    rng = np.random.default_rng(42)
    return {
        "unit_0": np.sort(rng.integers(0, 300000, size=100)),  # ~10s at 30kHz
        "unit_1": np.sort(rng.integers(0, 300000, size=50)),
        "unit_2": np.sort(rng.integers(0, 300000, size=200)),
    }


@pytest.fixture
def sample_counts():
    """Binned spike counts: 5 units × 1000 bins."""
    rng = np.random.default_rng(42)
    return rng.poisson(lam=2, size=(5, 1000)).astype(np.int32)


@pytest.fixture
def sample_metadata():
    """Metadata dict matching binning output."""
    return {
        "bin_width_ms": 10.0,
        "num_units": 5,
        "num_bins": 1000,
        "total_spikes": 500,
        "firing_rates_hz": [2.1, 3.5, 1.8, 4.2, 2.7],
    }


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------

class TestStyle:
    """Tests for style.py."""

    def test_color_palette_has_minimum_colors(self):
        """Should have at least 6 named colors."""
        assert len(COLORS) >= 6
        assert len(COLOR_CYCLE) >= 6

    def test_apply_style_changes_rcparams(self):
        """apply_style should modify Matplotlib rcParams."""
        reset_style()
        original_size = plt.rcParams["font.size"]
        apply_style()
        # Our style sets font.size to 10
        assert plt.rcParams["font.size"] == 10.0
        assert plt.rcParams["axes.spines.top"] is False

    def test_reset_style_restores_defaults(self):
        """reset_style should restore Matplotlib defaults."""
        apply_style()
        reset_style()
        # Default spine setting is True
        assert plt.rcParams["axes.spines.top"] is True

    def test_save_figure_creates_files(self, tmp_path):
        """save_figure should create PNG and PDF files."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        saved = save_figure(fig, "test_plot", tmp_path)
        assert (tmp_path / "test_plot.png").exists()
        assert (tmp_path / "test_plot.pdf").exists()
        assert len(saved) == 2

    def test_save_figure_creates_subdirectory(self, tmp_path):
        """save_figure should create output directory if needed."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        subdir = tmp_path / "nested" / "plots"
        save_figure(fig, "deep_plot", subdir)
        assert (subdir / "deep_plot.png").exists()

    def test_figure_helpers_return_correct_sizes(self):
        """Figure helper functions should set correct dimensions."""
        fig1, _ = figure_single_column()
        assert fig1.get_size_inches()[0] == pytest.approx(6.0)
        plt.close(fig1)

        fig2, _ = figure_full_width()
        assert fig2.get_size_inches()[0] == pytest.approx(12.0)
        plt.close(fig2)

        fig3, _ = figure_square()
        assert fig3.get_size_inches()[0] == pytest.approx(5.0)
        assert fig3.get_size_inches()[1] == pytest.approx(5.0)
        plt.close(fig3)


# ---------------------------------------------------------------------------
# Data plot tests
# ---------------------------------------------------------------------------

class TestPlotRaster:
    """Tests for plot_raster()."""

    def test_returns_figure(self, sample_spike_trains):
        """Should return a valid Matplotlib Figure."""
        fig = plot_raster(sample_spike_trains)
        assert isinstance(fig, plt.Figure)

    def test_accepts_time_range(self, sample_spike_trains):
        """Should work with a specified time range."""
        fig = plot_raster(sample_spike_trains, time_range=(0, 2.0))
        ax = fig.axes[0]
        assert ax.get_xlim()[0] >= 0
        assert ax.get_xlim()[1] <= 2.5  # Allow slight padding

    def test_draws_on_existing_axes(self, sample_spike_trains):
        """Should draw into a provided axes."""
        fig, ax = plt.subplots()
        result = plot_raster(sample_spike_trains, ax=ax)
        assert result is fig  # Same figure returned


class TestPlotVoltageHeatmap:
    """Tests for plot_voltage_heatmap()."""

    def test_returns_figure(self):
        """Should return a valid Figure."""
        traces = np.random.default_rng(42).standard_normal((30000, 10))
        fig = plot_voltage_heatmap(traces, sampling_frequency=30000)
        assert isinstance(fig, plt.Figure)

    def test_with_time_range(self):
        """Should handle a time range subset."""
        traces = np.random.default_rng(42).standard_normal((90000, 5))
        fig = plot_voltage_heatmap(
            traces, sampling_frequency=30000, time_range=(0.5, 1.5),
        )
        assert isinstance(fig, plt.Figure)


class TestPlotBinnedCountsHeatmap:
    """Tests for plot_binned_counts_heatmap()."""

    def test_returns_figure(self, sample_counts):
        """Should return a valid Figure."""
        fig = plot_binned_counts_heatmap(sample_counts)
        assert isinstance(fig, plt.Figure)

    def test_with_time_range(self, sample_counts):
        """Should handle time range subsetting."""
        fig = plot_binned_counts_heatmap(
            sample_counts, bin_width_ms=10.0, time_range=(1.0, 3.0),
        )
        assert isinstance(fig, plt.Figure)


class TestPlotFiringRateHistogram:
    """Tests for plot_firing_rate_histogram()."""

    def test_returns_figure(self):
        """Should return a valid Figure."""
        rates = np.array([2.1, 3.5, 1.8, 4.2, 2.7, 6.0, 1.2])
        fig = plot_firing_rate_histogram(rates)
        assert isinstance(fig, plt.Figure)


class TestPlotISIDistribution:
    """Tests for plot_isi_distribution()."""

    def test_returns_figure(self, sample_spike_trains):
        """Should return a valid Figure."""
        fig = plot_isi_distribution(sample_spike_trains)
        assert isinstance(fig, plt.Figure)

    def test_handles_empty_spikes(self):
        """Should handle units with no spikes gracefully."""
        trains = {"empty": np.array([], dtype=np.int64)}
        fig = plot_isi_distribution(trains)
        assert isinstance(fig, plt.Figure)


class TestPlotDataSummary:
    """Tests for plot_data_summary()."""

    def test_returns_four_panel_figure(
        self, sample_spike_trains, sample_counts, sample_metadata,
    ):
        """Should produce a 2×2 figure with 4+ axes."""
        fig = plot_data_summary(
            sample_spike_trains, sample_counts, sample_metadata,
            time_window=(0, 2.0),
        )
        assert isinstance(fig, plt.Figure)
        # 4 panels + any colorbar axes
        assert len(fig.axes) >= 4

    def test_saves_to_directory(
        self, sample_spike_trains, sample_counts, sample_metadata, tmp_path,
    ):
        """Should save PNG and PDF when output_dir is provided."""
        fig = plot_data_summary(
            sample_spike_trains, sample_counts, sample_metadata,
            output_dir=tmp_path,
        )
        assert (tmp_path / "data_summary.png").exists()
        assert (tmp_path / "data_summary.pdf").exists()
        plt.close(fig)
