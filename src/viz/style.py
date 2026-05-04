"""
Publication-quality style configuration for SpikeProphecy plots.

Sets consistent Matplotlib defaults for all visualizations:
- Fonts, sizes, and spacing tuned for single-column journal figures
- Colorblind-safe colormap defaults
- Export utilities for 300 DPI PNG + vector PDF

Usage:
    from src.viz.style import apply_style, save_figure, COLORS

    apply_style()  # Call once at the start of a plotting session
    fig, ax = plt.subplots()
    # ... plot ...
    save_figure(fig, "my_figure", output_dir="experiments/2026-02-12_run/plots")
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# =============================================================================
# Color palette (colorblind-safe, publication-friendly)
# =============================================================================

# 8-color qualitative palette (Wong, 2011 — widely used in neuroscience)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Ordered list for cycling through plot colors
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["cyan"],
]

# Default colormaps
CMAP_SEQUENTIAL = "viridis"        # For heatmaps, firing rates
CMAP_DIVERGING = "RdBu_r"         # For error maps, correlations
CMAP_SPIKE = "hot"                # For raster/spike density

# =============================================================================
# Style parameters
# =============================================================================

# Publication-ready defaults
STYLE_PARAMS = {
    # Figure
    "figure.figsize": (6, 4),          # Single-column default
    "figure.dpi": 150,                 # Screen display DPI
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "figure.autolayout": True,

    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,

    # Axes
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),

    # Ticks
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.fontsize": 8,
    "legend.frameon": False,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # Savefig
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
}


def apply_style():
    """
    Apply the publication-quality Matplotlib style.

    Call this once at the start of any plotting session.
    """
    mpl.rcParams.update(STYLE_PARAMS)
    logger.debug("Applied SpikeProphecy publication style")


def reset_style():
    """Reset Matplotlib to default style."""
    mpl.rcdefaults()
    logger.debug("Reset Matplotlib to defaults")


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Union[str, Path] = ".",
    formats: tuple = ("png", "pdf"),
    dpi: int = 300,
    close: bool = True,
) -> list:
    """
    Save a figure in multiple formats (PNG + PDF by default).

    Args:
        fig: Matplotlib Figure object.
        name: Base filename (without extension).
        output_dir: Directory to save into.
        formats: Tuple of file formats to save.
        dpi: Resolution for raster formats.
        close: If True, close the figure after saving to free memory.

    Returns:
        List of saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        filepath = out_path / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=dpi)
        saved.append(filepath)
        logger.info("Saved figure: %s", filepath)

    if close:
        plt.close(fig)

    return saved


def figure_single_column(**kwargs):
    """Create a single-column figure (6 × 4 inches)."""
    defaults = {"figsize": (6, 4)}
    defaults.update(kwargs)
    return plt.subplots(**defaults)


def figure_full_width(**kwargs):
    """Create a full-width figure (12 × 4 inches)."""
    defaults = {"figsize": (12, 4)}
    defaults.update(kwargs)
    return plt.subplots(**defaults)


def figure_square(**kwargs):
    """Create a square figure (5 × 5 inches)."""
    defaults = {"figsize": (5, 5)}
    defaults.update(kwargs)
    return plt.subplots(**defaults)
