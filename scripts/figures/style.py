"""
Shared figure style for SpikeProphecy NeurIPS paper.

Provides a unified visual identity across all paper figures:
  - Curated 5-architecture color palette
  - NeurIPS-quality typography (sans-serif, 8pt base)
  - L-shaped despined axes
  - Consistent panel labeling (bold lowercase: a., b., c.)
  - PDF + PNG dual export at 300 DPI

Usage:
    from figures.style import apply_style, COLORS, MARKERS, add_panel_label, save_figure
    apply_style()  # Call once at script start
"""

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Color palette — Wong (2011) colorblind-safe palette
# Ref: https://www.nature.com/articles/nmeth.1618
# Distinguishable under protanopia, deuteranopia, and tritanopia
# =============================================================================
COLORS = {
    # Primary architectures (6 models) — Wong palette selections.
    # Diagonal SSMs share the blue family; attention is reddish-purple;
    # gated RNN (LSTM) orange; spiking green.
    'Mamba':       '#D55E00',   # vermillion (warm, high-contrast)
    'HGRN2':       '#3a6db5',   # steel blue (diagonal SSM family)
    'Transformer': '#CC79A7',   # reddish purple
    'LRU':         '#0072B2',   # blue (diagonal SSM)
    'GatedDelta':  '#882255',   # dark magenta (non-diagonal SSM control)
    'LSTM':        '#E69F00',   # orange
    'SNN':         '#009E73',   # bluish green
    # Baselines & accents
    'GLM':         '#999999',   # grey (naive baseline)
    'Oracle':      '#BBBBBB',   # light grey (ceiling)
    'Ground Truth':'#444444',   # dark grey
    # Semantic accents
    'raw':         '#E69F00',   # orange for raw/unadjusted
    'adjusted':    '#0072B2',   # blue for adjusted
    'highlight':   '#F0E442',   # yellow for callouts
    'grid':        '#E8E8E8',   # very light grey
}

# Marker shapes per architecture
MARKERS = {
    'Mamba':       'D',    # diamond
    'HGRN2':       'P',    # plus (filled) — SNN's old shape; SNN gets new
    'Transformer': '^',    # triangle up
    'LRU':         's',    # square
    'GatedDelta':  'v',    # triangle down
    'LSTM':        'o',    # circle
    'SNN':         'X',    # X marker
}

# Short labels for tight spaces
SHORT_LABELS = {
    'Mamba': 'Mamba',
    'HGRN2': 'HGRN2',
    'Transformer': 'Xfmr',
    'LRU': 'LRU',
    'GatedDelta': 'GD',
    'LSTM': 'LSTM',
    'SNN': 'SNN',
}

# Ordered list (performance ranking on Steinmetz 39)
MODEL_ORDER = ['Mamba', 'HGRN2', 'Transformer', 'LRU', 'LSTM', 'SNN']


def apply_style():
    """
    Apply the unified NeurIPS figure style globally.

    Call once at the top of each figure script before creating figures.
    Sets matplotlib rcParams for fonts, axes, ticks, and output quality.
    """
    matplotlib.rcParams.update({
        # ---- Font (clean sans-serif) ----
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7,
        'legend.frameon': False,
        # ---- Axes: L-shaped, clean ----
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        # ---- Text color ----
        'text.color': '#333333',
        # ---- Output quality ----
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.transparent': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        # ---- Lines ----
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
    })


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """
    Add a bold lowercase panel label in NeurIPS style.

    Args:
        ax: matplotlib Axes
        label: Single character (e.g. 'a', 'b', 'c')
        x: Fractional x position in axes coordinates
        y: Fractional y position in axes coordinates
    """
    ax.text(x, y, f'{label}.',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            fontfamily='sans-serif',
            color='#222222',
            va='top', ha='left')


def despine(ax, left=True, bottom=True):
    """
    Remove top and right spines. Optionally keep/remove left and bottom.

    Args:
        ax: matplotlib Axes
        left: If True, keep left spine
        bottom: If True, keep bottom spine
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def save_figure(fig, name, out_dir=None):
    """
    Save figure as both PNG (300 DPI) and PDF.

    Args:
        fig: matplotlib Figure
        name: Base filename without extension
        out_dir: Output directory (default: docs/neurips_ed/figures/)
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / 'docs' / 'neurips_ed' / 'figures'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f'{name}.png'
    pdf_path = out_dir / f'{name}.pdf'

    fig.savefig(str(png_path), dpi=300, facecolor='white', edgecolor='none')
    fig.savefig(str(pdf_path), facecolor='white', edgecolor='none')

    print(f'  Saved: {png_path.name}')
    print(f'  Saved: {pdf_path.name}')


# NeurIPS text width for full-width figures
TEXT_WIDTH = 6.5   # inches
COLUMN_WIDTH = 3.25  # inches (for single-column appendix figures)
