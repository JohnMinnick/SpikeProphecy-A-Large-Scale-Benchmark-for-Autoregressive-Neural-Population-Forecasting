"""
Generate top-tier multi-panel figures for the SpikeProphecy NeurIPS E&D paper.

Composes publication-quality figures from existing experiment assets using
matplotlib gridspec. Matches NeurIPS text width (6.5in) with consistent
fonts, colors, and panel labels.

Outputs:
  - figure1_benchmark_overview.png/pdf   (Hero: raster + forecast + pop rate)
  - figure2_evaluation_protocol.png/pdf  (Pareto + radar + ceiling bars)
  - figure3_findings.png/pdf             (Region hierarchy + Fano + shuffle)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Source figure directory
SRC = Path(r"C:\AGCoding\SpikeProphecy\docs\paper\figures")
# Output directory
OUT = Path(r"C:\AGCoding\SpikeProphecy\docs\neurips_ed\figures")

# NeurIPS text width constraints
TEXT_WIDTH = 6.5  # inches (NeurIPS full text width)

# Consistent style for all figures
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Panel label style (bold, uppercase, top-left corner)
PANEL_LABEL_PROPS = dict(
    fontsize=14,
    fontweight='bold',
    fontfamily='sans-serif',
    va='top',
    ha='left',
)


def add_panel_label(ax, label, x=-0.05, y=1.05):
    """Add a bold panel label (a), (b), (c) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes, **PANEL_LABEL_PROPS)


def load_img(name):
    """Load an image from the source figures directory."""
    path = SRC / name
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return mpimg.imread(str(path))


def save_figure(fig, name):
    """Save figure as both PNG and PDF."""
    png_path = OUT / f"{name}.png"
    pdf_path = OUT / f"{name}.pdf"
    fig.savefig(str(png_path), dpi=300, facecolor='white', edgecolor='none')
    fig.savefig(str(pdf_path), facecolor='white', edgecolor='none')
    print(f"  Saved: {png_path.name} + {pdf_path.name}")


# =============================================================================
# FIGURE 1: Benchmark Overview (Hero)
# =============================================================================
# Layout:  (a) raster_5s  |  (b) GT vs Predicted heatmap
#          (c) population rate trace (full width)
# =============================================================================
def make_figure1():
    """Generate the hero benchmark overview figure."""
    print("Generating Figure 1: Benchmark Overview...")

    # Load source images
    img_raster = load_img("raster_5s.png")
    img_forecast = load_img("population_forecast.png")

    # The population_forecast image has two panels stacked vertically:
    # top = heatmaps (GT vs Predicted), bottom = population rate trace
    # Split them: top ~65% is heatmaps, bottom ~35% is pop rate
    h = img_forecast.shape[0]
    split_row = int(h * 0.62)
    img_heatmap = img_forecast[:split_row, :, :]
    img_poprate = img_forecast[split_row:, :, :]

    # Create the figure with tight layout
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1.0],
                           hspace=0.12, wspace=0.06)

    # Panel (a): Raw spike raster — shows what raw neural data looks like
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_raster, aspect='auto')
    ax_a.set_axis_off()
    add_panel_label(ax_a, 'a', x=-0.02, y=1.02)

    # Panel (b): GT vs Predicted heatmap — shows what the model produces
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_heatmap, aspect='auto')
    ax_b.set_axis_off()
    add_panel_label(ax_b, 'b', x=-0.02, y=1.02)

    # Panel (c): Population rate trace — GT vs Teacher vs SNN (full width)
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.imshow(img_poprate, aspect='auto')
    ax_c.set_axis_off()
    add_panel_label(ax_c, 'c', x=-0.01, y=1.02)

    save_figure(fig, "figure1_benchmark_overview")
    plt.close(fig)


# =============================================================================
# FIGURE 2: Evaluation Protocol in Action
# =============================================================================
# Layout:  (a) Pareto frontier  |  (b) Radar plot  |  (c) Ceiling bars
# =============================================================================
def make_figure2():
    """Generate the evaluation protocol figure."""
    print("Generating Figure 2: Evaluation Protocol...")

    # Load source images
    img_multipanel = load_img("architecture_multipanel.png")
    img_radar = mpimg.imread(str(OUT / "radar_benchmark.png"))

    # Split the architecture_multipanel into 3 panels
    # It's arranged as: (a) Pareto | (b) Bar chart | (c) Ceiling bars
    w = img_multipanel.shape[1]
    panel_w = w // 3
    img_pareto = img_multipanel[:, :panel_w, :]
    img_ceiling = img_multipanel[:, 2*panel_w:, :]

    # Create the figure
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 0.8],
                           wspace=0.08)

    # Panel (a): Pareto frontier
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_pareto, aspect='auto')
    ax_a.set_axis_off()
    add_panel_label(ax_a, 'a')

    # Panel (b): Radar plot
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_radar, aspect='auto')
    ax_b.set_axis_off()
    add_panel_label(ax_b, 'b')

    # Panel (c): Ceiling efficiency bars
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(img_ceiling, aspect='auto')
    ax_c.set_axis_off()
    add_panel_label(ax_c, 'c')

    save_figure(fig, "figure2_evaluation_protocol")
    plt.close(fig)


# =============================================================================
# FIGURE 3: Findings Enabled by the Benchmark
# =============================================================================
# Layout:  (a) Brain region hierarchy (ANCOVA bars)
#          (b) Fano factor stratification
#          (c) Population shuffle scatter
# =============================================================================
def make_figure3():
    """Generate the findings figure."""
    print("Generating Figure 3: Findings...")

    # Load source images
    img_hierarchy = load_img("kosmos_region_hierarchy.png")
    img_fano = load_img("kosmos_architecture_by_fano.png")
    img_shuffle = load_img("population_shuffle_aggregate.png")

    # Extract panel B (ANCOVA-adjusted bars) from the hierarchy image
    # The hierarchy image has two panels side by side: A (boxplots) | B (bars)
    # Add small left margin to avoid clipping the title
    w_h = img_hierarchy.shape[1]
    img_ancova = img_hierarchy[10:, int(w_h*0.48):, :]

    # Extract bottom-right panel from shuffle (Original vs Shuffled scatter)
    h_s = img_shuffle.shape[0]
    w_s = img_shuffle.shape[1]
    img_scatter = img_shuffle[h_s//2:, w_s//2:, :]

    # Create the figure — slightly taller to give panels breathing room
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.3, 0.9],
                           wspace=0.06)

    # Panel (a): ANCOVA-adjusted brain region bars
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_ancova, aspect='auto')
    ax_a.set_axis_off()
    add_panel_label(ax_a, 'a')

    # Panel (b): Fano factor stratification
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_fano, aspect='auto')
    ax_b.set_axis_off()
    add_panel_label(ax_b, 'b')

    # Panel (c): Shuffle scatter (r_original vs r_shuffled)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(img_scatter, aspect='auto')
    ax_c.set_axis_off()
    add_panel_label(ax_c, 'c')

    save_figure(fig, "figure3_findings")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Source: {SRC}")
    print(f"Output: {OUT}")
    print()

    make_figure1()
    make_figure2()
    make_figure3()

    print()
    print("All figures generated successfully.")
