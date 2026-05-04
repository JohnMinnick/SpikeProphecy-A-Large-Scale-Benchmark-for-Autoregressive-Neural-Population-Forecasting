"""
Diagonal-SSM signal figure — Variant 3: Multi-metric convergence.

Four-panel dot-plot (one panel per population metric) showing that
HGRN2 and Mamba are nearly indistinguishable on the population metrics
even though their weighted r differs modestly. This is the "HGRN2
matches Mamba" claim taken seriously across all four eval axes.

Strongest for rigor; weaker as a 3-second grab than v1.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from figures.style import apply_style, save_figure, add_panel_label


# From Table 1 + HGRN2 pop_metrics eval (Steinmetz 39)
METRICS = {
    # metric_name -> {model: value}
    "Weighted $r$": {
        "Mamba":       0.522,
        "HGRN2":       0.493,
        "LRU v2":      0.485,
        "Transformer": 0.491,
        "LSTM":        0.454,
        "SNN (2L)":    0.477,
    },
    "$r_\\mathrm{pop}$": {
        "Mamba":       0.756,
        "HGRN2":       0.740,
        "LRU v2":      0.716,
        "Transformer": 0.744,
        "LSTM":        0.702,
        "SNN (2L)":    0.596,
    },
    "$r_\\mathrm{spatial}$": {
        "Mamba":       0.551,
        "HGRN2":       0.544,
        "LRU v2":      0.535,
        "Transformer": 0.543,
        "LSTM":        0.494,
        "SNN (2L)":    0.506,
    },
    "Cosine sim.": {
        "Mamba":       0.626,
        "HGRN2":       0.621,
        "LRU v2":      0.614,
        "Transformer": 0.620,
        "LSTM":        0.583,
        "SNN (2L)":    0.592,
    },
}

# Consistent ordering (best→worst-ish on Wt-r)
MODEL_ORDER = ["Mamba", "HGRN2", "LRU v2", "Transformer", "LSTM", "SNN (2L)"]
MODEL_CLASS = {
    "Mamba":       "Diagonal SSM",
    "HGRN2":       "Diagonal SSM",
    "LRU v2":      "Diagonal SSM",
    "Transformer": "Attention",
    "LSTM":        "Gated RNN",
    "SNN (2L)":    "Spiking",
}
CLASS_COLORS = {
    "Diagonal SSM": "#0072B2",
    "Attention":    "#D55E00",
    "Gated RNN":    "#E69F00",
    "Spiking":      "#009E73",
}


def generate():
    apply_style()

    fig = plt.figure(figsize=(6.5, 3.0))
    gs = gridspec.GridSpec(
        1, 4, wspace=0.55,
        left=0.07, right=0.97, top=0.88, bottom=0.30,
    )

    metric_names = list(METRICS.keys())
    for col, metric in enumerate(metric_names):
        ax = fig.add_subplot(gs[0, col])
        vals = METRICS[metric]

        # y-axis: models in shared order
        y = np.arange(len(MODEL_ORDER))
        x_vals = [vals[m] for m in MODEL_ORDER]
        colors = [CLASS_COLORS[MODEL_CLASS[m]] for m in MODEL_ORDER]

        # Background band over diagonal SSMs (top 3 positions)
        ax.axhspan(-0.5, 2.5, alpha=0.08,
                   color=CLASS_COLORS["Diagonal SSM"], zorder=0)

        # Connecting line (emphasizes ranking)
        ax.plot(x_vals, y, color="#888888", linewidth=0.5, zorder=1,
                alpha=0.5)

        # Dots
        ax.scatter(x_vals, y, s=90, c=colors,
                   edgecolor="#222222", linewidth=0.8, zorder=3)

        # y-tick labels only on leftmost panel
        if col == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(MODEL_ORDER, fontsize=7.5)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])

        ax.invert_yaxis()
        ax.set_xlabel(metric, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(col == 0)

        # Set sensible x-limits per metric
        lo, hi = min(x_vals), max(x_vals)
        pad = (hi - lo) * 0.12 + 0.002
        ax.set_xlim(lo - pad, hi + pad)

        # Panel label
        add_panel_label(ax, chr(ord("a") + col), x=-0.08 if col == 0 else -0.02, y=1.08)

    # Top caption bar — class legend above panels
    legend_ax = fig.add_axes([0.07, 0.94, 0.90, 0.04])
    legend_ax.axis("off")
    class_order = ["Diagonal SSM", "Attention", "Gated RNN", "Spiking"]
    handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=CLASS_COLORS[c],
                   markeredgecolor="#222", markersize=7,
                   label=c)
        for c in class_order
    ]
    legend_ax.legend(
        handles=handles, loc="center", ncol=4, frameon=False,
        fontsize=7.5, handletextpad=0.3, columnspacing=1.5,
    )

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_diag_ssm_v3_multimetric", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
