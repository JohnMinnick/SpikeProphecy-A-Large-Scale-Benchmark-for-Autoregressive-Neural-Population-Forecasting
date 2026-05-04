"""
Diagonal-SSM signal figure — Variant 1: Grouped bars.

Single-panel grouped bar chart that makes the paper's narrative headline
visible at a glance: "All three diagonal SSMs (Mamba, HGRN2, LRU) cluster
above the non-diagonal baselines."

Design principles applied:
  - ONE message per figure (takeaway visible in <3 sec)
  - Color-codes by STRUCTURAL CLASS, not identity
  - Dashed band shows diagonal-SSM performance cluster
  - Wong colorblind-safe palette + hatching for redundancy
  - Mamba highlighted as reference (the established SOTA)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from figures.style import apply_style, save_figure, add_panel_label, COLORS


# Table 1 weighted r values (Steinmetz 39 sessions)
MODELS = [
    # (name, wt_r, params_M, class, mamba_ratio)
    ("Mamba",       0.522, 1.95, "Diagonal SSM"),
    ("HGRN2",       0.493, 1.82, "Diagonal SSM"),
    ("LRU v2",      0.485, 1.23, "Diagonal SSM"),
    ("Transformer", 0.491, 2.22, "Attention"),
    ("LSTM",        0.454, 2.22, "Gated RNN"),
    ("SNN (2L)",    0.477, 0.83, "Spiking"),
]

# Order: diagonal SSMs first (top cluster), then non-diagonal
ORDER = ["Mamba", "HGRN2", "LRU v2", "Transformer", "LSTM", "SNN (2L)"]

# Color by structural class — semantic, not decorative
CLASS_COLORS = {
    "Diagonal SSM": "#0072B2",    # blue (Wong)
    "Attention":    "#D55E00",    # vermillion (Wong)
    "Gated RNN":    "#E69F00",    # orange (Wong)
    "Spiking":      "#009E73",    # bluish green (Wong)
}
# Hatching for redundancy (colorblind backup)
CLASS_HATCH = {
    "Diagonal SSM": "",
    "Attention":    "//",
    "Gated RNN":    "\\\\",
    "Spiking":      "xx",
}


def generate():
    apply_style()

    name_to_row = {m[0]: m for m in MODELS}
    ordered = [name_to_row[n] for n in ORDER]

    fig = plt.figure(figsize=(6.5, 3.2))
    gs = gridspec.GridSpec(1, 1, left=0.08, right=0.78,
                           top=0.90, bottom=0.20)
    ax = fig.add_subplot(gs[0, 0])

    x = np.arange(len(ordered))
    heights = [r[1] for r in ordered]
    classes = [r[3] for r in ordered]
    colors = [CLASS_COLORS[c] for c in classes]
    hatches = [CLASS_HATCH[c] for c in classes]

    bars = ax.bar(
        x, heights, width=0.65,
        color=colors, edgecolor="#333333", linewidth=0.9,
        zorder=3,
    )
    # Apply hatching per-bar
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    # Reference line: Mamba's performance
    mamba_r = name_to_row["Mamba"][1]
    ax.axhline(mamba_r, color="#333333", linewidth=0.7, linestyle=":",
               zorder=2)
    ax.text(
        len(ordered) - 0.4, mamba_r + 0.004,
        f"Mamba r = {mamba_r:.3f}",
        ha="right", va="bottom", fontsize=7, color="#333333",
    )

    # Shaded band for diagonal-SSM cluster
    diag_rs = [r[1] for r in ordered if r[3] == "Diagonal SSM"]
    ax.axhspan(min(diag_rs) - 0.003, max(diag_rs) + 0.003,
               alpha=0.10, color=CLASS_COLORS["Diagonal SSM"],
               zorder=1)
    # Arrow annotation showing the cluster
    ax.annotate(
        "Diagonal\nSSMs", xy=(1, max(diag_rs)), xytext=(1.5, 0.545),
        fontsize=7.5, color=CLASS_COLORS["Diagonal SSM"],
        fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="-", color=CLASS_COLORS["Diagonal SSM"],
                        linewidth=0.6, alpha=0.7),
    )

    # Model labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in ordered], fontsize=8, rotation=0)

    # y-axis
    ax.set_ylabel("Weighted Pearson $r$ (val)")
    ax.set_ylim(0.43, 0.57)
    ax.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56])

    # Value labels on top of bars (makes numbers readable)
    for b, h_val in zip(bars, heights):
        ax.text(
            b.get_x() + b.get_width() / 2, h_val + 0.003,
            f"{h_val:.3f}",
            ha="center", va="bottom", fontsize=7, color="#333333",
        )

    # Remove top/right spines; clean grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

    # Legend outside right — class-level
    class_order = ["Diagonal SSM", "Attention", "Gated RNN", "Spiking"]
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=CLASS_COLORS[c], hatch=CLASS_HATCH[c],
                      edgecolor="#333333", linewidth=0.7)
        for c in class_order
    ]
    ax.legend(
        handles, class_order,
        title="Architecture class",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7, title_fontsize=7.5, frameon=False,
        borderaxespad=0.0, handlelength=1.5, handletextpad=0.5,
    )

    # Hide panel label (single-panel figure)
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_diag_ssm_v1_grouped_bars", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
