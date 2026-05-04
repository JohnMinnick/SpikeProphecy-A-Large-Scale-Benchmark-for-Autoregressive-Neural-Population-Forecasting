"""
Architecture clustering figure — all 7 baselines, grouped by structural class.

Single-panel grouped bar chart showing the paper's architectural
finding: the five modern-recurrence architectures (SSMs diagonal and
non-diagonal, Transformer) cluster tightly in a top tier
($r{=}0.485\text{--}0.522$); classical LSTM and the depth-matched
SNN fall meaningfully below. The diagonal vs non-diagonal distinction
doesn't cleanly separate performance within the top tier.

Design:
  - ONE message per figure: top-tier clustering + classical-RNN gap
  - Color-codes by STRUCTURAL CLASS, not identity
  - Shaded band highlights the modern-recurrence cluster
  - Wong colorblind-safe palette + hatching for redundancy
  - Mamba highlighted as reference (top of cluster)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from figures.style import apply_style, save_figure, add_panel_label, COLORS


# Table 1 weighted r values (Steinmetz 39 sessions, 3-layer models).
# Verified 2026-04-23 from each run's training metrics.json on S3.
#
# Color encodes the *only* tier distinction the paper claims:
# modern-recurrence cluster vs. classical RNN vs. spiking.  We do not
# color-code diagonal vs. non-diagonal SSMs because the paper does
# not claim that distinction is load-bearing (Appendix~A.6).
MODELS = [
    # (name, wt_r, params_M, class)
    ("Mamba",         0.500, 1.95, "Modern recurrence"),
    ("HGRN2",         0.493, 1.82, "Modern recurrence"),
    ("Transformer",   0.492, 2.22, "Modern recurrence"),
    ("GatedDeltaNet", 0.485, 1.43, "Modern recurrence"),
    ("LRU",           0.480, 1.23, "Modern recurrence"),
    ("LSTM",          0.441, 2.22, "Classical RNN"),
    ("SNN (3L)",      0.430, 0.97, "Spiking"),
]

# Order: top cluster first (ranked by Wt-r), then trailing baselines
ORDER = ["Mamba", "HGRN2", "Transformer", "GatedDeltaNet", "LRU", "LSTM", "SNN (3L)"]

CLASS_COLORS = {
    "Modern recurrence": "#0072B2",   # blue (Wong)
    "Classical RNN":     "#E69F00",   # orange (Wong)
    "Spiking":           "#009E73",   # bluish green (Wong)
}
# Hatching for colorblind redundancy
CLASS_HATCH = {
    "Modern recurrence": "",
    "Classical RNN":     "\\\\",
    "Spiking":           "xx",
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

    # Shaded band highlighting the top-tier cluster (all modern
    # recurrence: diagonal + non-diagonal SSMs + attention)
    top_tier_names = {"Mamba", "HGRN2", "Transformer",
                      "GatedDeltaNet", "LRU"}
    top_tier_rs = [r[1] for r in ordered if r[0] in top_tier_names]
    ax.axhspan(min(top_tier_rs) - 0.003, max(top_tier_rs) + 0.003,
               alpha=0.08, color="#555555", zorder=1)
    # Annotation: modern recurrence top tier
    ax.text(
        2, 0.513, "Modern recurrence cluster",
        fontsize=7.5, color="#555555",
        fontweight="bold", ha="center", style="italic",
    )

    # Model labels on x-axis — wrap GatedDeltaNet to 2 lines so it
    # doesn't collide with Transformer at paper-width
    display_map = {"GatedDeltaNet": "Gated\nDeltaNet"}
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_map.get(r[0], r[0]) for r in ordered],
        fontsize=8, rotation=0,
    )

    # y-axis — floor dropped below SNN (0.430) so its bar is still
    # visibly shaded rather than collapsing to a sliver
    ax.set_ylabel("Weighted Pearson $r$ (val)")
    ax.set_ylim(0.41, 0.52)
    ax.set_yticks([0.42, 0.44, 0.46, 0.48, 0.50, 0.52])

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
    class_order = ["Modern recurrence", "Classical RNN", "Spiking"]
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

    # Save to main figures dir (canonical version referenced from main.tex)
    save_figure(fig, "figure_arch_clustering")
    plt.close(fig)


if __name__ == "__main__":
    generate()
