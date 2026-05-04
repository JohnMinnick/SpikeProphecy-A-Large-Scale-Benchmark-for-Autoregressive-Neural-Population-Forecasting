"""
Figure 3 panel (a) — dot plot with error-bar-like range indicator.

Minimal, publication-grade. Each region is a row; raw + adjusted r
are two points connected by a line showing the ANCOVA shift. The
direction and magnitude of the shift is visually explicit.

Regions ordered by adjusted r. Makes the 'Motor Cortex and Midbrain
are most predictable post-correction' finding immediate.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, COLORS, save_figure


def generate():
    apply_style()

    from figures.data import REGION_ORDER, REGION_DATA

    items = [(r, REGION_DATA[r]["raw"], REGION_DATA[r]["adjusted"],
              REGION_DATA[r]["n"]) for r in REGION_ORDER]
    items.sort(key=lambda t: t[2])  # ascending so best adjusted at top
    names = [t[0].replace("\n", " ") for t in items]
    raw_vals = [t[1] for t in items]
    adj_vals = [t[2] for t in items]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.18)

    y = np.arange(len(names))
    # Draw connecting lines from raw to adjusted
    for yi, r, a in zip(y, raw_vals, adj_vals):
        ax.plot([r, a], [yi, yi], color="#AAAAAA", linewidth=1.2,
                zorder=1, solid_capstyle="round")

    # Raw points (orange, open)
    ax.scatter(raw_vals, y, s=60, color="white",
               edgecolor=COLORS["raw"], linewidth=1.5,
               zorder=2, label="Raw mean")
    # Adjusted points (blue, filled)
    ax.scatter(adj_vals, y, s=65, color=COLORS["adjusted"],
               edgecolor="#222", linewidth=0.7,
               zorder=3, label="ANCOVA-adjusted")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean per-neuron Pearson $r$")
    x_min = min(min(raw_vals), min(adj_vals)) - 0.01
    x_max = max(max(raw_vals), max(adj_vals)) + 0.02
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    ax.legend(fontsize=7.5, loc="lower right", frameon=False,
              handletextpad=0.3, borderaxespad=0.2)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure3_panela_v2_dotplot", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
