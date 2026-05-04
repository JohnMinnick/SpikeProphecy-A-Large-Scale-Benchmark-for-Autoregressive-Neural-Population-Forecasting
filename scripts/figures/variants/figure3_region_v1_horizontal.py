"""
Figure 3 panel (a) — horizontal bar variant.

Each region gets its own row with paired raw + ANCOVA-adjusted r
values side-by-side. Eliminates the rotated-label problem of the
vertical layout and reads more like a leaderboard.

Design principles:
  - Horizontal orientation — long region names fit cleanly
  - Raw shown as thin background bar, adjusted overlaid — visual diff
    between the two immediately readable
  - Regions sorted by ADJUSTED r (the statistic that actually matters
    after ANCOVA correction)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, COLORS, add_panel_label, save_figure
from figures.data import REGION_ORDER, REGION_DATA


def generate():
    apply_style()

    # Sort by ANCOVA-adjusted r (not raw)
    items = [(r, REGION_DATA[r]["raw"], REGION_DATA[r]["adjusted"],
              REGION_DATA[r]["n"]) for r in REGION_ORDER]
    items.sort(key=lambda t: t[2])  # ascending so best is at top
    names = [t[0].replace("\n", " ") for t in items]
    raw_vals = [t[1] for t in items]
    adj_vals = [t[2] for t in items]
    ns = [t[3] for t in items]

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.18)

    y = np.arange(len(names))
    bar_h = 0.38
    # Raw (lighter background bars)
    ax.barh(y + bar_h / 2, raw_vals, height=bar_h,
            color=COLORS["raw"], alpha=0.85,
            edgecolor="white", linewidth=0.4,
            label="Raw mean", zorder=2)
    # Adjusted (foreground, on top)
    ax.barh(y - bar_h / 2, adj_vals, height=bar_h,
            color=COLORS["adjusted"], alpha=0.9,
            edgecolor="white", linewidth=0.4,
            label="ANCOVA-adjusted", zorder=2)

    # Annotate n-neurons alongside region names
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n}  \\small ({nn:,}n)" for n, nn in zip(names, ns)],
        fontsize=7.5,
    )
    # Alternative: just region names, put n as separate annotation
    ax.set_yticklabels(names, fontsize=7.5)

    # Value labels at the end of each bar
    for yi, v_raw, v_adj, n in zip(y, raw_vals, adj_vals, ns):
        ax.text(max(v_raw, v_adj) + 0.003, yi + bar_h / 2,
                f"{v_raw:.3f}", va="center", ha="left",
                fontsize=6.5, color=COLORS["raw"])
        ax.text(max(v_raw, v_adj) + 0.003, yi - bar_h / 2,
                f"{v_adj:.3f}", va="center", ha="left",
                fontsize=6.5, color=COLORS["adjusted"])

    ax.set_xlabel("Mean per-neuron Pearson $r$")
    ax.set_xlim(0, max(max(raw_vals), max(adj_vals)) * 1.20)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    ax.legend(fontsize=7, loc="lower right", frameon=False)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure3_panela_v1_horizontal", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
