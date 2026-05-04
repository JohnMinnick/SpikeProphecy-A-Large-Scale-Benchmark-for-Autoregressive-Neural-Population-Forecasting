"""
NEW figure: Decomposition concept (the §1 contribution made visual).

Two-panel "before/after" view:
  (a) "Aggregate r alone": shows weighted r flat across 3 dataset
      scales — the standard reporting protocol.
  (b) "Decomposition reveals trade-off": same data, but split into
      pop_rate_r (climbing) and spatial_r (falling).

Reading goal: in 5 seconds, viewer understands "the standard metric
hides what we discovered, and our decomposition makes it visible."

This is the cleanest possible defense of the §1 primary contribution
("our metrics decompose into orthogonal axes"). Currently the paper
asserts this in prose; this figure makes it self-evident.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure, add_panel_label


SCALING = {
    "Steinmetz\n39 sess": {"wt_r": 0.522, "pop_r": 0.855, "spatial_r": 0.536},
    "IBL\n66 sess":       {"wt_r": 0.538, "pop_r": 0.924, "spatial_r": 0.406},
    "Combined\n105 sess": {"wt_r": 0.543, "pop_r": 0.929, "spatial_r": 0.373},
}


def generate():
    apply_style()

    labels = list(SCALING.keys())
    x = np.arange(len(labels))
    wt = [SCALING[l]["wt_r"] for l in labels]
    pop = [SCALING[l]["pop_r"] for l in labels]
    sp = [SCALING[l]["spatial_r"] for l in labels]

    fig = plt.figure(figsize=(6.5, 3.0))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.35,
        left=0.09, right=0.97, top=0.83, bottom=0.20,
    )

    # ----- Panel (a): Aggregate-only view -----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(x, wt, color="#444", linewidth=2.4, marker="o",
              markersize=10, markerfacecolor="white",
              markeredgewidth=1.8, zorder=3)
    # Value labels
    for xi, v in zip(x, wt):
        ax_a.text(xi, v + 0.012, f"{v:.3f}",
                  ha="center", fontsize=8.5, fontweight="bold",
                  color="#444")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=8)
    ax_a.set_ylabel("Aggregate weighted $r$")
    ax_a.set_ylim(0.40, 0.70)
    ax_a.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax_a.set_title("Standard reporting:\n+4% gain, looks flat",
                   fontsize=9, color="#444", pad=8)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_a.set_axisbelow(True)
    add_panel_label(ax_a, "a", x=-0.16, y=1.16)

    # ----- Panel (b): Decomposed view -----
    ax_b = fig.add_subplot(gs[0, 1])
    # Pop rate climbing
    ax_b.plot(x, pop, color="#0072B2", linewidth=2.4, marker="^",
              markersize=9, markeredgecolor="#222",
              markeredgewidth=0.7, zorder=4,
              label=f"$r_\\mathrm{{pop}}$ (temporal)")
    # Spatial dropping
    ax_b.plot(x, sp, color="#D55E00", linewidth=2.4, marker="v",
              markersize=9, markeredgecolor="#222",
              markeredgewidth=0.7, zorder=4,
              label=f"$r_\\mathrm{{spatial}}$ (spatial)")

    # Value labels
    for xi, v in zip(x, pop):
        ax_b.text(xi, v + 0.014, f"{v:.3f}",
                  ha="center", fontsize=7.5, color="#0072B2",
                  fontweight="bold")
    for xi, v in zip(x, sp):
        ax_b.text(xi, v - 0.024, f"{v:.3f}",
                  ha="center", fontsize=7.5, color="#D55E00",
                  fontweight="bold")

    # Direction-of-change annotations on the right edge
    ax_b.annotate(
        f"+{(pop[-1] / pop[0] - 1) * 100:.0f}%",
        xy=(2.05, pop[-1]), fontsize=10,
        color="#0072B2", fontweight="bold", va="center", ha="left",
    )
    ax_b.annotate(
        f"{(sp[-1] / sp[0] - 1) * 100:.0f}%",
        xy=(2.05, sp[-1]), fontsize=10,
        color="#D55E00", fontweight="bold", va="center", ha="left",
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_ylim(0.30, 1.00)
    ax_b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b.set_title(
        "Decomposed:\ntemporal $\\uparrow$, spatial $\\downarrow$",
        fontsize=9, color="#222", pad=8,
    )
    ax_b.legend(fontsize=7.5, loc="center left",
                frameon=False, handlelength=1.6,
                handletextpad=0.4, borderaxespad=0.4)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_b.set_axisbelow(True)
    add_panel_label(ax_b, "b", x=-0.10, y=1.16)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_new_decomp_concept", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
