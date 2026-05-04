"""
Decomposition schematic (NEW, content audit).

Illustrates §3.3's primary contribution: the population metric
decomposition. Currently the paper introduces pop_rate_r / spatial_r /
cosine_sim via equations but never shows WHY decomposing matters
visually.

Two-panel argument:
  (a) Aggregate view — a single bar "Weighted r = 0.52" — what the
      standard protocol reports. Looks fine, but uninformative.
  (b) Decomposed view — the same model's performance on three
      orthogonal axes. Now you see that temporal fidelity is near
      ceiling (0.76), spatial identity is moderate (0.55), cosine is
      high (0.63) — operationally different verdicts for different
      downstream use cases.

The takeaway: "aggregate r loses information the decomposition
recovers." This is the §1 primary-contribution claim as a picture.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure, add_panel_label, TEXT_WIDTH


def generate():
    apply_style()

    # Mamba numbers from Table 1 (Steinmetz 39)
    MAMBA = {
        "Weighted $r$": 0.522,
        "$r_\\mathrm{pop}$": 0.756,
        "$r_\\mathrm{spatial}$": 0.551,
        "Cosine sim.": 0.626,
    }

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.8))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[0.55, 1.45], wspace=0.20,
        left=0.07, right=0.97, top=0.86, bottom=0.22,
    )

    # ---- Panel (a): Aggregate view ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(
        [0], [MAMBA["Weighted $r$"]], width=0.5,
        color="#555555", edgecolor="#222", linewidth=0.7,
    )
    ax_a.text(0, MAMBA["Weighted $r$"] + 0.02,
              f"{MAMBA['Weighted $r$']:.3f}",
              ha="center", va="bottom",
              fontsize=9, fontweight="bold", color="#333333")
    ax_a.set_xticks([0])
    ax_a.set_xticklabels(["Weighted $r$"], fontsize=9)
    ax_a.set_ylabel("Metric value")
    ax_a.set_ylim(0, 0.90)
    ax_a.set_xlim(-0.7, 0.7)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_a.set_axisbelow(True)
    ax_a.set_title("Standard protocol:\none scalar", fontsize=8.5,
                   pad=6, color="#555555")
    add_panel_label(ax_a, "a", x=-0.25, y=1.18)

    # ---- Panel (b): Decomposed view ----
    ax_b = fig.add_subplot(gs[0, 1])
    metrics_order = ["$r_\\mathrm{pop}$",
                     "$r_\\mathrm{spatial}$",
                     "Cosine sim."]
    vals = [MAMBA[m] for m in metrics_order]
    # Include weighted r as faded reference for comparison
    ax_b.bar(
        [-1], [MAMBA["Weighted $r$"]], width=0.5,
        color="none", edgecolor="#555555", linewidth=1.0,
        linestyle="--", hatch="",
    )
    ax_b.text(-1, MAMBA["Weighted $r$"] + 0.02,
              f"{MAMBA['Weighted $r$']:.3f}",
              ha="center", va="bottom", fontsize=8, color="#555555",
              style="italic")
    ax_b.text(-1, 0.02, "Wt $r$\n(ref)", ha="center", va="bottom",
              fontsize=7, color="#888888", style="italic")

    palette = ["#0072B2", "#D55E00", "#CC79A7"]
    x_pos = np.arange(len(metrics_order))
    bars = ax_b.bar(
        x_pos, vals, width=0.6, color=palette,
        edgecolor="#222", linewidth=0.7,
    )
    for xi, v, c in zip(x_pos, vals, palette):
        ax_b.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                  fontsize=9, fontweight="bold", color=c)

    # Interpretation labels under each bar
    interpretations = [
        "temporal\nfidelity",
        "spatial\npattern",
        "magnitude-\ninvariant",
    ]
    for xi, interp, c in zip(x_pos, interpretations, palette):
        ax_b.text(xi, -0.07, interp, ha="center", va="top",
                  fontsize=7, color=c, style="italic",
                  transform=ax_b.get_xaxis_transform())

    ax_b.set_xticks(list(x_pos))
    ax_b.set_xticklabels(metrics_order, fontsize=9)
    ax_b.set_xlim(-1.6, len(metrics_order) - 0.3)
    ax_b.set_ylim(0, 0.90)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_b.set_axisbelow(True)
    ax_b.set_yticklabels([])
    ax_b.set_title("Our decomposition: three orthogonal axes",
                   fontsize=8.5, pad=6, color="#0072B2")
    add_panel_label(ax_b, "b", x=-0.05, y=1.18)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_decomp_schematic", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
