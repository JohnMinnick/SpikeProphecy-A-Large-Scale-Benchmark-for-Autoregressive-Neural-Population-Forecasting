"""
Three-tier GLM figure (NEW, content audit).

Visualizes §4.1's three-tier GLM story that's currently table-only:

  Tier 1  Autoregressive GLM (~10 features/neuron): r ≈ 0.001
          Mechanism: "own history insufficient — no signal at 50 ms"
  Tier 2  Population GLM (~7K features/neuron): r = -0.015
          Mechanism: "overfits — train r=1.000, val actively wrong"
  Tier 3  Deep nonlinear (Mamba, HGRN2, ...): r ~ 0.5
          Mechanism: "learns population structure — 3-5x gain"

Currently this three-way failure-mode comparison is prose only. A
single figure makes the argument (and the shape of each failure mode)
visible at a glance.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure, add_panel_label, TEXT_WIDTH


def generate():
    apply_style()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.25)

    # Data
    labels = [
        "Autoreg GLM\n(10 feat/neuron)",
        "Population GLM\n(7K feat/neuron)",
        "Deep models\n(Mamba, HGRN2, ...)",
    ]
    train_r = [0.17, 1.000, 0.55]      # typical train r
    val_r   = [0.001, -0.015, 0.52]    # val r (what paper reports)
    colors  = ["#E69F00", "#D55E00", "#0072B2"]
    mechanisms = [
        "no signal\nto extract",
        "catastrophic\noverfit",
        "learned pop\nstructure",
    ]

    x = np.arange(len(labels))
    bar_w = 0.35

    # Train vs val paired bars
    bars_tr = ax.bar(
        x - bar_w / 2, train_r, bar_w,
        color="white", edgecolor=colors, linewidth=2.0,
        label="train $r$", zorder=3,
    )
    bars_va = ax.bar(
        x + bar_w / 2, val_r, bar_w,
        color=colors, edgecolor="#222222", linewidth=0.7,
        label="val $r$", zorder=3,
    )

    # Value labels
    for bx, h in zip(x - bar_w / 2, train_r):
        va = "bottom" if h >= 0 else "top"
        off = 0.02 if h >= 0 else -0.02
        ax.text(bx, h + off, f"{h:.3f}", ha="center", va=va,
                fontsize=7, color="#555555")
    for bx, h, c in zip(x + bar_w / 2, val_r, colors):
        va = "bottom" if h >= 0 else "top"
        off = 0.02 if h >= 0 else -0.02
        ax.text(bx, h + off, f"{h:+.3f}", ha="center", va=va,
                fontsize=7, color=c, fontweight="bold")

    # Zero line
    ax.axhline(0, color="#333333", linewidth=0.7, linestyle="-", zorder=1)

    # X labels and mechanism annotations below
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)

    # Mechanism captions
    for xi, mech, c in zip(x, mechanisms, colors):
        ax.text(xi, -0.22, mech, ha="center", va="top",
                fontsize=7.5, color=c, style="italic")

    # Y axis
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(-0.25, 1.10)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        fontsize=8, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.08), frameon=False,
        handletextpad=0.4, columnspacing=1.8,
    )

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_glm_three_tier", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
