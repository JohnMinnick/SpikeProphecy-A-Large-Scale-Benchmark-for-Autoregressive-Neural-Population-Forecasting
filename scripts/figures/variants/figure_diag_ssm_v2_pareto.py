"""
Diagonal-SSM signal figure — Variant 2: Colored Pareto frontier.

Same message as v1 but as a scatter plot trading off parameter count
vs weighted r. Shades the diagonal-SSM Pareto region and uses shape +
color redundancy so the diagonal cluster is visually obvious.

Stronger when read by someone who thinks in efficiency/performance
tradeoffs; weaker as a single-glance takeaway than the grouped bars.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from figures.style import apply_style, save_figure


MODELS = [
    ("Mamba",       1.95, 0.522, "Diagonal SSM", "D"),
    ("HGRN2",       1.82, 0.493, "Diagonal SSM", "P"),
    ("LRU v2",      1.23, 0.485, "Diagonal SSM", "s"),
    ("Transformer", 2.22, 0.491, "Attention",    "^"),
    ("LSTM",        2.22, 0.454, "Gated RNN",    "o"),
    ("SNN (2L)",    0.83, 0.477, "Spiking",      "X"),
]

CLASS_COLORS = {
    "Diagonal SSM": "#0072B2",
    "Attention":    "#D55E00",
    "Gated RNN":    "#E69F00",
    "Spiking":      "#009E73",
}


def generate():
    apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.93, bottom=0.15)

    # Tight convex-hull ellipse around the three diagonal-SSM points
    # (no "dominance zone" extending to the chart edges — that was
    # misleading). This just draws a soft contour hugging the cluster.
    diag_pts = np.array([(m[1], m[2]) for m in MODELS if m[3] == "Diagonal SSM"])
    xs = [m[1] for m in MODELS]
    ys = [m[2] for m in MODELS]
    x_lo, x_hi = 0.4, max(xs) * 1.15
    y_lo, y_hi = min(ys) - 0.02, max(ys) + 0.01

    # Ellipse centered on the cluster centroid, sized to enclose all three
    # diagonal SSM points with a small buffer.
    from matplotlib.patches import Ellipse
    cx, cy = diag_pts.mean(axis=0)
    # Half-axes = spread of points + buffer
    half_w = (diag_pts[:, 0].max() - diag_pts[:, 0].min()) / 2 + 0.25
    half_h = (diag_pts[:, 1].max() - diag_pts[:, 1].min()) / 2 + 0.012
    ellipse = Ellipse(
        (cx, cy), 2 * half_w, 2 * half_h,
        facecolor=CLASS_COLORS["Diagonal SSM"], alpha=0.10,
        edgecolor=CLASS_COLORS["Diagonal SSM"], linewidth=0.6,
        linestyle="--", zorder=1,
    )
    ax.add_patch(ellipse)

    # Plot each model
    for name, p, r, klass, marker in MODELS:
        ax.scatter(
            p, r, s=130, marker=marker,
            color=CLASS_COLORS[klass], edgecolor="#222222", linewidth=0.9,
            zorder=3,
        )
        # Label each point
        if name == "Transformer":
            dx, dy, ha = -0.05, -0.006, "right"
        elif name == "HGRN2":
            dx, dy, ha = 0.05, 0.002, "left"
        elif name == "LRU v2":
            dx, dy, ha = 0.05, -0.003, "left"
        elif name == "Mamba":
            dx, dy, ha = 0.05, 0.002, "left"
        else:
            dx, dy, ha = 0.05, 0.003, "left"
        ax.annotate(name, xy=(p, r), xytext=(p + dx, r + dy),
                    fontsize=8, ha=ha, va="center",
                    color=CLASS_COLORS[klass], fontweight="bold")

    # Label the cluster ellipse
    ax.text(
        cx, cy + half_h + 0.004,
        "Diagonal-SSM cluster",
        ha="center", va="bottom", fontsize=8,
        color=CLASS_COLORS["Diagonal SSM"], style="italic", alpha=0.85,
        fontweight="bold",
    )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Weighted Pearson $r$ (val)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi + 0.002)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)

    # Class legend
    class_order = ["Diagonal SSM", "Attention", "Gated RNN", "Spiking"]
    handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=CLASS_COLORS[c],
                   markeredgecolor="#222", markersize=9,
                   label=c)
        for c in class_order
    ]
    ax.legend(
        handles=handles, title="Architecture class",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5, title_fontsize=8, frameon=False,
        borderaxespad=0.0,
    )

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_diag_ssm_v2_pareto", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
