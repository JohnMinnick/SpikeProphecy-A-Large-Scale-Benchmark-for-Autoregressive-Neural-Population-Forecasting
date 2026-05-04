"""
NEW figure: Cross-dataset scaling decomposition.

Visual proof-by-example of the paper's primary contribution claim
(§1: "decomposition reveals tradeoffs invisible to aggregate r"):

    Aggregate weighted r:   0.522 → 0.538 → 0.543 (+4%, looks flat)
    Population rate r:      0.855 → 0.924 → 0.929 (+9%, climbs)
    Spatial pattern r:      0.536 → 0.406 → 0.373 (−30%, falls hard)
    Cosine similarity:      0.580 → 0.475 → 0.442 (−24%, falls)

Currently this finding lives only in Table 5 in the appendix. It is
THE most concrete demonstration of why the decomposition matters.
Promoting it to a main-body figure directly supports the §1 primary
contribution.

Layout: single panel, 4 lines (one per metric) over 3 dataset scales.
Aggregate r in muted grey to make the contrast with the climbing /
falling decomposed metrics jump out.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure, COLORS


# Table 5 (scaling) values
SCALING = {
    "Steinmetz only": {"sessions": 39,  "wt_r": 0.522, "pop_r": 0.855, "spatial_r": 0.536, "cosine": 0.580},
    "IBL only":       {"sessions": 66,  "wt_r": 0.538, "pop_r": 0.924, "spatial_r": 0.406, "cosine": 0.475},
    "Combined":       {"sessions": 105, "wt_r": 0.543, "pop_r": 0.929, "spatial_r": 0.373, "cosine": 0.442},
}


def generate():
    apply_style()

    labels = list(SCALING.keys())
    n_sessions = [SCALING[l]["sessions"] for l in labels]
    wt = [SCALING[l]["wt_r"] for l in labels]
    pop = [SCALING[l]["pop_r"] for l in labels]
    spatial = [SCALING[l]["spatial_r"] for l in labels]
    cosine = [SCALING[l]["cosine"] for l in labels]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.18)

    x = np.arange(len(labels))

    # Aggregate r (the "headline number") — muted gray, plotted first
    ax.plot(x, wt, color="#888888", linewidth=2.0, marker="o",
            markersize=8, markerfacecolor="white", markeredgewidth=1.5,
            label=f"Aggregate weighted $r$  (+{(wt[-1]/wt[0]-1)*100:.0f}%)",
            zorder=2)

    # Decomposed metrics — vivid colors
    pop_pct = (pop[-1] / pop[0] - 1) * 100
    sp_pct = (spatial[-1] / spatial[0] - 1) * 100
    cos_pct = (cosine[-1] / cosine[0] - 1) * 100

    ax.plot(x, pop, color="#0072B2", linewidth=2.2, marker="^",
            markersize=8, markeredgecolor="#222", markeredgewidth=0.6,
            label=f"$r_\\mathrm{{pop}}$ (temporal fidelity)   ({pop_pct:+.0f}%)",
            zorder=4)
    ax.plot(x, spatial, color="#D55E00", linewidth=2.2, marker="v",
            markersize=8, markeredgecolor="#222", markeredgewidth=0.6,
            label=f"$r_\\mathrm{{spatial}}$ (spatial identity)  ({sp_pct:+.0f}%)",
            zorder=4)
    ax.plot(x, cosine, color="#CC79A7", linewidth=1.8, marker="s",
            markersize=7, markeredgecolor="#222", markeredgewidth=0.6,
            linestyle="--", alpha=0.85,
            label=f"Cosine similarity                  ({cos_pct:+.0f}%)",
            zorder=3)

    # Annotate dataset labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n({n} sessions)" for l, n in zip(labels, n_sessions)],
        fontsize=9,
    )
    ax.set_ylabel("Pearson $r$ (val)")
    ax.set_ylim(0.30, 1.00)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Annotate the trade-off finding directly on the plot
    # Down-arrow on spatial drop (between session 1 and session 3)
    ax.annotate(
        "", xy=(2, spatial[-1] + 0.01), xytext=(2, spatial[0] - 0.01),
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
    )
    ax.text(2.15, (spatial[0] + spatial[-1]) / 2,
            f"{sp_pct:+.0f}%", color="#D55E00",
            fontsize=10, fontweight="bold", va="center")

    ax.annotate(
        "", xy=(2, pop[-1] - 0.01), xytext=(2, pop[0] + 0.01),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.5),
    )
    ax.text(2.15, (pop[0] + pop[-1]) / 2,
            f"{pop_pct:+.0f}%", color="#0072B2",
            fontsize=10, fontweight="bold", va="center")

    # Highlight the punchline
    ax.text(
        0.02, 0.04,
        "Aggregate $r$ moves +4%, hiding\n"
        "a +9% temporal vs −30% spatial trade-off.",
        transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF7E0",
                  edgecolor="#E69F00", linewidth=0.6, alpha=0.95),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5, frameon=False, handlelength=2.0,
        handletextpad=0.5, borderaxespad=0.0,
    )

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_new_scaling_decomp", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
