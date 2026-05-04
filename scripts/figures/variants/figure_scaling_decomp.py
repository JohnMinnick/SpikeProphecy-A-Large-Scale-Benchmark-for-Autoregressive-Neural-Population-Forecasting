"""
Scaling decomposition figure (NEW, content audit).

Directly visualizes §4.2's argument: when data scales 39→66→105 sessions,
aggregate weighted r barely moves (+4%), masking a +9% pop-rate gain
and a -30% spatial-identity loss. Currently this story lives only in
prose — a figure makes it the argument it deserves to be.

Two panels side-by-side:
  (a) Aggregate weighted r vs session count — single flat line, looks
      like "scaling is saturated" if this is all you see.
  (b) Decomposed metrics vs session count — diverging lines showing
      the real structure: pop-rate up, spatial/cosine down.

The contrast between (a) and (b) is the figure's argument: the
decomposition is what the paper CLAIMS matters, and this figure shows
why — the same data looks different under the two evaluation protocols.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure, add_panel_label, TEXT_WIDTH


# From main.tex tab:scaling
SESSIONS = [39, 66, 105]
WT_R    = [0.522, 0.538, 0.543]
POP_R   = [0.855, 0.924, 0.929]
SPAT_R  = [0.536, 0.406, 0.373]
COSINE  = [0.580, 0.475, 0.442]


def generate():
    apply_style()

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.0, 1.3], wspace=0.35,
        left=0.08, right=0.97, top=0.92, bottom=0.16,
    )

    # ---- Panel (a): Aggregate-only view ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(SESSIONS, WT_R, "-o", color="#555555", linewidth=2.0,
              markersize=7, label="Weighted $r$")
    for s, r in zip(SESSIONS, WT_R):
        ax_a.text(s, r + 0.008, f"{r:.3f}", ha="center", va="bottom",
                  fontsize=7, color="#555555")
    ax_a.set_xlabel("Training sessions")
    ax_a.set_ylabel("Aggregate Weighted $r$")
    ax_a.set_xticks(SESSIONS)
    ax_a.set_xticklabels([f"{s}" for s in SESSIONS])
    ax_a.set_ylim(0.50, 0.57)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_a.set_axisbelow(True)
    ax_a.set_title("Aggregate view: \"saturated\"", fontsize=9, pad=6,
                   color="#555555")
    add_panel_label(ax_a, "a", x=-0.12, y=1.12)

    # Annotation: +4%
    ax_a.annotate(
        "+4% over 39→105",
        xy=(105, WT_R[-1]), xytext=(75, 0.510),
        fontsize=7.5, ha="center", color="#555555",
        arrowprops=dict(arrowstyle="->", color="#888888",
                        linewidth=0.6),
    )

    # ---- Panel (b): Decomposed view ----
    ax_b = fig.add_subplot(gs[0, 1])
    POP_COLOR = "#0072B2"
    SPAT_COLOR = "#D55E00"
    COSINE_COLOR = "#CC79A7"
    WT_COLOR = "#555555"

    ax_b.plot(SESSIONS, POP_R, "-o", color=POP_COLOR, linewidth=2.0,
              markersize=6, label="$r_\\mathrm{pop}$ (temporal)")
    ax_b.plot(SESSIONS, SPAT_R, "-s", color=SPAT_COLOR, linewidth=2.0,
              markersize=6, label="$r_\\mathrm{spatial}$")
    ax_b.plot(SESSIONS, COSINE, "-^", color=COSINE_COLOR, linewidth=2.0,
              markersize=6, label="Cosine sim.")
    ax_b.plot(SESSIONS, WT_R, ":d", color=WT_COLOR, linewidth=1.2,
              markersize=5, label="Weighted $r$ (aggregate)", alpha=0.7)

    ax_b.set_xlabel("Training sessions")
    ax_b.set_ylabel("Metric value")
    ax_b.set_xticks(SESSIONS)
    ax_b.set_xticklabels([f"{s}" for s in SESSIONS])
    ax_b.set_ylim(0.35, 1.0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=7, loc="center left",
                bbox_to_anchor=(1.01, 0.5), frameon=False,
                handlelength=1.6, handletextpad=0.4,
                borderaxespad=0.0)
    ax_b.set_title("Decomposed view: real trade-off revealed",
                   fontsize=9, pad=6, color=POP_COLOR)
    add_panel_label(ax_b, "b", x=-0.10, y=1.12)

    # Inline annotations on the diverging lines
    ax_b.annotate(
        "+9%", xy=(105, POP_R[-1]), xytext=(105, POP_R[-1] + 0.03),
        fontsize=7.5, ha="center", color=POP_COLOR, fontweight="bold",
    )
    ax_b.annotate(
        "-30%", xy=(105, SPAT_R[-1]), xytext=(105, SPAT_R[-1] - 0.04),
        fontsize=7.5, ha="center", color=SPAT_COLOR, fontweight="bold",
    )

    # Adjust right margin to let the legend fit
    gs.update(right=0.78)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_scaling_decomp", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
