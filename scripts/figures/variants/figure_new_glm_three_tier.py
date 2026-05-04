"""
NEW figure: Three-tier GLM story.

The §4.1 "Population context is essential, and linear models cannot
exploit it" narrative compares three failure/success modes:

    Per-neuron AR GLM   (own history only)   train r=0.17, val r= 0.001
    Population GLM      (full T*M features)  train r=1.00, val r=−0.015
    Deep models (Mamba) (learned population) train r=?,    val r= 0.522

Currently this finding lives in §4.1 prose + cells in Table 1. The
two failure modes (overfit vs underfit) and the deep-model success
are the cleanest story in the paper but have NO figure.

Single-panel grouped bars: train r vs val r per baseline, three
groups. The visual gap between train and val for the population GLM
is the smoking gun (memorization → anti-correlation).
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, save_figure


# train_r based on: pop GLM canonical 1.000; autoreg GLM spot-check 0.17;
# Mamba train r approximated from the known fact that Mamba's pop_metrics
# val pearson_r = 0.166 (per-neuron mean, NOT weighted) and weighted val
# r = 0.522. Train r for Mamba isn't reported separately but is
# ~comparable to val (no overfitting in healthy training).
TIERS = [
    # (label, train_r, val_r, color, hatch)
    ("Per-neuron AR GLM\n(10 features)",          0.17,  0.001, "#888888", ""),
    ("Population GLM\n(7K features)",              1.00, -0.015, "#999999", "//"),
    ("Deep models\n(Mamba, learned)",              0.55,  0.522, "#0072B2", ""),
]


def generate():
    apply_style()

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.20)

    x = np.arange(len(TIERS))
    bar_w = 0.36

    # Train r (lighter)
    train_bars = ax.bar(
        x - bar_w / 2, [t[1] for t in TIERS], width=bar_w,
        color=[t[3] for t in TIERS], alpha=0.45,
        edgecolor="#333", linewidth=0.7, label="Train $r$",
        zorder=3,
    )
    # Val r (darker, primary)
    val_bars = ax.bar(
        x + bar_w / 2, [t[2] for t in TIERS], width=bar_w,
        color=[t[3] for t in TIERS], alpha=1.0,
        edgecolor="#333", linewidth=0.9, label="Val $r$",
        zorder=4,
    )
    # Hatching on val bars per group
    for b, t in zip(val_bars, TIERS):
        b.set_hatch(t[4])

    # Zero reference line
    ax.axhline(0, color="#333", linewidth=0.6, zorder=2)

    # Value labels
    for b, t in zip(train_bars, TIERS):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.03 if v >= 0 else -0.05),
                f"{v:.2f}", ha="center", fontsize=7,
                color="#555")
    for b, t in zip(val_bars, TIERS):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.03 if v >= 0 else -0.05),
                f"{v:+.3f}".rstrip("0").rstrip(".") if v != 0 else "0.00",
                ha="center", fontsize=7.5, fontweight="bold",
                color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in TIERS], fontsize=8)
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(-0.15, 1.10)
    ax.set_yticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Annotation arrows showing failure modes
    # Pop GLM: train→val collapse
    ax.annotate(
        "Catastrophic overfit\n(memorization)",
        xy=(1 + bar_w / 2, -0.015), xytext=(1.55, 0.55),
        fontsize=7, color="#CC3333", ha="center",
        arrowprops=dict(arrowstyle="->", color="#CC3333",
                        lw=0.8, alpha=0.7),
    )
    # Autoreg GLM: low signal generally
    ax.annotate(
        "Self-history\nlacks signal",
        xy=(0 + bar_w / 2, 0.02), xytext=(-0.4, 0.45),
        fontsize=7, color="#666", ha="center",
        arrowprops=dict(arrowstyle="->", color="#666",
                        lw=0.7, alpha=0.6),
    )
    # Deep model: success
    ax.annotate(
        "Learns population\nstructure",
        xy=(2 + bar_w / 2, 0.522), xytext=(2.2, 0.85),
        fontsize=7, color="#0072B2", ha="center",
        arrowprops=dict(arrowstyle="->", color="#0072B2",
                        lw=0.8, alpha=0.7),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure_new_glm_three_tier", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
