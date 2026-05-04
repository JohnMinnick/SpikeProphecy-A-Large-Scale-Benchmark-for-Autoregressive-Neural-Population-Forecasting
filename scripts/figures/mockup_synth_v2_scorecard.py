"""Synthetic-validation variant: KS-distance scorecard.

Compact one-panel summary: for each population statistic (firing rate,
Fano factor, population rate, pairwise correlation recovery), show
KS-D (or 1-recovery) for each model with a colored "passes / partial /
fails" gradient. One look tells the reader which models preserve which
properties.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def main():
    apply_style()

    stats_path = (
        PROJECT_ROOT / "outputs" / "eval_local"
        / "synthetic_validation_stats.json"
    )
    s = json.load(open(stats_path))

    # Build a "metric × model" grid of KS-D (lower = better)
    # Plus pairwise correlation as 1 - mean_recovery (so lower = better)
    rows = [
        ("Firing rate dist.\n(per neuron)", "firing_rate", "D"),
        ("Fano factor dist.\n(per neuron)", "fano", "D"),
        ("Population rate\n(time series)", "pop_rate", "D"),
        ("Pairwise corr.\n(1 − recovery)", "pairwise", "loss"),
    ]
    models = [
        ("mamba_vs_gt", "Mamba", COLORS["Mamba"]),
        ("snn_vs_gt", "Spiking NN", COLORS["SNN"]),
    ]

    grid = np.zeros((len(rows), len(models)))
    for i, (label, key, m) in enumerate(rows):
        for j, (mkey, _, _) in enumerate(models):
            if key == "pairwise":
                # 1 - mean recovery
                v = s["pairwise_corr_recovery"][mkey]["mean"]
                grid[i, j] = 1 - v
            else:
                grid[i, j] = s["ks_tests"][key][mkey][m]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.8))

    # Bars side-by-side for each metric
    n_rows = len(rows)
    n_models = len(models)
    bar_h = 0.32
    y_centers = np.arange(n_rows)[::-1]

    for j, (mkey, mname, color) in enumerate(models):
        offset = (j - (n_models - 1) / 2) * bar_h * 1.05
        ys = y_centers + offset
        widths = grid[:, j]
        bars = ax.barh(
            ys, widths, height=bar_h,
            color=color, edgecolor="none",
            label=mname, alpha=0.88,
        )
        # Value labels
        for k, (yv, w) in enumerate(zip(ys, widths)):
            ax.text(
                w + 0.005, yv,
                f"{w:.3f}",
                va="center", ha="left",
                fontsize=7.5, color="#222222",
            )

    # Gradient threshold bands (reference)
    ax.axvspan(0, 0.05, color="#d8eed4", alpha=0.30, zorder=0)
    ax.axvspan(0.05, 0.15, color="#fff4d4", alpha=0.30, zorder=0)
    ax.axvspan(0.15, 1.0, color="#f4d4d4", alpha=0.30, zorder=0)

    # Threshold annotations
    ax.text(
        0.025, n_rows - 0.65, "indistinguishable\n(D < 0.05)",
        ha="center", va="top", fontsize=6.5, color="#1a6d2a",
    )
    ax.text(
        0.10, n_rows - 0.65, "small gap\n(0.05–0.15)",
        ha="center", va="top", fontsize=6.5, color="#a6740a",
    )
    ax.text(
        0.30, n_rows - 0.65, "meaningful gap\n(D > 0.15)",
        ha="center", va="top", fontsize=6.5, color="#a02929",
    )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(
        "Kolmogorov–Smirnov $D$ (rate / Fano / pop) "
        "or  1 − pairwise-corr recovery (lower = better)",
        fontsize=8.5,
    )
    ax.set_xlim(0, 0.50)
    ax.set_ylim(-0.5, n_rows - 0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax.set_title(
        "Distributional digital-twin scorecard "
        f"({s['n_neurons_total']:,} neurons, {s['n_sessions']} sessions)",
        fontsize=9, loc="left",
    )

    # Strip top/right
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    )
    save_figure(fig, "synth_v2_scorecard", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
