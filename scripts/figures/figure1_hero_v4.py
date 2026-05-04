"""
Figure 1 hero — multi-architecture small-multiples.

2x4 grid of pop-rate traces, one per architecture:
  Row 1 (modern recurrence cluster): Mamba | HGRN2 | Transformer | GatedDeltaNet
  Row 2 (rest):                       LRU   | LSTM  | SNN         | [legend]

Each panel shows:
  - GT (gray fill) as the reference signal
  - That architecture's predicted population rate (class-coloured line)
  - Title with architecture name and per-architecture r_pop on the
    representative session.

Color encodes structural class (matches Table 1 / clustering bar chart),
so the cluster-vs-classical pattern is visible at a glance: 5 cluster
panels look near-identical to GT, LSTM and SNN panels visibly differ.

Falls back to a Mamba+SNN-only 2-row layout if the multi-arch
prediction NPZ is not yet downloaded from S3.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (
    apply_style, COLORS, add_panel_label, save_figure, TEXT_WIDTH,
)


# Class colors mirror only the distinctions the paper claims as
# load-bearing: modern-recurrence cluster vs. classical RNN vs. spiking.
# The diagonal-vs-non-diagonal distinction is now an Appendix-only
# negative result and does NOT carry color in figures.
CLASS_COLOR = {
    "Modern recurrence": "#0072B2",   # blue
    "Classical RNN":     "#E69F00",   # orange
    "Spiking":           "#009E73",   # green
}
GT_COLOR = "#888888"

# Architectures, ordered so the modern-recurrence cluster fills row 1
# and the rest falls into row 2.
ARCHS = [
    # (npz key, display name, class)
    ("mamba_rates",       "Mamba",         "Modern recurrence"),
    ("hgrn2_rates",       "HGRN2",         "Modern recurrence"),
    ("transformer_rates", "Transformer",   "Modern recurrence"),
    ("gated_delta_rates", "GatedDeltaNet", "Modern recurrence"),
    ("lru_rates",         "LRU",           "Modern recurrence"),
    ("lstm_rates",        "LSTM",          "Classical RNN"),
    ("snn_rates",         "SNN",           "Spiking"),
]


def _try_load_multiarch():
    """Try local cache first, then S3."""
    local = (Path(__file__).resolve().parents[2]
             / "data" / "figure_cache" / "multi_arch_session4.npz")
    if local.exists():
        return np.load(str(local))
    try:
        import boto3
        s3 = boto3.client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")
        local.parent.mkdir(parents=True, exist_ok=True)
        key = ("<anon>/spike-prophecy/outputs/"
               "multi-arch-inference-session4/predictions.npz")
        s3.download_file("braingeneersdev", key, str(local))
        return np.load(str(local))
    except Exception as e:
        print(f"  multi-arch NPZ not yet available ({e})")
        return None


def _load_fallback_session(session_idx=4):
    """Fallback: cached Mamba+SNN-only NPZ."""
    from figures.data import load_prediction_arrays
    return load_prediction_arrays(session_idx)


def _plot_arch_panel(ax, time_s, gt_pop, pred_pop, name, color, r_pop,
                     show_ylabel=False, show_xlabel=False, ymax=None):
    ax.fill_between(time_s, gt_pop, alpha=0.30, color=GT_COLOR,
                    linewidth=0, zorder=1)
    ax.plot(time_s, gt_pop, color=GT_COLOR, linewidth=0.5,
            alpha=0.9, zorder=2)
    ax.plot(time_s, pred_pop, color=color, linewidth=1.0, zorder=3)
    ax.set_xlim(time_s[0], time_s[-1])
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_title(
        f"{name}  ($r_\\mathrm{{pop}}={r_pop:.2f}$)",
        fontsize=8.5, color=color, fontweight="bold", pad=3,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pop spike count", fontsize=7.5)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=6.5)


def generate():
    apply_style()
    npz = _try_load_multiarch()
    if npz is None:
        print("  Multi-arch data not available; rendering Mamba+SNN fallback")
        _generate_fallback()
        return

    gt = npz["gt"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt
    gt_pop = gt.sum(axis=1)
    ymax = float(gt_pop.max()) * 1.15

    # Per-arch population sums + r_pop
    archs_present = []
    for key, name, klass in ARCHS:
        if key not in npz.files:
            print(f"  skipping {name}: missing in NPZ")
            continue
        rates = npz[key]
        # rates may have different T due to history-window offset; align to gt
        if rates.shape[0] != gt.shape[0]:
            n = min(rates.shape[0], gt.shape[0])
            rates = rates[:n]
            gt_local = gt[:n]
        else:
            gt_local = gt
        pop = rates.sum(axis=1)
        r = pearsonr(gt_local.sum(axis=1), pop)[0]
        archs_present.append((key, name, klass, pop, r))

    n = len(archs_present)
    # Pick grid that fits the available archs without empty rows
    if n <= 6:
        rows, cols = 2, 3
    elif n <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.6 if rows == 2 else 5.4))
    gs = gridspec.GridSpec(
        rows, cols, hspace=0.55, wspace=0.30,
        left=0.06, right=0.985, top=0.92, bottom=0.12,
    )

    panel_axes = []
    for i, (key, name, klass, pop, r) in enumerate(archs_present):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        panel_axes.append(ax)
        _plot_arch_panel(
            ax, time_s, gt_pop, pop, name, CLASS_COLOR[klass], r,
            show_ylabel=(col == 0),
            show_xlabel=(row == rows - 1),
            ymax=ymax,
        )

    # Add panel label to the first content axis (not a new subplot)
    if panel_axes:
        panel_axes[0].text(
            -0.20, 1.18, "a.", transform=panel_axes[0].transAxes,
            fontsize=14, fontweight="bold", color="#222222",
            va="top", ha="left",
        )

    # Final cell: legend / class key
    if n < rows * cols:
        # Place legend in last cell of the grid
        leg_row = rows - 1
        leg_col = cols - 1
        # If the last cell is occupied by an arch, find first empty cell
        if n > leg_row * cols + leg_col:
            for ri in range(rows):
                for ci in range(cols):
                    if ri * cols + ci >= n:
                        leg_row, leg_col = ri, ci
                        break
                else:
                    continue
                break
        ax_leg = fig.add_subplot(gs[leg_row, leg_col])
        ax_leg.axis("off")
        ax_leg.text(
            0.0, 0.95, "Architecture class",
            transform=ax_leg.transAxes,
            fontsize=8, fontweight="bold", color="#222222",
            va="top",
        )
        y = 0.78
        for klass, color in CLASS_COLOR.items():
            ax_leg.plot([0.0, 0.18], [y, y], color=color,
                        linewidth=2.4, transform=ax_leg.transAxes,
                        clip_on=False)
            ax_leg.text(
                0.22, y, klass, transform=ax_leg.transAxes,
                fontsize=7.5, color="#222222", va="center",
            )
            y -= 0.14
        ax_leg.text(
            0.0, y - 0.04,
            "Each panel: GT (gray fill) +\nthat architecture's predicted\n"
            "population rate.  Single\nrepresentative Steinmetz session.",
            transform=ax_leg.transAxes,
            fontsize=6.5, color="#555555", va="top", style="italic",
        )

    save_figure(fig, "figure1_hero_v4")
    plt.close(fig)


def _generate_fallback():
    """Mamba+SNN 2-architecture pop-rate trace + 3 example forecasts.
    Used when the multi-arch NPZ has not yet been computed."""
    from figures.data import load_prediction_arrays
    data = load_prediction_arrays(4)
    gt, mamba, snn = data["gt"], data["mamba_rates"], data["snn_rates"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    pop_gt = gt.sum(axis=1)
    pop_m = mamba.sum(axis=1)
    pop_s = snn.sum(axis=1)
    r_m = pearsonr(pop_gt, pop_m)[0]
    r_s = pearsonr(pop_gt, pop_s)[0]

    ax.fill_between(time_s, pop_gt, alpha=0.22, color="#444444",
                    linewidth=0, zorder=1)
    ax.plot(time_s, pop_gt, color="#444444", linewidth=0.7,
            alpha=0.85, zorder=2, label="Ground truth")
    ax.plot(time_s, pop_m, color="#0072B2", linewidth=1.4, zorder=4,
            label=f"Mamba ($r_\\mathrm{{pop}}={r_m:.2f}$)")
    ax.plot(time_s, pop_s, color="#009E73", linewidth=1.1, zorder=3,
            linestyle=(0, (3, 1.6)),
            label=f"SNN ($r_\\mathrm{{pop}}={r_s:.2f}$)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population spike count")
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylim(0, pop_gt.max() * 1.15)
    ax.legend(fontsize=8, loc="upper right", ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.16)
    save_figure(fig, "figure1_hero_v4")
    plt.close(fig)


if __name__ == "__main__":
    generate()
