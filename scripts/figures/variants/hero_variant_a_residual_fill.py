"""Variant A: Residual fill.

Same hero layout as v5, but each panel's pop-rate trace is augmented
with a `fill_between(gt, pred)` colored by the architecture.  Where
the prediction matches GT, no fill.  Where they disagree, the area is
filled with the arch color — visually showing per-bin error magnitude.
SNN should display visibly more colored fill than cluster panels.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import (
    apply_style, COLORS, MARKERS, save_figure, TEXT_WIDTH,
)
from figures.data import TABLE1


CLUSTER_BG = "#EAF1F8"
GT_COLOR = "#888888"

ARCHS = [
    ("mamba_rates",       "Mamba",         "Mamba",        True),
    ("hgrn2_rates",       "HGRN2",         "HGRN2",        True),
    ("transformer_rates", "Transformer",   "Transformer",  True),
    ("gated_delta_rates", "GatedDeltaNet", "GatedDelta",   True),
    ("lru_rates",         "LRU",           "LRU",          True),
    ("lstm_rates",        "LSTM",          "LSTM",         False),
    ("snn_rates",         "SNN",           "SNN",          False),
]

ALL_MODELS = ["Mamba", "HGRN2", "Transformer", "GatedDelta",
              "LRU", "LSTM", "SNN"]
CLUSTER_MODELS = ["Mamba", "HGRN2", "Transformer", "GatedDelta", "LRU"]


def _load_multiarch():
    p = (Path(__file__).resolve().parents[3]
         / "data" / "figure_cache" / "multi_arch_session4.npz")
    return np.load(str(p))


def _plot_panel(ax, time_s, gt_pop, pred_pop, name, color, r_pop, r_neuron,
                cluster=False, show_ylabel=False, show_xlabel=False,
                ymax=None):
    if cluster:
        ax.set_facecolor(CLUSTER_BG)
    # GT envelope
    ax.fill_between(time_s, gt_pop, alpha=0.20, color=GT_COLOR,
                    linewidth=0, zorder=1)
    ax.plot(time_s, gt_pop, color=GT_COLOR, linewidth=0.5,
            alpha=0.9, zorder=2)
    # Residual fill — between GT and prediction.
    ax.fill_between(time_s, gt_pop, pred_pop, color=color, alpha=0.30,
                    linewidth=0, zorder=2.5)
    ax.plot(time_s, pred_pop, color=color, linewidth=0.9, zorder=3)
    ax.set_xlim(time_s[0], time_s[-1])
    if ymax is not None:
        ax.set_ylim(0, ymax)
    title = (f"{name}\n($r_\\mathrm{{pop}}{{=}}{r_pop:.2f}$, "
             f"$r_\\mathrm{{n}}{{=}}{r_neuron:.2f}$)")
    ax.set_title(title, fontsize=7.5, color=color,
                 fontweight="bold", pad=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pop spikes", fontsize=7)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)


def generate():
    apply_style()
    npz = _load_multiarch()
    gt = npz["gt"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt
    ymax = float(gt.sum(axis=1).max()) * 1.18

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    sm = gridspec.GridSpec(2, 4, hspace=0.65, wspace=0.32,
                           left=0.075, right=0.985,
                           top=0.92, bottom=0.13)

    for i, (key, name, t1key, cluster) in enumerate(ARCHS):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(sm[row, col])
        if key not in npz.files:
            continue
        rates = npz[key]
        n = min(rates.shape[0], gt.shape[0])
        rates_local = rates[:n]
        gt_local = gt[:n]
        gt_pop_loc = gt_local.sum(axis=1)
        pred_pop = rates_local.sum(axis=1)
        r = pearsonr(gt_pop_loc, pred_pop)[0]
        per_neuron_rs = []
        for j in range(gt_local.shape[1]):
            if gt_local[:, j].std() > 0 and rates_local[:, j].std() > 0:
                per_neuron_rs.append(
                    pearsonr(gt_local[:, j], rates_local[:, j])[0]
                )
        r_neuron = float(np.mean(per_neuron_rs))
        _plot_panel(ax, time_s[:n], gt_pop_loc, pred_pop, name,
                    COLORS[t1key], r, r_neuron, cluster=cluster,
                    show_ylabel=(col == 0),
                    show_xlabel=(row == 1),
                    ymax=ymax)

    save_figure(fig, "hero_variant_a_residual_fill",
                out_dir=Path(__file__).resolve().parents[2]
                / "figures" / "variants")
    plt.close(fig)


if __name__ == "__main__":
    generate()
