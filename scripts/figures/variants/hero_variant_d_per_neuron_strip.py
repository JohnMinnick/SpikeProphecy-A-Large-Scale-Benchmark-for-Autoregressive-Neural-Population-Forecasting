"""Variant D: Per-neuron r distribution strip below the small multiples.

Layout:
  Row 1-2: 7 small-multiple pop-rate panels (current v5 style)
  Row 3:   Wide horizontal strip showing per-arch per-neuron r
           distributions as horizontal violin plots.  Cluster archs
           cluster at higher r values; SNN shifts visibly left.

This decouples the discrimination story (distribution shift) from the
trace style (where pop-rate is least discriminative).
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


def _load_multiarch():
    p = (Path(__file__).resolve().parents[3]
         / "data" / "figure_cache" / "multi_arch_session4.npz")
    return np.load(str(p))


def _plot_panel(ax, time_s, gt_pop, pred_pop, name, color, r_pop, r_neuron,
                cluster=False, show_ylabel=False, show_xlabel=False,
                ymax=None):
    if cluster:
        ax.set_facecolor(CLUSTER_BG)
    ax.fill_between(time_s, gt_pop, alpha=0.30, color=GT_COLOR,
                    linewidth=0, zorder=1)
    ax.plot(time_s, gt_pop, color=GT_COLOR, linewidth=0.5, zorder=2)
    ax.plot(time_s, pred_pop, color=color, linewidth=1.0, zorder=3)
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


def _compute_per_neuron_r(gt_local, rates_local):
    pn_rs = []
    for j in range(gt_local.shape[1]):
        if gt_local[:, j].std() > 0 and rates_local[:, j].std() > 0:
            pn_rs.append(pearsonr(gt_local[:, j], rates_local[:, j])[0])
    return np.array(pn_rs)


def _plot_per_neuron_strip(ax, npz, gt):
    """Horizontal violin / strip chart of per-neuron r per arch."""
    distributions = []
    names = []
    colors_list = []
    means = []
    for key, name, t1key, _ in ARCHS:
        if key not in npz.files:
            continue
        rates = npz[key]
        n = min(rates.shape[0], gt.shape[0])
        pn_r = _compute_per_neuron_r(gt[:n], rates[:n])
        distributions.append(pn_r)
        names.append(name)
        colors_list.append(COLORS[t1key])
        means.append(np.mean(pn_r))

    # Plot as horizontal violins, top-down so cluster appears at top.
    positions = np.arange(len(distributions))[::-1]
    parts = ax.violinplot(
        distributions, positions=positions, vert=False,
        widths=0.85, showmeans=False, showmedians=False,
        showextrema=False,
    )
    for pc, c in zip(parts["bodies"], colors_list):
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(0.55)
        pc.set_linewidth(0.4)

    # Median markers
    for pos, dist, c in zip(positions, distributions, colors_list):
        med = np.median(dist)
        mean = np.mean(dist)
        ax.scatter([med], [pos], color=c, s=18, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.text(mean + 0.012, pos, f"{mean:.2f}",
                fontsize=6.5, color=c, fontweight="bold",
                va="center")

    # Cluster band: shade the y-region covering cluster archs
    cluster_positions = [p for p, n in zip(positions, names)
                         if n in ("Mamba", "HGRN2", "Transformer",
                                  "GatedDeltaNet", "LRU")]
    if cluster_positions:
        ax.axhspan(min(cluster_positions) - 0.45,
                   max(cluster_positions) + 0.45,
                   color="#0072B2", alpha=0.06, zorder=0, lw=0)

    # Empirical oracle ceiling annotation
    oracle = 0.170
    ax.axvline(oracle, color="#666666", linestyle="--", linewidth=0.7,
               zorder=2)
    ax.text(oracle + 0.005, len(distributions) - 0.2,
            f"empirical oracle\n($r{{=}}{oracle:.2f}$)",
            fontsize=5.5, color="#444444", style="italic",
            va="top", ha="left")

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-0.05, 0.45)
    ax.set_xlabel("Per-neuron Pearson $r$  (each violin = $\\sim$700 neurons in session 4)",
                  fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)


def generate():
    apply_style()
    npz = _load_multiarch()
    gt = npz["gt"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt
    ymax = float(gt.sum(axis=1).max()) * 1.18

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.5))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.6, 1.0],
        hspace=0.55, left=0.075, right=0.985,
        top=0.96, bottom=0.10,
    )

    # Top: 2x4 small multiples
    sm = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[0],
        hspace=0.65, wspace=0.32,
    )
    sm_axes = []
    for i, (key, name, t1key, cluster) in enumerate(ARCHS):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(sm[row, col])
        sm_axes.append(ax)
        if key not in npz.files:
            continue
        rates = npz[key]
        n = min(rates.shape[0], gt.shape[0])
        rates_local = rates[:n]
        gt_local = gt[:n]
        gt_pop_loc = gt_local.sum(axis=1)
        pred_pop = rates_local.sum(axis=1)
        r = pearsonr(gt_pop_loc, pred_pop)[0]
        pn_rs = _compute_per_neuron_r(gt_local, rates_local)
        r_neuron = float(np.mean(pn_rs))
        _plot_panel(ax, time_s[:n], gt_pop_loc, pred_pop, name,
                    COLORS[t1key], r, r_neuron, cluster=cluster,
                    show_ylabel=(col == 0),
                    show_xlabel=(row == 1),
                    ymax=ymax)

    # Panel labels
    sm_axes[0].text(
        -0.32, 1.30, "a.", transform=sm_axes[0].transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    # Bottom: per-neuron r distribution strip
    ax_strip = fig.add_subplot(outer[1])
    _plot_per_neuron_strip(ax_strip, npz, gt)
    ax_strip.text(
        -0.05, 1.10, "b.", transform=ax_strip.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    save_figure(fig, "hero_variant_d_per_neuron_strip",
                out_dir=Path(__file__).resolve().parents[2]
                / "figures" / "variants")
    plt.close(fig)


if __name__ == "__main__":
    generate()
