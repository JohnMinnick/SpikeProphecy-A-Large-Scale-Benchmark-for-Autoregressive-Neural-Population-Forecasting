"""Variant B: Per-neuron heatmap strip.

Each panel splits vertically into:
  - Top 60%: pop-rate trace (current view)
  - Bottom 40%: heatmap of top-30 most-active neurons over time, color
    encoding (gt - prediction) residual.  Diverging colormap centered
    on zero: cluster panels show pale heatmaps; SNN panel shows visibly
    higher residual structure.

This is the strongest visual differentiator because per-neuron
prediction quality is where the cluster-vs-trailing gap is largest.
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


def _plot_panel(fig, gs_cell, time_s, gt_local, rates_local, name, color,
                r_pop, r_neuron, top_idx,
                cluster=False, show_ylabel=False, show_xlabel=False,
                ymax=None, vmax=None):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_cell, hspace=0.10,
        height_ratios=[3, 2],
    )
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1])

    if cluster:
        ax_top.set_facecolor(CLUSTER_BG)
        ax_bot.set_facecolor(CLUSTER_BG)

    # Top: pop-rate trace
    gt_pop = gt_local.sum(axis=1)
    pred_pop = rates_local.sum(axis=1)
    ax_top.fill_between(time_s, gt_pop, alpha=0.25, color=GT_COLOR,
                        linewidth=0, zorder=1)
    ax_top.plot(time_s, gt_pop, color=GT_COLOR, linewidth=0.5, zorder=2)
    ax_top.plot(time_s, pred_pop, color=color, linewidth=1.0, zorder=3)
    ax_top.set_xlim(time_s[0], time_s[-1])
    if ymax is not None:
        ax_top.set_ylim(0, ymax)
    ax_top.set_xticks([])
    title = (f"{name} ($r_\\mathrm{{pop}}{{=}}{r_pop:.2f}$, "
             f"$r_\\mathrm{{n}}{{=}}{r_neuron:.2f}$)")
    ax_top.set_title(title, fontsize=7.0, color=color,
                     fontweight="bold", pad=2.5)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    if show_ylabel:
        ax_top.set_ylabel("Pop", fontsize=6.5)
    ax_top.tick_params(axis="y", labelsize=5.5)

    # Bottom: heatmap of |residual| for top-K active neurons.
    K = 30
    sub_gt = gt_local[:, top_idx]
    sub_pred = rates_local[:, top_idx]
    residual = sub_gt - sub_pred  # positive = under-predicted
    # Order rows by mean GT activity (highest at top)
    activity_order = np.argsort(-sub_gt.mean(axis=0))
    residual_ordered = residual[:, activity_order].T  # (K, T)
    if vmax is None:
        vmax = np.percentile(np.abs(residual_ordered), 95)
    im = ax_bot.imshow(
        residual_ordered, aspect="auto", origin="upper",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[time_s[0], time_s[-1], 0, K],
        interpolation="nearest",
    )
    ax_bot.set_xlim(time_s[0], time_s[-1])
    ax_bot.set_ylim(0, K)
    if show_ylabel:
        ax_bot.set_ylabel("top-30\nneurons", fontsize=5.5)
    ax_bot.set_yticks([])
    if show_xlabel:
        ax_bot.set_xlabel("Time (s)", fontsize=7)
    ax_bot.tick_params(axis="x", labelsize=6.0)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    return im


def generate():
    apply_style()
    npz = _load_multiarch()
    gt = npz["gt"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt
    ymax = float(gt.sum(axis=1).max()) * 1.18

    # Pick top-K most active neurons across the session for shared row
    # ordering so panels are directly comparable.
    activity = gt.mean(axis=0)
    top_idx = np.argsort(-activity)[:30]

    # Compute global residual scale across all archs for shared colormap
    all_residuals = []
    for key, *_ in ARCHS:
        if key in npz.files:
            r = npz[key][:T, :]
            res = gt[:r.shape[0], top_idx] - r[:, top_idx]
            all_residuals.append(np.abs(res))
    vmax = np.percentile(np.concatenate(all_residuals).ravel(), 95)
    print(f"Shared residual vmax (95th pct) = {vmax:.3f}")

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.2))
    sm = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.32,
                           left=0.075, right=0.985,
                           top=0.93, bottom=0.10)

    for i, (key, name, t1key, cluster) in enumerate(ARCHS):
        row, col = divmod(i, 4)
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
        _plot_panel(
            fig, sm[row, col], time_s[:n], gt_local, rates_local,
            name, COLORS[t1key], r, r_neuron, top_idx,
            cluster=cluster,
            show_ylabel=(col == 0),
            show_xlabel=(row == 1),
            ymax=ymax, vmax=vmax,
        )

    save_figure(fig, "hero_variant_b_heatmap_strip",
                out_dir=Path(__file__).resolve().parents[2]
                / "figures" / "variants")
    plt.close(fig)


if __name__ == "__main__":
    generate()
