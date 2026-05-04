"""
Figure 1 hero v6 — full architecture story with per-neuron strip.

Layout (top to bottom):
  (a)  2x4 small multiples of population-rate traces (per-arch colors,
       r_pop and r_neuron in titles, cluster panels blue-tinted)
  (b)  Per-neuron Pearson r distribution strip (horizontal violins)
       — the discriminating per-neuron view, anchored to the empirical
       oracle ceiling line
  (c)  Pareto: Wt-r vs parameter count, cluster band shaded
  (d)  Decomposition trace: 5-axis parallel coordinates incl.
       per-neuron r
  (e)  % of best Wt-r horizontal bars

Each architecture has its OWN color and marker, used consistently
across all five sub-panels.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    p = (Path(__file__).resolve().parents[2]
         / "data" / "figure_cache" / "multi_arch_session4.npz")
    return np.load(str(p))


def _per_neuron_r(gt_local, rates_local):
    pn_rs = []
    for j in range(gt_local.shape[1]):
        if gt_local[:, j].std() > 0 and rates_local[:, j].std() > 0:
            pn_rs.append(pearsonr(gt_local[:, j], rates_local[:, j])[0])
    return np.array(pn_rs)


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
    ax.set_title(title, fontsize=7.0, color=color,
                 fontweight="bold", pad=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pop", fontsize=7)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.0)


def _plot_per_neuron_strip(ax, npz, gt):
    # Shorter display labels for the y-axis so the panel letter has
    # room.  "GatedDeltaNet" -> "GatedDelta" matches the bars panel.
    SHORT_DISPLAY = {"GatedDeltaNet": "GatedDelta"}
    distributions = []
    names = []
    colors_list = []
    for key, name, t1key, _ in ARCHS:
        if key not in npz.files:
            continue
        rates = npz[key]
        n = min(rates.shape[0], gt.shape[0])
        pn_r = _per_neuron_r(gt[:n], rates[:n])
        distributions.append(pn_r)
        names.append(SHORT_DISPLAY.get(name, name))
        colors_list.append(COLORS[t1key])

    positions = np.arange(len(distributions))[::-1]
    parts = ax.violinplot(
        distributions, positions=positions, vert=False,
        widths=0.78, showmeans=False, showmedians=False,
        showextrema=False,
    )
    for pc, c in zip(parts["bodies"], colors_list):
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(0.55)
        pc.set_linewidth(0.4)

    for pos, dist, c in zip(positions, distributions, colors_list):
        med = np.median(dist)
        mean = np.mean(dist)
        ax.scatter([med], [pos], color=c, s=16, zorder=4,
                   edgecolors="white", linewidths=0.5)
        ax.text(mean + 0.015, pos, f"{mean:.2f}",
                fontsize=6.5, color=c, fontweight="bold",
                va="center", zorder=5)

    cluster_positions = [p for p, n in zip(positions, names)
                         if n in ("Mamba", "HGRN2", "Transformer",
                                  "GatedDelta", "LRU")]
    if cluster_positions:
        ax.axhspan(min(cluster_positions) - 0.45,
                   max(cluster_positions) + 0.45,
                   color="#0072B2", alpha=0.06, zorder=0, lw=0)

    oracle = 0.170
    ax.axvline(oracle, color="#666666", linestyle="--", linewidth=0.7,
               zorder=2)
    # Place the oracle annotation ABOVE the top violin (Mamba) so it
    # cannot collide with the per-arch mean labels.
    ax.text(oracle, len(distributions) - 0.10,
            f"oracle $r{{=}}{oracle:.2f}$",
            fontsize=5.5, color="#444444", style="italic",
            va="bottom", ha="center", zorder=5)

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(-0.05, 0.42)
    ax.set_xlabel("Per-neuron Pearson $r$",
                  fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)


def _plot_pareto(ax):
    cluster_rs = [TABLE1[n]["weighted_r"] for n in CLUSTER_MODELS]
    ax.axhspan(min(cluster_rs) - 0.004, max(cluster_rs) + 0.004,
               alpha=0.12, color="#0072B2", zorder=1, lw=0)

    for name in ALL_MODELS:
        d = TABLE1[name]
        ax.scatter(d["params_M"], d["weighted_r"],
                   c=COLORS[name], marker=MARKERS[name],
                   s=85, zorder=4, edgecolors="white", linewidths=0.7)

    ax.text(2.45, max(cluster_rs) + 0.011,
            "modern recurrence\ncluster (5 archs)",
            fontsize=6, color="#0072B2", style="italic",
            ha="right", va="bottom", zorder=5)

    ax.set_xlabel("Parameters (M)", fontsize=7.5)
    ax.set_ylabel("Weighted Pearson $r$", fontsize=7.5)
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(0.42, 0.52)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_decomp_trace(ax):
    metrics = [
        ("$r_\\mathrm{pop}$",     "pop_rate_r"),
        ("$r_\\mathrm{spatial}$", "spatial_r"),
        ("Cosine",                "cosine_sim"),
        ("Wt-$r$",                "weighted_r"),
        ("Per-neuron $r$",        "per_neuron_r"),
    ]
    x = np.arange(len(metrics))
    for xi in x:
        ax.axvline(xi, color=COLORS["grid"], linewidth=0.4, zorder=1)
    for name in ALL_MODELS:
        d = TABLE1[name]
        ys = [d[m[1]] for m in metrics]
        ax.plot(x, ys, color=COLORS[name], marker=MARKERS[name],
                linewidth=1.3, markersize=5,
                markeredgecolor="white", markeredgewidth=0.4,
                zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=6.5,
                       rotation=25, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Metric value", fontsize=7.5)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlim(-0.3, len(metrics) - 0.7)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    snn_pop = TABLE1["SNN"]["pop_rate_r"]
    ax.annotate(
        "SNN drops\non $r_\\mathrm{pop}$",
        xy=(0, snn_pop),
        xytext=(0.45, 0.30),
        fontsize=5.5, color=COLORS["SNN"], fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["SNN"],
                        linewidth=0.6, shrinkA=2, shrinkB=2),
        zorder=5,
    )
    ax.annotate(
        "aggregate $r$ hides\nper-neuron noise",
        xy=(4, TABLE1["Mamba"]["per_neuron_r"]),
        xytext=(3.05, 0.78),
        fontsize=5.5, color="#444444", fontweight="bold",
        ha="left", va="top", style="italic",
        arrowprops=dict(arrowstyle="->", color="#666666",
                        linewidth=0.5, shrinkA=3, shrinkB=2),
        zorder=5,
    )


def _plot_bars(ax):
    """Absolute weighted Pearson r per architecture, sorted descending.
    No 'best model' anchoring — the paper makes no within-cluster
    ordering claim and the values are reported in the standard scale
    that readers can compare to other benchmarks directly."""
    rs = {n: TABLE1[n]["weighted_r"] for n in ALL_MODELS}
    sorted_models = sorted(rs, key=rs.get, reverse=True)
    y = np.arange(len(sorted_models))
    bars = ax.barh(y, [rs[m] for m in sorted_models],
                   height=0.6, zorder=3,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="white", linewidth=0.6)
    # Numeric labels printed inside the bars for readability.
    for bar, n in zip(bars, sorted_models):
        ax.text(bar.get_width() - 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rs[n]:.3f}", fontsize=6.5, fontweight="bold",
                color="white", ha="right", va="center")
    # Cluster-vs-trailing divider line between LRU (last cluster) and LSTM
    ax.axhline(4.5, color="#0072B2", linewidth=0.7,
               linestyle=":", zorder=2, alpha=0.6)
    # Cluster band shading on the x-axis to echo panel (b).
    cluster_min = min(TABLE1[n]["weighted_r"] for n in CLUSTER_MODELS)
    cluster_max = max(TABLE1[n]["weighted_r"] for n in CLUSTER_MODELS)
    ax.axvspan(cluster_min - 0.003, cluster_max + 0.003,
               color="#0072B2", alpha=0.08, zorder=1, lw=0)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=7)
    ax.set_xlabel("Weighted Pearson $r$ (val)", fontsize=7.5)
    # Show full absolute scale from 0; readers can see differences AND
    # the absolute level rather than a misleading truncated range.
    ax.set_xlim(0, 0.55)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.tick_params(axis="x", labelsize=6.5)


def generate():
    apply_style()
    npz = _load_multiarch()
    gt = npz["gt"]
    T, _ = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt
    ymax = float(gt.sum(axis=1).max()) * 1.18

    # 6.3 in tall.  Front-loaded layout: 2x2 analytical summaries on
    # top, 2x4 small-multiples panel (with legend) on the bottom.
    # Reading order a/b/c/d (analytics) -> e (concrete predictions).
    fig = plt.figure(figsize=(TEXT_WIDTH, 6.3))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.55, 1.0],
        hspace=0.30, left=0.075, right=0.985,
        top=0.97, bottom=0.07,
    )

    # ---- Section 2 (bottom): 2x4 small multiples + legend ----
    sm = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[1],
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
        pn_rs = _per_neuron_r(gt_local, rates_local)
        r_neuron = float(np.mean(pn_rs))
        _plot_panel(ax, time_s[:n], gt_pop_loc, pred_pop, name,
                    COLORS[t1key], r, r_neuron, cluster=cluster,
                    show_ylabel=(col == 0),
                    show_xlabel=(row == 1),
                    ymax=ymax)

    # Legend cell at sm[1,3]
    ax_leg = fig.add_subplot(sm[1, 3])
    ax_leg.axis("off")
    ax_leg.text(0.50, 1.05, "Architecture",
                fontsize=8, fontweight="bold", color="#222222",
                transform=ax_leg.transAxes, va="top", ha="center")
    col1 = ["Mamba", "HGRN2", "Transformer", "GatedDelta", "LRU"]
    col2 = ["LSTM", "SNN"]

    def _entry(name, mx, tx, y):
        ax_leg.scatter(mx, y, c=COLORS[name], marker=MARKERS[name],
                       s=32, transform=ax_leg.transAxes,
                       edgecolors="white", linewidths=0.4, clip_on=False)
        ax_leg.text(tx, y, name, fontsize=6.5,
                    color=COLORS[name], fontweight="bold",
                    transform=ax_leg.transAxes, va="center")

    # Push entries down a bit so the "Architecture" header has clear
    # whitespace below it (otherwise its descenders crash into the
    # first row of markers).
    y = 0.74
    for name in col1:
        _entry(name, 0.02, 0.14, y)
        y -= 0.12
    y = 0.74
    for name in col2:
        _entry(name, 0.55, 0.67, y)
        y -= 0.12
    ax_leg.add_patch(plt.Rectangle(
        (0.02, 0.04), 0.18, 0.05,
        facecolor=CLUSTER_BG, edgecolor="#0072B2", linewidth=0.5,
        transform=ax_leg.transAxes, clip_on=False,
    ))
    ax_leg.text(0.24, 0.065, "cluster panel",
                fontsize=6.5, color="#0072B2",
                transform=ax_leg.transAxes, va="center", style="italic")

    sm_axes[0].text(
        -0.32, 1.30, "e.", transform=sm_axes[0].transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    # ---- Section 1 (top): 2x2 analytical-summary grid ----
    # Top row  = aggregate Wt-r story: (a) % bars, (b) Pareto.
    # Bot row  = decomposition story:  (c) decomp trace, (d) per-neuron
    # violins.  Reads L->R T->B with increasing analytic depth.
    sumgs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0],
        hspace=0.40, wspace=0.40,
    )

    # All four bottom-section panel labels share x=-0.18 for visual
    # alignment.  (a) and (d) have arch-name y-tick labels but the
    # offset is just enough to clear them after we shorten
    # "GatedDeltaNet" -> "GatedDelta" inside _plot_per_neuron_strip.
    LABEL_X = -0.18
    LABEL_Y = 1.10

    ax_bars = fig.add_subplot(sumgs[0, 0])
    _plot_bars(ax_bars)
    ax_bars.text(
        LABEL_X, LABEL_Y, "a.", transform=ax_bars.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    ax_pareto = fig.add_subplot(sumgs[0, 1])
    _plot_pareto(ax_pareto)
    ax_pareto.text(
        LABEL_X, LABEL_Y, "b.", transform=ax_pareto.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    ax_decomp = fig.add_subplot(sumgs[1, 0])
    _plot_decomp_trace(ax_decomp)
    ax_decomp.text(
        LABEL_X, LABEL_Y, "c.", transform=ax_decomp.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    ax_strip = fig.add_subplot(sumgs[1, 1])
    _plot_per_neuron_strip(ax_strip, npz, gt)
    ax_strip.text(
        LABEL_X, LABEL_Y, "d.", transform=ax_strip.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    save_figure(fig, "figure1_hero_v6")
    plt.close(fig)


if __name__ == "__main__":
    generate()
