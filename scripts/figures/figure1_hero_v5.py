"""
Figure 1 hero v5 — full architecture story in one figure.

Layout:
  (a)  2x4 small multiples of population-rate traces, one per architecture
  (b)  Pareto: Wt-r vs parameter count (cluster vs trailing visible)
  (c)  5-axis radar: per-architecture metric profiles
  (d)  % of best model bars

Each architecture has its OWN color (Wong palette) and its own marker shape;
the cluster vs trailing distinction is conveyed by:
  - a faint blue tinted background on the 5 cluster panels
  - a shaded cluster band on the Pareto chart

Data sources:
  - Multi-arch session-4 prediction NPZ (all 7 archs incl. Mamba+SNN)
  - TABLE1 (39-session weighted r values from data.py)
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


CLUSTER_BG = "#EAF1F8"      # very light blue tint for cluster panels
GT_COLOR = "#888888"

# Architecture order for small multiples (cluster top row, trailing bottom)
ARCHS = [
    # (npz key,           display name,    table1 key,    cluster?)
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
    if p.exists():
        return np.load(str(p))
    raise FileNotFoundError(f"multi-arch NPZ not found at {p}")


def _plot_panel(ax, time_s, gt_pop, pred_pop, name, color, r_pop,
                r_neuron=None, cluster=False, show_ylabel=False,
                show_xlabel=False, ymax=None):
    if cluster:
        ax.set_facecolor(CLUSTER_BG)
    ax.fill_between(time_s, gt_pop, alpha=0.30, color=GT_COLOR,
                    linewidth=0, zorder=1)
    ax.plot(time_s, gt_pop, color=GT_COLOR, linewidth=0.5,
            alpha=0.9, zorder=2)
    ax.plot(time_s, pred_pop, color=color, linewidth=1.0, zorder=3)
    ax.set_xlim(time_s[0], time_s[-1])
    if ymax is not None:
        ax.set_ylim(0, ymax)
    # Title: pop-rate r is the easy metric (cluster-similar); per-neuron r
    # is the discriminator (Mamba 0.15 vs SNN 0.08 on this session).
    if r_neuron is not None:
        title = (f"{name}\n($r_\\mathrm{{pop}}{{=}}{r_pop:.2f}$, "
                 f"$r_\\mathrm{{n}}{{=}}{r_neuron:.2f}$)")
    else:
        title = f"{name}  ($r_\\mathrm{{pop}}={r_pop:.2f}$)"
    ax.set_title(title, fontsize=7.5, color=color,
                 fontweight="bold", pad=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pop spikes", fontsize=7)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)


def _plot_pareto(ax):
    # Cluster band first (background)
    cluster_rs = [TABLE1[n]["weighted_r"] for n in CLUSTER_MODELS]
    ax.axhspan(min(cluster_rs) - 0.004, max(cluster_rs) + 0.004,
               alpha=0.12, color="#0072B2", zorder=1, lw=0)

    # Architecture markers — no per-cluster-member labels because the
    # legend in panel (a) maps marker shape -> architecture.  Crowded
    # labels in the 2-pp cluster band were unreadable.
    for name in ALL_MODELS:
        d = TABLE1[name]
        ax.scatter(d["params_M"], d["weighted_r"],
                   c=COLORS[name], marker=MARKERS[name],
                   s=85, zorder=4, edgecolors="white", linewidths=0.7)

    # No per-arch labels in the Pareto.  All architecture identity is
    # carried by marker shape + color (legend in panel a); only the
    # cluster band is annotated so the headline message stays visible.
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


def _plot_decomposition_trace(ax):
    """Parallel-coordinates view of the decomposition.

    Each arch is a line across 5 metric axes.  Cluster archs bunch
    together at the top of every axis; LSTM and SNN drop most on
    population metrics; per-neuron r at the right edge collapses by
    nearly $2{\\times}$ between cluster and SNN — the paper's headline
    "aggregate $r$ hides per-neuron noise" reveal.
    """
    metrics = [
        ("$r_\\mathrm{pop}$",     "pop_rate_r"),
        ("$r_\\mathrm{spatial}$", "spatial_r"),
        ("Cosine",                "cosine_sim"),
        ("Wt-$r$",                "weighted_r"),
        ("Per-neuron $r$",        "per_neuron_r"),
    ]
    x = np.arange(len(metrics))

    # Faint gridlines per metric
    for xi in x:
        ax.axvline(xi, color=COLORS["grid"], linewidth=0.4, zorder=1)

    for name in ALL_MODELS:
        d = TABLE1[name]
        ys = [d[m[1]] for m in metrics]
        ax.plot(x, ys, color=COLORS[name], marker=MARKERS[name],
                linewidth=1.3, markersize=5,
                markeredgecolor="white", markeredgewidth=0.4,
                zorder=3, label=name)

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

    # Annotation: SNN drops on pop_rate_r.
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
    # Annotation: per-neuron r is much lower than aggregate metrics
    # (the decomposition's main reveal).  Sit above the data in the
    # upper-right region of the panel so the lines descending into
    # per-neuron r aren't crossed.
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
    best_r = max(TABLE1[m]["weighted_r"] for m in ALL_MODELS)
    eff = {n: TABLE1[n]["weighted_r"] / best_r * 100 for n in ALL_MODELS}
    sorted_models = sorted(eff, key=eff.get, reverse=True)
    y = np.arange(len(sorted_models))
    bars = ax.barh(y, [eff[m] for m in sorted_models],
                   height=0.6, zorder=3,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="white", linewidth=0.6)
    for bar, n in zip(bars, sorted_models):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{eff[n]:.0f}%", fontsize=6.5, fontweight="bold",
                color="white", ha="right", va="center")
    ax.axvline(100, color="#666", linewidth=0.6, linestyle="--", zorder=2)

    # Cluster vs trailing divider
    n_cluster = sum(1 for m in sorted_models[:5] if m in CLUSTER_MODELS)
    if n_cluster == 5:
        ax.axhline(4.5, color="#0072B2", linewidth=0.7,
                   linestyle=":", zorder=2, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=7)
    ax.set_xlabel("% of best Wt-$r$", fontsize=7.5)
    ax.set_xlim(0, 110)
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
    gt_pop_full = gt.sum(axis=1)
    ymax = float(gt_pop_full.max()) * 1.18

    fig = plt.figure(figsize=(TEXT_WIDTH, 5.0))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.55, 1.0],
        hspace=0.50, left=0.075, right=0.985,
        top=0.96, bottom=0.085,
    )

    # ---- Top: 2x4 small multiples ----
    sm = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[0],
        hspace=0.55, wspace=0.32,
    )
    sm_axes = []
    for i, (key, name, t1key, cluster) in enumerate(ARCHS):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(sm[row, col])
        sm_axes.append(ax)
        if key not in npz.files:
            ax.text(0.5, 0.5, f"{name}\n(no data)",
                    ha="center", va="center", fontsize=8, color="#999")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        rates = npz[key]
        n = min(rates.shape[0], gt.shape[0])
        rates_local = rates[:n]
        gt_local = gt[:n]
        gt_pop_loc = gt_local.sum(axis=1)
        pred_pop = rates_local.sum(axis=1)
        r = pearsonr(gt_pop_loc, pred_pop)[0]
        # Per-neuron r — the most discriminative metric on this session.
        # Average Pearson r computed across neurons that fire in either
        # ground truth or predictions.
        per_neuron_rs = []
        for j in range(gt_local.shape[1]):
            if gt_local[:, j].std() > 0 and rates_local[:, j].std() > 0:
                per_neuron_rs.append(
                    pearsonr(gt_local[:, j], rates_local[:, j])[0]
                )
        r_neuron = (float(np.mean(per_neuron_rs))
                    if per_neuron_rs else float("nan"))
        _plot_panel(ax, time_s[:n], gt_pop_loc, pred_pop, name,
                    COLORS[t1key], r, r_neuron=r_neuron,
                    cluster=cluster,
                    show_ylabel=(col == 0),
                    show_xlabel=(row == 1),
                    ymax=ymax)

    # Legend in last cell — TWO-COLUMN layout because seven entries
    # don't fit vertically in a ~1.5" tall cell at 7pt font.
    # Col 1: cluster archs (Mamba, HGRN2, Transformer, GatedDelta, LRU)
    # Col 2: trailing archs (LSTM, SNN)
    ax_leg = fig.add_subplot(sm[1, 3])
    ax_leg.axis("off")
    ax_leg.text(0.50, 1.00, "Architecture",
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

    # Column 1 — cluster archs, top half of cell
    y = 0.83
    for name in col1:
        _entry(name, 0.02, 0.14, y)
        y -= 0.13

    # Column 2 — trailing archs, top of cell
    y = 0.83
    for name in col2:
        _entry(name, 0.55, 0.67, y)
        y -= 0.13

    # Cluster panel swatch — bottom of cell, full width
    ax_leg.add_patch(plt.Rectangle(
        (0.02, 0.04), 0.18, 0.05,
        facecolor=CLUSTER_BG, edgecolor="#0072B2", linewidth=0.5,
        transform=ax_leg.transAxes, clip_on=False,
    ))
    ax_leg.text(0.24, 0.065, "cluster panel",
                fontsize=6.5, color="#0072B2",
                transform=ax_leg.transAxes, va="center", style="italic")

    # Panel label 'a.' on first SM panel
    sm_axes[0].text(
        -0.32, 1.20, "a.", transform=sm_axes[0].transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    # ---- Bottom: summary 1x3 ----
    # Tighter wspace now that panel (c) is cartesian (no polar label
    # bleed) and (b) has no per-arch labels piling on the right edge.
    sumgs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.45,
        width_ratios=[1.0, 1.05, 0.95],
    )

    # Pareto (b)
    ax_pareto = fig.add_subplot(sumgs[0, 0])
    _plot_pareto(ax_pareto)
    ax_pareto.text(
        -0.22, 1.18, "b.", transform=ax_pareto.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    # Decomposition trace (c) — cartesian parallel-coordinates plot
    ax_decomp = fig.add_subplot(sumgs[0, 1])
    _plot_decomposition_trace(ax_decomp)
    ax_decomp.text(
        -0.18, 1.18, "c.", transform=ax_decomp.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    # % of best (d)
    ax_bars = fig.add_subplot(sumgs[0, 2])
    _plot_bars(ax_bars)
    ax_bars.text(
        -0.55, 1.18, "d.", transform=ax_bars.transAxes,
        fontsize=14, fontweight="bold", color="#222222",
        va="top", ha="left",
    )

    save_figure(fig, "figure1_hero_v5")
    plt.close(fig)


if __name__ == "__main__":
    generate()
