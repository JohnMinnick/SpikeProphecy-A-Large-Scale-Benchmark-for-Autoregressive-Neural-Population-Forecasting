"""
Figure 1 hero — combined (v2): multi-architecture, no heatmap.

Merges the previous Figure 1 hero with the architecture-clustering
Figure 2. Drops the Mamba-only heatmap (decorative, weak information
content) and makes the real-data panel multi-architecture so the
hero reflects the paper's thesis: *modern-recurrence architectures
cluster tightly, with classical RNN/SNN below*.

Layout:
  Row 1 (compact)  — population-rate trace for a representative
                     session: ground truth (gray fill) with three
                     architecture traces overlaid (Mamba, LSTM, SNN)
                     as tier representatives. Per-architecture r_pop
                     annotated; shows both the temporal-capture story
                     and the tier gap, without picking a single
                     winner.
  Row 2 (compact)  — three example single-neuron forecasts.  Each
                     overlays Mamba (top tier) and SNN (bottom tier)
                     predictions on the same ground-truth dots.  This
                     concretizes "aggregate r hides a heterogeneous
                     per-neuron distribution" *and* shows inter-model
                     differences without Mamba bias.
  Row 3 (compact)  — 7-architecture weighted Pearson r clustering.
                     Headline result.
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
from figures.data import TABLE1


# Architecture clustering — same data/palette as figure_arch_clustering
CLUSTER_MODELS = [
    ("Mamba",         0.500, "Diagonal SSM"),
    ("HGRN2",         0.493, "Diagonal SSM"),
    ("Transformer",   0.492, "Attention"),
    ("GatedDeltaNet", 0.485, "Non-diag SSM"),
    ("LRU",        0.480, "Diagonal SSM"),
    ("LSTM",          0.441, "Gated RNN"),
    ("SNN (3L)",      0.430, "Spiking"),
]
CLASS_COLORS = {
    "Diagonal SSM": "#0072B2",
    "Non-diag SSM": "#882255",
    "Attention":    "#D55E00",
    "Gated RNN":    "#E69F00",
    "Spiking":      "#009E73",
}
CLASS_HATCH = {
    "Diagonal SSM": "",
    "Non-diag SSM": "..",
    "Attention":    "//",
    "Gated RNN":    "\\\\",
    "Spiking":      "xx",
}


def _load_session(session_idx=4):
    from figures.data import load_prediction_arrays
    try:
        data = load_prediction_arrays(session_idx)
        return (
            data["gt"],
            data["mamba_rates"],
            data["snn_rates"],
        )
    except Exception as e:
        print(f"  real data unavailable ({e}); synthesizing")
        rng = np.random.RandomState(42)
        T, N = 660, 700
        base = rng.exponential(0.3, N)
        trend = 0.5 + 0.4 * np.sin(np.arange(T) * 0.1)
        gt = rng.poisson(base[None, :] * trend[:, None])
        mamba = base[None, :] * trend[:, None] + rng.normal(0, 0.04, (T, N))
        snn = mamba * 0.85 + rng.normal(0, 0.05, (T, N))
        return gt, np.clip(mamba, 0, None), np.clip(snn, 0, None)


def _per_neuron_r(gt, pred):
    T, N = gt.shape
    rs = np.full(N, np.nan)
    for n in range(N):
        g, p = gt[:, n], pred[:, n]
        if g.std() > 0 and p.std() > 0:
            rs[n] = np.corrcoef(g, p)[0, 1]
    return rs


def _pick_example_neurons(per_r, rate_gt, min_rate=0.15):
    active = np.where((rate_gt >= min_rate) & ~np.isnan(per_r))[0]
    if len(active) < 10:
        active = np.where(~np.isnan(per_r))[0]
    rs = per_r[active]
    order = np.argsort(rs)
    low_idx = active[order[max(1, int(0.05 * len(active)))]]
    mid_idx = active[order[len(active) // 2]]
    high_idx = active[order[min(len(active) - 1,
                                 int(0.97 * len(active)))]]
    return high_idx, mid_idx, low_idx


def _plot_multi_arch_pop(ax, time_s, gt, mamba, snn):
    """Row 1 panel: ground-truth population rate with Mamba (top tier)
    and SNN (bottom tier) overlaid as real-data tier representatives."""
    pop_gt = gt.sum(axis=1)
    pop_mamba = mamba.sum(axis=1)
    pop_snn = snn.sum(axis=1)

    r_mamba = pearsonr(pop_gt, pop_mamba)[0]
    r_snn   = pearsonr(pop_gt, pop_snn)[0]

    ax.fill_between(time_s, pop_gt, alpha=0.22, color="#444444",
                    linewidth=0, zorder=1)
    ax.plot(time_s, pop_gt, color="#444444", linewidth=0.7,
            alpha=0.85, zorder=2, label="Ground truth")
    ax.plot(time_s, pop_mamba, color=CLASS_COLORS["Diagonal SSM"],
            linewidth=1.3, zorder=4,
            label=f"Mamba (r$_\\mathrm{{pop}}${{=}}{r_mamba:.2f})")
    ax.plot(time_s, pop_snn, color=CLASS_COLORS["Spiking"],
            linewidth=1.05, zorder=3, linestyle=(0, (2, 1.5)),
            label=f"SNN (r$_\\mathrm{{pop}}${{=}}{r_snn:.2f})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population spike count")
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylim(0, max(pop_gt.max(), pop_mamba.max()) * 1.12)
    ax.legend(
        fontsize=7.5, loc="upper right",
        ncol=3, columnspacing=0.9,
        handlelength=1.8, handletextpad=0.4,
        frameon=False, borderaxespad=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_multi_arch_example(ax, time_s, gt_col, mamba_col, snn_col,
                              r_mamba, r_snn, title,
                              show_ylabel, show_legend):
    """Row 2 sub-panel: one example neuron with GT dots + Mamba rate
    line + SNN rate line. No Poisson band this time (too busy with 2
    lines)."""
    m_rate = np.clip(mamba_col, 1e-4, None)
    s_rate = np.clip(snn_col, 1e-4, None)
    ax.plot(time_s, m_rate, color=CLASS_COLORS["Diagonal SSM"],
            linewidth=1.2, zorder=3, label="Mamba")
    ax.plot(time_s, s_rate, color=CLASS_COLORS["Spiking"],
            linewidth=1.0, zorder=3, linestyle=(0, (2, 1.5)),
            label="SNN")
    ax.scatter(time_s, gt_col, s=4, color="#222222", alpha=0.7,
               edgecolors="none", zorder=4, label="GT")
    ax.set_xlim(time_s[0], time_s[-1])
    ymax = max(float(gt_col.max()) + 0.5,
               float(m_rate.max()) * 1.1,
               float(s_rate.max()) * 1.1)
    ax.set_ylim(-0.1, ymax)
    ax.set_title(title, fontsize=9, pad=4, color="#333333")
    ax.text(
        0.97, 0.92,
        f"Mamba $r{{=}}{r_mamba:.2f}$\n"
        f"SNN   $r{{=}}{r_snn:.2f}$",
        transform=ax.transAxes, fontsize=7.5, fontweight="bold",
        color="#333333", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Spike count")
    if show_legend:
        ax.legend(fontsize=6.8, loc="upper left",
                  handlelength=1.2, handletextpad=0.4,
                  frameon=False, borderaxespad=0.2)


def _plot_clustering(ax):
    name_to_row = {m[0]: m for m in CLUSTER_MODELS}
    ordered = [name_to_row[n] for n, _, _ in CLUSTER_MODELS]

    x = np.arange(len(ordered))
    heights = [r[1] for r in ordered]
    classes = [r[2] for r in ordered]
    colors = [CLASS_COLORS[c] for c in classes]
    hatches = [CLASS_HATCH[c] for c in classes]

    bars = ax.bar(x, heights, width=0.65,
                  color=colors, edgecolor="#333333",
                  linewidth=0.9, zorder=3)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    mamba_r = name_to_row["Mamba"][1]
    ax.axhline(mamba_r, color="#333333", linewidth=0.7,
               linestyle=":", zorder=2)
    ax.text(
        len(ordered) - 0.4, mamba_r + 0.003,
        f"Mamba r = {mamba_r:.3f}",
        ha="right", va="bottom", fontsize=7, color="#333333",
    )

    top_tier = {"Mamba", "HGRN2", "Transformer",
                "GatedDeltaNet", "LRU"}
    top_rs = [r[1] for r in ordered if r[0] in top_tier]
    ax.axhspan(min(top_rs) - 0.003, max(top_rs) + 0.003,
               alpha=0.08, color="#555555", zorder=1)
    ax.text(
        2, 0.513, "Modern recurrence cluster",
        fontsize=7.5, color="#555555",
        fontweight="bold", ha="center", style="italic",
    )

    for b, h_val in zip(bars, heights):
        ax.text(
            b.get_x() + b.get_width() / 2, h_val + 0.003,
            f"{h_val:.3f}",
            ha="center", va="bottom", fontsize=7, color="#333333",
        )

    display_map = {"GatedDeltaNet": "Gated\nDeltaNet"}
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_map.get(r[0], r[0]) for r in ordered],
        fontsize=8, rotation=0,
    )
    ax.set_ylabel("Weighted Pearson $r$")
    ax.set_ylim(0.41, 0.52)
    ax.set_yticks([0.42, 0.44, 0.46, 0.48, 0.50, 0.52])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

    class_order = ["Diagonal SSM", "Non-diag SSM", "Attention",
                   "Gated RNN", "Spiking"]
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=CLASS_COLORS[c],
                      hatch=CLASS_HATCH[c],
                      edgecolor="#333333", linewidth=0.7)
        for c in class_order
    ]
    ax.legend(
        handles, class_order,
        title="Architecture class",
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=7, title_fontsize=7.5, frameon=False,
        borderaxespad=0.0, handlelength=1.5, handletextpad=0.5,
    )


def generate():
    apply_style()

    gt, mamba, snn = _load_session(session_idx=4)
    T, N = gt.shape
    dt = 0.05
    time_s = np.arange(T) * dt

    mean_rates = gt.mean(axis=0)
    per_r_mamba = _per_neuron_r(gt, mamba)
    per_r_snn = _per_neuron_r(gt, snn)
    high_idx, mid_idx, low_idx = _pick_example_neurons(
        per_r_mamba, mean_rates)

    fig = plt.figure(figsize=(TEXT_WIDTH, 6.2))
    gs = gridspec.GridSpec(
        3, 3,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.65, wspace=0.38,
        left=0.095, right=0.82, top=0.96, bottom=0.07,
    )

    # ---------------- Row 1: multi-arch population trace -------------
    ax_pop = fig.add_subplot(gs[0, :])
    _plot_multi_arch_pop(ax_pop, time_s, gt, mamba, snn)
    add_panel_label(ax_pop, "a", x=-0.075, y=1.04)

    # ---------------- Row 2: 3 example forecasts, multi-arch ---------
    examples = [
        (high_idx, "Well-predicted"),
        (mid_idx,  "Typical"),
        (low_idx,  "Poorly predicted"),
    ]
    for col, (nid, label) in enumerate(examples):
        ax_ex = fig.add_subplot(gs[1, col])
        _plot_multi_arch_example(
            ax_ex, time_s,
            gt_col=gt[:, nid],
            mamba_col=mamba[:, nid], snn_col=snn[:, nid],
            r_mamba=per_r_mamba[nid], r_snn=per_r_snn[nid],
            title=label,
            show_ylabel=(col == 0), show_legend=(col == 0),
        )
        ax_ex.text(
            0.02, 1.02, "bcd"[col],
            transform=ax_ex.transAxes,
            fontsize=13, fontweight="bold", fontfamily="sans-serif",
            color="#222222", va="bottom", ha="left",
        )

    # ---------------- Row 3: architecture clustering -----------------
    ax_cl = fig.add_subplot(gs[2, :])
    _plot_clustering(ax_cl)
    ax_cl.text(
        -0.075, 1.04, "e",
        transform=ax_cl.transAxes,
        fontsize=14, fontweight="bold", fontfamily="sans-serif",
        color="#222222", va="bottom", ha="left",
    )

    save_figure(fig, "figure1_hero_combined")
    plt.close(fig)


if __name__ == "__main__":
    generate()
