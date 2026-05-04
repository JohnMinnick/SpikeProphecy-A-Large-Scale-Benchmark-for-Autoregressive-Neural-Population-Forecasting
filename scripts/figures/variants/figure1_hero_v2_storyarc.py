"""
Figure 1 hero — Variant 2: 3-panel left-to-right story arc.

Panels tell a story: (a) the data, (b) the prediction, (c) the evaluation.

  (a) Raw raster (left): zoomed 5-s window, ~150 top neurons, to give
      a visceral sense of the input spiking.
  (b) GT heatmap vs Mamba prediction heatmap (center + right): wider,
      full session, sorted by firing rate.
  (c) Population metrics bar triptych (bottom full-width): r_pop,
      r_spatial, cosine for the top-3 models (Mamba, HGRN2, Transformer).

Reading goal: tell a complete research narrative in one figure —
"this is the data, this is the prediction, this is how well it worked."
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.style import apply_style, COLORS, add_panel_label, save_figure, TEXT_WIDTH


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "spike_heat",
    ["#0d0221", "#2a0845", "#6b1d5e", "#c94277", "#f4a236", "#ffd166"],
    N=256,
)


def _load_session(session_idx=4):
    from figures.data import load_prediction_arrays
    try:
        data = load_prediction_arrays(session_idx)
        return data["gt"], data["mamba_rates"], data["snn_rates"]
    except Exception as e:
        print(f"  real data unavailable ({e}); synthesizing")
        rng = np.random.RandomState(42)
        T, N = 660, 700
        base = rng.exponential(0.3, N)
        trend = 0.5 + 0.4 * np.sin(np.arange(T) * 0.1)
        gt = rng.poisson(base[None, :] * trend[:, None])
        m = base[None, :] * trend[:, None] + rng.normal(0, 0.03, (T, N))
        s = m * 0.85 + rng.normal(0, 0.05, (T, N))
        return gt, np.clip(m, 0, None), np.clip(s, 0, None)


def generate():
    apply_style()

    gt, mamba, snn = _load_session(session_idx=4)
    T, N = gt.shape
    dt = 0.05

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.8))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.4, 1.0], width_ratios=[0.7, 1.0, 1.0],
        hspace=0.55, wspace=0.25,
        left=0.06, right=0.97, top=0.94, bottom=0.10,
    )

    # ----- Panel (a): Raster of top 150 neurons, first 5 s -----
    ax_raster = fig.add_subplot(gs[0, 0])
    t_end = min(100, T)  # 5 s
    n_show = min(150, N)
    # pick top-rate neurons for visual density
    mean_rates = gt.mean(axis=0)
    top = np.argsort(mean_rates)[::-1][:n_show]
    raster_data = gt[:t_end, top]

    # Plot as scatter of spike events, colored by firing rate
    for i in range(n_show):
        times = np.where(raster_data[:, i] > 0)[0]
        if len(times) == 0:
            continue
        rate_norm = min(mean_rates[top[i]] / max(mean_rates.max(), 1e-6), 1.0)
        color = HEATMAP_CMAP(rate_norm)
        ax_raster.scatter(times * dt, np.full(len(times), i, dtype=float),
                          s=3.5, c=[color], marker="|", linewidths=1.0)
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Neuron (by rate)")
    ax_raster.set_xlim(0, t_end * dt)
    ax_raster.set_ylim(-1, n_show)
    ax_raster.invert_yaxis()
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    add_panel_label(ax_raster, "a", x=-0.25, y=1.08)
    ax_raster.set_title(
        f"{n_show} neurons · 5 s",
        fontsize=7.5, color="#444444", pad=4,
    )

    # ----- Panel (b): GT heatmap (middle) -----
    n_show_hm = min(500, N)
    gt_sorted = gt[:, top[:n_show_hm] if n_show_hm <= n_show else np.argsort(mean_rates)[::-1][:n_show_hm]].T
    rng = np.random.RandomState(0)
    mamba_full_sorted_idx = np.argsort(mean_rates)[::-1][:n_show_hm]
    mamba_sampled = rng.poisson(np.clip(mamba[:, mamba_full_sorted_idx], 0, None)).T
    gt_sorted = gt[:, mamba_full_sorted_idx].T

    vmax = np.percentile(gt_sorted, 99)

    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.imshow(gt_sorted, aspect="auto", cmap=HEATMAP_CMAP,
                 vmin=0, vmax=vmax, interpolation="nearest")
    ax_gt.set_title("Ground truth", fontsize=8.5, color="#444444",
                    fontweight="bold", pad=4)
    ntks = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    ax_gt.set_xticks(ntks)
    ax_gt.set_xticklabels([f"{t * dt:.0f}" for t in ntks])
    ax_gt.set_xlabel("Time (s)")
    ax_gt.set_ylabel(f"{n_show_hm} neurons (by rate)")
    ax_gt.spines["top"].set_visible(False)
    ax_gt.spines["right"].set_visible(False)
    add_panel_label(ax_gt, "b", x=-0.18, y=1.08)

    # ----- Panel (b): Mamba heatmap (right) -----
    ax_pred = fig.add_subplot(gs[0, 2])
    im = ax_pred.imshow(mamba_sampled, aspect="auto", cmap=HEATMAP_CMAP,
                        vmin=0, vmax=vmax, interpolation="nearest")
    ax_pred.set_title("Mamba prediction", fontsize=8.5,
                      color=COLORS["Mamba"], fontweight="bold", pad=4)
    ax_pred.set_xticks(ntks)
    ax_pred.set_xticklabels([f"{t * dt:.0f}" for t in ntks])
    ax_pred.set_xlabel("Time (s)")
    ax_pred.set_yticklabels([])
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)

    cbar = plt.colorbar(im, ax=ax_pred, shrink=0.85, pad=0.04,
                        fraction=0.055)
    cbar.set_label("Spike count", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    # ----- Panel (c): Metric triptych (3 mini-panels spanning bottom) -----
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, :], wspace=0.5,
    )
    # Data for three top models
    metrics = {
        "$r_\\mathrm{pop}$":     {"Mamba": 0.756, "HGRN2": 0.740,
                                    "Transformer": 0.744, "LSTM": 0.702,
                                    "SNN": 0.596, "LRU v2": 0.716},
        "$r_\\mathrm{spatial}$": {"Mamba": 0.551, "HGRN2": 0.544,
                                    "Transformer": 0.543, "LSTM": 0.494,
                                    "SNN": 0.506, "LRU v2": 0.535},
        "Cosine similarity":     {"Mamba": 0.626, "HGRN2": 0.621,
                                    "Transformer": 0.620, "LSTM": 0.583,
                                    "SNN": 0.592, "LRU v2": 0.614},
    }

    MODEL_ORDER = ["Mamba", "HGRN2", "LRU v2", "Transformer", "LSTM", "SNN"]
    MODEL_COLOR_KEY = {
        "Mamba": "Mamba", "HGRN2": "LRU", "LRU v2": "LRU",
        "Transformer": "Transformer", "LSTM": "LSTM", "SNN": "SNN",
    }
    # HGRN2 same blue family as other diagonals
    M_COLORS = {
        "Mamba":       COLORS["Mamba"],
        "HGRN2":       "#3a6db5",  # different blue for HGRN2
        "LRU v2":      COLORS["LRU"],
        "Transformer": COLORS["Transformer"],
        "LSTM":        COLORS["LSTM"],
        "SNN":         COLORS["SNN"],
    }

    for col, (metric, vals) in enumerate(metrics.items()):
        ax_m = fig.add_subplot(gs_c[0, col])
        heights = [vals[m] for m in MODEL_ORDER]
        cs = [M_COLORS[m] for m in MODEL_ORDER]
        x = np.arange(len(MODEL_ORDER))
        bars = ax_m.bar(x, heights, width=0.65, color=cs,
                        edgecolor="#333", linewidth=0.6)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(MODEL_ORDER, fontsize=6.5, rotation=35, ha="right")
        ax_m.set_ylabel(metric, fontsize=8)
        lo = min(heights) - 0.04
        hi = max(heights) + 0.02
        ax_m.set_ylim(lo, hi)
        ax_m.spines["top"].set_visible(False)
        ax_m.spines["right"].set_visible(False)
        ax_m.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
        ax_m.set_axisbelow(True)
        if col == 0:
            add_panel_label(ax_m, "c", x=-0.22, y=1.15)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure1_hero_v2_storyarc", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
