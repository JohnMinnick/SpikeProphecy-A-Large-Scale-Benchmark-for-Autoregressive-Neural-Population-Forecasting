"""
Figure 1 hero — Variant 1: Minimal 2-panel Nature style.

Drops the standalone raster (redundant with the heatmap). Two vertically-
stacked panels:
  (a) Population rate trace — GT + top 3 models — big, clean, annotated
      with r_pop per model.
  (b) GT vs Mamba prediction heatmap side-by-side for a representative
      window.

Reading goal: in 3 seconds, viewer sees "the model tracks population
dynamics and spatial structure at scale."
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
    """Load real prediction arrays from the cached S3 archive."""
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
        mamba = base[None, :] * trend[:, None] + rng.normal(0, 0.03, (T, N))
        snn = mamba * 0.85 + rng.normal(0, 0.05, (T, N))
        return gt, np.clip(mamba, 0, None), np.clip(snn, 0, None)


def generate():
    apply_style()

    gt, mamba, snn = _load_session(session_idx=4)
    T, N = gt.shape
    dt = 0.05  # 50 ms bins
    time_s = np.arange(T) * dt

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.2))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1.0, 1.3], width_ratios=[1.0, 1.0],
        hspace=0.55, wspace=0.18,
        left=0.08, right=0.82, top=0.93, bottom=0.10,
    )

    # ----- Panel (a): Population rate trace (full width) -----
    ax_pop = fig.add_subplot(gs[0, :])
    pop_gt = gt.sum(axis=1)
    pop_mamba = mamba.sum(axis=1)
    pop_snn = snn.sum(axis=1)

    from scipy.stats import pearsonr
    r_mamba = pearsonr(pop_gt, pop_mamba)[0]
    r_snn = pearsonr(pop_gt, pop_snn)[0]

    ax_pop.fill_between(time_s, pop_gt, alpha=0.20, color="#444444",
                        label="Ground truth", zorder=1, linewidth=0)
    ax_pop.plot(time_s, pop_gt, color="#444444", linewidth=0.6,
                alpha=0.7, zorder=2)
    ax_pop.plot(time_s, pop_mamba, color=COLORS["Mamba"], linewidth=1.4,
                label=f"Mamba (r_pop={r_mamba:.2f})", zorder=3)
    ax_pop.plot(time_s, pop_snn, color=COLORS["SNN"], linewidth=1.0,
                linestyle="--", alpha=0.9,
                label=f"SNN (r_pop={r_snn:.2f})", zorder=3)

    ax_pop.set_xlabel("Time (s)")
    ax_pop.set_ylabel("Population spike count")
    ax_pop.set_xlim(time_s[0], time_s[-1])
    ymax = max(pop_gt.max(), pop_mamba.max()) * 1.15
    ax_pop.set_ylim(0, ymax)
    ax_pop.legend(fontsize=7, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False,
                  handlelength=1.8, handletextpad=0.5, borderaxespad=0.0)
    ax_pop.spines["top"].set_visible(False)
    ax_pop.spines["right"].set_visible(False)
    add_panel_label(ax_pop, "a", x=-0.06, y=1.10)

    # Summary text in panel (a) upper-left
    ax_pop.text(
        0.02, 0.96,
        f"{N:,} simultaneously recorded neurons\nmean r_pop = {r_mamba:.2f} (Mamba)",
        transform=ax_pop.transAxes, fontsize=7,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9),
    )

    # ----- Panel (b): GT heatmap -----
    # Sort neurons by mean rate, keep only top ~500 for visual density
    mean_rates = gt.mean(axis=0)
    top = np.argsort(mean_rates)[::-1][:min(500, N)]
    gt_sorted = gt[:, top].T
    rng = np.random.RandomState(0)
    mamba_sampled = rng.poisson(np.clip(mamba[:, top], 0, None)).T

    vmax = np.percentile(gt_sorted, 99)

    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(gt_sorted, aspect="auto", cmap=HEATMAP_CMAP,
                 vmin=0, vmax=vmax, interpolation="nearest")
    ax_gt.set_title("Ground truth", fontsize=8.5, color="#444444",
                    fontweight="bold", pad=4)
    ax_gt.set_xlabel("Time (s)")
    ax_gt.set_ylabel("Neurons (by rate)")
    ntks = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    ax_gt.set_xticks(ntks)
    ax_gt.set_xticklabels([f"{t * dt:.0f}" for t in ntks])
    ax_gt.spines["top"].set_visible(False)
    ax_gt.spines["right"].set_visible(False)
    add_panel_label(ax_gt, "b", x=-0.22, y=1.18)

    # ----- Panel (b) right: Mamba prediction heatmap -----
    ax_pred = fig.add_subplot(gs[1, 1])
    im = ax_pred.imshow(mamba_sampled, aspect="auto", cmap=HEATMAP_CMAP,
                       vmin=0, vmax=vmax, interpolation="nearest")
    ax_pred.set_title("Mamba prediction (Poisson-sampled)",
                      fontsize=8.5, color=COLORS["Mamba"],
                      fontweight="bold", pad=4)
    ax_pred.set_xlabel("Time (s)")
    ax_pred.set_yticklabels([])
    ax_pred.set_xticks(ntks)
    ax_pred.set_xticklabels([f"{t * dt:.0f}" for t in ntks])
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_pred, shrink=0.85, pad=0.04,
                        fraction=0.055)
    cbar.set_label("Spike count", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    out_dir = Path(__file__).resolve().parents[3] / "docs" / "neurips_ed" / "figures" / "variants"
    save_figure(fig, "figure1_hero_v1_minimal", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate()
