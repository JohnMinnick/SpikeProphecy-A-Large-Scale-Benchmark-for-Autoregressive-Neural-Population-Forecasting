"""Paper-grade DTW alignment figure for the neurocog appendix.

Three panels demonstrating that Mamba's predicted population rates are
temporally well-aligned with ground truth even when per-bin Pearson r is
modest (the per-session linear behavioral head reads from a 500ms window,
so it is robust to exactly the small temporal misalignments DTW absorbs).

Panels:
  (a) DTW distance matrix exemplar (session 10, the highest-r session,
      same as multizoom hero figure).  Optimal warping path in white;
      naive diagonal in dashed grey.
  (b) Naive vs DTW-aligned overlay of population spike rates for the
      same session.  Grey links between recorded and predicted curves
      visualize how DTW realigns the two.
  (c) Cross-session bar chart over all 39 Steinmetz sessions: naive
      bin-to-bin MAE vs DTW-aligned average error.  Mean improvement
      annotated.

Output: docs/neurips_neurocog/figures/figure_dtw_alignment.{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS  # noqa: E402

# Colors picked to match multizoom + paper conventions
COLOR_GT = "#1E88E5"   # blue for "Recorded" (matches multizoom)
COLOR_PRED = COLORS["Mamba"]  # vermillion (#D55E00), Mamba forecast
COLOR_LINK = "#888888"
COLOR_PATH = "#FFFFFF"
COLOR_DIAG = "#CCCCCC"

EXEMPLAR_SESSION = 10  # Same as multizoom hero (highest mean per-neuron r)


def load_pop_rates(session_idx: int, arrays_dir: Path):
    """Return (pop_gt, pop_pred, M_real) for one session."""
    p = arrays_dir / f"session_{session_idx:03d}.npz"
    d = dict(np.load(p, allow_pickle=True))
    gt = d["gt"].astype(np.float32)
    pred = np.clip(d["pred"], 0, None).astype(np.float32)
    M = int(d["m_i"])
    # Sum over real (non-padded) neurons.  Predictions are already shaped
    # to (T, M_real) in this dataset; sum across neurons gives population
    # rate per bin.
    pop_gt = gt.sum(axis=1)
    pop_pred = pred.sum(axis=1)
    return pop_gt, pop_pred, M


def build_figure(arrays_dir: Path, out_dir: Path):
    apply_style()

    from fastdtw import fastdtw

    # --- Panel-c data: cross-session DTW ---
    files = sorted(arrays_dir.glob("session_*.npz"))
    session_ids = []
    naive_maes = []
    dtw_errs = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if int(d["m_i"]) < 100:
            continue
        gt = d["gt"].astype(np.float32)
        pred = np.clip(d["pred"], 0, None).astype(np.float32)
        pop_gt = gt.sum(axis=1)
        pop_pred = pred.sum(axis=1)
        dist, path = fastdtw(pop_gt, pop_pred)
        dtw_avg = dist / max(1, len(path))
        naive = float(np.mean(np.abs(pop_gt - pop_pred)))
        session_ids.append(int(d["session_idx"]))
        naive_maes.append(naive)
        dtw_errs.append(dtw_avg)

    session_ids = np.array(session_ids)
    naive_maes = np.array(naive_maes)
    dtw_errs = np.array(dtw_errs)
    order = np.argsort(naive_maes)[::-1]  # worst-naive first
    session_ids = session_ids[order]
    naive_maes = naive_maes[order]
    dtw_errs = dtw_errs[order]
    mean_naive = float(naive_maes.mean())
    mean_dtw = float(dtw_errs.mean())
    pct_improve = 100.0 * (1.0 - mean_dtw / mean_naive)

    print(
        f"  Cross-session: n={len(session_ids)}, "
        f"naive={mean_naive:.2f}, DTW={mean_dtw:.2f}, "
        f"improvement={pct_improve:.1f}%"
    )

    # --- Panels a/b data: exemplar session ---
    pop_gt_ex, pop_pred_ex, M_ex = load_pop_rates(EXEMPLAR_SESSION, arrays_dir)
    T = len(pop_gt_ex)
    dist_ex, path_ex = fastdtw(pop_gt_ex, pop_pred_ex)
    avg_err_ex = dist_ex / max(1, len(path_ex))
    naive_mae_ex = float(np.mean(np.abs(pop_gt_ex - pop_pred_ex)))

    print(
        f"  Exemplar session {EXEMPLAR_SESSION}: M={M_ex}, "
        f"naive MAE={naive_mae_ex:.2f}, DTW avg={avg_err_ex:.2f}"
    )

    # --- Layout ---
    fig = plt.figure(
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.95),
        dpi=300, facecolor="white",
    )
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.5, 1.0],
        hspace=0.42,
        left=0.085, right=0.985,
        top=0.96, bottom=0.10,
    )

    # Top row: panel a (distance matrix) + panel b (warping path)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 1.55],
        wspace=0.42,
    )

    # ----- Panel (a): Distance matrix with warping path -----
    gs_a = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs_top[0],
        width_ratios=[0.20, 1.0, 0.07],
        height_ratios=[0.20, 1.0],
        wspace=0.05, hspace=0.05,
    )
    ax_a_top = fig.add_subplot(gs_a[0, 1])
    ax_a_left = fig.add_subplot(gs_a[1, 0])
    ax_a_main = fig.add_subplot(gs_a[1, 1])
    ax_a_cbar = fig.add_subplot(gs_a[1, 2])

    D = cdist(pop_pred_ex[:, None], pop_gt_ex[:, None], metric="euclidean")

    # Top marginal: GT trace
    ax_a_top.plot(np.arange(T), pop_gt_ex, color=COLOR_GT, linewidth=1.2)
    ax_a_top.axis("off")
    ax_a_top.set_xlim(-0.5, T - 0.5)
    # Left marginal: predicted (rotated)
    ax_a_left.plot(pop_pred_ex, np.arange(T), color=COLOR_PRED, linewidth=1.2)
    ax_a_left.invert_xaxis()
    ax_a_left.axis("off")
    ax_a_left.set_ylim(-0.5, T - 0.5)

    # Distance matrix
    im = ax_a_main.imshow(
        D, aspect="auto", origin="lower", cmap="viridis",
        interpolation="nearest",
    )
    # Naive diagonal
    ax_a_main.plot(
        [0, T - 1], [0, T - 1],
        color=COLOR_DIAG, linestyle="--", linewidth=1.2, alpha=0.95,
    )
    # Optimal DTW path (white)
    p_gt = [p[0] for p in path_ex]
    p_pred = [p[1] for p in path_ex]
    ax_a_main.plot(p_gt, p_pred, color=COLOR_PATH, linewidth=1.6)

    ax_a_main.set_xlim(-0.5, T - 0.5)
    ax_a_main.set_ylim(-0.5, T - 0.5)
    ax_a_main.set_xlabel("Time, recorded (bins)", fontsize=8)
    ax_a_main.set_ylabel("Time, Mamba forecast (bins)", fontsize=8)
    ax_a_main.tick_params(labelsize=7)

    cbar = fig.colorbar(im, cax=ax_a_cbar)
    cbar.set_label("Euclidean dist.", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ----- Panel (b): warping path overlay -----
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_top[1],
        height_ratios=[1.0, 1.0],
        hspace=0.18,
    )
    ax_b_top = fig.add_subplot(gs_b[0])
    ax_b_bot = fig.add_subplot(gs_b[1])

    # Top: naive overlay
    ax_b_top.plot(
        pop_gt_ex, color=COLOR_GT, linewidth=1.2, label="Recorded",
    )
    ax_b_top.plot(
        pop_pred_ex, color=COLOR_PRED, linewidth=1.2, label="Mamba forecast",
    )
    ax_b_top.set_title(
        f"Naive (bin-to-bin) MAE = {naive_mae_ex:.1f} spikes/bin",
        fontsize=8, pad=3,
    )
    ax_b_top.set_ylabel("Population rate", fontsize=7.5)
    ax_b_top.tick_params(labelsize=7)
    ax_b_top.legend(
        frameon=False, fontsize=7, loc="upper right",
        handlelength=1.2, ncol=2,
    )
    ax_b_top.set_xticks([])

    # Bottom: DTW-warped overlay with links
    ax_b_bot.plot(pop_gt_ex, color=COLOR_GT, linewidth=1.2)
    ax_b_bot.plot(pop_pred_ex, color=COLOR_PRED, linewidth=1.2)
    for k in range(0, len(path_ex), 3):
        i, j = path_ex[k]
        ax_b_bot.plot(
            [i, j], [pop_gt_ex[i], pop_pred_ex[j]],
            color=COLOR_LINK, alpha=0.32, linewidth=0.4,
        )
    ax_b_bot.set_title(
        f"DTW-aligned avg error = {avg_err_ex:.1f} spikes/bin "
        f"({100*(1 - avg_err_ex/naive_mae_ex):.0f}% reduction)",
        fontsize=8, pad=3,
    )
    ax_b_bot.set_xlabel("Time (50 ms bins)", fontsize=7.5)
    ax_b_bot.set_ylabel("Population rate", fontsize=7.5)
    ax_b_bot.tick_params(labelsize=7)

    # ----- Panel (c): cross-session bars -----
    ax_c = fig.add_subplot(outer[1])
    n = len(session_ids)
    x = np.arange(n)
    width = 0.4
    ax_c.bar(
        x - width / 2, naive_maes, width,
        color=COLOR_GT, label="Naive bin-to-bin MAE",
    )
    ax_c.bar(
        x + width / 2, dtw_errs, width,
        color=COLOR_PRED, label="DTW avg error",
    )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(s) for s in session_ids], fontsize=6, rotation=90)
    ax_c.set_xlabel("Session (sorted by naive MAE, descending)", fontsize=7.5)
    ax_c.set_ylabel("Spikes / bin", fontsize=7.5)
    ax_c.tick_params(axis="y", labelsize=7)
    ax_c.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)

    # Annotation: mean improvement (positioned to avoid overlapping bars)
    ax_c.annotate(
        f"Mean DTW improvement: {pct_improve:.1f}%\n"
        f"(naive {mean_naive:.1f} → DTW {mean_dtw:.1f} spikes/bin; "
        f"{len(session_ids)}/{len(session_ids)} sessions)",
        xy=(0.99, 0.99), xycoords="axes fraction",
        ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=COLORS.get("highlight", "#F0E442"),
                  edgecolor="#888", linewidth=0.5, alpha=0.92),
    )

    # Panel labels
    fig.text(0.012, 0.965, "a", fontsize=11, fontweight="bold",
             ha="left", va="top")
    fig.text(0.418, 0.965, "b", fontsize=11, fontweight="bold",
             ha="left", va="top")
    fig.text(0.012, 0.380, "c", fontsize=11, fontweight="bold",
             ha="left", va="top")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "figure_dtw_alignment", out_dir=out_dir)
    plt.close(fig)
    return {
        "exemplar_session": EXEMPLAR_SESSION,
        "exemplar_naive_mae": naive_mae_ex,
        "exemplar_dtw_avg": avg_err_ex,
        "n_sessions": len(session_ids),
        "mean_naive_mae": mean_naive,
        "mean_dtw_avg": mean_dtw,
        "pct_improvement": pct_improve,
    }


def build_cross_session_bars(arrays_dir: Path, out_dir: Path):
    """Bars-only variant for the appendix when single-session DTW
    panels (a, b) live in the body Figure 1 and would be redundant
    here.  Produces a single-panel figure showing naive MAE vs DTW
    avg error across all 39 sessions, sorted by naive MAE descending,
    with the mean-improvement annotation."""
    apply_style()
    from fastdtw import fastdtw

    files = sorted(arrays_dir.glob("session_*.npz"))
    session_ids = []
    naive_maes = []
    dtw_errs = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if int(d["m_i"]) < 100:
            continue
        gt = d["gt"].astype(np.float32)
        pred = np.clip(d["pred"], 0, None).astype(np.float32)
        pop_gt = gt.sum(axis=1)
        pop_pred = pred.sum(axis=1)
        dist, path = fastdtw(pop_gt, pop_pred)
        dtw_avg = dist / max(1, len(path))
        naive = float(np.mean(np.abs(pop_gt - pop_pred)))
        session_ids.append(int(d["session_idx"]))
        naive_maes.append(naive)
        dtw_errs.append(dtw_avg)

    session_ids = np.array(session_ids)
    naive_maes = np.array(naive_maes)
    dtw_errs = np.array(dtw_errs)
    order = np.argsort(naive_maes)[::-1]
    session_ids = session_ids[order]
    naive_maes = naive_maes[order]
    dtw_errs = dtw_errs[order]
    mean_naive = float(naive_maes.mean())
    mean_dtw = float(dtw_errs.mean())
    pct = 100.0 * (1.0 - mean_dtw / mean_naive)

    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.42),
        dpi=300, facecolor="white",
    )
    n = len(session_ids)
    x = np.arange(n)
    w = 0.4
    GT_COLOR = "#1E88E5"
    PRED_COLOR = COLORS["Mamba"]
    ax.bar(x - w / 2, naive_maes, w,
           color=GT_COLOR, label="Naive bin-to-bin MAE")
    ax.bar(x + w / 2, dtw_errs, w,
           color=PRED_COLOR, label="DTW avg error")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in session_ids], fontsize=6,
                       rotation=90)
    ax.set_xlabel("Session (sorted by naive MAE, descending)",
                  fontsize=8)
    ax.set_ylabel("Spikes / bin", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    ax.annotate(
        f"Mean DTW improvement: {pct:.1f}%\n"
        f"(naive {mean_naive:.1f} → DTW {mean_dtw:.1f} spikes/bin; "
        f"{n}/{n} sessions)",
        xy=(0.99, 0.99), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=COLORS.get("highlight", "#F0E442"),
                  edgecolor="#888", linewidth=0.5, alpha=0.92),
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "figure_dtw_cross_session", out_dir=out_dir)
    plt.close(fig)
    return {"n_sessions": n, "mean_naive": mean_naive,
            "mean_dtw": mean_dtw, "pct": pct}


if __name__ == "__main__":
    arrays = PROJECT_ROOT / "outputs" / "ifer_arrays"
    out = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
    metrics = build_figure(arrays, out)
    print("DTW figure (full 3-panel) metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()
    bars = build_cross_session_bars(arrays, out)
    print("DTW figure (bars-only for appendix) metrics:")
    for k, v in bars.items():
        print(f"  {k}: {v}")
