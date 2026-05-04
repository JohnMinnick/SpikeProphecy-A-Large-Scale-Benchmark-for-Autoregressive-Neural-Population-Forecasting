"""Combined hero figure: multizoom heatmap projection on top, DTW
population-rate temporal alignment on bottom — all anchored to the
same exemplar session (session 10, the highest mean per-neuron r).

Layout:
  (a) 5x5 grid of session pairs (recorded vs Mamba forecast)
  (b) Expanded view of the highlighted session (full probe)
  (c) 50-neuron, 2-second spike-level zoom
  ----
  (d) DTW pairwise distance matrix for the same session, warping
      path in white, naive diagonal in dashed grey.
  (e) Naive vs DTW-aligned population-rate overlay for the same
      session (top: bin-to-bin; bottom: with grey alignment links).

Pink projection lines connect (a)->(b); amber lines connect (b)->(c).
A teal strip on the right of (d)/(e) flags this as the same
exemplar session.

Output:
  docs/neurips_neurocog/figures/figure_hero_combined.{png,pdf}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS  # noqa: E402

# Pull data-loading helpers from the existing multizoom script.
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures"))
from figure_multizoom import (  # noqa: E402
    _load_session,
    _find_similar_zoom,
    _pick_most_similar_session,
)

# -----------------------------------------------------------------
# Style — match multizoom + DTW
# -----------------------------------------------------------------
GT_COLOR = "#1E88E5"           # Recorded (blue)
PRED_COLOR = COLORS["Mamba"]   # Mamba forecast (vermillion)
PROJ_A_TO_B = "#FF4081"        # pink: heatmap-grid -> session
PROJ_B_TO_C = "#FFB300"        # amber: session -> spike zoom
COLOR_LINK = "#888888"
COLOR_PATH = "#FFFFFF"
COLOR_DIAG = "#CCCCCC"
BG = "white"
CMAP = "magma"


def _draw_session_pair(ax_gt, ax_pred, gt, pred, vmax,
                       title_gt=None, title_pred=None):
    ax_gt.imshow(gt.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
                 interpolation="nearest")
    ax_pred.imshow(pred.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
                   interpolation="nearest")
    for a in (ax_gt, ax_pred):
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    if title_gt:
        ax_gt.set_title(title_gt, fontsize=6.5, color=GT_COLOR, pad=2,
                        fontweight="bold")
    if title_pred:
        ax_pred.set_title(title_pred, fontsize=6.5, color=PRED_COLOR,
                          pad=2, fontweight="bold")


def build_combined(arrays_dir: Path, out_dir: Path, grid_n: int = 5,
                   highlighted_session: int | None = None,
                   n_neurons_zoom: int = 50, n_bins_zoom: int = 40):
    apply_style()

    # ---- Auto-pick exemplar (same as multizoom default) ----
    if highlighted_session is None:
        highlighted_session, hi_r = _pick_most_similar_session(arrays_dir)
        print(f"  exemplar session: {highlighted_session} "
              f"(mean per-neuron r = {hi_r:.4f})")

    # ---- Load + select 25 sessions for grid ----
    files = sorted(arrays_dir.glob("session_*.npz"))
    sess_meta = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if int(d["m_i"]) < 100:
            continue
        sess_meta.append((int(d["session_idx"]),
                          float(np.nanmean(d["per_neuron_r"]))))
    sess_meta.sort(key=lambda x: -x[1])
    picked = [s[0] for s in sess_meta[: grid_n * grid_n]]
    if highlighted_session not in picked:
        picked[-1] = highlighted_session
    picked.sort()
    if highlighted_session in picked:
        picked.remove(highlighted_session)
        picked.insert(grid_n, highlighted_session)

    loaded = {}
    all_gt = []
    for si in picked:
        s = _load_session(si, arrays_dir)
        loaded[si] = s
        all_gt.append(s["gt"].flatten())
    vmax = max(float(np.percentile(np.concatenate(all_gt), 99)), 1.0)

    # ---- Compute DTW for exemplar ----
    # Important: use the *raw* predicted rates (not the Poisson-sampled
    # version that the heatmap panels use) so the DTW numbers match the
    # standalone DTW appendix figure.
    from fastdtw import fastdtw

    s_hi = loaded[highlighted_session]
    raw_npz = dict(np.load(arrays_dir /
                           f"session_{highlighted_session:03d}.npz",
                           allow_pickle=True))
    raw_pred = np.clip(raw_npz["pred"], 0, None).astype(np.float32)
    pop_gt = s_hi["gt"].sum(axis=1).astype(np.float32)
    pop_pred = raw_pred.sum(axis=1)
    dist_dtw, path_dtw = fastdtw(pop_gt, pop_pred)
    avg_err = dist_dtw / max(1, len(path_dtw))
    naive_mae = float(np.mean(np.abs(pop_gt - pop_pred)))
    pct_red = 100.0 * (1.0 - avg_err / naive_mae)
    T = len(pop_gt)
    D_cost = cdist(pop_pred[:, None], pop_gt[:, None], metric="euclidean")

    print(f"  DTW: T={T}, naive MAE={naive_mae:.2f}, "
          f"DTW avg={avg_err:.2f}, reduction={pct_red:.1f}%")

    # ---- Canvas ----
    # Width matches existing multizoom (TEXT_WIDTH * 1.35 ~= 8.78").
    # Height tuned so the figure fits on a single page with surrounding
    # text rather than getting deferred by LaTeX's float placement.
    # 5x5 grid ratio reduced from 2.30 -> 1.85 so the bottom DTW row
    # gets proportionally more vertical real estate.
    fig_w = TEXT_WIDTH * 1.35
    fig_h = TEXT_WIDTH * 1.18
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor=BG)

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.85, 1.00, 1.05],
        hspace=0.32,
        left=0.045, right=0.985,
        top=0.97, bottom=0.045,
    )

    # ============= Panel A: 5x5 grid =============
    gs_a = gridspec.GridSpecFromSubplotSpec(
        grid_n, grid_n,
        subplot_spec=outer[0],
        wspace=0.22, hspace=0.18,
    )
    highlighted_axes = None
    for i, si in enumerate(picked):
        row, col = divmod(i, grid_n)
        cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_a[row, col], wspace=0.04,
        )
        ax_gt = fig.add_subplot(cell[0, 0])
        ax_pred = fig.add_subplot(cell[0, 1])
        _draw_session_pair(
            ax_gt, ax_pred, loaded[si]["gt"], loaded[si]["pred"], vmax,
            title_gt="Recorded" if row == 0 else None,
            title_pred="Mamba forecast" if row == 0 else None,
        )
        if si == highlighted_session:
            highlighted_axes = (ax_gt, ax_pred)

    # ============= Panel B: expanded session =============
    # Panel C: spike-level zoom side-by-side
    gs_bc = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[1.15, 1.0], wspace=0.28,
    )

    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_bc[0], wspace=0.10,
    )
    ax_b_gt = fig.add_subplot(gs_b[0, 0])
    ax_b_pr = fig.add_subplot(gs_b[0, 1])

    gt_b = s_hi["gt"]
    pred_b = s_hi["pred"]
    T_b, M_b = gt_b.shape
    vmax_b = max(float(np.percentile(gt_b, 99)), 1.0)
    ax_b_gt.imshow(gt_b.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_b,
                   interpolation="nearest")
    ax_b_pr.imshow(pred_b.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_b,
                   interpolation="nearest")
    for a, label in [(ax_b_gt, "Recorded"), (ax_b_pr, "Mamba forecast")]:
        a.set_title(label, fontsize=8.5,
                    color=GT_COLOR if "Recorded" in label else PRED_COLOR,
                    fontweight="bold", pad=3)
        a.tick_params(labelsize=6.5, colors="#444")
        for sp in a.spines.values():
            sp.set_visible(False)
    ax_b_gt.set_ylabel("Neurons (probe depth)", fontsize=7, color="#333")
    T_to_s = 0.05
    ax_b_gt.set_xticks([0, T_b // 2, T_b - 1])
    ax_b_gt.set_xticklabels(
        [f"{0:.0f}", f"{T_b/2*T_to_s:.0f}", f"{T_b*T_to_s:.0f}"],
        fontsize=6.5,
    )
    ax_b_pr.set_xticks([0, T_b // 2, T_b - 1])
    ax_b_pr.set_xticklabels(
        [f"{0:.0f}", f"{T_b/2*T_to_s:.0f}", f"{T_b*T_to_s:.0f}"],
        fontsize=6.5,
    )
    ax_b_gt.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_b_pr.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_b_pr.set_yticks([])

    # Panel C: zoom
    t0, t1, n0, n1 = _find_similar_zoom(gt_b, pred_b, n_neurons_zoom,
                                        n_bins_zoom)
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_bc[1], wspace=0.10,
    )
    ax_c_gt = fig.add_subplot(gs_c[0, 0])
    ax_c_pr = fig.add_subplot(gs_c[0, 1])
    gt_c = gt_b[t0:t1, n0:n1]
    pred_c = pred_b[t0:t1, n0:n1]
    vmax_c = max(float(np.percentile(gt_c, 99)),
                 float(gt_c.max()) * 0.9, 2.0)
    ax_c_gt.imshow(gt_c.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_c,
                   interpolation="nearest")
    ax_c_pr.imshow(pred_c.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_c,
                   interpolation="nearest")
    for a, label in [(ax_c_gt, "Recorded"), (ax_c_pr, "Mamba forecast")]:
        a.set_title(label, fontsize=8.5,
                    color=GT_COLOR if "Recorded" in label else PRED_COLOR,
                    fontweight="bold", pad=3)
        a.tick_params(labelsize=6.5, colors="#444")
        for sp in a.spines.values():
            sp.set_visible(False)
    n_bins_shown = n_bins_zoom
    ts0 = t0 * T_to_s
    ts1 = t1 * T_to_s
    ax_c_gt.set_xticks([0, n_bins_shown / 2, n_bins_shown - 1])
    ax_c_gt.set_xticklabels(
        [f"{ts0:.1f}", f"{(ts0 + ts1)/2:.1f}", f"{ts1:.1f}"], fontsize=6.5,
    )
    ax_c_gt.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_c_pr.set_xticks([0, n_bins_shown / 2, n_bins_shown - 1])
    ax_c_pr.set_xticklabels(
        [f"{ts0:.1f}", f"{(ts0 + ts1)/2:.1f}", f"{ts1:.1f}"], fontsize=6.5,
    )
    ax_c_pr.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_c_gt.set_ylabel(f"Neurons {n0}–{n1-1}", fontsize=7, color="#333")
    ax_c_pr.set_yticks([])

    # ============= Panel D + E: DTW row =============
    gs_de = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.0, 1.5], wspace=0.40,
    )

    # ---- Panel D: Distance matrix with warping path ----
    # wspace + ax_d_left.margins below give the heatmap's tick labels
    # room without overlapping the orange predicted-rate marginal.
    gs_d = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs_de[0],
        width_ratios=[0.20, 1.0, 0.06],
        height_ratios=[0.18, 1.0],
        wspace=0.40, hspace=0.10,
    )
    ax_d_top = fig.add_subplot(gs_d[0, 1])
    ax_d_left = fig.add_subplot(gs_d[1, 0])
    ax_d_main = fig.add_subplot(gs_d[1, 1])
    ax_d_cbar = fig.add_subplot(gs_d[1, 2])

    ax_d_top.plot(np.arange(T), pop_gt, color=GT_COLOR, linewidth=1.0)
    ax_d_top.axis("off")
    ax_d_top.set_xlim(-0.5, T - 0.5)
    ax_d_left.plot(pop_pred, np.arange(T), color=PRED_COLOR, linewidth=1.0)
    # Explicit right-side padding (the rightmost point of the orange
    # trace was butting up against ax_d_main's tick labels even with
    # generous wspace).  Set the xlim with extra room on the right
    # side so the line tapers away from the heatmap edge.
    pp_min = float(pop_pred.min())
    pp_max = float(pop_pred.max())
    pp_pad = (pp_max - pp_min) * 0.18
    ax_d_left.set_xlim(pp_max + pp_pad * 0.2, pp_min - pp_pad)
    ax_d_left.axis("off")
    ax_d_left.set_ylim(-0.5, T - 0.5)

    im = ax_d_main.imshow(D_cost, aspect="auto", origin="lower",
                          cmap="viridis", interpolation="nearest")
    ax_d_main.plot([0, T - 1], [0, T - 1], color=COLOR_DIAG,
                   linestyle="--", linewidth=1.0, alpha=0.95)
    p_gt = [p[0] for p in path_dtw]
    p_pred = [p[1] for p in path_dtw]
    ax_d_main.plot(p_gt, p_pred, color=COLOR_PATH, linewidth=1.4)
    ax_d_main.set_xlim(-0.5, T - 0.5)
    ax_d_main.set_ylim(-0.5, T - 0.5)
    ax_d_main.set_xlabel("Time, recorded (bins)", fontsize=7.5)
    # NOTE: the y-axis label is placed via fig.text below (after draw)
    # because ax_d_main's natural y-label position would land on top of
    # ax_d_left's marginal predicted-rate trace (orange line).
    ax_d_main.tick_params(labelsize=6.5)
    cbar = fig.colorbar(im, cax=ax_d_cbar)
    cbar.set_label("Euclidean dist.", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    # ---- Panel E: warping-path overlay ----
    gs_e = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_de[1],
        height_ratios=[1.0, 1.0], hspace=0.20,
    )
    ax_e_top = fig.add_subplot(gs_e[0])
    ax_e_bot = fig.add_subplot(gs_e[1])

    ax_e_top.plot(pop_gt, color=GT_COLOR, linewidth=1.1, label="Recorded")
    ax_e_top.plot(pop_pred, color=PRED_COLOR, linewidth=1.1,
                  label="Mamba forecast")
    ax_e_top.set_title(
        f"Naive (bin-to-bin) MAE = {naive_mae:.1f} spikes/bin",
        fontsize=8, pad=3,
    )
    ax_e_top.set_ylabel("Population rate", fontsize=7.5)
    ax_e_top.tick_params(labelsize=6.5)
    ax_e_top.legend(frameon=False, fontsize=7, loc="upper right",
                    handlelength=1.2, ncol=2)
    ax_e_top.set_xticks([])
    for sp in ("top", "right"):
        ax_e_top.spines[sp].set_visible(False)

    ax_e_bot.plot(pop_gt, color=GT_COLOR, linewidth=1.1)
    ax_e_bot.plot(pop_pred, color=PRED_COLOR, linewidth=1.1)
    for k in range(0, len(path_dtw), 3):
        i, j = path_dtw[k]
        ax_e_bot.plot([i, j], [pop_gt[i], pop_pred[j]],
                      color=COLOR_LINK, alpha=0.30, linewidth=0.4)
    ax_e_bot.set_title(
        f"DTW-aligned avg error = {avg_err:.1f} spikes/bin "
        f"({pct_red:.0f}% reduction)",
        fontsize=8, pad=3,
    )
    ax_e_bot.set_xlabel("Time (50 ms bins)", fontsize=7.5)
    ax_e_bot.set_ylabel("Population rate", fontsize=7.5)
    ax_e_bot.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax_e_bot.spines[sp].set_visible(False)

    # ============= Projection overlays =============
    fig.canvas.draw()

    title_clearance = 0.018
    xlabel_clearance = 0.040
    pad = 0.004

    ax_a_left, ax_a_right = highlighted_axes
    pos_al = ax_a_left.get_position()
    pos_ar = ax_a_right.get_position()
    a_x0 = pos_al.x0
    a_x1 = pos_ar.x1
    a_y0 = pos_al.y0
    a_y1 = pos_al.y1

    rect_a = FancyBboxPatch(
        (a_x0 - pad, a_y0 - pad),
        (a_x1 - a_x0) + 2 * pad, (a_y1 - a_y0) + 2 * pad,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_A_TO_B,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_a)

    pos_bl = ax_b_gt.get_position()
    pos_br = ax_b_pr.get_position()
    b_x0 = pos_bl.x0 - pad
    b_x1 = pos_br.x1 + pad
    b_top = pos_bl.y1 + title_clearance
    b_bot = pos_bl.y0 - xlabel_clearance
    for x_frac in [0.0, 1.0]:
        x_a = (a_x0 - pad) + x_frac * ((a_x1 - a_x0) + 2 * pad)
        y_a = a_y0 - pad
        x_b = b_x0 + x_frac * (b_x1 - b_x0)
        line = mlines.Line2D(
            [x_a, x_b], [y_a, b_top],
            transform=fig.transFigure, clip_on=False,
            color=PROJ_A_TO_B, linewidth=1.0,
            linestyle="--", alpha=0.85,
        )
        fig.lines.append(line)

    rect_b = FancyBboxPatch(
        (b_x0, b_bot),
        b_x1 - b_x0, b_top - b_bot,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_A_TO_B,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_b)

    n_frac_start = n0 / M_b
    n_frac_end = n1 / M_b
    t_frac_start = t0 / T_b
    t_frac_end = t1 / T_b
    zb_x = pos_bl.x0 + t_frac_start * pos_bl.width
    zb_w = (t_frac_end - t_frac_start) * pos_bl.width
    zb_y = pos_bl.y0 + (1.0 - n_frac_end) * pos_bl.height
    zb_h = (n_frac_end - n_frac_start) * pos_bl.height
    rect_b_zoom = FancyBboxPatch(
        (zb_x, zb_y), zb_w, zb_h,
        boxstyle="round,pad=0.001",
        linewidth=1.5, edgecolor=PROJ_B_TO_C,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_b_zoom)

    pos_cl = ax_c_gt.get_position()
    pos_cr = ax_c_pr.get_position()
    c_top = pos_cl.y1 + title_clearance
    c_bot = pos_cl.y0 - xlabel_clearance
    c_xl = pos_cl.x0 - pad
    c_xr = pos_cr.x1 + pad
    for y_frac, y_c in [(1.0, c_top), (0.0, c_bot)]:
        x_from = zb_x + zb_w
        y_from = zb_y + y_frac * zb_h
        line = mlines.Line2D(
            [x_from, c_xl], [y_from, y_c],
            transform=fig.transFigure, clip_on=False,
            color=PROJ_B_TO_C, linewidth=1.0,
            linestyle="--", alpha=0.85,
        )
        fig.lines.append(line)
    rect_c = FancyBboxPatch(
        (c_xl, c_bot),
        c_xr - c_xl, c_top - c_bot,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_B_TO_C,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_c)

    # Panel labels (a. b. c. d. e. — period style matches the
    # add_panel_label convention used elsewhere in the paper).
    # Each label is positioned just above its panel's top edge using
    # the actual axes positions after layout, so labels stay anchored
    # to their panels rather than to figure-coordinate guesses.
    LABEL_GAP = 0.012
    LABEL_FS = 12

    # a. — above the top-left cell of the 5x5 grid
    grid_first_ax = fig.axes[0]  # top-left ax in panel a's grid
    pos_a_first = grid_first_ax.get_position()
    fig.text(0.012, pos_a_first.y1 + LABEL_GAP, "a.",
             fontsize=LABEL_FS, fontweight="bold",
             ha="left", va="bottom", color="#222")

    # b. — above panel b's expanded session
    pos_b = ax_b_gt.get_position()
    fig.text(0.012, pos_b.y1 + LABEL_GAP, "b.",
             fontsize=LABEL_FS, fontweight="bold",
             ha="left", va="bottom", color="#222")

    # c. — above panel c's spike zoom (small extra up-nudge to balance
    # the left shift)
    pos_c = ax_c_gt.get_position()
    fig.text(pos_c.x0 - 0.030, pos_c.y1 + LABEL_GAP + 0.005, "c.",
             fontsize=LABEL_FS, fontweight="bold",
             ha="left", va="bottom", color="#222")

    # d. — above panel d's top blue marginal trace
    pos_d_top_ax = ax_d_top.get_position()
    fig.text(0.012, pos_d_top_ax.y1 + LABEL_GAP, "d.",
             fontsize=LABEL_FS, fontweight="bold",
             ha="left", va="bottom", color="#222")

    # e. — above panel e's top trace
    pos_e_top = ax_e_top.get_position()
    fig.text(pos_e_top.x0 - 0.02, pos_e_top.y1 + LABEL_GAP, "e.",
             fontsize=LABEL_FS, fontweight="bold",
             ha="left", va="bottom", color="#222")

    # Custom y-label for panel (d): placed via fig.text so it sits to
    # the left of ax_d_left's orange marginal trace rather than
    # overlapping it (the natural ax_d_main.set_ylabel position would
    # land on the orange line).
    pos_d_main = ax_d_main.get_position()
    pos_d_left = ax_d_left.get_position()
    y_label_x = pos_d_left.x0 - 0.012
    y_label_y = (pos_d_main.y0 + pos_d_main.y1) / 2.0
    fig.text(
        y_label_x, y_label_y,
        "Time, Mamba forecast (bins)",
        fontsize=7.5, color="#333",
        ha="center", va="center", rotation=90,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "figure_hero_combined", out_dir=out_dir)
    plt.close(fig)
    return {
        "exemplar_session": highlighted_session,
        "naive_mae": naive_mae,
        "dtw_avg": avg_err,
        "pct_reduction": pct_red,
    }


if __name__ == "__main__":
    arrays = PROJECT_ROOT / "outputs" / "ifer_arrays"
    out = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
    metrics = build_combined(arrays, out)
    print("Combined hero metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
