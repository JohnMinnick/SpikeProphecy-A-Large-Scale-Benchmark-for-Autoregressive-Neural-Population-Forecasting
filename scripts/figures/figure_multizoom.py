"""Multi-zoom hero figure: multi-session grid \u2192 single session \u2192 spike-level.

Ports the IFER "multizoom" design to the neurocog paper framing. Labels are
"Recorded / Mamba forecast" instead of "Recorded / Twin prediction."

Uses:
  - outputs/ifer_arrays/session_NNN.npz (200-bin 10s Mamba predictions on 39 Steinmetz sessions)

Output:
  docs/neurips_neurocog/figure_candidates/F1_hero/v8_multizoom_mamba.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH  # noqa: E402


# Colors
GT_COLOR = "#1E88E5"         # blue for "Recorded"
PRED_COLOR = "#D55E00"       # vermillion for "Mamba forecast"
PROJ_A_TO_B = "#FF4081"      # magenta/pink for A\u2192B
PROJ_B_TO_C = "#FFB300"      # amber for B\u2192C
BG = "white"
CMAP = "magma"


def _load_session(session_idx: int, arrays_dir: Path):
    p = arrays_dir / f"session_{session_idx:03d}.npz"
    d = dict(np.load(p, allow_pickle=True))
    gt = d["gt"].astype(np.float32)            # (T, M)
    pred_rates = np.clip(d["pred"], 0, None).astype(np.float32)
    np.random.seed(42 + session_idx)
    pred = np.random.poisson(lam=pred_rates).astype(np.float32)
    m_i = int(d["m_i"])
    per_r = d["per_neuron_r"]
    return {"gt": gt, "pred": pred, "m_i": m_i, "per_r": per_r}


def _sort_neurons(gt, pred):
    """Identity — preserves physical probe depth order (NWB unit order)."""
    return gt, pred


def _draw_pair(ax_gt, ax_pred, gt, pred, vmax, show_axes=False,
               title_gt=None, title_pred=None):
    """Render (GT, pred) as side-by-side heatmaps on two axes."""
    ax_gt.imshow(
        gt.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
        interpolation="nearest",
    )
    ax_pred.imshow(
        pred.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
        interpolation="nearest",
    )
    for a in (ax_gt, ax_pred):
        if not show_axes:
            a.set_xticks([])
            a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
    if title_gt:
        ax_gt.set_title(title_gt, fontsize=6.5, color=GT_COLOR, pad=2)
    if title_pred:
        ax_pred.set_title(title_pred, fontsize=6.5, color=PRED_COLOR, pad=2)


def _find_active_zoom(gt, n_neurons=50, n_bins=40):
    """Find the most active (n_neurons, n_bins) window in gt (T, M)."""
    T, M = gt.shape
    rates = gt.mean(axis=0)
    best_n, best_s = 0, -1
    for ns in range(0, max(1, M - n_neurons + 1), max(1, n_neurons // 10)):
        s = rates[ns:ns + n_neurons].sum()
        if s > best_s:
            best_s, best_n = s, ns
    sub = gt[:, best_n:best_n + n_neurons]
    best_t, best_ts = 0, -1
    for ts in range(0, max(1, T - n_bins + 1)):
        s = sub[ts:ts + n_bins, :].sum()
        if s > best_ts:
            best_ts, best_t = s, ts
    return best_t, best_t + n_bins, best_n, best_n + n_neurons


def _find_similar_zoom(gt, pred, n_neurons=50, n_bins=40, min_spikes=20):
    """Find the contiguous (n_neurons, n_bins) window where gt and pred agree
    best by cosine similarity. Preserves depth ordering by searching over
    contiguous neuron slices.

    Args:
        gt, pred: (T, M) ground truth and prediction arrays.
        n_neurons, n_bins: window dimensions.
        min_spikes: skip windows with fewer than this many gt spikes
                    (avoids selecting an empty quiet window that scores 1.0).

    Returns: (t0, t1, n0, n1) bounding box.
    """
    T, M = gt.shape
    best_score = -np.inf
    best_n, best_t = 0, 0
    n_step = max(1, n_neurons // 10)
    t_step = max(1, n_bins // 10)
    for ns in range(0, max(1, M - n_neurons + 1), n_step):
        for ts in range(0, max(1, T - n_bins + 1), t_step):
            g = gt[ts:ts + n_bins, ns:ns + n_neurons].ravel()
            if g.sum() < min_spikes:
                continue
            p = pred[ts:ts + n_bins, ns:ns + n_neurons].ravel()
            denom = float(np.linalg.norm(g) * np.linalg.norm(p))
            if denom < 1e-9:
                continue
            score = float(np.dot(g, p) / denom)
            if score > best_score:
                best_score = score
                best_n, best_t = ns, ts
    return best_t, best_t + n_bins, best_n, best_n + n_neurons


def _pick_most_similar_session(arrays_dir: Path, min_neurons: int = 100):
    """Return the session index with the highest mean per-neuron Pearson r."""
    best_idx, best_r = None, -np.inf
    for f in sorted(arrays_dir.glob("session_*.npz")):
        d = dict(np.load(f, allow_pickle=True))
        if int(d["m_i"]) < min_neurons:
            continue
        r = float(np.nanmean(d["per_neuron_r"]))
        if r > best_r:
            best_r = r
            best_idx = int(d["session_idx"])
    return best_idx, best_r


def build_multizoom(
    arrays_dir: Path,
    out_dir: Path,
    grid_n: int = 5,
    highlighted_session: int | None = None,
    n_neurons_zoom: int = 50,
    n_bins_zoom: int = 40,
    zoom_mode: str = "similar",
):
    """Build the multi-zoom hero figure.

    Args:
        highlighted_session: session index to expand in panel B. If None,
            auto-pick the session with the highest mean per-neuron r
            (most similar between gt and Mamba prediction).
        zoom_mode: "similar" (default) picks the (n_neurons x n_bins)
            sub-rectangle of panel B with the highest local cosine
            similarity between gt and pred; "active" reproduces the
            original behavior (highest gt activity).
    """
    apply_style()

    # Auto-select the highlighted session as the one Mamba predicts best.
    if highlighted_session is None:
        highlighted_session, hi_r = _pick_most_similar_session(arrays_dir)
        print(
            f"  auto-picked highlighted session: {highlighted_session}"
            f" (mean per-neuron r = {hi_r:.4f})"
        )

    # Load all sessions with at least m_i neurons > 100
    files = sorted(arrays_dir.glob("session_*.npz"))
    sessions = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if int(d["m_i"]) < 100:
            continue
        idx = int(d["session_idx"])
        mean_r = float(np.nanmean(d["per_neuron_r"]))
        sessions.append((idx, mean_r))
    # Sort by r desc then take 25 and include the highlighted one
    sessions.sort(key=lambda x: -x[1])
    picked = [s[0] for s in sessions[: grid_n * grid_n]]
    if highlighted_session not in picked:
        picked[-1] = highlighted_session
    picked.sort()
    # Put highlighted at row 1 col 0 to match IFER Fig 2 visual design
    if highlighted_session in picked:
        picked.remove(highlighted_session)
        picked.insert(grid_n, highlighted_session)

    # Canvas
    fig_w, fig_h = TEXT_WIDTH * 1.35, TEXT_WIDTH * 1.05
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor=BG)

    # Outer layout: A on top, B+C below
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.3, 1.0],
        hspace=0.22,
        left=0.035, right=0.99,
        top=0.96, bottom=0.045,
    )

    # --- Panel A: 5x5 grid ---
    gs_a = gridspec.GridSpecFromSubplotSpec(
        grid_n, grid_n,
        subplot_spec=outer[0],
        wspace=0.22, hspace=0.18,
    )

    # Global vmax
    all_gt = []
    loaded = {}
    for si in picked:
        s = _load_session(si, arrays_dir)
        loaded[si] = s
        all_gt.append(s["gt"].flatten())
    vmax = float(np.percentile(np.concatenate(all_gt), 99))
    vmax = max(vmax, 1.0)

    # Track highlighted cell axes to compute its position later
    highlighted_axes = None

    for i, si in enumerate(picked):
        row, col = divmod(i, grid_n)
        # Sub-gridspec inside the cell for GT/Pred
        cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_a[row, col], wspace=0.04,
        )
        ax_gt = fig.add_subplot(cell[0, 0])
        ax_pred = fig.add_subplot(cell[0, 1])
        s = loaded[si]
        gt_s, pred_s = _sort_neurons(s["gt"], s["pred"])
        _draw_pair(ax_gt, ax_pred, gt_s, pred_s, vmax)
        if row == 0:
            ax_gt.set_title(
                "Recorded", fontsize=6.5, color=GT_COLOR, pad=2,
                fontweight="bold",
            )
            ax_pred.set_title(
                "Mamba forecast", fontsize=6.5, color=PRED_COLOR, pad=2,
                fontweight="bold",
            )
        if si == highlighted_session:
            highlighted_axes = (ax_gt, ax_pred)

    # --- Panel B: single session (the highlighted one, full view) ---
    # --- Panel C: spike-level zoom from B ---
    gs_bc = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[1.15, 1.0],
        wspace=0.28,
    )
    # B: 2 heatmaps side-by-side for full session
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_bc[0], wspace=0.10,
    )
    ax_b_gt = fig.add_subplot(gs_b[0, 0])
    ax_b_pr = fig.add_subplot(gs_b[0, 1])
    s_hi = loaded[highlighted_session]
    gt_b, pred_b = _sort_neurons(s_hi["gt"], s_hi["pred"])
    vmax_b = max(float(np.percentile(gt_b, 99)), 1.0)
    ax_b_gt.imshow(
        gt_b.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_b,
        interpolation="nearest",
    )
    ax_b_pr.imshow(
        pred_b.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_b,
        interpolation="nearest",
    )
    T_b, M_b = gt_b.shape
    # Axis labels
    for a, label in [(ax_b_gt, "Recorded"), (ax_b_pr, "Mamba forecast")]:
        a.set_title(
            label, fontsize=8.5,
            color=GT_COLOR if "Recorded" in label else PRED_COLOR,
            fontweight="bold", pad=3,
        )
        a.tick_params(labelsize=6.5, colors="#444")
        for spine in a.spines.values():
            spine.set_visible(False)
    ax_b_gt.set_ylabel(
        "Neurons (by probe depth)", fontsize=7, color="#333",
    )
    # x-axis in seconds
    T_to_s = 0.05  # 50 ms bins
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

    # C: spike-level zoom (50 neurons, 40 bins = 2 s) — pick the
    # sub-rectangle where Mamba and ground truth agree best (or fall
    # back to the most active region if requested).
    if zoom_mode == "similar":
        t0, t1, n0, n1 = _find_similar_zoom(
            gt_b, pred_b, n_neurons_zoom, n_bins_zoom,
        )
    else:
        t0, t1, n0, n1 = _find_active_zoom(gt_b, n_neurons_zoom, n_bins_zoom)
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_bc[1], wspace=0.10,
    )
    ax_c_gt = fig.add_subplot(gs_c[0, 0])
    ax_c_pr = fig.add_subplot(gs_c[0, 1])
    gt_c = gt_b[t0:t1, n0:n1]
    pred_c = pred_b[t0:t1, n0:n1]
    vmax_c = max(float(np.percentile(gt_c, 99)), float(gt_c.max()) * 0.9, 2.0)
    ax_c_gt.imshow(
        gt_c.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_c,
        interpolation="nearest",
    )
    ax_c_pr.imshow(
        pred_c.T, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax_c,
        interpolation="nearest",
    )
    for a, label in [(ax_c_gt, "Recorded"), (ax_c_pr, "Mamba forecast")]:
        a.set_title(
            label, fontsize=8.5,
            color=GT_COLOR if "Recorded" in label else PRED_COLOR,
            fontweight="bold", pad=3,
        )
        a.tick_params(labelsize=6.5, colors="#444")
        for spine in a.spines.values():
            spine.set_visible(False)
    n_bins_shown = n_bins_zoom
    t_sec_start = t0 * T_to_s
    t_sec_end = t1 * T_to_s
    ax_c_gt.set_xticks([0, n_bins_shown / 2, n_bins_shown - 1])
    ax_c_gt.set_xticklabels(
        [f"{t_sec_start:.1f}", f"{(t_sec_start + t_sec_end)/2:.1f}",
         f"{t_sec_end:.1f}"], fontsize=6.5,
    )
    ax_c_gt.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_c_pr.set_xticks([0, n_bins_shown / 2, n_bins_shown - 1])
    ax_c_pr.set_xticklabels(
        [f"{t_sec_start:.1f}", f"{(t_sec_start + t_sec_end)/2:.1f}",
         f"{t_sec_end:.1f}"], fontsize=6.5,
    )
    ax_c_pr.set_xlabel("Time (s)", fontsize=7, color="#333")
    ax_c_gt.set_ylabel(f"Neurons {n0}\u2013{n1-1}", fontsize=7, color="#333")
    ax_c_pr.set_yticks([])

    # --- Projection lines ---
    fig.canvas.draw()

    # Clearance above/below axes to route lines past title/xlabel text
    # (titles sit above y1; xlabels + tick labels sit below y0)
    title_clearance = 0.028
    xlabel_clearance = 0.045

    # Get positions of the highlighted A cell and Panel B axes
    ax_a_left, ax_a_right = highlighted_axes
    pos_a_left = ax_a_left.get_position()
    pos_a_right = ax_a_right.get_position()
    # Bounding box of the highlighted pair
    a_x0 = pos_a_left.x0
    a_x1 = pos_a_right.x1
    a_y0 = pos_a_left.y0
    a_y1 = pos_a_left.y1

    # Pink highlight box around A pair (tight — grid cells have no titles)
    pad = 0.004
    rect_a = FancyBboxPatch(
        (a_x0 - pad, a_y0 - pad),
        (a_x1 - a_x0) + 2 * pad, (a_y1 - a_y0) + 2 * pad,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_A_TO_B,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_a)

    # Projection lines: bottom of A-highlight -> top of B border
    # B border wraps around titles, so top of border is above titles.
    pos_b_left = ax_b_gt.get_position()
    pos_b_right = ax_b_pr.get_position()
    b_x0 = pos_b_left.x0 - pad
    b_x1 = pos_b_right.x1 + pad
    b_border_y_top = pos_b_left.y1 + title_clearance
    b_border_y_bot = pos_b_left.y0 - xlabel_clearance
    # Two lines (arriving at top-left and top-right corners of B border)
    for x_frac in [0.0, 1.0]:
        x_a = (a_x0 - pad) + x_frac * ((a_x1 - a_x0) + 2 * pad)
        y_a = a_y0 - pad
        x_b = b_x0 + x_frac * (b_x1 - b_x0)
        line = mlines.Line2D(
            [x_a, x_b], [y_a, b_border_y_top],
            transform=fig.transFigure, clip_on=False,
            color=PROJ_A_TO_B, linewidth=1.2,
            linestyle="--", alpha=0.85,
        )
        fig.lines.append(line)

    # Pink border around Panel B including titles above
    rect_b = FancyBboxPatch(
        (b_x0, b_border_y_bot),
        b_x1 - b_x0, b_border_y_top - b_border_y_bot,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_A_TO_B,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_b)

    # Amber highlight in B: the zoom region (time range × neuron range)
    # On ax_b_gt, the zoom is neurons n0:n1 out of M_b, time t0:t1 out of T_b
    n_frac_start = n0 / M_b
    n_frac_end = n1 / M_b
    t_frac_start = t0 / T_b
    t_frac_end = t1 / T_b
    # Use ax_b_gt position
    zb_x = pos_b_left.x0 + t_frac_start * pos_b_left.width
    zb_w = (t_frac_end - t_frac_start) * pos_b_left.width
    zb_y = pos_b_left.y0 + (1.0 - n_frac_end) * pos_b_left.height
    zb_h = (n_frac_end - n_frac_start) * pos_b_left.height
    rect_b_zoom = FancyBboxPatch(
        (zb_x, zb_y), zb_w, zb_h,
        boxstyle="round,pad=0.001",
        linewidth=1.5, edgecolor=PROJ_B_TO_C,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_b_zoom)

    # Projection lines from zoom rect (right side) to left edge of Panel C border
    pos_c_left = ax_c_gt.get_position()
    pos_c_right = ax_c_pr.get_position()
    # C border wraps titles + xlabels
    c_border_y_top = pos_c_left.y1 + title_clearance
    c_border_y_bot = pos_c_left.y0 - xlabel_clearance
    c_x_left = pos_c_left.x0 - pad
    c_x_right = pos_c_right.x1 + pad
    # Top-right of zoom box -> top-left of C border; bot-right -> bot-left of C border
    for y_frac, y_c in [(1.0, c_border_y_top), (0.0, c_border_y_bot)]:
        x_from = zb_x + zb_w
        y_from = zb_y + y_frac * zb_h
        line = mlines.Line2D(
            [x_from, c_x_left], [y_from, y_c],
            transform=fig.transFigure, clip_on=False,
            color=PROJ_B_TO_C, linewidth=1.2,
            linestyle="--", alpha=0.85,
        )
        fig.lines.append(line)

    # Amber border around Panel C including titles + xlabels
    rect_c = FancyBboxPatch(
        (c_x_left, c_border_y_bot),
        c_x_right - c_x_left,
        c_border_y_top - c_border_y_bot,
        boxstyle="round,pad=0.003",
        linewidth=2.0, edgecolor=PROJ_B_TO_C,
        facecolor="none", alpha=0.9,
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_c)

    # Panel labels (a, b, c)
    fig.text(
        0.01, 0.96, "a", fontsize=14, fontweight="bold",
        ha="left", va="top", color="#222",
    )
    fig.text(
        0.01, 0.33, "b", fontsize=14, fontweight="bold",
        ha="left", va="top", color="#222",
    )
    fig.text(
        0.535, 0.33, "c", fontsize=14, fontweight="bold",
        ha="left", va="top", color="#222",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "v8_multizoom_mamba", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    import shutil

    arrays = PROJECT_ROOT / "outputs" / "ifer_arrays"
    out = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figure_candidates" / "F1_hero"
    build_multizoom(arrays, out)
    print(f"Wrote {out / 'v8_multizoom_mamba.png'} and .pdf")

    # Mirror to the paper's figures directory under the canonical name.
    paper_figs = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
    paper_figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = out / f"v8_multizoom_mamba.{ext}"
        dst = paper_figs / f"figure_multizoom.{ext}"
        shutil.copyfile(src, dst)
        print(f"  copied -> {dst}")
